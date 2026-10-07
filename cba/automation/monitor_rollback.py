from __future__ import annotations

import json
import time
import uuid
from dataclasses import asdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from cba.automation.queue_manager import CloudQueueManager
from cba.core.models import ActionExecution, ActionType, CloudProvider, HealthCheck, HealthStatus, RecommendationStatus


class AutoRollbackMonitor:
    """Continuous Post-Optimization Health Monitoring and Automatic Rollback Engine."""

    def __init__(self, state_dir: str | Path = "outputs/live/actions"):
        self.state_dir = Path(state_dir)
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.queue_mgr = CloudQueueManager()

    @staticmethod
    def _utc() -> datetime:
        return datetime.now(timezone.utc)

    def _save_action(self, action: ActionExecution) -> None:
        file_path = self.state_dir / f"{action.action_id}.json"
        file_path.write_text(json.dumps(action.to_dict(), indent=2), encoding="utf-8")

    def get_action(self, action_id: str) -> Optional[Dict[str, Any]]:
        file_path = self.state_dir / f"{action_id}.json"
        if file_path.exists():
            return json.loads(file_path.read_text(encoding="utf-8"))
        return None

    def list_actions(self) -> List[Dict[str, Any]]:
        actions = []
        for p in self.state_dir.glob("*.json"):
            try:
                actions.append(json.loads(p.read_text(encoding="utf-8")))
            except Exception:
                pass
        return sorted(actions, key=lambda x: x.get("started_at", ""), reverse=True)

    def register_execution(
        self,
        provider: str,
        resource_id: str,
        action_type: str,
        target_size: str,
        original_state: Dict[str, Any],
        observation_minutes: int = 10,
        native_message_id: Optional[str] = None,
        cloud_config: Optional[Dict[str, Any]] = None,
    ) -> ActionExecution:
        """Registers a new dispatched optimization action and sets the observation deadline."""
        action_id = f"act-{uuid.uuid4().hex[:10]}"
        now = self._utc()
        deadline = now + timedelta(minutes=observation_minutes)

        prov = CloudProvider.AWS
        if "azure" in provider.lower():
            prov = CloudProvider.AZURE
        elif "gcp" in provider.lower():
            prov = CloudProvider.GCP

        execution = ActionExecution(
            action_id=action_id,
            recommendation_id=f"rec-{action_id}",
            started_at=now,
            completed_at=None,
            provider=prov,
            resource_id=resource_id,
            action=ActionType.SCALE_DOWN if "down" in action_type.lower() else ActionType.SCALE_UP,
            status=RecommendationStatus.OBSERVING,
            native_message_id=native_message_id,
            before_state=original_state,
            after_state={"target_size": target_size},
            observation_deadline=deadline,
            metadata={
                "observation_minutes": observation_minutes,
                "target_size": target_size,
                "cloud_config": cloud_config or {},
            },
        )
        self._save_action(execution)
        return execution

    def execute_cloud_resize(
        self,
        provider: str,
        resource_id: str,
        target_size: str,
        cloud_config: Dict[str, Any],
        action_type: str = "scale_down",
    ) -> Tuple[bool, str]:
        """Performs actual live cloud API mutation (resize, start, stop, EBS optimize) using production adapters."""
        prov = provider.lower()
        act = (action_type or "scale_down").lower()
        try:
            if "aws" in prov:
                from cba.automation.aws_adapter import AWSAutomationAdapter
                creds = cloud_config.get("aws_credentials") or cloud_config.get("credentials")
                region = cloud_config.get("region")
                if not region and creds and isinstance(creds, dict):
                    region = creds.get("region")
                if not region:
                    try:
                        from pathlib import Path
                        import pandas as pd
                        csv_p = Path("outputs/live/telemetry.csv")
                        if csv_p.exists():
                            tdf = pd.read_csv(csv_p)
                            match = tdf[tdf["resource_id"].astype(str) == str(resource_id)]
                            if not match.empty and "region" in match.columns:
                                region = str(match.iloc[0]["region"])
                    except Exception:
                        pass
                if not region:
                    region = os.environ.get("AWS_DEFAULT_REGION") or os.environ.get("AWS_REGION") or "ap-southeast-2"

                is_dry = bool(cloud_config.get("dry_run", False))
                adapter = AWSAutomationAdapter(credentials=creds, region=region, dry_run=is_dry)

                if "stop" in act:
                    res = adapter.stop_instance(instance_id=resource_id, dry_run=is_dry)
                    if res.get("success"):
                        return True, res.get("message", f"AWS EC2 instance {resource_id} successfully stopped.")
                    return False, res.get("error") or res.get("message", "AWS EC2 stop failed.")

                elif "start" in act:
                    res = adapter.start_instance(instance_id=resource_id, dry_run=is_dry)
                    if res.get("success"):
                        return True, res.get("message", f"AWS EC2 instance {resource_id} successfully started.")
                    return False, res.get("error") or res.get("message", "AWS EC2 start failed.")

                elif "storage" in act or "ebs" in act or "gp3" in act:
                    vol_type = target_size if ("gp" in target_size or "io" in target_size) else "gp3"
                    res = adapter.modify_ebs_volume(volume_id=resource_id, volume_type=vol_type, dry_run=is_dry)
                    if res.get("success"):
                        return True, res.get("message", f"AWS EBS volume {resource_id} optimization to {vol_type} submitted.")
                    return False, res.get("error") or res.get("message", "AWS EBS optimization failed.")

                else:
                    res = adapter.resize_instance(
                        instance_id=resource_id,
                        target_sku=target_size,
                        restart_after=True,
                        dry_run=is_dry,
                    )
                    if res.get("success"):
                        return True, res.get("message", f"AWS EC2 instance {resource_id} successfully resized to {target_size}.")
                    return False, res.get("error") or res.get("message", "AWS EC2 resize failed.")

            elif "azure" in prov:
                from azure.identity import ClientSecretCredential
                from azure.mgmt.compute import ComputeManagementClient

                cred = ClientSecretCredential(
                    tenant_id=cloud_config["tenant_id"],
                    client_id=cloud_config["client_id"],
                    client_secret=cloud_config["client_secret"],
                )
                compute = ComputeManagementClient(cred, cloud_config["subscription_id"])
                rg = cloud_config.get("resource_group", "default")
                compute.virtual_machines.begin_update(
                    resource_group_name=rg,
                    vm_name=resource_id,
                    parameters={"hardware_profile": {"vm_size": target_size}},
                ).result(timeout=180)
                return True, f"Azure VM {resource_id} successfully resized to {target_size}."

            elif "gcp" in prov or "google" in prov:
                from google.oauth2 import service_account
                from google.cloud import compute_v1

                info = json.loads(cloud_config["service_account_json"])
                credentials = service_account.Credentials.from_service_account_info(info)
                client = compute_v1.InstancesClient(credentials=credentials)
                project = cloud_config["project_id"]
                zone = cloud_config.get("zone", "us-central1-a")

                req = compute_v1.SetMachineTypeInstanceRequest(
                    project=project,
                    zone=zone,
                    instance=resource_id,
                    instances_set_machine_type_request_resource=compute_v1.InstancesSetMachineTypeRequest(
                        machine_type=f"zones/{zone}/machineTypes/{target_size}"
                    ),
                )
                client.set_machine_type(request=req)
                return True, f"GCP GCE VM {resource_id} successfully resized to {target_size}."

            return False, f"Unsupported provider: {provider}"

        except Exception as exc:
            return False, f"Cloud mutation failed: {exc}"

    def rollback_action(
        self,
        action_id: str,
        reason: str,
        cloud_config: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Immediately executes rollback to the exact previous state."""
        action_data = self.get_action(action_id)
        if not action_data:
            raise ValueError(f"Action {action_id} not found.")

        before_state = action_data.get("before_state", {})
        orig_size = before_state.get("instance_type") or before_state.get("vm_size") or before_state.get("machine_type")
        provider = action_data.get("provider", "aws")
        resource_id = action_data.get("resource_id", "")

        # Automatically fall back to persisted cloud_config if not supplied in call
        if not cloud_config:
            cloud_config = action_data.get("metadata", {}).get("cloud_config")

        # Execute cloud rollback if config supplied
        mutation_msg = "Rollback state updated."
        if cloud_config and orig_size:
            success, mutation_msg = self.execute_cloud_resize(provider, resource_id, orig_size, cloud_config)

        # Enqueue rollback message
        rollback_payload = {
            "event_type": "ROLLBACK_ACTION",
            "action_id": action_id,
            "resource_id": resource_id,
            "provider": provider,
            "restored_size": orig_size,
            "reason": reason,
            "timestamp": self._utc().isoformat(),
        }
        self.queue_mgr._log_local(rollback_payload, "OUTBOUND")

        # Update record
        action_data["status"] = RecommendationStatus.ROLLED_BACK.value
        action_data["completed_at"] = self._utc().isoformat()
        action_data["rollback_reason"] = reason
        action_data["metadata"]["rollback_note"] = mutation_msg

        (self.state_dir / f"{action_id}.json").write_text(json.dumps(action_data, indent=2), encoding="utf-8")
        return action_data

    def commit_permanently(self, action_id: str) -> Dict[str, Any]:
        """Marks optimization permanently committed when the observation window passes with no issues."""
        action_data = self.get_action(action_id)
        if not action_data:
            raise ValueError(f"Action {action_id} not found.")

        action_data["status"] = RecommendationStatus.COMMITTED_PERMANENTLY.value
        action_data["completed_at"] = self._utc().isoformat()

        commit_payload = {
            "event_type": "COMMIT_PERMANENT",
            "action_id": action_id,
            "resource_id": action_data.get("resource_id"),
            "provider": action_data.get("provider"),
            "final_size": action_data.get("after_state", {}).get("target_size"),
            "timestamp": self._utc().isoformat(),
        }
        self.queue_mgr._log_local(commit_payload, "OUTBOUND")

        (self.state_dir / f"{action_id}.json").write_text(json.dumps(action_data, indent=2), encoding="utf-8")
        return action_data

    def run_health_check_step(
        self,
        action_id: str,
        current_cpu: float = 0.0,
        current_mem: float = 0.0,
        cloud_config: Optional[Dict[str, Any]] = None,
        cpu_threshold: float = 85.0,
        observed_cpu: Optional[float] = None,
        observed_memory: Optional[float] = None,
        **kwargs: Any,
    ) -> Tuple[str, Optional[str]]:
        """Evaluates health during the observation window. Returns (status, rollback_reason_if_failed)."""
        if observed_cpu is not None:
            current_cpu = float(observed_cpu)
        if observed_memory is not None:
            current_mem = float(observed_memory)
        action_data = self.get_action(action_id)
        if not action_data:
            return "NOT_FOUND", "Action record missing."

        status = action_data.get("status")
        if status in [RecommendationStatus.COMMITTED_PERMANENTLY.value, RecommendationStatus.ROLLED_BACK.value]:
            return status, None

        if kwargs.get("force_complete") or kwargs.get("commit_early"):
            self.commit_permanently(action_id)
            return "COMMITTED_PERMANENTLY", None

        deadline_str = action_data.get("observation_deadline")
        now = self._utc()

        # Check 1: CPU spike / performance degradation
        if current_cpu > cpu_threshold:
            reason = f"Safety breach: CPU reached {current_cpu:.1f}%, exceeding {cpu_threshold:.1f}% threshold."
            self.rollback_action(action_id, reason=reason, cloud_config=cloud_config)
            return "ROLLED_BACK", reason

        # Check 2: Deadline reached without issues -> Commit permanently
        if deadline_str:
            deadline = datetime.fromisoformat(deadline_str)
            if now >= deadline:
                self.commit_permanently(action_id)
                return "COMMITTED_PERMANENTLY", None

        # Still within observation window
        return "OBSERVING", None
