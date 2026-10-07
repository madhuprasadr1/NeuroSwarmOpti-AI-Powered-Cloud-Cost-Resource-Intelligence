"""Production-oriented live cloud collection, recommendation, and dispatch.

Credentials are obtained only from each provider's standard credential chain.
Secrets are never accepted by the UI, persisted, or written to reports.
"""
from __future__ import annotations

import json
import os
import time
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable

import pandas as pd


SUPPORTED_PROVIDERS = {"aws", "azure", "gcp"}


class LiveCloudError(RuntimeError):
    """A real provider connection, authorization, or action failure."""


@dataclass
class ActionRecord:
    action_id: str
    provider: str
    resource_id: str
    operation: str
    status: str
    dispatched_at: str
    native_execution_id: str | None = None
    original_state: dict[str, Any] | None = None
    message: str = ""


class LiveCloudService:
    """Provider-neutral service for compute telemetry and native runbook dispatch.

    Automatic execution dispatches a pre-approved provider-native runbook. The
    runbook performs the mutation and consumes the supplied original state for
    rollback. This service never guesses a new VM SKU or stores credentials.
    """

    def __init__(self, state_dir: str | Path = "outputs/live") -> None:
        self.state_dir = Path(state_dir)
        self.state_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _provider(provider: str) -> str:
        value = provider.strip().lower()
        if value not in SUPPORTED_PROVIDERS:
            raise LiveCloudError("Provider must be one of: aws, azure, gcp.")
        return value

    @staticmethod
    def _utc() -> datetime:
        return datetime.now(timezone.utc)

    @staticmethod
    def _avg(datapoints: Iterable[dict[str, Any]], key: str = "Average") -> float | None:
        values = [float(p[key]) for p in datapoints if p.get(key) is not None]
        return round(sum(values) / len(values), 3) if values else None

    def collect_compute(self, provider: str, scope: str, region: str | None = None,
                        lookback_minutes: int = 30) -> pd.DataFrame:
        provider = self._provider(provider)
        if lookback_minutes < 5 or lookback_minutes > 1440:
            raise LiveCloudError("lookback_minutes must be between 5 and 1440.")
        if provider == "aws":
            rows = self._collect_aws(scope, region, lookback_minutes)
        elif provider == "azure":
            rows = self._collect_azure(scope, lookback_minutes)
        else:
            rows = self._collect_gcp(scope, region, lookback_minutes)
        frame = pd.DataFrame(rows)
        if frame.empty:
            return pd.DataFrame(columns=["provider", "resource_id", "resource_name", "region", "state", "instance_type", "cpu_usage", "network_in_bytes", "network_out_bytes", "collected_at"])
        return frame

    def _collect_aws(self, account_id: str, region: str | None, lookback: int) -> list[dict[str, Any]]:
        try:
            import boto3
        except ImportError as exc:
            raise LiveCloudError("Install requirements-cloud.txt to use AWS.") from exc
        session = boto3.Session(region_name=region or os.getenv("AWS_REGION") or os.getenv("AWS_DEFAULT_REGION"))
        sts = session.client("sts")
        identity = sts.get_caller_identity()
        real_account = str(identity["Account"])
        if account_id and account_id != real_account:
            raise LiveCloudError("AWS account scope does not match authenticated account.")
        active_region = session.region_name
        if not active_region:
            raise LiveCloudError("Set --region or AWS_REGION for AWS collection.")
        ec2 = session.client("ec2", region_name=active_region)
        cw = session.client("cloudwatch", region_name=active_region)
        end = self._utc(); start = end - timedelta(minutes=lookback)
        rows: list[dict[str, Any]] = []
        for page in ec2.get_paginator("describe_instances").paginate():
            for reservation in page.get("Reservations", []):
                for item in reservation.get("Instances", []):
                    instance_id = item["InstanceId"]
                    metrics = cw.get_metric_data(
                        MetricDataQueries=[
                            {"Id": "cpu", "MetricStat": {"Metric": {"Namespace": "AWS/EC2", "MetricName": "CPUUtilization", "Dimensions": [{"Name": "InstanceId", "Value": instance_id}]}, "Period": 300, "Stat": "Average"}, "ReturnData": True},
                            {"Id": "netin", "MetricStat": {"Metric": {"Namespace": "AWS/EC2", "MetricName": "NetworkIn", "Dimensions": [{"Name": "InstanceId", "Value": instance_id}]}, "Period": 300, "Stat": "Sum"}, "ReturnData": True},
                            {"Id": "netout", "MetricStat": {"Metric": {"Namespace": "AWS/EC2", "MetricName": "NetworkOut", "Dimensions": [{"Name": "InstanceId", "Value": instance_id}]}, "Period": 300, "Stat": "Sum"}, "ReturnData": True},
                        ], StartTime=start, EndTime=end, ScanBy="TimestampAscending"
                    )
                    values = {x["Id"]: self._avg([{"Average": v} for v in x.get("Values", [])]) for x in metrics.get("MetricDataResults", [])}
                    tags = {t["Key"]: t["Value"] for t in item.get("Tags", [])}
                    rows.append({"provider": "aws", "account_scope": real_account, "resource_id": instance_id, "resource_name": tags.get("Name", instance_id), "region": active_region, "state": item["State"]["Name"], "instance_type": item["InstanceType"], "cpu_usage": values.get("cpu"), "network_in_bytes": values.get("netin"), "network_out_bytes": values.get("netout"), "collected_at": end.isoformat(), "original_state": {"instance_type": item["InstanceType"], "state": item["State"]["Name"], "region": active_region}})
        return rows

    def _collect_azure(self, subscription_id: str, lookback: int) -> list[dict[str, Any]]:
        try:
            from azure.identity import DefaultAzureCredential
            from azure.mgmt.compute import ComputeManagementClient
            from azure.mgmt.monitor import MonitorManagementClient
        except ImportError as exc:
            raise LiveCloudError("Install requirements-cloud.txt to use Azure.") from exc
        credential = DefaultAzureCredential(exclude_interactive_browser_credential=True)
        compute = ComputeManagementClient(credential, subscription_id)
        monitor = MonitorManagementClient(credential, subscription_id)
        end = self._utc(); start = end - timedelta(minutes=lookback)
        rows: list[dict[str, Any]] = []
        for vm in compute.virtual_machines.list_all():
            resource_id = vm.id
            cpu = None
            try:
                result = monitor.metrics.list(resource_id, timespan=f"{start.isoformat()}/{end.isoformat()}", interval="PT5M", metricnames="Percentage CPU", aggregation="Average")
                points = [p for m in result.value for ts in m.timeseries for p in ts.data]
                cpu = self._avg([{"Average": p.average} for p in points if p.average is not None])
            except Exception:
                # A missing metric permission must not turn resource discovery into invented telemetry.
                cpu = None
            rows.append({"provider": "azure", "account_scope": subscription_id, "resource_id": resource_id, "resource_name": vm.name, "region": vm.location, "state": "unknown", "instance_type": getattr(vm.hardware_profile, "vm_size", None), "cpu_usage": cpu, "network_in_bytes": None, "network_out_bytes": None, "collected_at": end.isoformat(), "original_state": {"vm_size": getattr(vm.hardware_profile, "vm_size", None), "resource_group": resource_id.split("/")[4], "location": vm.location}})
        return rows

    def _collect_gcp(self, project_id: str, zone: str | None, lookback: int) -> list[dict[str, Any]]:
        try:
            from google.cloud import compute_v1, monitoring_v3
        except ImportError as exc:
            raise LiveCloudError("Install requirements-cloud.txt to use GCP.") from exc
        instances = compute_v1.InstancesClient(); monitoring = monitoring_v3.MetricServiceClient()
        end = self._utc(); start = end - timedelta(minutes=lookback)
        interval = monitoring_v3.TimeInterval(start_time={"seconds": int(start.timestamp())}, end_time={"seconds": int(end.timestamp())})
        rows: list[dict[str, Any]] = []
        for scoped in instances.aggregated_list(project=project_id):
            zone_name, scoped_list = scoped
            if not scoped_list.instances or (zone and not zone_name.endswith(zone)):
                continue
            for vm in scoped_list.instances:
                cpu = None
                try:
                    series = monitoring.list_time_series(request={"name": f"projects/{project_id}", "filter": f'metric.type="compute.googleapis.com/instance/cpu/utilization" AND resource.labels.instance_id="{vm.id}"', "interval": interval, "view": monitoring_v3.ListTimeSeriesRequest.TimeSeriesView.FULL})
                    values = [point.value.double_value * 100 for item in series for point in item.points]
                    cpu = round(sum(values) / len(values), 3) if values else None
                except Exception:
                    cpu = None
                actual_zone = vm.zone.rsplit("/", 1)[-1]
                rows.append({"provider": "gcp", "account_scope": project_id, "resource_id": str(vm.id), "resource_name": vm.name, "region": actual_zone, "state": vm.status, "instance_type": vm.machine_type.rsplit("/", 1)[-1], "cpu_usage": cpu, "network_in_bytes": None, "network_out_bytes": None, "collected_at": end.isoformat(), "original_state": {"machine_type": vm.machine_type, "zone": actual_zone, "status": vm.status}})
        return rows

    def recommend(self, telemetry: pd.DataFrame) -> pd.DataFrame:
        """Make conservative, explainable recommendations from real measurements only."""
        if telemetry.empty:
            return telemetry.assign(action=pd.Series(dtype=str), risk=pd.Series(dtype=str), explanation=pd.Series(dtype=str))
        result = telemetry.copy()
        def decision(row: pd.Series) -> tuple[str, str, str]:
            cpu = row.get("cpu_usage")
            if pd.isna(cpu):
                return "COLLECT_MORE", "low", "CPU metric is unavailable; no optimization is safe."
            if str(row.get("state", "")).lower() not in {"running", "succeeded"}:
                return "NO_ACTION", "low", "Resource is not in an active compute state."
            if float(cpu) < 10:
                return "RIGHTSIZE_CANDIDATE", "medium", f"Average CPU is {float(cpu):.1f}% across the selected live window; validate memory and application SLOs before resizing."
            if float(cpu) > 85:
                return "SCALE_UP_CANDIDATE", "high", f"Average CPU is {float(cpu):.1f}%, above the 85% safety threshold; approval is required."
            return "NO_ACTION", "low", f"Average CPU is {float(cpu):.1f}%, within the conservative operating envelope."
        values = result.apply(decision, axis=1, result_type="expand")
        result[["action", "risk", "explanation"]] = values
        result["approval_required"] = result["risk"].ne("low")
        return result

    def dispatch(self, recommendation: dict[str, Any], automatic: bool = False) -> ActionRecord:
        provider = self._provider(str(recommendation["provider"]))
        risk = str(recommendation.get("risk", "high")).lower()
        if automatic and risk != "low":
            raise LiveCloudError("Only low-risk actions can run automatically; this action requires explicit approval.")
        action_id = f"act-{uuid.uuid4().hex}"
        payload = {"action_id": action_id, "operation": recommendation.get("action"), "resource_id": recommendation["resource_id"], "original_state": recommendation.get("original_state", {}), "requested_at": self._utc().isoformat()}
        native_id = self._dispatch_native(provider, payload)
        record = ActionRecord(action_id, provider, payload["resource_id"], str(payload["operation"]), "DISPATCHED", payload["requested_at"], native_id, payload["original_state"], "Provider-native runbook dispatched.")
        self._save(record)
        self._notify(record, "action_dispatched")
        return record

    def rollback(self, action_id: str) -> ActionRecord:
        source = self._read(action_id)
        payload = {"action_id": action_id, "operation": "ROLLBACK", "resource_id": source["resource_id"], "original_state": source.get("original_state", {}), "requested_at": self._utc().isoformat()}
        native_id = self._dispatch_native(source["provider"], payload)
        record = ActionRecord(f"rb-{uuid.uuid4().hex}", source["provider"], source["resource_id"], "ROLLBACK", "DISPATCHED", payload["requested_at"], native_id, source.get("original_state", {}), "Rollback runbook dispatched.")
        self._save(record)
        self._notify(record, "rollback_dispatched")
        return record

    def _dispatch_native(self, provider: str, payload: dict[str, Any]) -> str:
        if provider == "aws":
            import boto3
            document = os.getenv("CLOUDOPT_AWS_SSM_DOCUMENT")
            if not document:
                raise LiveCloudError("Set CLOUDOPT_AWS_SSM_DOCUMENT to an approved SSM Automation document name.")
            response = boto3.client("ssm", region_name=os.getenv("AWS_REGION") or os.getenv("AWS_DEFAULT_REGION")).start_automation_execution(DocumentName=document, Parameters={"CloudOptPayload": [json.dumps(payload)]})
            return str(response["AutomationExecutionId"])
        if provider == "azure":
            import urllib.request
            webhook = os.getenv("CLOUDOPT_AZURE_AUTOMATION_WEBHOOK")
            if not webhook:
                raise LiveCloudError("Set CLOUDOPT_AZURE_AUTOMATION_WEBHOOK to an approved Azure Automation webhook.")
            request = urllib.request.Request(webhook, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"}, method="POST")
            with urllib.request.urlopen(request, timeout=20) as response:
                return response.headers.get("x-ms-request-id", "azure-webhook-dispatched")
        from google.cloud import pubsub_v1
        topic = os.getenv("CLOUDOPT_GCP_PUBSUB_TOPIC")
        if not topic:
            raise LiveCloudError("Set CLOUDOPT_GCP_PUBSUB_TOPIC to a fully-qualified approved Pub/Sub topic.")
        return pubsub_v1.PublisherClient().publish(topic, json.dumps(payload).encode(), source="cloudopt").result(timeout=30)

    def _notify(self, record: ActionRecord, event_type: str) -> None:
        """Publish AWS events to the configured SQS queue; other providers use their native dispatcher."""
        if record.provider != "aws":
            return
        queue_url = os.getenv("CLOUDOPT_AWS_SQS_QUEUE_URL")
        if not queue_url:
            return
        try:
            import boto3
            boto3.client("sqs", region_name=os.getenv("AWS_REGION") or os.getenv("AWS_DEFAULT_REGION")).send_message(
                QueueUrl=queue_url,
                MessageBody=json.dumps({"event_type": event_type, **asdict(record)}),
            )
        except Exception as exc:
            # An alert transport issue must be visible and must not be mistaken for a completed action.
            raise LiveCloudError(f"AWS SQS notification failed: {exc}") from exc

    def _save(self, record: ActionRecord) -> None:
        (self.state_dir / f"{record.action_id}.json").write_text(json.dumps(asdict(record), indent=2), encoding="utf-8")

    def _read(self, action_id: str) -> dict[str, Any]:
        path = self.state_dir / f"{action_id}.json"
        if not path.exists():
            raise LiveCloudError(f"Action record not found: {action_id}")
        return json.loads(path.read_text(encoding="utf-8"))

    def monitor(self, action_id: str, minutes: int = 15, interval_seconds: int = 300) -> list[dict[str, Any]]:
        if minutes < 5 or interval_seconds < 60:
            raise LiveCloudError("Monitoring requires at least 5 minutes and a 60 second interval.")
        source = self._read(action_id); deadline = time.monotonic() + minutes * 60; snapshots = []
        while time.monotonic() < deadline:
            frame = self.collect_compute(source["provider"], "", source.get("original_state", {}).get("region"), min(interval_seconds, 30))
            match = frame[frame["resource_id"].astype(str) == str(source["resource_id"])]
            if not match.empty:
                snapshots.append(match.iloc[0].to_dict())
                if source["provider"] == "aws":
                    self._notify(ActionRecord(action_id, "aws", source["resource_id"], "HEALTH_CHECK", "OBSERVED", self._utc().isoformat(), message="Post-action health metric collected."), "health_observed")
                cpu = match.iloc[0].get("cpu_usage")
                if cpu is not None and not pd.isna(cpu) and float(cpu) > 90:
                    self.rollback(action_id)
                    raise LiveCloudError("Post-action CPU exceeded 90%; rollback was dispatched.")
            time.sleep(interval_seconds)
        return snapshots
