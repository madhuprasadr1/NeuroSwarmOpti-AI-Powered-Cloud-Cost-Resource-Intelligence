from __future__ import annotations

import copy
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from cba.core.models import (
    ActionExecution,
    ActionType,
    CloudProvider,
    HealthCheck,
    HealthStatus,
    OptimizationRecommendation,
    RecommendationStatus,
    RiskLevel,
)


class AutomationEngine:
    def __init__(
        self,
        dry_run: bool = True,
        require_approval: bool = True,
        verify_health: bool = True,
        allow_high_risk: bool = False,
    ):
        self.dry_run = bool(dry_run)
        self.require_approval = bool(require_approval)
        self.verify_health = bool(verify_health)
        self.allow_high_risk = bool(allow_high_risk)
        self.execution_history: List[Dict[str, Any]] = []
        self.rollback_store: Dict[str, Dict[str, Any]] = {}

    def _now(self) -> str:
        return datetime.now(timezone.utc).isoformat()

    def _get(
        self,
        obj: Any,
        key: str,
        default: Any = None,
    ) -> Any:
        if obj is None:
            return default

        if isinstance(obj, dict):
            return obj.get(key, default)

        return getattr(obj, key, default)

    def _enum_value(
        self,
        value: Any,
        default: str = "",
    ) -> str:
        if value is None:
            return default

        if hasattr(value, "value"):
            return str(value.value)

        return str(value)

    def _provider(
        self,
        recommendation: Any,
    ) -> CloudProvider:
        provider = self._get(
            recommendation,
            "provider",
            CloudProvider.SIMULATION,
        )

        if isinstance(provider, CloudProvider):
            return provider

        normalized = str(provider).lower().strip()

        mapping = {
            "aws": CloudProvider.AWS,
            "azure": CloudProvider.AZURE,
            "gcp": CloudProvider.GCP,
            "google": CloudProvider.GCP,
            "simulation": CloudProvider.SIMULATION,
        }

        return mapping.get(
            normalized,
            CloudProvider.SIMULATION,
        )

    def _action(
        self,
        recommendation: Any,
    ) -> ActionType:
        action = self._get(
            recommendation,
            "action",
            ActionType.NO_ACTION,
        )

        if isinstance(action, ActionType):
            return action

        try:
            return ActionType(str(action))
        except ValueError:
            return ActionType.NO_ACTION

    def _risk_level(
        self,
        recommendation: Any,
    ) -> RiskLevel:
        risk = self._get(
            recommendation,
            "risk_level",
            RiskLevel.MEDIUM,
        )

        if isinstance(risk, RiskLevel):
            return risk

        try:
            return RiskLevel(str(risk))
        except ValueError:
            return RiskLevel.MEDIUM

    def _confidence(
        self,
        recommendation: Any,
    ) -> float:
        try:
            value = float(
                self._get(
                    recommendation,
                    "confidence",
                    0.0,
                )
            )
        except (TypeError, ValueError):
            value = 0.0

        if value > 1:
            value /= 100.0

        return max(
            0.0,
            min(
                1.0,
                value,
            ),
        )

    def _resource_id(
        self,
        recommendation: Any,
    ) -> str:
        value = self._get(
            recommendation,
            "resource_id",
        )

        if value:
            return str(value)

        return "unknown-resource"

    def _execution_id(self) -> str:
        return f"exec-{uuid.uuid4().hex[:16]}"

    def _rollback_id(self) -> str:
        return f"rollback-{uuid.uuid4().hex[:16]}"

    def validate_recommendation(
        self,
        recommendation: OptimizationRecommendation,
    ) -> Dict[str, Any]:
        errors: List[str] = []
        warnings: List[str] = []

        action = self._action(
            recommendation
        )

        provider = self._provider(
            recommendation
        )

        risk = self._risk_level(
            recommendation
        )

        confidence = self._confidence(
            recommendation
        )

        resource_id = self._resource_id(
            recommendation
        )

        status = self._enum_value(
            self._get(
                recommendation,
                "status",
            )
        )

        if not resource_id or resource_id == "unknown-resource":
            errors.append(
                "Resource identifier is missing."
            )

        if action == ActionType.NO_ACTION:
            errors.append(
                "Recommendation contains no executable action."
            )

        if status in {
            RecommendationStatus.REJECTED.value,
            RecommendationStatus.FAILED.value,
            RecommendationStatus.ROLLED_BACK.value,
        }:
            errors.append(
                f"Recommendation status is {status}."
            )

        if confidence < 0.60:
            errors.append(
                "Model confidence is below the automation threshold."
            )

        if risk == RiskLevel.CRITICAL:
            errors.append(
                "Critical-risk recommendations cannot be automated."
            )

        if risk == RiskLevel.HIGH and not self.allow_high_risk:
            errors.append(
                "High-risk automation is disabled."
            )

        if provider == CloudProvider.SIMULATION:
            warnings.append(
                "Simulation provider detected; no real cloud mutation will occur."
            )

        if action in {
            ActionType.DELETE,
            ActionType.STOP,
        }:
            warnings.append(
                "Destructive or availability-affecting action requires additional verification."
            )

        return {
            "valid": len(errors) == 0,
            "errors": errors,
            "warnings": warnings,
            "resource_id": resource_id,
            "provider": provider.value,
            "action": action.value,
            "risk_level": risk.value,
            "confidence": confidence,
        }

    def approve(
        self,
        recommendation: OptimizationRecommendation,
    ) -> OptimizationRecommendation:
        if isinstance(
            recommendation,
            OptimizationRecommendation,
        ):
            recommendation.status = (
                RecommendationStatus.APPROVED
            )

        return recommendation

    def reject(
        self,
        recommendation: OptimizationRecommendation,
    ) -> OptimizationRecommendation:
        if isinstance(
            recommendation,
            OptimizationRecommendation,
        ):
            recommendation.status = (
                RecommendationStatus.REJECTED
            )

        return recommendation

    def _capture_state(
        self,
        resource: Optional[Dict[str, Any]],
        recommendation: OptimizationRecommendation,
    ) -> Dict[str, Any]:
        resource_state = copy.deepcopy(
            resource or {}
        )

        return {
            "rollback_id": self._rollback_id(),
            "resource_id": self._resource_id(
                recommendation
            ),
            "provider": self._provider(
                recommendation
            ).value,
            "captured_at": self._now(),
            "original_state": resource_state,
            "recommendation": self._serialize(
                recommendation
            ),
        }

    def _serialize(
        self,
        value: Any,
    ) -> Any:
        if value is None:
            return None

        if hasattr(value, "to_dict"):
            return value.to_dict()

        if hasattr(value, "value"):
            return value.value

        if isinstance(value, dict):
            return {
                str(key): self._serialize(item)
                for key, item in value.items()
            }

        if isinstance(value, list):
            return [
                self._serialize(item)
                for item in value
            ]

        if isinstance(value, tuple):
            return [
                self._serialize(item)
                for item in value
            ]

        return value

    def _simulated_mutation(
        self,
        resource: Dict[str, Any],
        recommendation: OptimizationRecommendation,
    ) -> Dict[str, Any]:
        state = copy.deepcopy(
            resource or {}
        )

        action = self._action(
            recommendation
        )

        current_vcpu = state.get(
            "vcpu",
            state.get(
                "vCPU",
                4,
            ),
        )

        current_ram = state.get(
            "ram_gb",
            state.get(
                "RAM_GB",
                8.0,
            ),
        )

        current_price = state.get(
            "price_per_hour",
            0.10,
        )

        try:
            current_vcpu = max(
                1,
                int(current_vcpu),
            )
        except (TypeError, ValueError):
            current_vcpu = 4

        try:
            current_ram = max(
                1.0,
                float(current_ram),
            )
        except (TypeError, ValueError):
            current_ram = 8.0

        try:
            current_price = max(
                0.001,
                float(current_price),
            )
        except (TypeError, ValueError):
            current_price = 0.10

        if action == ActionType.SCALE_DOWN:
            new_vcpu = max(
                1,
                int(round(current_vcpu * 0.50)),
            )

            new_ram = max(
                1.0,
                current_ram * 0.50,
            )

            new_price = current_price * 0.55

            state["vcpu"] = new_vcpu
            state["vCPU"] = new_vcpu
            state["ram_gb"] = new_ram
            state["RAM_GB"] = new_ram
            state["price_per_hour"] = new_price

        elif action == ActionType.SCALE_UP:
            new_vcpu = max(
                current_vcpu + 1,
                int(round(current_vcpu * 1.50)),
            )

            new_ram = max(
                current_ram + 1.0,
                current_ram * 1.50,
            )

            new_price = current_price * 1.45

            state["vcpu"] = new_vcpu
            state["vCPU"] = new_vcpu
            state["ram_gb"] = new_ram
            state["RAM_GB"] = new_ram
            state["price_per_hour"] = new_price

        elif action == ActionType.RIGHTSIZING:
            new_vcpu = max(
                1,
                int(round(current_vcpu * 0.70)),
            )

            new_ram = max(
                1.0,
                current_ram * 0.70,
            )

            new_price = current_price * 0.72

            state["vcpu"] = new_vcpu
            state["vCPU"] = new_vcpu
            state["ram_gb"] = new_ram
            state["RAM_GB"] = new_ram
            state["price_per_hour"] = new_price

        elif action == ActionType.STORAGE_OPTIMIZATION:
            state["storage_optimized"] = True
            state["price_per_hour"] = (
                current_price * 0.75
            )

        elif action == ActionType.RESERVED_CAPACITY:
            state["reserved_capacity"] = True
            state["price_per_hour"] = (
                current_price * 0.82
            )

        elif action == ActionType.SPOT:
            state["spot_capacity"] = True
            state["price_per_hour"] = (
                current_price * 0.35
            )

        elif action == ActionType.STOP:
            state["active"] = False
            state["price_per_hour"] = 0.0

        elif action == ActionType.START:
            state["active"] = True

        elif action == ActionType.DELETE:
            state["deleted"] = True
            state["active"] = False
            state["price_per_hour"] = 0.0

        elif action == ActionType.SCHEDULE:
            state["scheduled_optimization"] = True
            state["price_per_hour"] = (
                current_price * 0.65
            )

        return state

    def _health_metrics(
        self,
        before: Dict[str, Any],
        after: Dict[str, Any],
    ) -> Dict[str, float]:
        def number(
            value: Any,
            default: float = 0.0,
        ) -> float:
            try:
                result = float(value)

                if result != result:
                    return default

                return result
            except (TypeError, ValueError):
                return default

        before_cpu = number(
            before.get(
                "cpu_usage",
                before.get(
                    "utilization",
                    0.0,
                ),
            )
        )

        after_cpu = number(
            after.get(
                "cpu_usage",
                after.get(
                    "utilization",
                    before_cpu,
                ),
            )
        )

        if before_cpu > 1:
            before_cpu /= 100.0

        if after_cpu > 1:
            after_cpu /= 100.0

        before_latency = number(
            before.get(
                "latency_ms",
                100.0,
            ),
            100.0,
        )

        after_latency = number(
            after.get(
                "latency_ms",
                before_latency,
            ),
            before_latency,
        )

        before_throughput = number(
            before.get(
                "throughput",
                100.0,
            ),
            100.0,
        )

        after_throughput = number(
            after.get(
                "throughput",
                before_throughput,
            ),
            before_throughput,
        )

        return {
            "before_cpu": before_cpu,
            "after_cpu": after_cpu,
            "before_latency": before_latency,
            "after_latency": after_latency,
            "before_throughput": before_throughput,
            "after_throughput": after_throughput,
        }

    def verify_health(
        self,
        resource: Dict[str, Any],
        previous_resource: Optional[
            Dict[str, Any]
        ] = None,
        recommendation: Optional[
            OptimizationRecommendation
        ] = None,
    ) -> HealthCheck:
        previous = previous_resource or {}

        metrics = self._health_metrics(
            previous,
            resource,
        )

        cpu = metrics["after_cpu"]
        latency = metrics["after_latency"]
        throughput = metrics["after_throughput"]

        issues = []

        if cpu >= 0.95:
            issues.append(
                "CPU utilization is critically high."
            )

        if latency >= 250:
            issues.append(
                "Latency exceeds the operational threshold."
            )

        before_throughput = metrics[
            "before_throughput"
        ]

        if (
            before_throughput > 0
            and throughput
            < before_throughput * 0.70
        ):
            issues.append(
                "Throughput decreased materially."
            )

        if issues:
            status = HealthStatus.DEGRADED
        else:
            status = HealthStatus.HEALTHY

        resource_id = (
            self._resource_id(
                recommendation
            )
            if recommendation is not None
            else str(
                resource.get(
                    "resource_id",
                    "unknown-resource",
                )
            )
        )

        return HealthCheck(
            resource_id=resource_id,
            status=status,
            cpu_utilization=cpu,
            memory_utilization=float(
                resource.get(
                    "memory_usage",
                    0.0,
                )
            )
            / (
                100.0
                if float(
                    resource.get(
                        "memory_usage",
                        0.0,
                    )
                )
                > 1
                else 1.0
            ),
            latency_ms=latency,
            throughput=throughput,
            issues=issues,
            checked_at=self._now(),
        )

    def _execution(
        self,
        recommendation: OptimizationRecommendation,
        status: RecommendationStatus,
        success: bool,
        message: str,
        health: Optional[
            HealthCheck
        ] = None,
        rollback_id: Optional[str] = None,
    ) -> ActionExecution:
        return ActionExecution(
            execution_id=self._execution_id(),
            recommendation_id=str(
                self._get(
                    recommendation,
                    "recommendation_id",
                    "",
                )
            ),
            resource_id=self._resource_id(
                recommendation
            ),
            provider=self._provider(
                recommendation
            ),
            action=self._action(
                recommendation
            ),
            status=status,
            success=success,
            message=message,
            health_status=(
                health.status
                if health is not None
                else HealthStatus.UNKNOWN
            ),
            rollback_id=rollback_id,
            executed_at=self._now(),
        )

    def execute(
        self,
        recommendation: OptimizationRecommendation,
        resource: Optional[
            Dict[str, Any]
        ] = None,
        force: bool = False,
    ) -> ActionExecution:
        validation = self.validate_recommendation(
            recommendation
        )

        if not validation["valid"] and not force:
            execution = self._execution(
                recommendation,
                RecommendationStatus.FAILED,
                False,
                "; ".join(
                    validation["errors"]
                ),
            )

            self.execution_history.append(
                self._serialize(
                    execution
                )
            )

            return execution

        action = self._action(
            recommendation
        )

        provider = self._provider(
            recommendation
        )

        previous_state = copy.deepcopy(
            resource or {}
        )

        rollback_state = self._capture_state(
            resource,
            recommendation,
        )

        rollback_id = rollback_state[
            "rollback_id"
        ]

        self.rollback_store[
            rollback_id
        ] = rollback_state

        if self.require_approval and not force:
            status = self._enum_value(
                self._get(
                    recommendation,
                    "status",
                )
            )

            if status != RecommendationStatus.APPROVED.value:
                execution = self._execution(
                    recommendation,
                    RecommendationStatus.FAILED,
                    False,
                    "Manual approval is required before execution.",
                    rollback_id=rollback_id,
                )

                self.execution_history.append(
                    self._serialize(
                        execution
                    )
                )

                return execution

        if self.dry_run or provider == CloudProvider.SIMULATION:
            simulated_state = (
                self._simulated_mutation(
                    resource or {},
                    recommendation,
                )
            )

            health = self.verify_health(
                simulated_state,
                previous_state,
                recommendation,
            )

            if (
                self.verify_health
                and health.status
                != HealthStatus.HEALTHY
            ):
                execution = self._execution(
                    recommendation,
                    RecommendationStatus.FAILED,
                    False,
                    "Dry-run action failed health verification.",
                    health=health,
                    rollback_id=rollback_id,
                )

                self.execution_history.append(
                    self._serialize(
                        execution
                    )
                )

                return execution

            execution = self._execution(
                recommendation,
                RecommendationStatus.SIMULATED,
                True,
                f"Action {action.value} simulated successfully.",
                health=health,
                rollback_id=rollback_id,
            )

            self.execution_history.append(
                self._serialize(
                    execution
                )
            )

            return execution

        execution = self._execute_provider_action(
            recommendation,
            resource or {},
            rollback_id,
        )

        if execution.success:
            execution.status = (
                RecommendationStatus.EXECUTED
            )

        self.execution_history.append(
            self._serialize(
                execution
            )
        )

        return execution

    def _execute_provider_action(
        self,
        recommendation: OptimizationRecommendation,
        resource: Dict[str, Any],
        rollback_id: str,
    ) -> ActionExecution:
        provider = self._provider(
            recommendation
        )

        action = self._action(
            recommendation
        )

        provider_method = {
            CloudProvider.AWS: self._execute_aws,
            CloudProvider.AZURE: self._execute_azure,
            CloudProvider.GCP: self._execute_gcp,
        }.get(provider)

        if provider_method is None:
            return self._execution(
                recommendation,
                RecommendationStatus.FAILED,
                False,
                "Unsupported cloud provider.",
                rollback_id=rollback_id,
            )

        try:
            return provider_method(
                recommendation,
                resource,
                rollback_id,
            )
        except Exception as exc:
            return self._execution(
                recommendation,
                RecommendationStatus.FAILED,
                False,
                f"Provider execution failed: {exc}",
                rollback_id=rollback_id,
            )

    def _execute_aws(
        self,
        recommendation: OptimizationRecommendation,
        resource: Dict[str, Any],
        rollback_id: str,
    ) -> ActionExecution:
        action = self._action(recommendation)
        resource_id = self._resource_id(recommendation)

        try:
            from cba.automation.aws_adapter import AWSAutomationAdapter
        except ImportError:
            return self._execution(
                recommendation,
                RecommendationStatus.FAILED,
                False,
                "AWS automation adapter could not be imported.",
                rollback_id=rollback_id,
            )

        region = resource.get("region") or getattr(recommendation, "region", None) or "us-east-1"
        creds = resource.get("credentials") or resource.get("aws_credentials")

        try:
            adapter = AWSAutomationAdapter(
                credentials=creds,
                region=str(region),
                dry_run=self.dry_run,
            )

            # 1. Power Operations
            if action == ActionType.START:
                res = adapter.start_instance(resource_id, dry_run=self.dry_run)
                if not res.get("success"):
                    return self._execution(recommendation, RecommendationStatus.FAILED, False, res.get("error", "Failed to start instance"), rollback_id=rollback_id)
                return self._execution(recommendation, RecommendationStatus.EXECUTED, True, res.get("message", f"AWS instance {resource_id} started."), rollback_id=rollback_id)

            if action == ActionType.STOP:
                res = adapter.stop_instance(resource_id, dry_run=self.dry_run)
                if not res.get("success"):
                    return self._execution(recommendation, RecommendationStatus.FAILED, False, res.get("error", "Failed to stop instance"), rollback_id=rollback_id)
                return self._execution(recommendation, RecommendationStatus.EXECUTED, True, res.get("message", f"AWS instance {resource_id} stopped."), rollback_id=rollback_id)

            # 2. Resizing / Rightsizing Operations
            if action in {ActionType.SCALE_DOWN, ActionType.SCALE_UP, ActionType.RIGHTSIZING}:
                details = getattr(recommendation, "details", {}) or {}
                meta = getattr(recommendation, "metadata", {}) or {}
                target_sku = (
                    getattr(recommendation, "recommended_size", None)
                    or (details.get("target_sku") if isinstance(details, dict) else None)
                    or (details.get("target_size") if isinstance(details, dict) else None)
                    or (meta.get("target_sku") if isinstance(meta, dict) else None)
                    or (meta.get("target_size") if isinstance(meta, dict) else None)
                    or resource.get("target_sku")
                    or resource.get("target_size")
                )
                if not target_sku:
                    # Resolve target SKU from resource model catalog
                    from cba.intelligence.resource_model import ResourceIntelligenceModel
                    rm = ResourceIntelligenceModel()
                    curr_sku = resource.get("instance_type", "t3.medium")
                    target_sku, _, _ = rm.suggest_target_sku("aws", curr_sku, action.value)

                res = adapter.resize_instance(
                    instance_id=resource_id,
                    target_sku=target_sku,
                    restart_after=True,
                    dry_run=self.dry_run,
                )

                if not res.get("success"):
                    return self._execution(
                        recommendation,
                        RecommendationStatus.FAILED,
                        False,
                        res.get("error") or res.get("message", "Resize operation failed."),
                        rollback_id=rollback_id,
                    )

                stat = RecommendationStatus.SIMULATED if self.dry_run else RecommendationStatus.EXECUTED
                return self._execution(
                    recommendation,
                    stat,
                    True,
                    res.get("message", f"AWS instance {resource_id} resized to {target_sku}."),
                    rollback_id=rollback_id,
                )

            # 3. Storage Optimization
            if action == ActionType.STORAGE_OPTIMIZATION:
                vol_id = resource.get("volume_id")
                if not vol_id:
                    # Look up attached volume
                    inst_det = adapter.get_instance_details(resource_id)
                    vols = inst_det.get("volumes", [])
                    vol_id = vols[0].get("volume_id") if vols else None

                if not vol_id:
                    return self._execution(recommendation, RecommendationStatus.FAILED, False, "No EBS volume found to optimize.", rollback_id=rollback_id)

                res = adapter.modify_ebs_volume(volume_id=vol_id, volume_type="gp3", dry_run=self.dry_run)
                if not res.get("success"):
                    return self._execution(recommendation, RecommendationStatus.FAILED, False, res.get("error", "EBS modification failed."), rollback_id=rollback_id)
                stat = RecommendationStatus.SIMULATED if self.dry_run else RecommendationStatus.EXECUTED
                return self._execution(recommendation, stat, True, res.get("message", f"EBS volume {vol_id} optimized to gp3."), rollback_id=rollback_id)

            return self._execution(
                recommendation,
                RecommendationStatus.FAILED,
                False,
                f"Unsupported AWS action type: {action.value}",
                rollback_id=rollback_id,
            )

        except Exception as exc:
            return self._execution(
                recommendation,
                RecommendationStatus.FAILED,
                False,
                f"AWS adapter execution failed: {exc}",
                rollback_id=rollback_id,
            )

    def _execute_azure(
        self,
        recommendation: OptimizationRecommendation,
        resource: Dict[str, Any],
        rollback_id: str,
    ) -> ActionExecution:
        action = self._action(
            recommendation
        )

        resource_id = self._resource_id(
            recommendation
        )

        if action not in {
            ActionType.START,
            ActionType.STOP,
        }:
            return self._execution(
                recommendation,
                RecommendationStatus.FAILED,
                False,
                "Azure execution adapter currently permits only safe start/stop actions.",
                rollback_id=rollback_id,
            )

        try:
            from azure.identity import DefaultAzureCredential
            from azure.mgmt.compute import ComputeManagementClient
        except ImportError:
            return self._execution(
                recommendation,
                RecommendationStatus.FAILED,
                False,
                "Azure SDK dependencies are not installed.",
                rollback_id=rollback_id,
            )

        subscription_id = resource.get(
            "subscription_id"
        )

        resource_group = resource.get(
            "resource_group"
        )

        vm_name = resource.get(
            "vm_name",
            resource_id,
        )

        if not subscription_id or not resource_group:
            return self._execution(
                recommendation,
                RecommendationStatus.FAILED,
                False,
                "Azure subscription_id and resource_group are required.",
                rollback_id=rollback_id,
            )

        try:
            credential = (
                DefaultAzureCredential()
            )

            client = ComputeManagementClient(
                credential,
                subscription_id,
            )

            if action == ActionType.START:
                client.virtual_machines.begin_start(
                    resource_group,
                    vm_name,
                ).result()
            else:
                client.virtual_machines.begin_deallocate(
                    resource_group,
                    vm_name,
                ).result()

            return self._execution(
                recommendation,
                RecommendationStatus.EXECUTED,
                True,
                f"Azure {action.value} executed for {vm_name}.",
                rollback_id=rollback_id,
            )

        except Exception as exc:
            return self._execution(
                recommendation,
                RecommendationStatus.FAILED,
                False,
                f"Azure action failed: {exc}",
                rollback_id=rollback_id,
            )

    def _execute_gcp(
        self,
        recommendation: OptimizationRecommendation,
        resource: Dict[str, Any],
        rollback_id: str,
    ) -> ActionExecution:
        action = self._action(
            recommendation
        )

        resource_id = self._resource_id(
            recommendation
        )

        if action not in {
            ActionType.START,
            ActionType.STOP,
        }:
            return self._execution(
                recommendation,
                RecommendationStatus.FAILED,
                False,
                "GCP execution adapter currently permits only safe start/stop actions.",
                rollback_id=rollback_id,
            )

        try:
            from google.cloud import compute_v1
        except ImportError:
            return self._execution(
                recommendation,
                RecommendationStatus.FAILED,
                False,
                "GCP Compute SDK is not installed.",
                rollback_id=rollback_id,
            )

        project = resource.get(
            "project_id"
        )

        zone = resource.get(
            "zone"
        )

        instance = resource.get(
            "instance_name",
            resource_id,
        )

        if not project or not zone:
            return self._execution(
                recommendation,
                RecommendationStatus.FAILED,
                False,
                "GCP project_id and zone are required.",
                rollback_id=rollback_id,
            )

        try:
            client = (
                compute_v1.InstancesClient()
            )

            if action == ActionType.START:
                operation = client.start(
                    project=project,
                    zone=zone,
                    instance=instance,
                )
            else:
                operation = client.stop(
                    project=project,
                    zone=zone,
                    instance=instance,
                )

            operation.result()

            return self._execution(
                recommendation,
                RecommendationStatus.EXECUTED,
                True,
                f"GCP {action.value} executed for {instance}.",
                rollback_id=rollback_id,
            )

        except Exception as exc:
            return self._execution(
                recommendation,
                RecommendationStatus.FAILED,
                False,
                f"GCP action failed: {exc}",
                rollback_id=rollback_id,
            )

    def rollback(
        self,
        execution: ActionExecution,
        resource: Optional[
            Dict[str, Any]
        ] = None,
    ) -> Dict[str, Any]:
        rollback_id = self._get(
            execution,
            "rollback_id",
        )

        if not rollback_id:
            return {
                "success": False,
                "message": "No rollback state is associated with this execution.",
            }

        state = self.rollback_store.get(
            str(rollback_id)
        )

        if state is None:
            return {
                "success": False,
                "message": "Rollback state was not found.",
            }

        original_state = copy.deepcopy(
            state.get(
                "original_state",
                {},
            )
        )

        if self.dry_run:
            return {
                "success": True,
                "mode": "dry_run",
                "resource_id": state.get(
                    "resource_id"
                ),
                "restored_state": original_state,
                "message": "Rollback simulated successfully.",
                "rolled_back_at": self._now(),
            }

        provider = str(state.get("provider", "simulation")).lower()

        if provider in {CloudProvider.SIMULATION.value, "simulation"}:
            return {
                "success": True,
                "mode": "simulation",
                "resource_id": state.get(
                    "resource_id"
                ),
                "restored_state": original_state,
                "message": "Simulation state restored.",
                "rolled_back_at": self._now(),
            }

        if provider in {CloudProvider.AWS.value, "aws"}:
            resource_id = state.get("resource_id")
            original_sku = original_state.get("instance_type") or original_state.get("current_size")
            if resource_id and original_sku:
                try:
                    from cba.automation.aws_adapter import AWSAutomationAdapter
                    region = (
                        (resource or {}).get("region")
                        or original_state.get("region")
                        or "us-east-1"
                    )
                    creds = (resource or {}).get("credentials") or (resource or {}).get("aws_credentials")
                    adapter = AWSAutomationAdapter(credentials=creds, region=str(region), dry_run=self.dry_run)
                    res = adapter.rollback_instance(resource_id=resource_id, original_sku=original_sku, dry_run=self.dry_run)
                    if res.get("success"):
                        return {
                            "success": True,
                            "mode": "live",
                            "resource_id": resource_id,
                            "restored_state": original_state,
                            "message": res.get("message", f"AWS instance {resource_id} restored to {original_sku}."),
                            "rolled_back_at": self._now(),
                        }
                    return {
                        "success": False,
                        "mode": "live",
                        "resource_id": resource_id,
                        "error": res.get("error"),
                        "message": res.get("message", f"AWS rollback failed: {res.get('error')}"),
                        "rolled_back_at": self._now(),
                    }
                except Exception as exc:
                    return {
                        "success": False,
                        "mode": "live",
                        "resource_id": resource_id,
                        "error": str(exc),
                        "message": f"AWS rollback adapter failure: {exc}",
                        "rolled_back_at": self._now(),
                    }

        return {
            "success": False,
            "mode": "live",
            "resource_id": state.get(
                "resource_id"
            ),
            "message": (
                "Live rollback requires provider-specific "
                "state restoration and has not been enabled "
                "for this resource."
            ),
            "rolled_back_at": self._now(),
        }

    def execute_batch(
        self,
        recommendations: List[
            OptimizationRecommendation
        ],
        resources: Optional[
            Dict[str, Dict[str, Any]]
        ] = None,
        approved_only: bool = True,
        max_actions: int = 20,
    ) -> List[ActionExecution]:
        resources = resources or {}

        executions = []

        ordered = sorted(
            recommendations,
            key=lambda item: (
                self._confidence(item),
                self._risk_rank(
                    self._risk_level(item)
                ),
            ),
            reverse=True,
        )

        count = 0

        for recommendation in ordered:
            if count >= max_actions:
                break

            status = self._enum_value(
                self._get(
                    recommendation,
                    "status",
                )
            )

            if (
                approved_only
                and status
                != RecommendationStatus.APPROVED.value
            ):
                continue

            resource_id = self._resource_id(
                recommendation
            )

            execution = self.execute(
                recommendation,
                resources.get(
                    resource_id,
                    {},
                ),
            )

            executions.append(
                execution
            )

            count += 1

        return executions

    def _risk_rank(
        self,
        risk: RiskLevel,
    ) -> int:
        ranking = {
            RiskLevel.LOW: 1,
            RiskLevel.MEDIUM: 2,
            RiskLevel.HIGH: 3,
            RiskLevel.CRITICAL: 4,
        }

        return ranking.get(
            risk,
            2,
        )

    def history(
        self,
        limit: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        history = list(
            self.execution_history
        )

        if limit is not None:
            return history[
                -max(1, int(limit)) :
            ]

        return history

    def summary(self) -> Dict[str, Any]:
        total = len(
            self.execution_history
        )

        successful = 0
        failed = 0
        simulated = 0
        executed = 0
        rolled_back = 0

        for item in self.execution_history:
            status = str(
                item.get(
                    "status",
                    "",
                )
            )

            success = bool(
                item.get(
                    "success",
                    False,
                )
            )

            if success:
                successful += 1
            else:
                failed += 1

            if status == RecommendationStatus.SIMULATED.value:
                simulated += 1

            if status == RecommendationStatus.EXECUTED.value:
                executed += 1

            if status == RecommendationStatus.ROLLED_BACK.value:
                rolled_back += 1

        return {
            "total_actions": total,
            "successful": successful,
            "failed": failed,
            "simulated": simulated,
            "executed": executed,
            "rolled_back": rolled_back,
            "success_rate": (
                successful / total
                if total
                else 0.0
            ),
            "dry_run": self.dry_run,
            "require_approval": self.require_approval,
            "verify_health": self.verify_health,
        }


ActionManager = AutomationEngine
