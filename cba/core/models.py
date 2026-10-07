from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional


class CloudProvider(str, Enum):
    AWS = "aws"
    AZURE = "azure"
    GCP = "gcp"
    SIMULATION = "simulation"


class ResourceType(str, Enum):
    COMPUTE = "compute"
    DATABASE = "database"
    STORAGE = "storage"
    NETWORK = "network"
    CONTAINER = "container"
    SERVERLESS = "serverless"
    OTHER = "other"


class ActionType(str, Enum):
    NO_ACTION = "no_action"
    RIGHTSIZING = "rightsizing"
    SCALE_UP = "scale_up"
    SCALE_DOWN = "scale_down"
    STORAGE_OPTIMIZATION = "storage_optimization"
    RESERVED_CAPACITY = "reserved_capacity"
    SPOT = "spot"
    STOP = "stop"
    START = "start"
    DELETE = "delete"
    SCHEDULE = "schedule"
    ROLLBACK = "rollback"


class RiskLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class RecommendationStatus(str, Enum):
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    QUEUED = "queued"
    EXECUTING = "executing"
    EXECUTED = "executed"
    SIMULATED = "simulated"
    OBSERVING = "observing"
    COMMITTED_PERMANENTLY = "committed_permanently"
    ROLLED_BACK = "rolled_back"
    FAILED = "failed"


class HealthStatus(str, Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    CRITICAL = "critical"
    UNKNOWN = "unknown"


class EnvironmentTier(str, Enum):
    PRODUCTION = "production"
    STAGING = "staging"
    DEVELOPMENT = "development"
    UNTAGGED = "untagged"


def normalize_environment(val: Optional[str]) -> str:
    """Normalizes arbitrary cloud tag strings into standard environment tiers."""
    if not val:
        return EnvironmentTier.UNTAGGED.value
    v = str(val).strip().lower().replace("_", "-").replace(" ", "-")
    if any(v.startswith(prefix) for prefix in ["prod", "prd", "live"]):
        return EnvironmentTier.PRODUCTION.value
    if any(v.startswith(prefix) for prefix in ["stag", "stg", "uat", "preprod", "pre-prod"]):
        return EnvironmentTier.STAGING.value
    if any(v.startswith(prefix) for prefix in ["dev", "test", "qa", "sandbox", "poc", "nonprod", "non-prod"]):
        return EnvironmentTier.DEVELOPMENT.value
    return EnvironmentTier.UNTAGGED.value


def detect_environment_from_tags(tags: Optional[Dict[str, Any]]) -> str:
    """Extracts and normalizes environment from cloud tag key-value dictionary."""
    if not tags or not isinstance(tags, dict):
        return EnvironmentTier.UNTAGGED.value

    # Direct match for known environment keys (case-insensitive)
    env_keys = {"environment", "env", "stage", "tier", "deployment-stage", "deployment_stage", "env_type"}
    for k, v in tags.items():
        k_clean = str(k).strip().lower().replace("_", "-")
        if k_clean in env_keys:
            tier = normalize_environment(v)
            if tier != EnvironmentTier.UNTAGGED.value:
                return tier

    # Fallback: check all values in case key was slightly different (e.g. "EnvironmentName", "app_env")
    for k, v in tags.items():
        k_clean = str(k).strip().lower()
        if "env" in k_clean or "stage" in k_clean or "tier" in k_clean:
            tier = normalize_environment(v)
            if tier != EnvironmentTier.UNTAGGED.value:
                return tier

    return EnvironmentTier.UNTAGGED.value


def detect_resize_intent_from_tags(tags: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Inspects cloud tags to identify explicit right-sizing, auto-optimization, or resize recommendations.
    
    Supports keys like 'ResizeRecommendation', 'RightSize', 'AutoOptimize', 'TargetSKU',
    'CloudOpt:Recommendation', and parses target SKUs or action directives.
    """
    default_result: Dict[str, Any] = {
        "has_resize_intent": False,
        "intent_type": None,
        "action": None,
        "target_sku": None,
        "tag_key": None,
        "matched_key": None,
        "tag_value": None,
        "directive_value": None,
        "directive_label": None,
        "reason": None,
    }
    if not tags or not isinstance(tags, dict):
        return default_result

    resize_keys = {
        "resizerecommendation", "resize-recommendation", "resize_recommendation",
        "rightsize", "right-size", "right_size",
        "autooptimize", "auto-optimize", "auto_optimize",
        "targetsku", "target-sku", "target_sku",
        "cloudopt:recommendation", "cloudopt-recommendation", "cloudopt:target",
        "desiredsku", "desired-sku", "desired_sku",
        "recommendation", "resize", "scale-down", "scale_down"
    }

    # Known common SKUs to detect if embedded in tag value
    known_skus = [
        "t3.nano", "t3.micro", "t3.small", "t3.medium", "t3.large", "t3.xlarge", "t3.2xlarge",
        "t2.nano", "t2.micro", "t2.small", "t2.medium", "t2.large", "t2.xlarge",
        "c5.large", "c5.xlarge", "c5.2xlarge", "c5.4xlarge",
        "m5.large", "m5.xlarge", "m5.2xlarge",
        "standard_b1s", "standard_b1ms", "standard_b2s", "standard_d2s_v5",
        "e2-micro", "e2-small", "e2-medium", "e2-standard-2"
    ]

    for raw_k, raw_v in tags.items():
        k_clean = str(raw_k).strip().lower()
        v_str = str(raw_v).strip()
        v_clean = v_str.lower()

        # Check if key matches or contains resize indicators
        is_target_key = k_clean in resize_keys or any(rk in k_clean for rk in ["resize", "rightsize", "autoopt", "target-sku", "target_sku"])

        # Check if value mentions a specific SKU
        matched_sku = None
        for sku in known_skus:
            if sku in v_clean:
                matched_sku = sku
                break

        if is_target_key or matched_sku:
            intent_type = "scale_down"
            if "scale_up" in v_clean or "scale-up" in v_clean or "upsize" in v_clean or "enlarge" in v_clean:
                intent_type = "scale_up"
            elif "no_action" in v_clean or "lock" in v_clean or "keep" in v_clean:
                intent_type = "no_action"

            label = f"Tag [{raw_k}={v_str}]"
            reason = f"Tagged with '{raw_k}: {v_str}' directing {intent_type}"
            if matched_sku:
                reason += f" to target SKU {matched_sku}"

            return {
                "has_resize_intent": True,
                "intent_type": intent_type,
                "action": intent_type,
                "target_sku": matched_sku,
                "tag_key": str(raw_k),
                "matched_key": str(raw_k),
                "tag_value": v_str,
                "directive_value": v_str,
                "directive_label": label,
                "reason": reason,
            }

    return default_result


@dataclass
class ResourceTelemetry:
    timestamp: datetime
    provider: CloudProvider
    resource_id: str
    resource_type: ResourceType = ResourceType.COMPUTE
    region: Optional[str] = None
    resource_name: Optional[str] = None
    vm_type: Optional[str] = None
    cpu_usage: float = 0.0
    memory_usage: float = 0.0
    network_in: float = 0.0
    network_out: float = 0.0
    disk_read: float = 0.0
    disk_write: float = 0.0
    latency_ms: float = 0.0
    throughput: float = 0.0
    active_requests: float = 0.0
    error_rate: float = 0.0
    vcpu: float = 0.0
    ram_gb: float = 0.0
    hourly_cost: float = 0.0
    daily_cost: float = 0.0
    monthly_cost: float = 0.0
    utilization: float = 0.0
    state: str = "running"
    environment: str = "untagged"
    tags: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["timestamp"] = self.timestamp.isoformat() if hasattr(self.timestamp, "isoformat") else str(self.timestamp)
        data["provider"] = self.provider.value if hasattr(self.provider, "value") else str(self.provider)
        data["resource_type"] = self.resource_type.value if hasattr(self.resource_type, "value") else str(self.resource_type)
        return data


@dataclass
class WorkloadForecast:
    resource_id: str
    forecast_timestamp: datetime
    horizon_minutes: int
    predicted_cpu: float
    predicted_memory: float
    failure_probability: float = 0.0
    resilience_score: float = 100.0
    confidence: float = 0.0
    lower_bound: Optional[float] = None
    upper_bound: Optional[float] = None
    model_name: str = ""

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["forecast_timestamp"] = self.forecast_timestamp.isoformat()
        return data


@dataclass
class CostForecast:
    provider: CloudProvider
    forecast_date: datetime
    horizon_days: int
    predicted_cost: float
    lower_bound: Optional[float] = None
    upper_bound: Optional[float] = None
    confidence: float = 0.0
    model_name: str = ""

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["forecast_date"] = self.forecast_date.isoformat()
        data["provider"] = self.provider.value
        return data


@dataclass
class ResourceCandidate:
    provider: CloudProvider
    resource_type: ResourceType = ResourceType.COMPUTE
    resource_id: str = ""
    resource_name: Optional[str] = None
    instance_type: Optional[str] = None
    vcpu: float = 0.0
    ram_gb: float = 0.0
    hourly_cost: float = 0.0
    monthly_cost: float = 0.0
    predicted_cpu: float = 0.0
    predicted_memory: float = 0.0
    predicted_latency_ms: float = 0.0
    performance_margin: float = 0.0
    estimated_savings: float = 0.0
    savings_percentage: float = 0.0
    feasible: bool = True
    risk_level: RiskLevel = RiskLevel.LOW
    reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["provider"] = self.provider.value if hasattr(self.provider, "value") else str(self.provider)
        data["resource_type"] = self.resource_type.value if hasattr(self.resource_type, "value") else str(self.resource_type)
        data["risk_level"] = self.risk_level.value if hasattr(self.risk_level, "value") else str(self.risk_level)
        return data


@dataclass
class AnomalyResult:
    resource_id: str
    timestamp: datetime
    is_anomaly: bool
    anomaly_score: float
    anomaly_probability: float
    severity: RiskLevel
    contributing_features: List[str] = field(default_factory=list)
    explanation: str = ""
    model_name: str = ""

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["timestamp"] = self.timestamp.isoformat()
        data["severity"] = self.severity.value
        return data


@dataclass
class OptimizationRecommendation:
    recommendation_id: str
    resource_id: str
    provider: CloudProvider
    action: ActionType
    risk_level: RiskLevel
    confidence: float
    estimated_monthly_savings: float
    current_utilization: float
    expected_utilization: float
    current_size: Optional[str]
    recommended_size: Optional[str]
    rationale: str
    status: RecommendationStatus = RecommendationStatus.PENDING
    environment: str = "untagged"
    approval_required: bool = False
    resilience_score: float = 100.0
    feature_importances: Dict[str, float] = field(default_factory=dict)
    original_state: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["provider"] = self.provider.value if hasattr(self.provider, "value") else str(self.provider)
        data["action"] = self.action.value if hasattr(self.action, "value") else str(self.action)
        data["status"] = self.status.value if hasattr(self.status, "value") else str(self.status)
        data["risk_level"] = self.risk_level.value if hasattr(self.risk_level, "value") else str(self.risk_level)
        data["environment"] = self.environment
        return data


@dataclass
class HealthCheck:
    resource_id: str
    checked_at: Any
    status: HealthStatus
    cpu_usage: float = 0.0
    memory_usage: float = 0.0
    latency_ms: float = 0.0
    error_rate: float = 0.0
    throughput: float = 0.0
    sla_violation: bool = False
    message: str = ""
    cpu_utilization: Optional[float] = None
    memory_utilization: Optional[float] = None
    issues: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if isinstance(self.checked_at, str):
            try:
                self.checked_at = datetime.fromisoformat(self.checked_at)
            except Exception:
                self.checked_at = datetime.now(timezone.utc)
        if self.cpu_utilization is not None and self.cpu_usage == 0.0:
            self.cpu_usage = self.cpu_utilization
        if self.cpu_utilization is None:
            self.cpu_utilization = self.cpu_usage
        if self.memory_utilization is not None and self.memory_usage == 0.0:
            self.memory_usage = self.memory_utilization
        if self.memory_utilization is None:
            self.memory_utilization = self.memory_usage

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["checked_at"] = self.checked_at.isoformat() if hasattr(self.checked_at, "isoformat") else str(self.checked_at)
        data["status"] = self.status.value if hasattr(self.status, "value") else str(self.status)
        return data


@dataclass
class ActionExecution:
    action_id: str = ""
    recommendation_id: str = ""
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    provider: CloudProvider = CloudProvider.AWS
    resource_id: str = ""
    action: ActionType = ActionType.NO_ACTION
    status: RecommendationStatus = RecommendationStatus.PENDING
    queue_name: str = ""
    native_message_id: Optional[str] = None
    before_state: Dict[str, Any] = field(default_factory=dict)
    after_state: Dict[str, Any] = field(default_factory=dict)
    health_check: Optional[HealthCheck] = None
    observation_deadline: Optional[datetime] = None
    rollback_reason: Optional[str] = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    execution_id: Optional[str] = None
    success: bool = True
    message: str = ""
    health_status: Optional[HealthStatus] = None
    rollback_id: Optional[str] = None
    executed_at: Optional[str] = None

    def __post_init__(self) -> None:
        if self.execution_id and not self.action_id:
            self.action_id = self.execution_id
        elif self.action_id and not self.execution_id:
            self.execution_id = self.action_id

        if not self.started_at:
            if self.executed_at:
                try:
                    self.started_at = datetime.fromisoformat(self.executed_at)
                except Exception:
                    self.started_at = datetime.now(timezone.utc)
            else:
                self.started_at = datetime.now(timezone.utc)

        if not self.executed_at:
            self.executed_at = self.started_at.isoformat()

        if self.rollback_id and "rollback_id" not in self.metadata:
            self.metadata["rollback_id"] = self.rollback_id
        if self.message and "message" not in self.metadata:
            self.metadata["message"] = self.message

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["started_at"] = self.started_at.isoformat() if self.started_at else None
        data["completed_at"] = self.completed_at.isoformat() if self.completed_at else None
        data["observation_deadline"] = self.observation_deadline.isoformat() if self.observation_deadline else None
        data["provider"] = self.provider.value if hasattr(self.provider, "value") else str(self.provider)
        data["action"] = self.action.value if hasattr(self.action, "value") else str(self.action)
        data["status"] = self.status.value if hasattr(self.status, "value") else str(self.status)
        data["health_check"] = self.health_check.to_dict() if self.health_check else None
        data["target_size"] = self.after_state.get("target_size", "") if isinstance(self.after_state, dict) else ""
        data["execution_id"] = self.execution_id
        data["success"] = self.success
        data["message"] = self.message
        data["rollback_id"] = self.rollback_id
        data["executed_at"] = self.executed_at
        return data


@dataclass
class QueueMessagePayload:
    message_id: str
    event_type: str  # e.g., 'OPTIMIZE_ACTION', 'ROLLBACK_ACTION', 'ALERT'
    provider: str
    resource_id: str
    action: str
    current_size: str
    target_size: str
    risk_level: str
    original_state: Dict[str, Any]
    dispatched_at: str
    observation_window_minutes: int = 10

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def resource_telemetry_from_dict(data: Dict[str, Any]) -> ResourceTelemetry:
    timestamp = data.get("timestamp") or data.get("collected_at")
    if isinstance(timestamp, str):
        try:
            timestamp = datetime.fromisoformat(timestamp)
        except Exception:
            timestamp = datetime.now(timezone.utc)
    elif not isinstance(timestamp, datetime):
        timestamp = datetime.now(timezone.utc)

    provider_raw = str(data.get("provider", "aws")).lower()
    if "azure" in provider_raw:
        provider = CloudProvider.AZURE
    elif "gcp" in provider_raw or "google" in provider_raw:
        provider = CloudProvider.GCP
    else:
        provider = CloudProvider.AWS

    return ResourceTelemetry(
        timestamp=timestamp,
        provider=provider,
        resource_id=str(data.get("resource_id", "")),
        region=data.get("region"),
        resource_name=data.get("resource_name") or data.get("name"),
        vm_type=data.get("vm_type") or data.get("instance_type"),
        cpu_usage=float(data.get("cpu_usage", 0.0) or 0.0),
        memory_usage=float(data.get("memory_usage", 0.0) or 0.0),
        network_in=float(data.get("network_in", 0.0) or data.get("network_in_bytes", 0.0) or 0.0),
        network_out=float(data.get("network_out", 0.0) or data.get("network_out_bytes", 0.0) or 0.0),
        disk_read=float(data.get("disk_read", 0.0) or 0.0),
        disk_write=float(data.get("disk_write", 0.0) or 0.0),
        latency_ms=float(data.get("latency_ms", 0.0) or 0.0),
        throughput=float(data.get("throughput", 0.0) or 0.0),
        vcpu=float(data.get("vcpu", 0.0) or data.get("vCPU", 0.0) or 0.0),
        ram_gb=float(data.get("ram_gb", 0.0) or data.get("RAM_GB", 0.0) or 0.0),
        hourly_cost=float(data.get("hourly_cost", 0.0) or data.get("price_per_hour", 0.0) or 0.0),
        monthly_cost=float(data.get("monthly_cost", 0.0) or 0.0),
        utilization=float(data.get("utilization", 0.0) or data.get("cpu_usage", 0.0) or 0.0),
        state=str(data.get("state", "running")),
        tags=data.get("tags") or {},
        metadata=data.get("metadata") or data.get("original_state") or {},
    )
