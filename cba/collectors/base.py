from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

from ..core.exceptions import CollectorError


@dataclass
class BillingData:
    provider: str
    account_id: str
    service: str
    region: str
    resource_id: str
    resource_name: str
    usage_type: str
    usage_amount: float
    usage_unit: str
    cost: float
    currency: str
    start_time: datetime
    end_time: datetime
    tags: Dict[str, str] = field(default_factory=dict)
    environment: Optional[str] = None
    cost_center: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        result["start_time"] = self.start_time.isoformat()
        result["end_time"] = self.end_time.isoformat()
        return result


@dataclass
class ResourceData:
    provider: str
    account_id: str
    resource_id: str
    resource_name: str
    resource_type: str
    region: str
    state: str
    creation_time: datetime
    tags: Dict[str, str] = field(default_factory=dict)
    environment: Optional[str] = None
    cost_center: Optional[str] = None
    is_idle: bool = False
    last_used_time: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        result["creation_time"] = self.creation_time.isoformat()

        if self.last_used_time:
            result["last_used_time"] = (
                self.last_used_time.isoformat()
            )

        return result


class BaseCollector(ABC):
    def __init__(
        self,
        config: Any,
        credentials: Optional[Dict[str, str]] = None,
    ):
        self.config = config
        self.credentials = credentials or {}

        class_name = self.__class__.__name__
        self.provider_name = class_name.replace(
            "Collector",
            "",
        ).lower()

        self.authenticated = False
        self.last_authentication_time: Optional[
            datetime
        ] = None

    @abstractmethod
    def authenticate(self) -> None:
        pass

    @abstractmethod
    def collect_billing_data(
        self,
        start_date: datetime,
        end_date: datetime,
    ) -> List[BillingData]:
        pass

    @abstractmethod
    def collect_resource_data(
        self,
    ) -> List[ResourceData]:
        pass

    @abstractmethod
    def get_cost_breakdown(
        self,
        start_date: datetime,
        end_date: datetime,
        group_by: str = "service",
    ) -> Dict[str, float]:
        pass

    def validate_date_range(
        self,
        start_date: datetime,
        end_date: datetime,
    ) -> None:
        if not isinstance(start_date, datetime):
            raise CollectorError(
                "start_date must be a datetime."
            )

        if not isinstance(end_date, datetime):
            raise CollectorError(
                "end_date must be a datetime."
            )

        start = self._normalize_datetime(start_date)
        end = self._normalize_datetime(end_date)
        now = datetime.now(timezone.utc)

        if start >= end:
            raise CollectorError(
                "Start date must be before end date."
            )

        if end > now:
            raise CollectorError(
                "End date cannot be in the future."
            )

        if end - start > timedelta(days=365):
            raise CollectorError(
                "Date range cannot exceed 365 days."
            )

    def _normalize_datetime(
        self,
        value: datetime,
    ) -> datetime:
        if value.tzinfo is None:
            return value.replace(
                tzinfo=timezone.utc
            )

        return value.astimezone(timezone.utc)

    def ensure_authenticated(self) -> None:
        if not self.authenticated:
            self.authenticate()

        if not self.authenticated:
            raise CollectorError(
                f"{self.provider_name} collector "
                "authentication failed."
            )

    def filter_data_by_region(
        self,
        data: List[Any],
        regions: Optional[List[str]],
    ) -> List[Any]:
        if not regions:
            return data

        normalized_regions = {
            str(region).strip().lower()
            for region in regions
            if region is not None
        }

        return [
            item
            for item in data
            if str(
                getattr(item, "region", "")
            ).strip().lower()
            in normalized_regions
        ]

    def filter_data_by_tags(
        self,
        data: List[Any],
        required_tags: Optional[List[str]],
    ) -> List[Any]:
        if not required_tags:
            return data

        required = {
            str(tag).strip()
            for tag in required_tags
            if tag is not None
        }

        if not required:
            return data

        filtered_data = []

        for item in data:
            tags = getattr(
                item,
                "tags",
                {},
            ) or {}

            normalized_tags = {
                str(key).strip().lower()
                for key in tags
            }

            if all(
                str(tag).strip().lower()
                in normalized_tags
                for tag in required
            ):
                filtered_data.append(item)

        return filtered_data

    def filter_data_by_environment(
        self,
        data: List[Any],
        environment: Optional[str],
    ) -> List[Any]:
        if not environment:
            return data

        target = str(environment).strip().lower()

        return [
            item
            for item in data
            if str(
                getattr(item, "environment", "")
                or ""
            ).strip().lower()
            == target
        ]

    def filter_data_by_cost_center(
        self,
        data: List[Any],
        cost_center: Optional[str],
    ) -> List[Any]:
        if not cost_center:
            return data

        target = str(cost_center).strip().lower()

        return [
            item
            for item in data
            if str(
                getattr(item, "cost_center", "")
                or ""
            ).strip().lower()
            == target
        ]

    def extract_environment_from_tags(
        self,
        tags: Dict[str, Any],
        environment_tag: str = "Environment",
    ) -> Optional[str]:
        if not tags:
            return None

        if environment_tag in tags:
            value = tags[environment_tag]
            return (
                str(value)
                if value is not None
                else None
            )

        target = environment_tag.strip().lower()

        for key, value in tags.items():
            if str(key).strip().lower() == target:
                return (
                    str(value)
                    if value is not None
                    else None
                )

        aliases = {
            "environment",
            "env",
            "environmentname",
            "environment_name",
        }

        for key, value in tags.items():
            normalized = (
                str(key)
                .strip()
                .lower()
                .replace("-", "_")
                .replace(" ", "_")
            )

            if normalized in aliases:
                return (
                    str(value)
                    if value is not None
                    else None
                )

        return None

    def extract_cost_center_from_tags(
        self,
        tags: Dict[str, Any],
        cost_center_tag: str = "CostCenter",
    ) -> Optional[str]:
        if not tags:
            return None

        if cost_center_tag in tags:
            value = tags[cost_center_tag]
            return (
                str(value)
                if value is not None
                else None
            )

        target = cost_center_tag.strip().lower()

        for key, value in tags.items():
            if str(key).strip().lower() == target:
                return (
                    str(value)
                    if value is not None
                    else None
                )

        aliases = {
            "costcenter",
            "cost_center",
            "cost-centre",
            "cost_centre",
        }

        for key, value in tags.items():
            normalized = (
                str(key)
                .strip()
                .lower()
                .replace("-", "_")
                .replace(" ", "_")
            )

            if normalized in aliases:
                return (
                    str(value)
                    if value is not None
                    else None
                )

        return None

    def standardize_tags(
        self,
        raw_tags: Optional[Dict[str, Any]],
    ) -> Dict[str, str]:
        if not raw_tags:
            return {}

        standardized: Dict[str, str] = {}

        for key, value in raw_tags.items():
            if key is None or value is None:
                continue

            normalized_key = str(key).strip()

            if not normalized_key:
                continue

            standardized[
                normalized_key
            ] = str(value).strip()

        return standardized

    def calculate_daily_costs(
        self,
        billing_data: List[BillingData],
    ) -> Dict[str, float]:
        daily_costs: Dict[str, float] = {}

        for data in billing_data:
            timestamp = self._normalize_datetime(
                data.start_time
            )

            date_str = timestamp.strftime(
                "%Y-%m-%d"
            )

            daily_costs[date_str] = (
                daily_costs.get(date_str, 0.0)
                + max(0.0, float(data.cost))
            )

        return dict(
            sorted(
                daily_costs.items()
            )
        )

    def calculate_daily_costs_by_service(
        self,
        billing_data: List[BillingData],
    ) -> Dict[str, Dict[str, float]]:
        result: Dict[
            str,
            Dict[str, float],
        ] = {}

        for data in billing_data:
            date_str = (
                self._normalize_datetime(
                    data.start_time
                ).strftime("%Y-%m-%d")
            )

            service = str(
                data.service or "Unknown"
            )

            if date_str not in result:
                result[date_str] = {}

            result[date_str][service] = (
                result[date_str].get(
                    service,
                    0.0,
                )
                + max(0.0, float(data.cost))
            )

        return dict(
            sorted(
                result.items()
            )
        )

    def get_top_services_by_cost(
        self,
        billing_data: List[BillingData],
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        if limit <= 0:
            return []

        service_costs: Dict[str, float] = {}

        for data in billing_data:
            service = str(
                data.service or "Unknown"
            )

            service_costs[service] = (
                service_costs.get(
                    service,
                    0.0,
                )
                + max(0.0, float(data.cost))
            )

        sorted_services = sorted(
            service_costs.items(),
            key=lambda item: item[1],
            reverse=True,
        )

        total_cost = sum(
            service_costs.values()
        )

        return [
            {
                "service": service,
                "cost": cost,
                "percentage": (
                    (cost / total_cost * 100)
                    if total_cost > 0
                    else 0.0
                ),
            }
            for service, cost
            in sorted_services[:limit]
        ]

    def get_top_resources_by_cost(
        self,
        billing_data: List[BillingData],
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        if limit <= 0:
            return []

        resource_costs: Dict[str, Dict[str, Any]] = {}

        for data in billing_data:
            resource_id = str(
                data.resource_id
                or data.resource_name
                or "unknown"
            )

            if resource_id not in resource_costs:
                resource_costs[resource_id] = {
                    "resource_id": resource_id,
                    "resource_name": data.resource_name,
                    "service": data.service,
                    "region": data.region,
                    "cost": 0.0,
                    "currency": data.currency,
                }

            resource_costs[
                resource_id
            ]["cost"] += max(
                0.0,
                float(data.cost),
            )

        return sorted(
            resource_costs.values(),
            key=lambda item: item["cost"],
            reverse=True,
        )[:limit]

    def get_cost_anomaly_candidates(
        self,
        billing_data: List[BillingData],
        threshold_multiplier: float = 2.0,
    ) -> List[BillingData]:
        if threshold_multiplier <= 0:
            raise CollectorError(
                "threshold_multiplier must be greater than zero."
            )

        service_costs: Dict[str, List[float]] = {}

        for data in billing_data:
            service = str(
                data.service or "Unknown"
            )

            service_costs.setdefault(
                service,
                [],
            ).append(
                max(
                    0.0,
                    float(data.cost),
                )
            )

        service_avg_costs = {
            service: (
                sum(costs) / len(costs)
                if costs
                else 0.0
            )
            for service, costs
            in service_costs.items()
        }

        anomalies = []

        for data in billing_data:
            service = str(
                data.service or "Unknown"
            )

            avg_cost = service_avg_costs.get(
                service,
                0.0,
            )

            if (
                avg_cost > 0
                and float(data.cost)
                > avg_cost * threshold_multiplier
            ):
                anomalies.append(data)

        return anomalies

    def get_idle_resources(
        self,
        resources: List[ResourceData],
    ) -> List[ResourceData]:
        return [
            resource
            for resource in resources
            if bool(
                getattr(
                    resource,
                    "is_idle",
                    False,
                )
            )
        ]

    def get_resource_summary(
        self,
        resources: List[ResourceData],
    ) -> Dict[str, Any]:
        if not resources:
            return {
                "total_resources": 0,
                "idle_resources": 0,
                "active_resources": 0,
                "idle_percentage": 0.0,
                "by_type": {},
                "by_region": {},
                "by_state": {},
            }

        by_type: Dict[str, int] = {}
        by_region: Dict[str, int] = {}
        by_state: Dict[str, int] = {}

        idle_resources = 0

        for resource in resources:
            resource_type = str(
                resource.resource_type
                or "unknown"
            )

            region = str(
                resource.region
                or "unknown"
            )

            state = str(
                resource.state
                or "unknown"
            )

            by_type[resource_type] = (
                by_type.get(
                    resource_type,
                    0,
                )
                + 1
            )

            by_region[region] = (
                by_region.get(
                    region,
                    0,
                )
                + 1
            )

            by_state[state] = (
                by_state.get(
                    state,
                    0,
                )
                + 1
            )

            if resource.is_idle:
                idle_resources += 1

        total = len(resources)

        return {
            "total_resources": total,
            "idle_resources": idle_resources,
            "active_resources": total - idle_resources,
            "idle_percentage": (
                idle_resources / total * 100
            ),
            "by_type": by_type,
            "by_region": by_region,
            "by_state": by_state,
        }

    def get_cost_breakdown_from_data(
        self,
        billing_data: List[BillingData],
        group_by: str = "service",
    ) -> Dict[str, float]:
        supported_fields = {
            "service",
            "region",
            "account_id",
            "resource_id",
            "resource_name",
            "usage_type",
            "environment",
            "cost_center",
            "currency",
            "provider",
        }

        group_by = str(
            group_by
        ).strip().lower()

        if group_by not in supported_fields:
            raise CollectorError(
                f"Unsupported cost grouping: {group_by}"
            )

        result: Dict[str, float] = {}

        for data in billing_data:
            value = getattr(
                data,
                group_by,
                None,
            )

            if value is None or str(value).strip() == "":
                value = "unknown"

            key = str(value)

            result[key] = (
                result.get(
                    key,
                    0.0,
                )
                + max(
                    0.0,
                    float(data.cost),
                )
            )

        return dict(
            sorted(
                result.items(),
                key=lambda item: item[1],
                reverse=True,
            )
        )

    def get_total_cost(
        self,
        billing_data: List[BillingData],
    ) -> float:
        return sum(
            max(
                0.0,
                float(data.cost),
            )
            for data in billing_data
        )

    def get_average_daily_cost(
        self,
        billing_data: List[BillingData],
    ) -> float:
        daily_costs = self.calculate_daily_costs(
            billing_data
        )

        if not daily_costs:
            return 0.0

        return (
            sum(daily_costs.values())
            / len(daily_costs)
        )

    def get_provider_summary(
        self,
        billing_data: List[BillingData],
    ) -> Dict[str, Dict[str, Any]]:
        summary: Dict[
            str,
            Dict[str, Any],
        ] = {}

        for data in billing_data:
            provider = str(
                data.provider or "unknown"
            ).lower()

            if provider not in summary:
                summary[provider] = {
                    "provider": provider,
                    "cost": 0.0,
                    "records": 0,
                    "services": set(),
                    "regions": set(),
                }

            summary[provider]["cost"] += max(
                0.0,
                float(data.cost),
            )

            summary[provider]["records"] += 1
            summary[provider]["services"].add(
                str(data.service)
            )
            summary[provider]["regions"].add(
                str(data.region)
            )

        for provider, data in summary.items():
            data["services"] = sorted(
                data["services"]
            )
            data["regions"] = sorted(
                data["regions"]
            )

        return summary

    def get_collection_metadata(
        self,
        billing_data: List[BillingData],
        resource_data: Optional[
            List[ResourceData]
        ] = None,
    ) -> Dict[str, Any]:
        timestamps = [
            self._normalize_datetime(
                item.start_time
            )
            for item in billing_data
            if item.start_time
        ]

        providers = sorted(
            {
                str(item.provider).lower()
                for item in billing_data
                if item.provider
            }
        )

        return {
            "provider": self.provider_name,
            "billing_records": len(
                billing_data
            ),
            "resource_records": len(
                resource_data or []
            ),
            "providers": providers,
            "start_time": (
                min(timestamps).isoformat()
                if timestamps
                else None
            ),
            "end_time": (
                max(timestamps).isoformat()
                if timestamps
                else None
            ),
            "total_cost": self.get_total_cost(
                billing_data
            ),
            "currency": (
                billing_data[0].currency
                if billing_data
                else None
            ),
        }