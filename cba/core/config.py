from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from .exceptions import ConfigurationError


@dataclass
class CloudProviderConfig:
    name: str
    enabled: bool = False
    account_id: Optional[str] = None
    subscription_id: Optional[str] = None
    project_id: Optional[str] = None
    regions: List[str] = field(default_factory=list)
    tags_required: List[str] = field(default_factory=list)
    cost_center_tag: str = "CostCenter"
    environment_tag: str = "Environment"
    read_only: bool = True
    automation_enabled: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class BudgetConfig:
    monthly_limit: float = 1000.0
    warning_threshold: float = 0.80
    critical_threshold: float = 0.95
    currency: str = "USD"
    alert_emails: List[str] = field(default_factory=list)
    alert_webhooks: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ReportConfig:
    output_dir: str = "reports"
    formats: List[str] = field(
        default_factory=lambda: ["json", "csv", "html"]
    )
    schedule: str = "daily"
    include_charts: bool = True
    email_reports: bool = False
    email_recipients: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class SimulationConfig:
    enabled: bool = True
    default_days: int = 30
    default_resources: int = 20
    enforce_safety_checks: bool = True
    preserve_sla: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class AutomationConfig:
    enabled: bool = False
    require_approval: bool = True
    dry_run: bool = True
    max_risk_level: str = "low"
    minimum_confidence: float = 0.80
    verify_health_after_action: bool = True
    rollback_on_health_failure: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class Config:
    config_file: Optional[str] = None
    debug: bool = False
    log_level: str = "INFO"
    data_retention_days: int = 90

    aws: CloudProviderConfig = field(
        default_factory=lambda: CloudProviderConfig(name="aws")
    )
    azure: CloudProviderConfig = field(
        default_factory=lambda: CloudProviderConfig(name="azure")
    )
    gcp: CloudProviderConfig = field(
        default_factory=lambda: CloudProviderConfig(name="gcp")
    )

    budget: BudgetConfig = field(default_factory=BudgetConfig)
    reports: ReportConfig = field(default_factory=ReportConfig)
    simulation: SimulationConfig = field(default_factory=SimulationConfig)
    automation: AutomationConfig = field(default_factory=AutomationConfig)

    @classmethod
    def from_file(cls, config_path: str) -> "Config":
        path = Path(config_path)

        if not path.exists():
            raise ConfigurationError(
                f"Configuration file not found: {config_path}"
            )

        try:
            with path.open("r", encoding="utf-8") as file:
                data = yaml.safe_load(file) or {}
        except yaml.YAMLError as exc:
            raise ConfigurationError(
                f"Invalid YAML in configuration file: {exc}"
            ) from exc
        except OSError as exc:
            raise ConfigurationError(
                f"Unable to read configuration file: {exc}"
            ) from exc

        config = cls.from_dict(data)
        config.config_file = str(path)
        return config

    @classmethod
    def from_dict(cls, config_data: Optional[Dict[str, Any]]) -> "Config":
        if config_data is None:
            config_data = {}

        if not isinstance(config_data, dict):
            raise ConfigurationError(
                "Configuration root must be a dictionary"
            )

        config = cls()

        config.debug = bool(config_data.get("debug", False))
        config.log_level = str(
            config_data.get("log_level", "INFO")
        ).upper()
        config.data_retention_days = int(
            config_data.get("data_retention_days", 90)
        )

        providers = config_data.get("providers", {}) or {}

        config.aws = cls._provider_from_dict(
            "aws",
            providers.get("aws", {}),
        )
        config.azure = cls._provider_from_dict(
            "azure",
            providers.get("azure", {}),
        )
        config.gcp = cls._provider_from_dict(
            "gcp",
            providers.get("gcp", {}),
        )

        budget_data = config_data.get("budget", {}) or {}

        config.budget = BudgetConfig(
            monthly_limit=float(
                budget_data.get("monthly_limit", 1000.0)
            ),
            warning_threshold=float(
                budget_data.get("warning_threshold", 0.80)
            ),
            critical_threshold=float(
                budget_data.get("critical_threshold", 0.95)
            ),
            currency=str(
                budget_data.get("currency", "USD")
            ),
            alert_emails=list(
                budget_data.get("alert_emails", [])
            ),
            alert_webhooks=list(
                budget_data.get("alert_webhooks", [])
            ),
        )

        report_data = config_data.get("reports", {}) or {}

        config.reports = ReportConfig(
            output_dir=str(
                report_data.get("output_dir", "reports")
            ),
            formats=list(
                report_data.get(
                    "formats",
                    ["json", "csv", "html"],
                )
            ),
            schedule=str(
                report_data.get("schedule", "daily")
            ),
            include_charts=bool(
                report_data.get("include_charts", True)
            ),
            email_reports=bool(
                report_data.get("email_reports", False)
            ),
            email_recipients=list(
                report_data.get("email_recipients", [])
            ),
        )

        simulation_data = config_data.get("simulation", {}) or {}

        config.simulation = SimulationConfig(
            enabled=bool(
                simulation_data.get("enabled", True)
            ),
            default_days=int(
                simulation_data.get("default_days", 30)
            ),
            default_resources=int(
                simulation_data.get("default_resources", 20)
            ),
            enforce_safety_checks=bool(
                simulation_data.get(
                    "enforce_safety_checks",
                    True,
                )
            ),
            preserve_sla=bool(
                simulation_data.get("preserve_sla", True)
            ),
        )

        automation_data = config_data.get("automation", {}) or {}

        config.automation = AutomationConfig(
            enabled=bool(
                automation_data.get("enabled", False)
            ),
            require_approval=bool(
                automation_data.get("require_approval", True)
            ),
            dry_run=bool(
                automation_data.get("dry_run", True)
            ),
            max_risk_level=str(
                automation_data.get(
                    "max_risk_level",
                    "low",
                )
            ).lower(),
            minimum_confidence=float(
                automation_data.get(
                    "minimum_confidence",
                    0.80,
                )
            ),
            verify_health_after_action=bool(
                automation_data.get(
                    "verify_health_after_action",
                    True,
                )
            ),
            rollback_on_health_failure=bool(
                automation_data.get(
                    "rollback_on_health_failure",
                    True,
                )
            ),
        )

        return config

    @staticmethod
    def _provider_from_dict(
        name: str,
        data: Optional[Dict[str, Any]],
    ) -> CloudProviderConfig:
        data = data or {}

        return CloudProviderConfig(
            name=name,
            enabled=bool(data.get("enabled", False)),
            account_id=data.get("account_id"),
            subscription_id=data.get("subscription_id"),
            project_id=data.get("project_id"),
            regions=list(data.get("regions", [])),
            tags_required=list(
                data.get("tags_required", [])
            ),
            cost_center_tag=str(
                data.get("cost_center_tag", "CostCenter")
            ),
            environment_tag=str(
                data.get("environment_tag", "Environment")
            ),
            read_only=bool(
                data.get("read_only", True)
            ),
            automation_enabled=bool(
                data.get("automation_enabled", False)
            ),
        )

    @classmethod
    def from_env(cls) -> "Config":
        config = cls()

        config.debug = os.getenv(
            "CBA_DEBUG",
            "false",
        ).lower() in {"1", "true", "yes", "on"}

        config.log_level = os.getenv(
            "CBA_LOG_LEVEL",
            "INFO",
        ).upper()

        config.data_retention_days = int(
            os.getenv(
                "CBA_DATA_RETENTION_DAYS",
                "90",
            )
        )

        if os.getenv("CBA_BUDGET_MONTHLY_LIMIT"):
            config.budget.monthly_limit = float(
                os.getenv("CBA_BUDGET_MONTHLY_LIMIT")
            )

        if os.getenv("CBA_BUDGET_WARNING_THRESHOLD"):
            config.budget.warning_threshold = float(
                os.getenv("CBA_BUDGET_WARNING_THRESHOLD")
            )

        if os.getenv("CBA_BUDGET_CRITICAL_THRESHOLD"):
            config.budget.critical_threshold = float(
                os.getenv("CBA_BUDGET_CRITICAL_THRESHOLD")
            )

        config.automation.enabled = (
            os.getenv(
                "CBA_AUTOMATION_ENABLED",
                "false",
            ).lower()
            in {"1", "true", "yes", "on"}
        )

        config.automation.dry_run = (
            os.getenv(
                "CBA_AUTOMATION_DRY_RUN",
                "true",
            ).lower()
            in {"1", "true", "yes", "on"}
        )

        config.automation.require_approval = (
            os.getenv(
                "CBA_AUTOMATION_REQUIRE_APPROVAL",
                "true",
            ).lower()
            in {"1", "true", "yes", "on"}
        )

        return config

    def validate(self) -> None:
        if self.data_retention_days <= 0:
            raise ConfigurationError(
                "Data retention days must be positive"
            )

        if not self.log_level:
            raise ConfigurationError(
                "Log level cannot be empty"
            )

        if self.budget.monthly_limit <= 0:
            raise ConfigurationError(
                "Budget monthly limit must be positive"
            )

        if not 0 < self.budget.warning_threshold < 1:
            raise ConfigurationError(
                "Budget warning threshold must be between 0 and 1"
            )

        if not 0 < self.budget.critical_threshold < 1:
            raise ConfigurationError(
                "Budget critical threshold must be between 0 and 1"
            )

        if (
            self.budget.warning_threshold
            >= self.budget.critical_threshold
        ):
            raise ConfigurationError(
                "Warning threshold must be less than critical threshold"
            )

        if self.simulation.default_days <= 0:
            raise ConfigurationError(
                "Simulation duration must be positive"
            )

        if self.simulation.default_resources <= 0:
            raise ConfigurationError(
                "Simulation resource count must be positive"
            )

        if not 0 <= self.automation.minimum_confidence <= 1:
            raise ConfigurationError(
                "Automation minimum confidence must be between 0 and 1"
            )

        valid_risk_levels = {
            "low",
            "medium",
            "high",
            "critical",
        }

        if self.automation.max_risk_level not in valid_risk_levels:
            raise ConfigurationError(
                "Automation max risk level is invalid"
            )

        self._validate_provider(self.aws)
        self._validate_provider(self.azure)
        self._validate_provider(self.gcp)

    @staticmethod
    def _validate_provider(
        provider: CloudProviderConfig,
    ) -> None:
        if not provider.enabled:
            return

        if provider.name == "aws" and not provider.account_id:
            raise ConfigurationError(
                "AWS account ID is required when AWS is enabled"
            )

        if (
            provider.name == "azure"
            and not provider.subscription_id
        ):
            raise ConfigurationError(
                "Azure subscription ID is required when Azure is enabled"
            )

        if provider.name == "gcp" and not provider.project_id:
            raise ConfigurationError(
                "GCP project ID is required when GCP is enabled"
            )

        if provider.automation_enabled and provider.read_only:
            raise ConfigurationError(
                f"{provider.name.upper()} cannot enable automation "
                "while configured as read-only"
            )

    def enabled_providers(self) -> List[CloudProviderConfig]:
        return [
            provider
            for provider in (
                self.aws,
                self.azure,
                self.gcp,
            )
            if provider.enabled
        ]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "debug": self.debug,
            "log_level": self.log_level,
            "data_retention_days": self.data_retention_days,
            "providers": {
                "aws": self.aws.to_dict(),
                "azure": self.azure.to_dict(),
                "gcp": self.gcp.to_dict(),
            },
            "budget": self.budget.to_dict(),
            "reports": self.reports.to_dict(),
            "simulation": self.simulation.to_dict(),
            "automation": self.automation.to_dict(),
        }

    def save(self, config_path: str) -> None:
        path = Path(config_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with path.open("w", encoding="utf-8") as file:
            yaml.safe_dump(
                self.to_dict(),
                file,
                sort_keys=False,
                default_flow_style=False,
            )

        self.config_file = str(path)