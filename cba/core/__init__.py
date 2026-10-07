# Keep the data models usable for the CSV-first MVP without requiring live
# cloud credential packages.  These two integrations are optional extras.
try:
    from .config import Config
except ImportError:  # pragma: no cover - depends on optional installation
    Config = None  # type: ignore[assignment,misc]

try:
    from .credentials import CredentialManager
except ImportError:  # pragma: no cover - depends on optional installation
    CredentialManager = None  # type: ignore[assignment,misc]
from .exceptions import (
    CloudBillingError,
    CloudBenefitAutomationError,
    ConfigurationError,
    CredentialError,
    CollectorError,
    AuthenticationError,
    AuthorizationError,
    APIError,
    DataProcessingError,
    AnalyzerError,
    ModelError,
    ForecastError,
    AnomalyDetectionError,
    OptimizationError,
    SafetyCheckError,
    AutomationError,
    ApprovalRequiredError,
    ActionExecutionError,
    HealthCheckError,
    RollbackError,
    SimulationError,
    AlertError,
    BudgetExceededError,
    ReportingError,
    ReportError,
    PersistenceError,
)

__all__ = [
    "Config",
    "CredentialManager",
    "CloudBillingError",
    "CloudBenefitAutomationError",
    "ConfigurationError",
    "CredentialError",
    "CollectorError",
    "AuthenticationError",
    "AuthorizationError",
    "APIError",
    "DataProcessingError",
    "AnalyzerError",
    "ModelError",
    "ForecastError",
    "AnomalyDetectionError",
    "OptimizationError",
    "SafetyCheckError",
    "AutomationError",
    "ApprovalRequiredError",
    "ActionExecutionError",
    "HealthCheckError",
    "RollbackError",
    "SimulationError",
    "AlertError",
    "BudgetExceededError",
    "ReportingError",
    "ReportError",
    "PersistenceError",
]
