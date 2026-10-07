class CloudBillingError(Exception):
    """Base exception for the cloud billing automation platform."""


class CloudBenefitAutomationError(CloudBillingError):
    """Base exception for the application."""


class ConfigurationError(CloudBenefitAutomationError):
    """Raised when application configuration is invalid."""


class CredentialError(CloudBenefitAutomationError):
    """Raised when credential storage or retrieval fails."""


class ValidationError(CloudBenefitAutomationError):
    """Raised when input data fails validation."""


class CollectorError(CloudBenefitAutomationError):
    """Raised when cloud data collection fails."""


class AuthenticationError(CollectorError):
    """Raised when cloud authentication fails."""


class AuthorizationError(CollectorError):
    """Raised when required cloud permissions are unavailable."""


class APIError(CollectorError):
    """Raised when an external cloud API call fails."""


class DataProcessingError(CloudBenefitAutomationError):
    """Raised when data processing or feature engineering fails."""


class AnalyzerError(CloudBenefitAutomationError):
    """Raised when cost or trend analysis fails."""


class ModelError(CloudBenefitAutomationError):
    """Raised when an ML model cannot be loaded, trained, or executed."""


class ForecastError(ModelError):
    """Raised when forecasting fails."""


class AnomalyDetectionError(ModelError):
    """Raised when anomaly detection fails."""


class OptimizationError(CloudBenefitAutomationError):
    """Raised when optimization fails."""


class SafetyCheckError(CloudBenefitAutomationError):
    """Raised when an optimization action fails a safety check."""


class AutomationError(CloudBenefitAutomationError):
    """Raised when an automation action fails."""


class ApprovalRequiredError(AutomationError):
    """Raised when an action requires approval before execution."""


class ActionExecutionError(AutomationError):
    """Raised when a cloud optimization action cannot be executed."""


class HealthCheckError(AutomationError):
    """Raised when post-action health verification fails."""


class RollbackError(AutomationError):
    """Raised when an optimization action cannot be rolled back."""


class SimulationError(CloudBenefitAutomationError):
    """Raised when simulation execution fails."""


class AlertError(CloudBenefitAutomationError):
    """Raised when alert processing or delivery fails."""


class BudgetExceededError(AlertError):
    """Raised when spend exceeds the configured budget limit."""


class ReportingError(CloudBenefitAutomationError):
    """Raised when report generation fails."""


class ReportError(ReportingError):
    """Backward-compatible report exception."""


class PersistenceError(CloudBenefitAutomationError):
    """Raised when persistent storage operations fail."""