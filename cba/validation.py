import json
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from urllib.parse import urlparse

from .core.exceptions import CloudBillingError, ValidationError


class ValidationUtils:
    @staticmethod
    def validate_email(email: str) -> bool:
        if not isinstance(email, str):
            return False

        email = email.strip()

        if len(email) > 254:
            return False

        pattern = r"^[A-Za-z0-9.!#$%&'*+/=?^_`{|}~-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)+$"
        return re.fullmatch(pattern, email) is not None

    @staticmethod
    def validate_url(url: str) -> bool:
        if not isinstance(url, str):
            return False

        try:
            parsed = urlparse(url.strip())

            if parsed.scheme not in {"http", "https"}:
                return False

            if not parsed.netloc:
                return False

            if any(char in url for char in "\r\n\t"):
                return False

            return True
        except Exception:
            return False

    @staticmethod
    def validate_aws_account_id(account_id: str) -> bool:
        return (
            isinstance(account_id, str)
            and re.fullmatch(r"\d{12}", account_id.strip())
            is not None
        )

    @staticmethod
    def validate_azure_subscription_id(
        subscription_id: str,
    ) -> bool:
        if not isinstance(subscription_id, str):
            return False

        pattern = (
            r"^[0-9a-fA-F]{8}-"
            r"[0-9a-fA-F]{4}-"
            r"[0-9a-fA-F]{4}-"
            r"[0-9a-fA-F]{4}-"
            r"[0-9a-fA-F]{12}$"
        )

        return re.fullmatch(
            pattern,
            subscription_id.strip(),
        ) is not None

    @staticmethod
    def validate_gcp_project_id(
        project_id: str,
    ) -> bool:
        if not isinstance(project_id, str):
            return False

        project_id = project_id.strip()

        if not 6 <= len(project_id) <= 30:
            return False

        return (
            re.fullmatch(
                r"[a-z][a-z0-9-]*[a-z0-9]",
                project_id,
            )
            is not None
        )

    @staticmethod
    def validate_resource_id(
        resource_id: str,
        provider: str,
    ) -> bool:
        if not isinstance(resource_id, str):
            return False

        resource_id = resource_id.strip()

        if not resource_id or len(resource_id) > 2048:
            return False

        provider = str(provider).strip().lower()

        if provider == "aws":
            return bool(
                re.fullmatch(
                    r"[A-Za-z0-9:/_.+=,@-]+",
                    resource_id,
                )
            )

        if provider == "azure":
            return resource_id.startswith("/subscriptions/")

        if provider == "gcp":
            return bool(
                re.fullmatch(
                    r"[A-Za-z0-9:/_.+=,@-]+",
                    resource_id,
                )
            )

        return False

    @staticmethod
    def validate_tag_key(tag_key: str) -> bool:
        if not isinstance(tag_key, str):
            return False

        if not 1 <= len(tag_key) <= 128:
            return False

        if "\x00" in tag_key:
            return False

        invalid_chars = {"<", ">", "&", "\\", '"', "?"}

        return not any(
            char in tag_key
            for char in invalid_chars
        )

    @staticmethod
    def validate_tag_value(tag_value: str) -> bool:
        if not isinstance(tag_value, str):
            return False

        if len(tag_value) > 256:
            return False

        if "\x00" in tag_value:
            return False

        invalid_chars = {"<", ">", "&", "\\", '"'}

        return not any(
            char in tag_value
            for char in invalid_chars
        )

    @staticmethod
    def validate_cost_amount(
        amount: Union[str, float, int],
    ) -> bool:
        try:
            cost = float(amount)

            return (
                cost >= 0
                and cost < 1_000_000_000
            )
        except (ValueError, TypeError, OverflowError):
            return False

    @staticmethod
    def validate_percentage(
        percentage: Union[str, float, int],
    ) -> bool:
        try:
            value = float(percentage)

            return (
                value >= 0
                and value <= 100
            )
        except (ValueError, TypeError, OverflowError):
            return False

    @staticmethod
    def validate_date_range(
        start_date: str,
        end_date: str,
    ) -> bool:
        try:
            start = ValidationUtils.parse_datetime(
                start_date
            )
            end = ValidationUtils.parse_datetime(
                end_date
            )

            if start >= end:
                return False

            return (
                end - start
            ) <= timedelta(days=365)

        except (ValueError, TypeError):
            return False

    @staticmethod
    def parse_datetime(
        value: Union[str, datetime],
    ) -> datetime:
        if isinstance(value, datetime):
            return value

        if not isinstance(value, str):
            raise ValueError(
                "Datetime value must be a string or datetime."
            )

        value = value.strip()

        if not value:
            raise ValueError(
                "Datetime value cannot be empty."
            )

        normalized = value.replace(
            "Z",
            "+00:00",
        )

        try:
            return datetime.fromisoformat(
                normalized
            )
        except ValueError:
            formats = (
                "%Y-%m-%d",
                "%Y-%m-%d %H:%M",
                "%Y-%m-%d %H:%M:%S",
                "%Y-%m-%dT%H:%M",
                "%Y-%m-%dT%H:%M:%S",
                "%Y-%m-%dT%H:%M:%S.%f",
            )

            for date_format in formats:
                try:
                    return datetime.strptime(
                        value,
                        date_format,
                    )
                except ValueError:
                    continue

        raise ValueError(
            f"Invalid datetime value: {value}"
        )

    @staticmethod
    def validate_region(
        region: str,
        provider: str,
    ) -> bool:
        if not isinstance(region, str):
            return False

        region = region.strip().lower()
        provider = str(provider).strip().lower()

        if not region:
            return False

        if provider == "aws":
            return (
                re.fullmatch(
                    r"[a-z]{2}(?:-gov)?-[a-z0-9-]+-\d+",
                    region,
                )
                is not None
            )

        if provider == "azure":
            return (
                re.fullmatch(
                    r"[a-z0-9]+(?:-[a-z0-9]+)*",
                    region,
                )
                is not None
            )

        if provider == "gcp":
            return (
                re.fullmatch(
                    r"[a-z]+-[a-z0-9]+[0-9]",
                    region,
                )
                is not None
            )

        return False

    @staticmethod
    def validate_service_name(
        service: str,
        provider: str,
    ) -> bool:
        if not isinstance(service, str):
            return False

        service = service.strip()

        if not service or len(service) > 100:
            return False

        provider = str(provider).strip().lower()

        known_services = {
            "aws": {
                "Amazon EC2",
                "Amazon RDS",
                "Amazon S3",
                "AWS Lambda",
                "Amazon CloudFront",
                "Amazon Route 53",
                "AWS CloudTrail",
                "Amazon VPC",
                "Elastic Load Balancing",
                "Amazon DynamoDB",
            },
            "azure": {
                "Microsoft.Compute/virtualMachines",
                "Microsoft.Storage/storageAccounts",
                "Microsoft.Sql/servers",
                "Microsoft.Network/virtualNetworks",
                "Microsoft.Web/sites",
                "Microsoft.KeyVault/vaults",
            },
            "gcp": {
                "Compute Engine",
                "Cloud Storage",
                "Cloud SQL",
                "Cloud Functions",
                "Cloud Load Balancing",
                "Cloud DNS",
                "Cloud IAM",
                "Cloud Logging",
            },
        }

        if provider in known_services:
            return service in known_services[provider]

        return False

    @staticmethod
    def validate_json_schema(
        data: Dict[str, Any],
        schema: Dict[str, Any],
    ) -> tuple[bool, List[str]]:
        if not isinstance(data, dict):
            return False, ["Data must be an object."]

        if not isinstance(schema, dict):
            return False, ["Schema must be an object."]

        errors: List[str] = []

        type_validators = {
            "string": lambda value: isinstance(
                value,
                str,
            ),
            "number": lambda value: (
                isinstance(value, (int, float))
                and not isinstance(value, bool)
            ),
            "integer": lambda value: (
                isinstance(value, int)
                and not isinstance(value, bool)
            ),
            "boolean": lambda value: isinstance(
                value,
                bool,
            ),
            "array": lambda value: isinstance(
                value,
                list,
            ),
            "object": lambda value: isinstance(
                value,
                dict,
            ),
            "null": lambda value: value is None,
        }

        for field, expected_type in schema.items():
            if field not in data:
                errors.append(
                    f"Missing required field: {field}"
                )
                continue

            value = data[field]

            if callable(expected_type):
                try:
                    valid = bool(expected_type(value))
                except Exception:
                    valid = False

                if not valid:
                    errors.append(
                        f"Invalid value for field: {field}"
                    )

                continue

            validator = type_validators.get(
                str(expected_type).lower()
            )

            if validator is None:
                errors.append(
                    f"Unsupported schema type for {field}: "
                    f"{expected_type}"
                )
                continue

            if not validator(value):
                errors.append(
                    f"Field {field} must be "
                    f"{expected_type}"
                )

        return len(errors) == 0, errors

    @staticmethod
    def sanitize_string(
        input_string: str,
        max_length: int = 1000,
    ) -> str:
        if input_string is None:
            return ""

        if not isinstance(input_string, str):
            input_string = str(input_string)

        max_length = max(0, int(max_length))

        sanitized = re.sub(
            r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]",
            "",
            input_string,
        )

        sanitized = re.sub(
            r'[<>"\'&\\]',
            "",
            sanitized,
        )

        return sanitized[:max_length].strip()

    @staticmethod
    def sanitize_filename(
        filename: str,
    ) -> str:
        if filename is None:
            return "unnamed"

        filename = str(filename)

        filename = re.sub(
            r'[\x00-\x1f<>:"/\\|?*]',
            "",
            filename,
        )

        filename = re.sub(
            r"\s+",
            "_",
            filename,
        )

        filename = filename.strip(". ")

        filename = filename[:255]

        if not filename:
            return "unnamed"

        reserved_names = {
            "CON",
            "PRN",
            "AUX",
            "NUL",
            *(f"COM{i}" for i in range(1, 10)),
            *(f"LPT{i}" for i in range(1, 10)),
        }

        stem = Path(filename).stem.upper()

        if stem in reserved_names:
            filename = f"_{filename}"

        return filename

    @staticmethod
    def validate_file_path(
        file_path: str,
        check_exists: bool = False,
    ) -> bool:
        if not isinstance(file_path, (str, Path)):
            return False

        try:
            path = Path(file_path)

            if not str(path).strip():
                return False

            if "\x00" in str(path):
                return False

            if check_exists and not path.exists():
                return False

            return True

        except (ValueError, OSError, TypeError):
            return False

    @staticmethod
    def validate_api_key(
        api_key: str,
        provider: str,
    ) -> bool:
        if not isinstance(api_key, str):
            return False

        api_key = api_key.strip()
        provider = str(provider).strip().lower()

        if not api_key:
            return False

        if provider == "aws":
            return bool(
                re.fullmatch(
                    r"(?:AKIA|ASIA)[A-Z0-9]{16}",
                    api_key,
                )
            )

        if provider == "azure":
            return len(api_key) >= 20

        if provider == "gcp":
            try:
                value = json.loads(api_key)

                return (
                    isinstance(value, dict)
                    and value.get("type")
                    == "service_account"
                    and bool(value.get("project_id"))
                    and bool(value.get("private_key"))
                    and bool(value.get("client_email"))
                )
            except (json.JSONDecodeError, TypeError):
                return False

        return len(api_key) >= 10

    @staticmethod
    def validate_budget_threshold(
        threshold: float,
    ) -> bool:
        try:
            value = float(threshold)

            return (
                value > 0
                and value <= 1
            )
        except (ValueError, TypeError, OverflowError):
            return False

    @staticmethod
    def validate_currency(
        currency: str,
    ) -> bool:
        if not isinstance(currency, str):
            return False

        valid_currencies = {
            "USD",
            "EUR",
            "GBP",
            "JPY",
            "CAD",
            "AUD",
            "CHF",
            "CNY",
            "SEK",
            "NOK",
            "DKK",
            "PLN",
            "CZK",
            "HUF",
            "RON",
            "BGN",
            "HRK",
            "RUB",
            "TRY",
            "INR",
            "SGD",
            "HKD",
            "NZD",
            "ZAR",
            "MXN",
            "BRL",
            "ARS",
            "CLP",
            "COP",
            "PEN",
            "UYU",
            "VES",
        }

        return currency.strip().upper() in valid_currencies

    @staticmethod
    def validate_cron_expression(
        cron_expr: str,
    ) -> bool:
        if not isinstance(cron_expr, str):
            return False

        parts = cron_expr.strip().split()

        if len(parts) != 5:
            return False

        ranges = (
            (0, 59),
            (0, 23),
            (1, 31),
            (1, 12),
            (0, 7),
        )

        for part, (
            minimum,
            maximum,
        ) in zip(parts, ranges):
            if not ValidationUtils._validate_cron_field(
                part,
                minimum,
                maximum,
            ):
                return False

        return True

    @staticmethod
    def _validate_cron_field(
        field: str,
        minimum: int,
        maximum: int,
    ) -> bool:
        if field == "*":
            return True

        for segment in field.split(","):
            if not segment:
                return False

            if "/" in segment:
                base, step = segment.split(
                    "/",
                    1,
                )

                if not step.isdigit() or int(step) <= 0:
                    return False

                if base != "*" and not ValidationUtils._validate_cron_range(
                    base,
                    minimum,
                    maximum,
                ):
                    return False

            elif not ValidationUtils._validate_cron_range(
                segment,
                minimum,
                maximum,
            ):
                return False

        return True

    @staticmethod
    def _validate_cron_range(
        value: str,
        minimum: int,
        maximum: int,
    ) -> bool:
        if "-" in value:
            parts = value.split("-")

            if len(parts) != 2:
                return False

            if not all(part.isdigit() for part in parts):
                return False

            start = int(parts[0])
            end = int(parts[1])

            return (
                minimum <= start <= maximum
                and minimum <= end <= maximum
                and start <= end
            )

        if not value.isdigit():
            return False

        number = int(value)

        return minimum <= number <= maximum

    @staticmethod
    def validate_webhook_url(
        url: str,
    ) -> bool:
        return (
            ValidationUtils.validate_url(url)
            and url.strip().lower().startswith(
                ("http://", "https://")
            )
        )

    @staticmethod
    def validate_slack_webhook(
        webhook_url: str,
    ) -> bool:
        if not isinstance(webhook_url, str):
            return False

        return webhook_url.strip().startswith(
            "https://hooks.slack.com/"
        )

    @staticmethod
    def validate_configuration_structure(
        config: Dict[str, Any],
    ) -> tuple[bool, List[str]]:
        errors: List[str] = []

        if not isinstance(config, dict):
            return False, ["Configuration must be an object."]

        required_sections = {
            "providers",
            "budget",
        }

        for section in required_sections:
            if section not in config:
                errors.append(
                    f"Missing required section: {section}"
                )

        providers = config.get("providers")

        if providers is not None:
            if not isinstance(providers, dict):
                errors.append(
                    "providers must be an object."
                )
            else:
                for provider in (
                    "aws",
                    "azure",
                    "gcp",
                ):
                    provider_config = providers.get(
                        provider
                    )

                    if provider_config is None:
                        continue

                    if not isinstance(
                        provider_config,
                        dict,
                    ):
                        errors.append(
                            f"{provider} provider configuration "
                            f"must be an object."
                        )
                        continue

                    if provider == "aws":
                        account_id = provider_config.get(
                            "account_id"
                        )

                        if (
                            account_id is not None
                            and not ValidationUtils.validate_aws_account_id(
                                str(account_id)
                            )
                        ):
                            errors.append(
                                f"Invalid AWS account ID: "
                                f"{account_id}"
                            )

                    elif provider == "azure":
                        subscription_id = provider_config.get(
                            "subscription_id"
                        )

                        if (
                            subscription_id is not None
                            and not ValidationUtils.validate_azure_subscription_id(
                                str(subscription_id)
                            )
                        ):
                            errors.append(
                                f"Invalid Azure subscription ID: "
                                f"{subscription_id}"
                            )

                    elif provider == "gcp":
                        project_id = provider_config.get(
                            "project_id"
                        )

                        if (
                            project_id is not None
                            and not ValidationUtils.validate_gcp_project_id(
                                str(project_id)
                            )
                        ):
                            errors.append(
                                f"Invalid GCP project ID: "
                                f"{project_id}"
                            )

                    regions = provider_config.get(
                        "regions",
                        [],
                    )

                    if regions is not None:
                        if not isinstance(
                            regions,
                            list,
                        ):
                            errors.append(
                                f"{provider} regions must be a list."
                            )
                        else:
                            for region in regions:
                                if not ValidationUtils.validate_region(
                                    str(region),
                                    provider,
                                ):
                                    errors.append(
                                        f"Invalid {provider} "
                                        f"region: {region}"
                                    )

        budget = config.get("budget")

        if budget is not None:
            if not isinstance(budget, dict):
                errors.append(
                    "budget must be an object."
                )
            else:
                if "monthly_limit" in budget:
                    if not ValidationUtils.validate_cost_amount(
                        budget["monthly_limit"]
                    ):
                        errors.append(
                            "Invalid monthly budget limit: "
                            f"{budget['monthly_limit']}"
                        )

                for field in (
                    "warning_threshold",
                    "critical_threshold",
                ):
                    if field in budget:
                        if not ValidationUtils.validate_budget_threshold(
                            budget[field]
                        ):
                            errors.append(
                                f"Invalid {field}: "
                                f"{budget[field]}"
                            )

                if (
                    "warning_threshold" in budget
                    and "critical_threshold" in budget
                ):
                    try:
                        if (
                            float(budget["warning_threshold"])
                            >= float(
                                budget["critical_threshold"]
                            )
                        ):
                            errors.append(
                                "warning_threshold must be "
                                "less than critical_threshold."
                            )
                    except (
                        TypeError,
                        ValueError,
                    ):
                        pass

                if "currency" in budget:
                    if not ValidationUtils.validate_currency(
                        budget["currency"]
                    ):
                        errors.append(
                            f"Invalid currency: "
                            f"{budget['currency']}"
                        )

        return len(errors) == 0, errors

    @staticmethod
    def validate_billing_data(
        billing_data: List[Dict[str, Any]],
    ) -> tuple[bool, List[str]]:
        errors: List[str] = []

        if not isinstance(billing_data, list):
            return False, [
                "Billing data must be a list."
            ]

        required_fields = {
            "provider",
            "account_id",
            "service",
            "cost",
            "currency",
            "timestamp",
        }

        for index, data in enumerate(billing_data):
            prefix = f"Record {index}"

            if not isinstance(data, dict):
                errors.append(
                    f"{prefix}: must be an object."
                )
                continue

            missing = required_fields - set(
                data.keys()
            )

            for field in sorted(missing):
                errors.append(
                    f"{prefix}: Missing required field: "
                    f"{field}"
                )

            if "cost" in data:
                if not ValidationUtils.validate_cost_amount(
                    data["cost"]
                ):
                    errors.append(
                        f"{prefix}: Invalid cost amount: "
                        f"{data['cost']}"
                    )

            if "currency" in data:
                if not ValidationUtils.validate_currency(
                    str(data["currency"])
                ):
                    errors.append(
                        f"{prefix}: Invalid currency: "
                        f"{data['currency']}"
                    )

            if "provider" in data:
                provider = str(
                    data["provider"]
                ).strip().lower()

                if provider not in {
                    "aws",
                    "azure",
                    "gcp",
                }:
                    errors.append(
                        f"{prefix}: Invalid provider: "
                        f"{data['provider']}"
                    )

            if "timestamp" in data:
                try:
                    ValidationUtils.parse_datetime(
                        data["timestamp"]
                    )
                except (ValueError, TypeError):
                    errors.append(
                        f"{prefix}: Invalid timestamp: "
                        f"{data['timestamp']}"
                    )

        return len(errors) == 0, errors

    @staticmethod
    def validate_alert_config(
        alert_config: Dict[str, Any],
    ) -> tuple[bool, List[str]]:
        errors: List[str] = []

        if not isinstance(alert_config, dict):
            return False, [
                "Alert configuration must be an object."
            ]

        if "type" not in alert_config:
            errors.append(
                "Missing alert type"
            )
        else:
            alert_type = str(
                alert_config["type"]
            ).strip().lower()

            valid_types = {
                "email",
                "webhook",
                "slack",
            }

            if alert_type not in valid_types:
                errors.append(
                    f"Invalid alert type: "
                    f"{alert_config['type']}"
                )

        if "enabled" in alert_config:
            if not isinstance(
                alert_config["enabled"],
                bool,
            ):
                errors.append(
                    "Enabled field must be boolean"
                )

        if "severity" in alert_config:
            severity = str(
                alert_config["severity"]
            ).strip().lower()

            if severity not in {
                "low",
                "medium",
                "high",
                "critical",
            }:
                errors.append(
                    f"Invalid severity: "
                    f"{alert_config['severity']}"
                )

        alert_type = str(
            alert_config.get("type", "")
        ).strip().lower()

        if alert_type == "email":
            required_fields = (
                "smtp_server",
                "username",
                "from_email",
                "to_emails",
            )

            for field in required_fields:
                if field not in alert_config:
                    errors.append(
                        f"Email alert missing required "
                        f"field: {field}"
                    )

            if "from_email" in alert_config:
                if not ValidationUtils.validate_email(
                    alert_config["from_email"]
                ):
                    errors.append(
                        "Invalid from_email address."
                    )

            if "to_emails" in alert_config:
                recipients = alert_config["to_emails"]

                if not isinstance(
                    recipients,
                    list,
                ):
                    errors.append(
                        "to_emails must be a list"
                    )
                else:
                    for email in recipients:
                        if not ValidationUtils.validate_email(
                            email
                        ):
                            errors.append(
                                f"Invalid email address: "
                                f"{email}"
                            )

        elif alert_type == "webhook":
            if "url" not in alert_config:
                errors.append(
                    "Webhook alert missing required "
                    "field: url"
                )
            elif not ValidationUtils.validate_webhook_url(
                alert_config["url"]
            ):
                errors.append(
                    f"Invalid webhook URL: "
                    f"{alert_config['url']}"
                )

        elif alert_type == "slack":
            if "webhook_url" not in alert_config:
                errors.append(
                    "Slack alert missing required "
                    "field: webhook_url"
                )
            elif not ValidationUtils.validate_slack_webhook(
                alert_config["webhook_url"]
            ):
                errors.append(
                    f"Invalid Slack webhook URL: "
                    f"{alert_config['webhook_url']}"
                )

        return len(errors) == 0, errors

    @staticmethod
    def require_valid(
        condition: bool,
        message: str,
    ) -> None:
        if not condition:
            raise ValidationError(message)

    @staticmethod
    def validate_provider(
        provider: str,
    ) -> bool:
        return str(provider).strip().lower() in {
            "aws",
            "azure",
            "gcp",
        }

    @staticmethod
    def validate_risk_level(
        risk_level: str,
    ) -> bool:
        return str(risk_level).strip().lower() in {
            "low",
            "medium",
            "high",
            "critical",
        }

    @staticmethod
    def validate_confidence(
        confidence: Union[str, float, int],
    ) -> bool:
        try:
            value = float(confidence)
            return 0 <= value <= 1
        except (
            ValueError,
            TypeError,
            OverflowError,
        ):
            return False