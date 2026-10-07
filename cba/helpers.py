import csv
import io
import json
import math
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from .core.exceptions import CloudBillingError


class FormatUtils:
    @staticmethod
    def format_currency(
        amount: float,
        currency: str = "USD",
        include_symbol: bool = True,
    ) -> str:
        try:
            amount = float(amount)
        except (TypeError, ValueError):
            amount = 0.0

        currency = str(currency).upper()

        if not include_symbol:
            return f"{amount:,.2f} {currency}"

        symbols = {
            "USD": "$",
            "EUR": "€",
            "GBP": "£",
            "JPY": "¥",
            "CAD": "C$",
            "AUD": "A$",
            "CHF": "Fr",
            "CNY": "¥",
            "INR": "₹",
        }

        return f"{symbols.get(currency, currency + ' ')}{amount:,.2f}"

    @staticmethod
    def format_percentage(
        value: float,
        decimal_places: int = 1,
    ) -> str:
        try:
            value = float(value)
        except (TypeError, ValueError):
            value = 0.0

        return f"{value:.{max(0, decimal_places)}f}%"

    @staticmethod
    def format_number(
        number: Union[int, float],
        decimal_places: int = 2,
    ) -> str:
        if isinstance(number, bool):
            return str(number)

        if isinstance(number, int):
            return f"{number:,}"

        try:
            return f"{float(number):,.{max(0, decimal_places)}f}"
        except (TypeError, ValueError):
            return str(number)

    @staticmethod
    def format_bytes(bytes_value: Union[int, float]) -> str:
        try:
            value = float(bytes_value)
        except (TypeError, ValueError):
            return "0 B"

        if value <= 0:
            return "0 B"

        units = ("B", "KB", "MB", "GB", "TB", "PB")
        index = min(
            int(math.floor(math.log(value, 1024))),
            len(units) - 1,
        )

        size = value / (1024 ** index)

        if index == 0:
            return f"{int(value)} B"

        return f"{size:.1f} {units[index]}"

    @staticmethod
    def format_duration(
        start_time: datetime,
        end_time: datetime,
    ) -> str:
        duration = end_time - start_time
        total_seconds = max(0, int(duration.total_seconds()))

        days, remainder = divmod(total_seconds, 86400)
        hours, remainder = divmod(remainder, 3600)
        minutes, seconds = divmod(remainder, 60)

        if days:
            return f"{days}d {hours}h"

        if hours:
            return f"{hours}h {minutes}m"

        if minutes:
            return f"{minutes}m {seconds}s"

        return f"{seconds}s"

    @staticmethod
    def format_relative_time(timestamp: datetime) -> str:
        if timestamp.tzinfo is not None:
            timestamp = timestamp.astimezone(timezone.utc).replace(
                tzinfo=None
            )

        now = datetime.utcnow()
        diff_seconds = int((now - timestamp).total_seconds())

        if diff_seconds < 0:
            future_seconds = abs(diff_seconds)

            if future_seconds >= 86400:
                return f"in {future_seconds // 86400} days"

            if future_seconds >= 3600:
                return f"in {future_seconds // 3600} hours"

            if future_seconds >= 60:
                return f"in {future_seconds // 60} minutes"

            return f"in {future_seconds} seconds"

        if diff_seconds >= 86400:
            days = diff_seconds // 86400
            return f"{days} day{'s' if days != 1 else ''} ago"

        if diff_seconds >= 3600:
            hours = diff_seconds // 3600
            return f"{hours} hour{'s' if hours != 1 else ''} ago"

        if diff_seconds >= 60:
            minutes = diff_seconds // 60
            return f"{minutes} minute{'s' if minutes != 1 else ''} ago"

        return f"{diff_seconds} second{'s' if diff_seconds != 1 else ''} ago"

    @staticmethod
    def truncate_string(
        text: str,
        max_length: int = 50,
        suffix: str = "...",
    ) -> str:
        text = str(text)
        suffix = str(suffix)

        if max_length <= 0:
            return ""

        if len(text) <= max_length:
            return text

        if len(suffix) >= max_length:
            return suffix[:max_length]

        return text[: max_length - len(suffix)] + suffix

    @staticmethod
    def format_list(
        items: List[str],
        max_items: int = 3,
        item_prefix: str = "• ",
    ) -> str:
        if not items:
            return ""

        max_items = max(0, max_items)
        shown = items[:max_items]

        lines = [
            f"{item_prefix}{item}"
            for item in shown
        ]

        remaining = len(items) - len(shown)

        if remaining > 0:
            lines.append(
                f"{item_prefix}... and {remaining} more"
            )

        return "\n".join(lines)

    @staticmethod
    def format_tags(
        tags: Dict[str, str],
        max_tags: int = 5,
    ) -> str:
        if not tags:
            return "No tags"

        items = [
            f"{key}: {value}"
            for key, value in list(tags.items())[:max_tags]
        ]

        remaining = len(tags) - len(items)

        if remaining > 0:
            items.append(f"... and {remaining} more")

        return ", ".join(items)

    @staticmethod
    def format_table_data(
        data: List[Dict[str, Any]],
        headers: List[str],
    ) -> List[List[str]]:
        result = [list(headers)]

        for row in data:
            formatted_row = []

            for header in headers:
                value = row.get(header, "")

                if isinstance(value, bool):
                    formatted_value = "✓" if value else "✗"
                elif isinstance(value, int):
                    formatted_value = f"{value:,}"
                elif isinstance(value, float):
                    formatted_value = f"{value:,.2f}"
                elif isinstance(value, datetime):
                    formatted_value = value.strftime(
                        "%Y-%m-%d %H:%M"
                    )
                elif isinstance(value, list):
                    formatted_value = f"[{len(value)} items]"
                elif isinstance(value, dict):
                    formatted_value = f"[{len(value)} keys]"
                else:
                    formatted_value = str(value)

                formatted_row.append(
                    FormatUtils.truncate_string(
                        formatted_value,
                        30,
                    )
                )

            result.append(formatted_row)

        return result

    @staticmethod
    def format_error_message(
        error: Exception,
        include_traceback: bool = False,
    ) -> str:
        message = f"{type(error).__name__}: {error}"

        if include_traceback:
            import traceback

            message += (
                "\n\nTraceback:\n"
                + traceback.format_exc()
            )

        return message

    @staticmethod
    def format_progress_bar(
        current: int,
        total: int,
        width: int = 50,
    ) -> str:
        width = max(1, width)

        if total <= 0:
            return "[" + "=" * width + "] 100.0%"

        current = max(0, min(current, total))
        filled = int(width * current / total)
        empty = width - filled
        percentage = current / total * 100

        bar = "[" + "=" * filled + " " * empty + "]"

        return f"{bar} {percentage:.1f}%"

    @staticmethod
    def format_status_badge(
        status: str,
        color_map: Optional[Dict[str, str]] = None,
    ) -> str:
        status = str(status)
        mapping = color_map or {
            "active": "🟢",
            "inactive": "🔴",
            "pending": "🟡",
            "completed": "✅",
            "failed": "❌",
            "warning": "⚠️",
            "info": "ℹ️",
            "success": "✅",
            "error": "❌",
        }

        icon = mapping.get(status.lower(), "")
        return f"{icon} {status.title()}".strip()


class DateUtils:
    @staticmethod
    def get_date_range(
        period: str,
        end_date: Optional[datetime] = None,
    ) -> tuple[datetime, datetime]:
        if end_date is None:
            end_date = datetime.now()

        period = str(period).strip().lower()

        if period == "today":
            start_date = end_date.replace(
                hour=0,
                minute=0,
                second=0,
                microsecond=0,
            )

        elif period == "yesterday":
            start_date = (
                end_date - timedelta(days=1)
            ).replace(
                hour=0,
                minute=0,
                second=0,
                microsecond=0,
            )

            end_date = start_date + timedelta(days=1) - timedelta(
                microseconds=1
            )

        elif period == "this_week":
            start_date = (
                end_date - timedelta(days=end_date.weekday())
            ).replace(
                hour=0,
                minute=0,
                second=0,
                microsecond=0,
            )

        elif period == "last_week":
            start_date = (
                end_date
                - timedelta(days=end_date.weekday() + 7)
            ).replace(
                hour=0,
                minute=0,
                second=0,
                microsecond=0,
            )

            end_date = start_date + timedelta(days=7) - timedelta(
                microseconds=1
            )

        elif period == "this_month":
            start_date = end_date.replace(
                day=1,
                hour=0,
                minute=0,
                second=0,
                microsecond=0,
            )

        elif period == "last_month":
            current_month_start = end_date.replace(
                day=1,
                hour=0,
                minute=0,
                second=0,
                microsecond=0,
            )

            end_date = current_month_start - timedelta(
                microseconds=1
            )

            start_date = end_date.replace(
                day=1,
                hour=0,
                minute=0,
                second=0,
                microsecond=0,
            )

        elif period == "this_year":
            start_date = end_date.replace(
                month=1,
                day=1,
                hour=0,
                minute=0,
                second=0,
                microsecond=0,
            )

        elif period == "last_year":
            start_date = end_date.replace(
                year=end_date.year - 1,
                month=1,
                day=1,
                hour=0,
                minute=0,
                second=0,
                microsecond=0,
            )

            end_date = start_date.replace(
                year=start_date.year + 1
            ) - timedelta(microseconds=1)

        else:
            try:
                days = int(period)

                if days < 0:
                    raise ValueError

                start_date = end_date - timedelta(days=days)
                start_date = start_date.replace(
                    hour=0,
                    minute=0,
                    second=0,
                    microsecond=0,
                )
            except ValueError as exc:
                raise CloudBillingError(
                    f"Invalid period: {period}"
                ) from exc

        return start_date, end_date

    @staticmethod
    def parse_date_string(date_str: str) -> datetime:
        if isinstance(date_str, datetime):
            return date_str

        value = str(date_str).strip()

        formats = (
            "%Y-%m-%d",
            "%Y-%m-%d %H:%M:%S",
            "%Y-%m-%dT%H:%M:%S",
            "%Y-%m-%d %H:%M",
            "%Y-%m-%dT%H:%M",
            "%Y-%m-%dT%H:%M:%S.%f",
        )

        normalized = value.replace("Z", "+00:00")

        try:
            return datetime.fromisoformat(normalized)
        except ValueError:
            pass

        for fmt in formats:
            try:
                return datetime.strptime(value, fmt)
            except ValueError:
                continue

        raise CloudBillingError(
            f"Unable to parse date string: {date_str}"
        )

    @staticmethod
    def get_month_start_end(
        date: datetime,
    ) -> tuple[datetime, datetime]:
        start = date.replace(
            day=1,
            hour=0,
            minute=0,
            second=0,
            microsecond=0,
        )

        if start.month == 12:
            next_month = start.replace(
                year=start.year + 1,
                month=1,
            )
        else:
            next_month = start.replace(
                month=start.month + 1
            )

        end = next_month - timedelta(microseconds=1)

        return start, end

    @staticmethod
    def get_week_start_end(
        date: datetime,
    ) -> tuple[datetime, datetime]:
        start = (
            date - timedelta(days=date.weekday())
        ).replace(
            hour=0,
            minute=0,
            second=0,
            microsecond=0,
        )

        end = start + timedelta(days=7) - timedelta(
            microseconds=1
        )

        return start, end

    @staticmethod
    def is_weekend(date: datetime) -> bool:
        return date.weekday() >= 5

    @staticmethod
    def is_business_day(date: datetime) -> bool:
        return not DateUtils.is_weekend(date)

    @staticmethod
    def get_business_days(
        start_date: datetime,
        end_date: datetime,
    ) -> int:
        if start_date > end_date:
            return 0

        count = 0
        current = start_date

        while current.date() <= end_date.date():
            if DateUtils.is_business_day(current):
                count += 1

            current += timedelta(days=1)

        return count

    @staticmethod
    def add_business_days(
        date: datetime,
        days: int,
    ) -> datetime:
        if days < 0:
            raise CloudBillingError(
                "days must be non-negative."
            )

        current = date
        added = 0

        while added < days:
            current += timedelta(days=1)

            if DateUtils.is_business_day(current):
                added += 1

        return current

    @staticmethod
    def get_quarter_start_end(
        date: datetime,
    ) -> tuple[datetime, datetime]:
        quarter = (date.month - 1) // 3
        start_month = quarter * 3 + 1

        start = datetime(
            date.year,
            start_month,
            1,
        )

        if start_month == 10:
            next_quarter = datetime(
                date.year + 1,
                1,
                1,
            )
        else:
            next_quarter = datetime(
                date.year,
                start_month + 3,
                1,
            )

        end = next_quarter - timedelta(
            microseconds=1
        )

        return start, end

    @staticmethod
    def get_fiscal_year_start_end(
        fy_start_month: int = 1,
        fy_start_day: int = 1,
        current_date: Optional[datetime] = None,
    ) -> tuple[datetime, datetime]:
        if not 1 <= fy_start_month <= 12:
            raise CloudBillingError(
                "fy_start_month must be between 1 and 12."
            )

        current_date = current_date or datetime.now()

        try:
            fiscal_start = datetime(
                current_date.year,
                fy_start_month,
                fy_start_day,
            )
        except ValueError as exc:
            raise CloudBillingError(
                "Invalid fiscal year start date."
            ) from exc

        if current_date < fiscal_start:
            fiscal_start = fiscal_start.replace(
                year=fiscal_start.year - 1
            )

        fiscal_end = fiscal_start.replace(
            year=fiscal_start.year + 1
        ) - timedelta(microseconds=1)

        return fiscal_start, fiscal_end


class DataUtils:
    @staticmethod
    def deep_merge_dict(
        dict1: Dict[str, Any],
        dict2: Dict[str, Any],
    ) -> Dict[str, Any]:
        result = dict(dict1)

        for key, value in dict2.items():
            if (
                key in result
                and isinstance(result[key], dict)
                and isinstance(value, dict)
            ):
                result[key] = DataUtils.deep_merge_dict(
                    result[key],
                    value,
                )
            else:
                result[key] = value

        return result

    @staticmethod
    def flatten_dict(
        d: Dict[str, Any],
        parent_key: str = "",
        sep: str = ".",
    ) -> Dict[str, Any]:
        result: Dict[str, Any] = {}

        for key, value in d.items():
            new_key = (
                f"{parent_key}{sep}{key}"
                if parent_key
                else str(key)
            )

            if isinstance(value, dict):
                result.update(
                    DataUtils.flatten_dict(
                        value,
                        new_key,
                        sep,
                    )
                )
            else:
                result[new_key] = value

        return result

    @staticmethod
    def safe_get(
        data: Dict[str, Any],
        key: str,
        default: Any = None,
    ) -> Any:
        if not key:
            return data

        current: Any = data

        for part in str(key).split("."):
            if isinstance(current, dict) and part in current:
                current = current[part]
            else:
                return default

        return current

    @staticmethod
    def safe_set(
        data: Dict[str, Any],
        key: str,
        value: Any,
    ) -> None:
        parts = str(key).split(".")

        if not parts or not parts[0]:
            raise CloudBillingError(
                "Key must not be empty."
            )

        current = data

        for part in parts[:-1]:
            if part not in current:
                current[part] = {}

            if not isinstance(current[part], dict):
                raise CloudBillingError(
                    f"Cannot set nested key through non-dict value: {part}"
                )

            current = current[part]

        current[parts[-1]] = value

    @staticmethod
    def convert_to_csv(
        data: List[Dict[str, Any]],
    ) -> str:
        if not data:
            return ""

        headers = list(
            dict.fromkeys(
                key
                for row in data
                for key in row.keys()
            )
        )

        output = io.StringIO()
        writer = csv.writer(output)

        writer.writerow(headers)

        for row in data:
            values = []

            for header in headers:
                value = row.get(header, "")

                if isinstance(value, (dict, list, tuple)):
                    value = json.dumps(
                        value,
                        default=str,
                    )

                elif isinstance(value, datetime):
                    value = value.isoformat()

                values.append(value)

            writer.writerow(values)

        return output.getvalue().rstrip("\r\n")

    @staticmethod
    def convert_to_json(
        data: Any,
        indent: int = 2,
    ) -> str:
        return json.dumps(
            data,
            indent=indent,
            default=str,
            ensure_ascii=False,
        )

    @staticmethod
    def calculate_percent_change(
        old_value: float,
        new_value: float,
    ) -> float:
        if old_value == 0:
            if new_value == 0:
                return 0.0

            return float("inf") if new_value > 0 else float("-inf")

        return (
            (new_value - old_value)
            / abs(old_value)
        ) * 100

    @staticmethod
    def calculate_average(
        values: List[float],
    ) -> float:
        if not values:
            return 0.0

        return sum(values) / len(values)

    @staticmethod
    def calculate_median(
        values: List[float],
    ) -> float:
        if not values:
            return 0.0

        ordered = sorted(values)
        count = len(ordered)
        middle = count // 2

        if count % 2 == 0:
            return (
                ordered[middle - 1]
                + ordered[middle]
            ) / 2

        return ordered[middle]

    @staticmethod
    def calculate_std_dev(
        values: List[float],
    ) -> float:
        if len(values) < 2:
            return 0.0

        mean = DataUtils.calculate_average(values)

        variance = sum(
            (value - mean) ** 2
            for value in values
        ) / len(values)

        return math.sqrt(variance)

    @staticmethod
    def round_to_significant(
        value: float,
        sig_figs: int = 3,
    ) -> float:
        if value == 0:
            return 0.0

        if sig_figs <= 0:
            raise CloudBillingError(
                "sig_figs must be greater than zero."
            )

        magnitude = 10 ** (
            sig_figs
            - 1
            - int(math.floor(math.log10(abs(value))))
        )

        return round(value * magnitude) / magnitude

    @staticmethod
    def safe_divide(
        numerator: float,
        denominator: float,
        default: float = 0.0,
    ) -> float:
        if denominator == 0:
            return default

        return numerator / denominator