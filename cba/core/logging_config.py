import logging
import logging.handlers
import sys
from pathlib import Path
from typing import Optional

from rich.console import Console
from rich.logging import RichHandler


def setup_logging(
    level: str = "INFO",
    log_file: Optional[str] = None,
    enable_rich: bool = True,
    debug: bool = False,
) -> None:
    numeric_level = getattr(logging, str(level).upper(), logging.INFO)

    if debug:
        numeric_level = logging.DEBUG

    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)

    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
        try:
            handler.close()
        except Exception:
            pass

    if enable_rich:
        console_handler = RichHandler(
            console=Console(stderr=True),
            show_time=True,
            show_path=debug,
            markup=True,
            rich_tracebacks=True,
        )
    else:
        console_handler = logging.StreamHandler(sys.stderr)
        console_handler.setFormatter(
            logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
        )

    console_handler.setLevel(numeric_level)
    root_logger.addHandler(console_handler)

    if log_file:
        log_path = Path(log_file).expanduser()
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.handlers.RotatingFileHandler(
            filename=log_path,
            maxBytes=10 * 1024 * 1024,
            backupCount=5,
            encoding="utf-8",
        )

        file_handler.setLevel(numeric_level)
        file_handler.setFormatter(
            logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - "
                "%(pathname)s:%(lineno)d - %(message)s"
            )
        )

        root_logger.addHandler(file_handler)

    noisy_loggers = (
        "boto3",
        "botocore",
        "urllib3",
        "azure",
        "google",
        "google.auth",
        "google.api_core",
        "httpx",
        "httpcore",
    )

    for logger_name in noisy_loggers:
        logging.getLogger(logger_name).setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)