import logging
import os
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional

from rich.console import Console
from rich.logging import RichHandler
from rich.theme import Theme
from rich.traceback import install

install(show_locals=True)

custom_theme = Theme(
    {
        "info": "cyan",
        "warning": "yellow",
        "error": "red",
        "critical": "red reverse",
        "debug": "grey70",
        "timestamp": "green",
    }
)

console = Console(theme=custom_theme)


class RichRotatingFileHandler(RotatingFileHandler):
    """File handler that strips rich markup from log messages."""

    def emit(self, record: logging.LogRecord) -> None:
        from rich.markup import escape

        record.msg = escape(str(record.msg))
        super().emit(record)


def setup_logger(
    name: str,
    log_file: Optional[Path] = None,
    level: Optional[int] = None,
    rich_tracebacks: bool = True,
) -> logging.Logger:
    """Configure and return a logger with rich formatting.

    Args:
        name: Logger name (typically __name__ from the calling module).
        log_file: Optional path to write log file.
        level: Logging level; defaults to environment-based level.
        rich_tracebacks: Enable rich traceback formatting.

    Returns:
        Configured logger instance.
    """
    env_level = {
        "development": logging.DEBUG,
        "production": logging.INFO,
        "testing": logging.WARNING,
    }.get(os.getenv("ENVIRONMENT", "development"), logging.INFO)

    level = level or env_level

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.handlers = []

    rich_handler = RichHandler(
        console=console,
        rich_tracebacks=rich_tracebacks,
        markup=True,
        show_time=True,
        show_level=True,
        show_path=True,
        enable_link_path=True,
    )
    rich_handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(rich_handler)

    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = RichRotatingFileHandler(
            log_file, maxBytes=10 * 1024 * 1024, backupCount=5
        )
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        )
        logger.addHandler(file_handler)

    return logger


# default_logger = setup_logger("eo_pipe")


def log_step_start(logger: logging.Logger, step_name: str, **kwargs: object) -> None:
    """Log the start of a processing step."""
    params = " ".join(f"{k}={v}" for k, v in kwargs.items())
    logger.info(f"[bold blue]Starting {step_name}[/] {params}")


def log_step_complete(
    logger: logging.Logger, step_name: str, duration: Optional[float] = None
) -> None:
    """Log the completion of a processing step."""
    duration_str = f" ([green]{duration:.2f}s[/])" if duration is not None else ""
    logger.info(f"[bold green]Completed {step_name}{duration_str}[/]")


def log_error(
    logger: logging.Logger, message: str, error: Optional[Exception] = None
) -> None:
    """Log an error with optional exception details."""
    if error:
        logger.error(f"[red]{message}[/]: {str(error)}")
    else:
        logger.error(f"[red]{message}[/]")
