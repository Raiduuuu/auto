"""
Logging configuration for TradingAgents
"""
import sys
from loguru import logger


def setup_logger(
    level: str = "INFO",
    log_file: str = "trading_agents.log"
) -> None:
    """Configure loguru logger."""
    # Remove default handler
    logger.remove()

    # Console handler
    logger.add(
        sys.stderr,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
        level=level,
        colorize=True
    )

    # File handler
    logger.add(
        log_file,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
        level=level,
        rotation="10 MB",
        retention="7 days",
        compression="zip"
    )

    logger.info(f"Logger initialized with level: {level}")
