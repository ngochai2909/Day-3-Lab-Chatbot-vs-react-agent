import logging
import json
import os
from datetime import datetime
from typing import Any, Dict


class IndustryLogger:
    """
    Structured logger that simulates industry practices.

    - File handler: writes full JSON events to logs/<date>.log (for telemetry analysis).
    - Console handler: stays QUIET by default (only warnings/errors) so the
      terminal isn't flooded with JSON. Set LOG_CONSOLE=1 to echo JSON to console.
    """

    def __init__(self, name: str = "AI-Lab-Agent", log_dir: str = "logs"):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.INFO)
        self.logger.propagate = False

        # Avoid duplicate handlers if the logger is re-instantiated.
        if self.logger.handlers:
            self.logger.handlers.clear()

        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        # File Handler (full JSON, always on)
        log_file = os.path.join(log_dir, f"{datetime.now().strftime('%Y-%m-%d')}.log")
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(logging.INFO)
        self.logger.addHandler(file_handler)

        # Console Handler (quiet by default)
        console_handler = logging.StreamHandler()
        verbose = os.getenv("LOG_CONSOLE", "0") == "1"
        console_handler.setLevel(logging.INFO if verbose else logging.WARNING)
        self.logger.addHandler(console_handler)

    def log_event(self, event_type: str, data: Dict[str, Any]):
        """Logs an event with a timestamp and type (JSON) to the file."""
        payload = {
            "timestamp": datetime.utcnow().isoformat(),
            "event": event_type,
            "data": data,
        }
        self.logger.info(json.dumps(payload, ensure_ascii=False))

    def info(self, msg: str):
        self.logger.info(msg)

    def error(self, msg: str, exc_info=True):
        self.logger.error(msg, exc_info=exc_info)


# Global logger instance
logger = IndustryLogger()
