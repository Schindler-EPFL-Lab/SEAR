import logging
from pathlib import Path

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

formatter = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s", "%Y-%m-%d %H:%M:%S"
)

console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)

logger.addHandler(console_handler)

log_file = Path("outputs/sear.log")
log_file.parent.mkdir(parents=True, exist_ok=True)
file_handler = logging.FileHandler(log_file)
file_handler.setFormatter(formatter)

logger.addHandler(file_handler)
