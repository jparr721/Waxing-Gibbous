import os
from time import sleep

from loguru import logger


def normalize(value: float, min_: float, max_: float) -> float:
    return (value - min_) / (max_ - min_)


def denormalize(value: float, min_: float, max_: float) -> float:
    return (value + min_) * (max_ - min_)


def poll_ram():
    while True:
        sleep(1)

        try:
            used = int(
                os.popen("free --giga --total | awk '/^Total:/ {print $3}'").read()
            )

            if used > 31:
                logger.error("Memory limit reached, killing process")
                logger.warning("Memory will not be cleaned up, I suggest a reboot")
                os._exit(1)
        except Exception as e:
            logger.error(f"Error getting memory reading! This is dangerous! {e}")
