import logging

from config import LOG_FORMAT, LOG_LEVEL


class Logger:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            logging.basicConfig(
                level=getattr(logging, LOG_LEVEL.upper()),
                format=LOG_FORMAT,
                handlers=[
                    logging.StreamHandler(),
                    logging.FileHandler("tech_radar_chat.log"),
                ],
            )
            cls._instance = logging.getLogger("tech_radar")
        return cls._instance

# Create singleton instance
logger = Logger()
