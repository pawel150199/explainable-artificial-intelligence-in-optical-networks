import sys
import logging
from typing import Any

def configureLogger(debug=False) -> Any:
    """
    Function configure logger

    Returns:
        Any: Logger from colorlog class
    
    Usage::

        from loggers import configureLogger

        logger = configureLogger()
        logger.info("Logger is working fine!")

    """  
    logger= logging.getLogger('xai-experiment')
    logger.setLevel(logging.DEBUG if debug else logging.INFO)
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("%(asctime)s — %(levelname)s — %(message)s", "%Y-%M-%d %H:%M:%S"))
    logger.addHandler(handler)
    
    return logger