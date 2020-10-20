import logging

import coloredlogs
from rich.console import Console
from rich.traceback import install


logger = logging.getLogger(__name__)


def setup_logger(logger=None, log_level=logging.INFO):
    console = Console()
    install()

    # install logger
    if logger is not None:
        # Clear any previous handlers
        if logger.hasHandlers():
            logger.handlers.clear()

        # Configure colored logging
        coloredlogs.install(
            logger=logger,
            format="%(levelname)-8s %(name)-40s:%(lineno)4d - %(message)-50s",
            level=log_level,
        )

    return console


console = setup_logger(log_level=logger)
