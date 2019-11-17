import logging
import logzero
import os
from logzero import logger

class Logger(object):
    def __init__(self, log_dir):
        """Create a summary writer logging to log_dir."""
        # formatter = logging.Formatter('%(name)s - %(asctime)-15s - %(levelname)s: %(message)s');
        # logzero.formatter(formatter)
        logzero.logfile(os.path.join(log_dir, 'rotate-log.log'), maxBytes=1e8, loglevel=logging.INFO)

    def scalar_summary(self, tag, value, step):
        """Log a scalar variable."""
        logger.info("step: %d \ttag: %s\t value: %s" % (step, tag, value))

    def list_of_scalars_summary(self, tag_value_pairs, step):
        """Log scalar variables."""
        logger.info("step: %d \ttag and value: %s" % (step, tag_value_pairs))
    def info(self, msg):
        logger.info(msg)
