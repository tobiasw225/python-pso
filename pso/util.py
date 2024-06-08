import logging
import logging.config


def singleton(c):
    """
        very simple singleton

    :param c:
    :return:
    """
    instance = None

    def init(*args, **kwargs):
        nonlocal instance
        if not instance:
            instance = c(*args, **kwargs)
        return instance

    return init


def get_my_logger(fout_name="", log_level=logging.INFO):
    """

    :param fout_name:
    :param log_level:
    :return:
    """
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logger = logging.getLogger(__name__)
    logger.setLevel(log_level)
    handler = logging.handlers.RotatingFileHandler(fout_name)
    handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter(log_format)
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logging.basicConfig(level=log_level, format=log_format)
    return logger
