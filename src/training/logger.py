import logging


def setup_logging(log_file, level):
    stream_handler = logging.StreamHandler()
    file_handler = logging.FileHandler(filename=log_file)

    formatter = logging.Formatter(
        '%(asctime)s | %(levelname)s | %(message)s', 
        datefmt='%Y-%m-%d,%H:%M:%S')

    stream_handler.setFormatter(formatter)
    stream_handler.setLevel(level)
    file_handler.setFormatter(formatter)
    file_handler.setLevel(level)

    root_logger = logging.getLogger()
    root_logger.addHandler(stream_handler)
    root_logger.addHandler(file_handler)
    root_logger.setLevel(level)
