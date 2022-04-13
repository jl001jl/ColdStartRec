import logging
import os
import time


def check_and_mkdir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

def set_log(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s', log_data_dir=None):

    if log_data_dir:
        log_file = os.path.join(log_data_dir, 'info.log')
        logging.basicConfig(level=level,
                            format=format,
                            handlers=[logging.FileHandler(log_file, mode='w'),
                                      logging.StreamHandler()])
    else:
        logging.basicConfig(level=level,
                            format=format,
                            handlers=[logging.StreamHandler()])