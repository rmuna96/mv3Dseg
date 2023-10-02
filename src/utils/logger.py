import sys
import logging
import os
import os.path as osp

class Logger():
    def __init__(self, odir, name):
        self.odir = odir
        self.name = name

    def get_logger(self,):
        log_dir = osp.join(self.odir)
        os.makedirs(log_dir, exist_ok=True)
        logger = logging.getLogger('log')
        logger.setLevel(logging.DEBUG)
        if osp.exists(osp.join(log_dir, f'{self.name}.log')):
            os.remove(osp.join(log_dir, f'{self.name}.log'))
        output_file_handler = logging.FileHandler(osp.join(log_dir, f'{self.name}.log'))
        logger.addHandler(output_file_handler)
        logger.addHandler(logging.StreamHandler(sys.stdout))
        return logger

