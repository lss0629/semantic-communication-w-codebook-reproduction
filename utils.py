import numpy as np
import math
import torch
import random
import os
import logging
import time


def windows_safe_timestamp():
    """Return a timestamp that is valid in Windows file and directory names."""
    return time.strftime('%Y-%m-%d_%H-%M-%S')


def _sanitize_filename_component(value):
    return str(value).replace(':', '-').replace(' ', '_')


class AverageMeter:
    """Compute running average."""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def clear(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0


def logger_configuration(config, save_log=False, test_mode=False):
    if hasattr(config, 'filename'):
        safe_filename = _sanitize_filename_component(config.filename)
        if safe_filename != config.filename:
            config.filename = safe_filename
            config.workdir = os.path.join('.', 'history', safe_filename)
            config.log = os.path.join(config.workdir, 'Log_{}.log'.format(safe_filename))
            config.samples = os.path.join(config.workdir, 'samples')
            config.models = os.path.join(config.workdir, 'models')
    # 配置 logger
    logger = logging.getLogger("Deep joint source channel coder")
    if test_mode:
        config.workdir += '_test'
    if save_log:
        makedirs(config.workdir)
        makedirs(config.samples)
        makedirs(config.models)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s] %(message)s')
    stdhandler = logging.StreamHandler()
    stdhandler.setLevel(logging.INFO)
    stdhandler.setFormatter(formatter)
    logger.addHandler(stdhandler)
    if save_log:
        filehandler = logging.FileHandler(config.log)
        filehandler.setLevel(logging.INFO)
        filehandler.setFormatter(formatter)
        logger.addHandler(filehandler)
    logger.setLevel(logging.INFO)
    config.logger = logger
    return config.logger

def makedirs(directory):
    os.makedirs(directory, exist_ok=True)

def save_model(model, save_path):
    parent = os.path.dirname(os.path.abspath(save_path))
    makedirs(parent)
    torch.save(model.state_dict(), save_path)


def seed_torch(seed=1029):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)  # 为了禁止hash随机化，使得实验可复现
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
