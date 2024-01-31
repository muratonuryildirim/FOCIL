import os
import copy
import random
import warnings
import optuna
import datetime
import logging
import hashlib
import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

logger = None

def setup(args):
    if not os.path.exists('./models'): os.mkdir('./models')
    if not os.path.exists('./logs'): os.mkdir('./logs')
    if not os.path.exists('./masks'): os.mkdir('./masks')

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    warnings.filterwarnings("ignore", category=UserWarning)

    set_logger(args)

    print_and_log(args)
    print_and_log('=' * 60)
    print_and_log('Dataset: {0}, #Task:{1}, #ClassPerTask:{2}'.format(args.dataset, args.num_tasks, args.num_classes_per_task))
    print_and_log('Model: {0}'.format(args.backbone))
    print_and_log('Epochs: {0}'.format(args.epochs))
    print_and_log('Batch Size: {0}'.format(args.batch_train))
    print_and_log('Optimizer: {0}'.format(args.optimizer))
    print_and_log('=' * 60)

def set_logger(args):
    global logger
    if logger == None:
        logger = logging.getLogger()
    else:  # wish there was a logger.close()
        for handler in logger.handlers[:]:  # make a copy of the list
            logger.removeHandler(handler)

    args_copy = copy.deepcopy(args)
    # copy to get a clean hash
    # use the same log file hash if iterations or verbose are different
    # these flags do not change the results
    args_copy.iters = 1
    args_copy.verbose = False
    args_copy.log_interval = 1
    args_copy.seed = 0

    current_datetime = datetime.datetime.now().strftime("%Y-%m-%d_%H.%M.%S")
    log_path = './logs/{0}_{1}task_{2}_{3}.log'.format(args.dataset,
                                                       args.num_tasks,
                                                       args.backbone,
                                                       current_datetime,
                                                       hashlib.md5(str(args_copy).encode('utf-8')).hexdigest()[:8])
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(fmt='%(asctime)s: %(message)s', datefmt='%H:%M:%S')
    fh = logging.FileHandler(log_path)
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    optuna.logging.enable_propagation()

def print_and_log(msg):
    global logger
    print(msg)
    logger.info(msg)

class CosineDecay:
    def __init__(self, death_rate, T_max, eta_min=0.005, last_epoch=-1):
        self.sgd = optim.SGD(torch.nn.ParameterList([torch.nn.Parameter(torch.zeros(1))]), lr=death_rate)
        self.cosine_stepper = torch.optim.lr_scheduler.CosineAnnealingLR(self.sgd, T_max, eta_min, last_epoch)

    def step(self):
        self.cosine_stepper.step()

    def get_dr(self):
        return self.sgd.param_groups[0]['lr']

class LinearDecay(object):
    def __init__(self, death_rate, factor=0.99, frequency=600):
        self.factor = factor
        self.steps = 0
        self.frequency = frequency

    def step(self):
        self.steps += 1

    def get_dr(self, death_rate):
        if self.steps > 0 and self.steps % self.frequency == 0:
            return death_rate*self.factor
        else:
            return death_rate