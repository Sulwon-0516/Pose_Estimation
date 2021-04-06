import torch
import torch.nn as nn
import os

from . import _init_paths
from baseline import baseline
from HRNet import HRNet
from .functions._test import _test
from .dataloader.coco_data_loader import COCO_DataLoader

def test(config,device):
    if config.TEST.IS_TEST:
        # Make test data loader.
        print("No test data set yet")
        assert(0)
    else:
        # valid data loader
        dataset = COCO_DataLoader(False,config)
    
    batch_size = config.TEST.BATCH_SIZE

    if config.TRAIN.LOSS == 'MSE':
        criterion = nn.MSELoss(reduction = 'none')
    else:
        print("wronge loss function")
        assert(0)

    ''' should be changed, when model changed'''
    if config.MODEL == "baseline":
        model = baseline()
    elif config.MODEL == "HRNet":
        model = HRNet(device)
    else:
        print("wrong model name : {}".format(config.MODEL))
        assert(0)
    model.to(device)
    
    M_PATH = config.TEST.MODEL_PATH%config.MODEL
    if not os.path.isdir(M_PATH):
        print("Invalid model path {}".format(M_PATH))
        assert(0)
    M_PATH = os.path.join(M_PATH, config.TEST.MODEL_FILE)
    if not os.path.isfile(M_PATH):
        print("Invalid model path {}".format(M_PATH))
        assert(0)
    _test(
            dataset = dataset,
            batch_size = batch_size,
            criterion = criterion, 
            model = model, 
            M_PATH = M_PATH, 
            PATH = config.PATH, 
            TITLE = config.THEME, 
            config = config.TEST, 
            num_worker = config.TEST.NUM_WORKER,
            device = device)
