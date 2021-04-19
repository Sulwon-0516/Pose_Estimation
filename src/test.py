import torch
import torch.nn as nn
import os

from . import _init_paths
from baseline import baseline
from HRNet import HRNet
from HR_official import get_pose_net

from .functions._test import _test
from .utils.tools import coco_loss_mask
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
        model = HRNet(device,False)
    else:
        print("wrong model name : {}".format(config.MODEL))
        assert(0)
    
    
    M_PATH = config.TEST.MODEL_PATH%config.MODEL
    if not os.path.isdir(M_PATH):
        print("Invalid model path {}".format(M_PATH))
        assert(0)
    M_PATH = os.path.join(M_PATH, config.TEST.MODEL_FILE)
    if not os.path.isfile(M_PATH):
        print("Invalid model path {}".format(M_PATH))
        assert(0)
        
        
    checkpoint = torch.load(M_PATH)
    model.load_state_dict(checkpoint['model_state_dict'], strict = False)
    model.to(device)
    model.eval()
    
    _test(
            dataset = dataset,
            batch_size = batch_size,
            criterion = criterion, 
            model = model, 
            PATH = config.PATH, 
            TITLE = config.THEME, 
            config = config.TEST, 
            num_worker = config.TEST.NUM_WORKER,
            device = device,
            loss_mask = coco_loss_mask)
