import torch
import torchvision
import torch.optim as optims
from torch.utils.data import DataLoader
from pytictoc import TicToc
from . import _init_paths
import numpy as np
import torch.nn as nn
import os
from baseline.baseline import baseline
from HRNet.HRNet import HRNet, HigherHRNet
# import all model files right know.
from .functions._train import _train
from .utils.tools import coco_loss_mask, load_model, save_model
from .dataloader.coco_data_loader import COCO_DataLoader


def train(config,device):
    # Define the training
    # is it right?
    start_epoch = 0
    lowest_loss = 100000
    EPOCH = config.TRAIN.EPOCH
    t = TicToc()

    if config.MODEL == "baseline":
        model = baseline()
    elif config.MODEL == "HRNet":
        model = HRNet(device)
    elif config.MODEL == "HigherHRNet":
        model = HigherHRNet(device)
    else:
        print("wrong model name : {}".format(config.MODEL))
        assert(0)
    model.to(device)
    
    if config.TRAIN.OPTIM == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr = config.TRAIN.LR)
    elif config.TRAIN.OPTIM == 'RMSProp':
        optimizer = torch.optim.RMSprop(model.parameters(), lr = config.TRAIN.LR)
    else:
        print("Invalid optimizer name")
        assert(0)

    # load the model
    if config.TRAIN.IS_SCHED:
        if config.TRAIN.SCHED == 'MultiStepLR':
            scheduler = optims.lr_scheduler.MultiStepLR(optimizer, milestones = config.TRAIN.MILTESTONES, gamma = config.TRAIN.DECAY_RATE)
    else:
        scheduler = None
        
    if config.TRAIN.LOAD_PREV:
        # Load the previous model.
        path = os.path.join(config.PATH.RESULT_PATH,config.TRAIN.PREV_PATH%(config.MODEL))
        file_name = config.TRAIN.PREV_MODEL
        model, optimizer, scheduler, start_epoch, lowest_loss = load_model(path,file_name,model,optimizer,scheduler, config.TRAIN, True)

    # set criterion
    if config.TRAIN.LOSS == 'MSE':
        criterion = nn.MSELoss(reduction = 'none')
    else:
        print("Invalid criterion name")
        assert(0)

    
    # train_dataset = coco_data_loader.DataLoader(True)
    train_dataset = COCO_DataLoader(False,config)
    train_dataloader = DataLoader(dataset = train_dataset, batch_size = config.TRAIN.BATCH_SIZE, shuffle = config.TRAIN.IS_SHUFFLE, num_workers = config.TRAIN.NUM_WORKER)


    #check_grad_mode(model.parameters())
        
    n_images = train_dataset.__len__()
    n_steps = train_dataset.__len__()//config.TRAIN.BATCH_SIZE + 1
    highest_acc = 0

    
    for epoch in range(start_epoch,EPOCH):
        t.tic()
        lr_state_log = 'Epoch {}, lr {}'.format(epoch, optimizer.param_groups[0]['lr'])
        print(lr_state_log)
        lr_state_log = lr_state_log + "\n"
        
        if epoch <=start_epoch:
            is_first=True
        else:
            is_first=False
        
        if config.TRAIN.TEST_EPOCH != 0:
            if epoch % config.TRAIN.TEST_EPOCH == 0:
                is_debug = True
            else:
                is_debug = False
        else:
            is_debug = False

        avg_loss = _train(model = model, 
                            criterion = criterion, 
                            optimizer = optimizer, 
                            dataset = train_dataset,
                            train_dataloader = train_dataloader, 
                            epoch = epoch,
                            config = config,
                            n_steps = n_steps,
                            lr_log = lr_state_log,
                            device = device,
                            is_first=is_first,
                            loss_mask = coco_loss_mask,
                            debug = is_debug)
        t.toc()
        epoch_time = t.tocvalue()
        print("[%d/%d] epochs loss : %f"%(epoch,EPOCH,avg_loss))
        
        # Call Tensor board.

        # save model
        MODEL_NAME = config.THEME + "_" + config.MODEL
        if(avg_loss < lowest_loss):
            BEST_MODEL_PATH = os.path.join(config.PATH.RESULT_PATH,config.PATH.MODEL)
            if not os.path.isdir(BEST_MODEL_PATH):
                print("Invalid save path")
                assert(0)
            BEST_MODEL_PATH = os.path.join(BEST_MODEL_PATH,config.PATH.BEST_MODEL_PATH)
            '''
            if not os.path.isdir(BEST_MODEL_PATH):
                print("Invalid save path")
                assert(0)
            '''
            lowest_loss = avg_loss
            save_model(BEST_MODEL_PATH,model,optimizer,scheduler,avg_loss,epoch,EPOCH,n_images,config.PATH.BEST_FILE,MODEL_NAME,True)
        # save every 10 epochs.
        if epoch%config.TRAIN.CHECK_FREQ==0 and epoch!=0:
            CHECKPOINT_PATH = os.path.join(config.PATH.RESULT_PATH,config.PATH.MODEL)
            if not os.path.isdir(CHECKPOINT_PATH):
                print("Invalid save path")
                assert(0)
            CHECKPOINT_PATH = os.path.join(CHECKPOINT_PATH,config.PATH.CHECKPOINT_PATH)
            
            save_model(CHECKPOINT_PATH,model,optimizer,scheduler,avg_loss,epoch,EPOCH,n_images,config.PATH.CHECKPOINT_FILE,MODEL_NAME)
        
        if config.TRAIN.IS_SCHED:
            scheduler.step()

