import torch
import torchvision
import torch.optim as optims
from torch.utils.data import DataLoader
from pytictoc import TicToc
from . import _init_paths
import numpy as np
import torch.nn as nn
import os
from pathlib import Path
import visdom
import json
from baseline import baseline
from HRNet import HRNet, HigherHRNet
from HR_official import get_pose_net
# import all model files right know.
from .functions._train import _train
from .functions._train_BU import _train_BU
from .utils.tools import coco_loss_mask, load_model, save_model, load_pretrain_model
from .utils.monitor import VisdomLinePlotter
from .utils.bottom_up_utils import AELoss
from .dataloader.coco_data_loader import COCO_DataLoader, Bottom_Up_COCO_DataLoader
from .dataloader import coco_data_loader
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

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
        model = HRNet(device,False)
    elif config.MODEL == "HigherHRNet":
        model = HigherHRNet(device,False)
        if config.TRAIN.PRETRAIN:
            path = os.path.join(
                        config.PATH.RESULT_PATH,
                        config.TRAIN.PRETRAIN_PATH%(config.TRAIN.PRETRAIN_MODEL_NAME))
            file_name = config.TRAIN.PRETRAIN_MODEL
            model = load_pretrain_model(path=path,
                                        file=file_name,
                                        model=model)
        model.train()
            
        
    else:
        print("wrong model name : {}".format(config.MODEL))
        assert(0)
    model.to(device)
    
    if config.TRAIN.OPTIM == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), 
                                     lr = config.TRAIN.LR)
        
    elif config.TRAIN.OPTIM == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(),
                                    lr = config.TRAIN.LR,
                                    momentum = 0.9,
                                    weight_decay = 0.0001,
                                    nesterov = False)
                                    
    elif config.TRAIN.OPTIM == 'RMSProp':
        optimizer = torch.optim.RMSprop(model.parameters(), 
                                        lr = config.TRAIN.LR)
        
    else:
        print("Invalid optimizer name")
        assert(0)

    
    if config.TRAIN.IS_SCHED:
        if config.TRAIN.SCHED == 'MultiStepLR':
            scheduler = optims.lr_scheduler.MultiStepLR(
                            optimizer, 
                            milestones = config.TRAIN.MILTESTONES, 
                            gamma = config.TRAIN.DECAY_RATE)
        elif config.TRAIN.SCHED == 'ReduceLROnPlateau':
            scheduler = optims.lr_scheduler.ReduceLROnPlateau(
                            optimizer = optimizer,
                            mode = 'min',
                            factor = 0.1,
                            patience = 5,
                            cooldown = 10,
                            min_lr = config.TRAIN.LR*(0.1**3))
        else:
            print("Invalid scheduler name")
            assert(0)
    else:
        scheduler = None
    
    
    ''' load the model '''
    if config.TRAIN.LOAD_PREV:
        path = os.path.join(
                        config.PATH.RESULT_PATH,
                        config.TRAIN.PREV_PATH%(config.MODEL))
        file_name = config.TRAIN.PREV_MODEL
        model, optimizer, scheduler, start_epoch, lowest_loss = load_model(
                                                                    path,
                                                                    file_name,
                                                                    model,
                                                                    optimizer,
                                                                    scheduler, 
                                                                    config.TRAIN, 
                                                                    True)

    
    if config.TRAIN.LOSS == 'MSE':
        criterion = nn.MSELoss(reduction = 'none')
    else:
        print("Invalid criterion name")
        assert(0)

    if config.IS_BU:
        # train_dataset = Bottom_Up_COCO_DataLoader(True,config)
        train_dataset = Bottom_Up_COCO_DataLoader(False,config)
        train_dataloader = DataLoader(
                                dataset = train_dataset,
                                batch_size = config.TRAIN.BATCH_SIZE,
                                shuffle = config.TRAIN.IS_SHUFFLE,
                                num_workers = config.TRAIN.NUM_WORKER,
                                collate_fn = train_dataset.COCO_BU_collate_fn)
        loss_func = AELoss(B_size = config.TRAIN.BATCH_SIZE,
                           NUM_RES = [4,2],
                           device = device)
        loss_func.to(device)
        
    else:
        # train_dataset = Bottom_Up_COCO_DataLoader(True,config)
        train_dataset = COCO_DataLoader(False,config)
        train_dataloader = DataLoader(
                                dataset = train_dataset,
                                batch_size = config.TRAIN.BATCH_SIZE,
                                shuffle = config.TRAIN.IS_SHUFFLE,
                                num_workers = config.TRAIN.NUM_WORKER)

        
    n_images = train_dataset.__len__()
    n_steps = train_dataset.__len__()//config.TRAIN.BATCH_SIZE + 1
    highest_acc = 0


    ''' Visdom relevant '''
    if config.VIS.IS_USE:
        vis_graph = VisdomLinePlotter(env_name = "loss_functions")
        init_x = torch.Tensor([0])
        init_y = torch.Tensor([1])
        vis_graph.plot("main_loss",
                       "mean loss",
                       "log10_loss"+config.THEME,
                       init_x,
                       init_y)
        
        vis_graph.plot("epoch_loss",
                       "epoch_loss",
                       "epoch_loss"+config.THEME,
                       init_x,
                       init_y)



    if config.IS_PARALLEL:
        print("using %d GPUs for training" % torch.cuda.device_count())
        model = nn.DataParallel(model)
        


    for epoch in range(start_epoch,EPOCH):
        t.tic()
        lr_state_log = 'Epoch {}, lr {}'.format(epoch, 
                                                optimizer.param_groups[0]['lr'])
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


        if config.IS_BU:
            avg_loss = _train_BU(
                                visualizer = vis_graph,
                                model = model,
                                criterion = criterion,
                                optimizer = optimizer,
                                dataset = train_dataset,
                                train_dataloader = train_dataloader,
                                epoch = epoch,
                                config = config,
                                lr_log = lr_state_log,
                                device = device,
                                loss_func = loss_func,
                                is_first = is_first,
                                loss_mask = coco_loss_mask,
                                debug = is_debug)
            
        else:
            avg_loss = _train(
                                visualizer = vis_graph,
                                model = model, 
                                criterion = criterion, 
                                optimizer = optimizer, 
                                dataset = train_dataset,
                                train_dataloader = train_dataloader, 
                                epoch = epoch,
                                config = config,
                                lr_log = lr_state_log,
                                device = device,
                                is_first=is_first,
                                loss_mask = coco_loss_mask,
                                debug = is_debug)
        t.toc()
        epoch_time = t.tocvalue()
        print("[%d/%d] epochs loss : %f"%(epoch,EPOCH,avg_loss))

        '''Plot epoch loss '''
        if config.VIS.IS_USE:
            vis_x = torch.Tensor([epoch])
            vis_y = avg_loss.unsqueeze(0)
            vis_graph.plot("epoch_loss","epoch_loss","log10_loss",vis_x,torch.log10(vis_y))

        

        '''save model'''
        MODEL_NAME = config.THEME + "_" + config.MODEL
        if(avg_loss < lowest_loss):
            BEST_MODEL_PATH = os.path.join(config.PATH.RESULT_PATH,config.MODEL)
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
            save_model(
                    BEST_MODEL_PATH,
                    model,
                    optimizer,
                    scheduler,
                    avg_loss,
                    epoch,
                    EPOCH,
                    n_images,
                    config.PATH.BEST_FILE,MODEL_NAME,
                    True)
        # save every 10 epochs.
        if epoch%config.TRAIN.CHECK_FREQ==0 and epoch!=0:
            CHECKPOINT_PATH = os.path.join(config.PATH.RESULT_PATH,config.MODEL)
            if not os.path.isdir(CHECKPOINT_PATH):
                print("Invalid save path")
                assert(0)
            CHECKPOINT_PATH = os.path.join(CHECKPOINT_PATH,config.PATH.CHECKPOINT_PATH)
            
            save_model(
                    CHECKPOINT_PATH,
                    model,
                    optimizer,
                    scheduler,
                    avg_loss,
                    epoch,
                    EPOCH,
                    n_images,
                    config.PATH.CHECKPOINT_FILE,
                    MODEL_NAME)
        
        if config.TRAIN.IS_SCHED:
            if config.TRAIN.SCHED == 'ReduceLROnPlateau':
                scheduler.step(avg_loss)
            else:
                scheduler.step()

            

        if epoch%10==0:
            with torch.no_grad():
                model.eval()
                dataset = train_dataset
            
            
                n_keys = dataset.__len__()
                tot_data = []
                tot_loss = 0
                tot_OKS = 0
                t = TicToc()
                save_predict_heatmap = config.TEST.SAVE_HEATMAP
                for i, data in enumerate(train_dataloader):
                    t.tic()
                    imgs, heatmaps, old_bboxs, ids , keypoints, n_keys, _ = data
                    #print(old_bboxs)
                    old_bboxs, keypoints = old_bboxs.to(device), keypoints.to(device)

                    
                    imgs, heatmaps = imgs.float().to(device), heatmaps.float().to(device)


                    p_heatmaps = model(imgs)
                    loss = criterion(heatmaps,p_heatmaps)
                    loss = 0.5*loss.mean(axis = (2,3))
                    
                    cpu_p_heatmaps = p_heatmaps.cpu()
                    for j in range(ids.shape[0]):
                        re_keypoints = coco_data_loader.restore_heatmap_gpu(p_heatmaps[j],
                                                                            old_bboxs[j],
                                                                            keypoints[j],
                                                                            device)
                        
                        if config.TEST.SAVE_PREDICTED and j<config.TEST.SAVE_IMG_PER_BATCH:
                            re_anns = dataset.save_key_img(ids[j],True,re_keypoints.cpu(),False)
                        else:
                            re_anns = dataset.save_key_img(ids[j],True,re_keypoints.cpu())
                        if save_predict_heatmap > 0 and n_keys[j]>8:
                            print("called")
                            dataset.show_heatmaps(ids[j],cpu_p_heatmaps[j],imgs[j].cpu(),False)
                            save_predict_heatmap -= 1
                        
                        re_anns['score'] = - loss[j].cpu().sum().item()
                        tot_data.append(re_anns)
                    
                    tot_loss += loss.cpu().sum(axis = (0,1))

                    print("step %d, loss : %f" %(i,loss.mean()))
                    t.toc() 
                tot_loss = tot_loss/n_keys


                # check the path exists.
                Path(config.PATH.RESULT_PATH).mkdir(exist_ok=True)
                f_name = os.path.join(config.PATH.RESULT_PATH,config.PATH.MODEL)
                Path(f_name).mkdir(exist_ok=True)
                f_name = os.path.join(f_name,config.PATH.PRED_PATH)
                Path(f_name).mkdir(exist_ok=True)
                n_file = len(os.listdir(f_name))
                file_path = os.path.join(f_name,config.PATH.PRED_NAME%("temp",n_file+1))
                
                with open(file_path,"w") as res_file:
                    json.dump(tot_data,res_file)

            model.train()
            
            val_dataset = COCO_DataLoader(config.VAL.IS_TRAIN,config)

            annType = ['segm','bbox','keypoints']
            annType = annType[config.TYPE]      #specify type here
            prefix = 'person_keypoints' if annType=='keypoints' else 'instances'

            # GT means Ground Truth
            cocoGT = val_dataset.coco

            cocoDT = cocoGT.loadRes(file_path)

            # Get the Img Ids
            imgIds = sorted(val_dataset.get_imgIds())
            # print(imgIds)

            # Evaluation
            cocoEval = COCOeval(cocoGT,cocoDT,annType)
            cocoEval.params.imgIds = imgIds
            cocoEval.evaluate()
            cocoEval.accumulate()
            cocoEval.summarize()