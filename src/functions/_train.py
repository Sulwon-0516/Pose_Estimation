import torch
import torch.utils.data
from torch.autograd import Variable
from ..utils.tools import logger

def _train(model,criterion,optimizer,train_dataloader, epoch, config, n_steps, lr_log, device, loss_mask = None):
    epoch_loss = 0
    n_data = 0
    for i,data in enumerate(train_dataloader):
        image, heatmaps, _ , _, keypoints, _ = data
        image, heatmaps = Variable(image).float(), Variable(heatmaps).float()
        image.to(device)
        heatmaps.to(device)

        output = model(image)
        step_loss = criterion(output,heatmaps)
        step_loss = step_loss.sum(axis = (2,3))
        
        if loss_mask != None:
            mask = loss_mask(keypoints)
            step_loss = step_loss * mask.float()
        
        optimizer.zero_grad()
        step_loss = step_loss.sum(axis=(0,1))
        step_loss.backward()
        optimizer.step()

        #### This part need to be changed into Logger
        if i%config.LOG.FREQ == 0:
            log = config.LOG.STEP_FORMAT%(epoch,config.TRAIN.EPOCH,i,step_loss/image.shape[0])
            logger(lr_log, config)
            logger(log, config)

        epoch_loss += step_loss.detach()
        n_data += image.shape[0]

    epoch_loss /= n_data
        
    log = config.LOG.EPOCH_FORMAT%(epoch,config.TRAIN.EPOCH,epoch_loss)
    logger(log,config)
    ### Logger should be called.
    # log for epoch_loss

    return epoch_loss
