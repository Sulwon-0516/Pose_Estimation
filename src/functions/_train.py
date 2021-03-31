import torch
import torch.utils.data
from torch.autograd import Variable
from ..utils.tools import logger

def _train(model,criterion,optimizer,dataset,train_dataloader, epoch, config, n_steps, lr_log, device, is_first=False, loss_mask = None, debug = False):
    epoch_loss = 0
    n_data = 0
    n_debug = config.TRAIN.TEST_PER_BATCH
    for i,data in enumerate(train_dataloader):
        image, heatmaps, _ , ids, keypoints, _ = data
        image, heatmaps = image.float().to(device), heatmaps.float().to(device)

        output = model(image)
        step_loss = criterion(output,heatmaps)
        step_loss = step_loss.mean(axis = (2,3))
        
        if loss_mask != None:
            mask = loss_mask(keypoints).to(device)
            step_loss = step_loss * mask.float()
        
        optimizer.zero_grad()
        step_loss = step_loss.mean(axis=(0,1))
        step_loss.backward()
        optimizer.step()


        '''debug'''
        if debug:
            cpu_output = output.detach().cpu()
            for j in range(ids.shape[0]):
                if n_debug>0:
                    dataset.show_heatmaps(ids[j],cpu_output[j],image[j].cpu(),False,epoch)
                    n_debug -= 1

        if is_first and False:
            print(ids)
        

        ''' This part need to be changed into Logger '''
        

        if i%config.LOG.FREQ == 0:
            log = config.LOG.STEP_FORMAT%(epoch,config.TRAIN.EPOCH,i,step_loss/image.shape[0])
            logger(lr_log, config)
            logger(log, config)

        epoch_loss += step_loss.detach()* image.shape[0]
        n_data += image.shape[0]

    epoch_loss /= n_data
        
    log = config.LOG.EPOCH_FORMAT%(epoch,config.TRAIN.EPOCH,epoch_loss)
    logger(log,config)
    ### Logger should be called.
    # log for epoch_loss

    return epoch_loss
