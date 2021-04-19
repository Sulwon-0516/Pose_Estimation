import torch
import torch.utils.data
import visdom
import json


from pytictoc import TicToc
from torch.autograd import Variable
from ..utils.tools import logger
#from ..utils.monitor import VisdomLinePlotter

def _train(
                visualizer,
                model,
                criterion,
                optimizer,
                dataset,
                train_dataloader,
                epoch, 
                config,  
                lr_log, 
                device, 
                is_first=False, 
                loss_mask = None, 
                debug = False):
    
    t = TicToc()
    epoch_loss = 0
    n_data = 0
    n_debug = config.TRAIN.TEST_PER_BATCH
    for i,data in enumerate(train_dataloader):
        t.tic()
        image, heatmaps, _ , ids, keypoints, _, img_ids = data
        image, heatmaps = image.float().to(device), heatmaps.float().to(device)


        output = model(image)
        step_loss = criterion(output,heatmaps)
        step_loss = step_loss.mean(axis = (2,3))
        
        if False:
            debug_HR(model)
        if loss_mask != None:
            mask = loss_mask(keypoints).to(device)
            step_loss = step_loss * mask.float()
        
        optimizer.zero_grad()
        step_loss = 0.5 * step_loss.mean(axis=(0,1))
        step_loss.backward()
        optimizer.step()


        if epoch == 1000:
            print("hello")
        
        '''debug''' 
        # draw output heatmaps
        if debug and config.VIS.IS_USE:
            vis_heat = visdom.Visdom(env="heatmap")
            if epoch==0:
                for j in range(config.TRAIN.TEST_PER_BATCH):
                    vis_heat.delete_env(env = "heatmap_"+str(j))
            cpu_output = output.detach().cpu()
            cpu_output = torch.abs(cpu_output.unsqueeze(1))
            opts=dict(title="{}_{}_heatmap".format(epoch,step_loss))
            for j in range(ids.shape[0]):
                if n_debug>0:
                    vis_heat.images(cpu_output[j],env = "heatmap_"+str(j),opts=opts)
                    '''dataset.show_heatmaps(ids[j],
                                          cpu_output[j],
                                          image[j].cpu(),
                                          False,
                                          epoch)'''
                    n_debug -= 1


    
        if is_first and False:
            # Things to do in the first epoch
            # why doesn't use epoch==0 : to solve the case of loading + training
            print(ids)
        

        ''' This part need to be changed into Logger '''
        


        if i%config.LOG.FREQ == 0:
            log = config.LOG.STEP_FORMAT%(epoch,
                                          config.TRAIN.EPOCH,
                                          i,
                                          step_loss)
            logger(lr_log, config)
            logger(log, config)
            print(log)

        # draw graph loss
        if config.VIS.IS_USE:
            if i%config.VIS.STEP_FREQ == 0:
                vis_x = torch.Tensor([epoch*len(train_dataloader)+i])
                vis_y = step_loss.unsqueeze(0)
                visualizer.plot("main_loss","mean loss","log10_loss",vis_x,torch.log10(vis_y))


        epoch_loss += step_loss.detach()* image.shape[0]
        n_data += image.shape[0]
        
        
        t.toc()

    epoch_loss =  epoch_loss/n_data
        
    log = config.LOG.EPOCH_FORMAT%(epoch,config.TRAIN.EPOCH,epoch_loss)
    logger(log,config)
    ### Logger should be called.
    # log for epoch_loss

    return epoch_loss.cpu()




def debug_HR(model):
    grads = []
    
    for param in model.parameters():
        print(param.shape)
        grads.append(param.grad.view(-1))
        print(torch.mean(grads[-1]))