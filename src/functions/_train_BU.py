import torch
import torch.utils.data
import visdom


from torch.autograd import Variable
from ..utils.tools import logger
from ..utils.bottom_up_utils import AELoss
#from ..utils.monitor import VisdomLinePlotter

NUM_RES = [4,2]

def _train_BU(
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
                loss_func = None,
                is_first=False, 
                loss_mask = None, 
                debug = False):
    

    epoch_loss = 0
    n_data = 0
    n_debug = config.TRAIN.TEST_PER_BATCH
    for i,data in enumerate(train_dataloader):
        image, heatmaps, keypointss, num_keys, img_ids= data
        image= image.float().to(device)
        for i in range(len(heatmaps)):
            heatmaps[i] = heatmaps[i].float().to(device)
        keypointss = keypointss.int().to(device)

        outputs = model(image)
        
        
        '''With this array, maybe it won't work on GPU'''
        heatmap_outputs = []
        AE_outputs = []
        for i in range(len(NUM_RES)):
            dim_first_outputs = outputs[i].permute(1,0,2,3)
            heatmap_output = (dim_first_outputs[17:]).permute(1,0,2,3)
            AE_output = (dim_first_outputs[0:17]).permute(1,0,2,3)
            heatmap_outputs.append(heatmap_output)
            AE_outputs.append(AE_output)
        
        
        # debug -------------------
        for i in range(len(NUM_RES)):
            print(heatmap_output[i].device)
            print(AE_outputs[i].device)
                
        
        '''Calculate Heatmap Loss First'''
        heat_loss = 0
        AE_loss = 0
        for i in range(len(NUM_RES)):
            temp = criterion(heatmap_outputs[i], heatmaps[i])
            heat_loss += temp.mean(axis=(2,3))
            
        AE_loss += AELoss(keypointss=keypointss,
                          tag_mapsss=AE_outputs,
                          scope = NUM_RES,
                          device = device)
        
            
        
        
        
        if loss_mask != None:
            mask = loss_mask(num_keys, NUM_RES).to(device)
            heat_loss = heat_loss * mask.float()
        
        optimizer.zero_grad()
        heat_loss = heat_loss.mean(axis=(0,1))
        heat_loss.backward(retain_graph = True)
        AE_loss = AE_loss.mean()
        AE_loss.backward()
        optimizer.step()

        step_loss = heat_loss.detach() + AE_loss.detach()
        step_loss = step_loss.cpu()

        
        
        '''debug''' 
        # draw output heatmaps
        if debug and config.VIS.IS_USE:
            vis_heat = visdom.Visdom(env="heatmap")
            for res in range(len(NUM_RES)):
                if epoch==0:
                    for j in range(config.TRAIN.TEST_PER_BATCH):
                        vis_heat.delete_env(env = "heatmap_"+str(j)+\
                                                        "_res_"+str(res+1))
                cpu_output = heatmap_outputs[res].detach().cpu()
                cpu_output = torch.abs(cpu_output)
                opts=dict(title="{}_{}_heatmap".format(epoch,step_loss))
                for j in range(img_ids.shape[0]):
                    if n_debug>0:
                        vis_heat.images(cpu_output[j].unsqueeze(1),env = "heatmap_"+str(j)+\
                                                "_res_"+str(res+1),opts=opts)
                        
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

       



        # get model kernel images


        # draw step-wise loss



        epoch_loss += step_loss * image.shape[0]
        n_data += image.shape[0]

    epoch_loss /= n_data
        
    log = config.LOG.EPOCH_FORMAT%(epoch,config.TRAIN.EPOCH,epoch_loss)
    logger(log,config)
    ### Logger should be called.
    # log for epoch_loss

    return epoch_loss
