import torch
import torch.nn as nn
import os

SIGMA = 1



class AELoss(nn.Module):
    def __init__(self, B_size, NUM_RES, device, sigma = SIGMA, N_joints = 17):
        super(AELoss, self).__init__()
        self.b_size = B_size
        self.NUM_RES = NUM_RES
        self.N_JOITNS = N_joints
        self.sigma = sigma
        self.device = device
    
    def SingleOutput_AELoss(self, keypoints, tag_maps):
        num_humans = 0
    
        loss = 0 
        avg_human = []
        for i, human in enumerate(keypoints):
            tags = []
            for j in range(tag_maps.shape[0]):
                if human[j*3+2] != 0:
                    tags.append(tag_maps[j,human[j*3],human[j*3+1]])
            
            if len(tags) == 0:
                avg_human.append(torch.zeros(1, device = self.device).squeeze())
            else:
                tags = torch.stack(tags)
                avg_tag = torch.mean(tags)
                avg_human.append(avg_tag)
                num_humans +=1

                '''calculate each human tag loss'''
                tags = tags - avg_tag
                tags = tags ** 2
                loss += torch.sum(tags)
        loss = loss / num_humans
        
        avg_human = torch.stack(avg_human)
        ''' calculate dividing tag loss'''
        grid_x, grid_y = torch.meshgrid(avg_human,avg_human)
        diff = grid_x-grid_y
        loss_map = torch.exp(-diff**2/(2*self.sigma**2))
        loss += torch.sum(loss_map)/(num_humans**2)

        return loss
    
    def forward(self, keypointss, tag_mapsss):
        
        '''
        keypoints : single batch keypoints.
        tag_mapss : single batch list of list of tag maps
        sigma : sigma of the ae loss
        IS_SUM : if it's true, sum all loss from all levels and return
        (when enought GPU Mems)
        '''
        
        losses = torch.zeros(self.b_size,device = self.device)
        

        for i in range(self.b_size):
            #tags = [tag_mapsss[0][i], tag_mapsss[1][i]]
            
            for j,ratio in enumerate(self.NUM_RES):
                losses[i] += self.SingleOutput_AELoss(keypointss[i][j],
                                                        tag_mapsss[j][i])
                
        return losses
        
        
        

def SingleOutput_AELoss(keypoints, tag_maps, sigma, device):
    '''
    <Input>
    keypoint : [[human1 keypoints],[human2 keypoints]...]  positions of the GT keypoint 
    -> human keypoints are given same as COCO dataset
    tag : # of joints by H by W tensor output.

    <what it do>
    1. Calculate the average tags per human with calculating loss.
    2. 

    '''
    num_humans = len(keypoints)
    
    loss = 0 
    avg_human = []
    sum_tags = torch.autograd.Variable(torch.zeros(num_humans,2,device = device))
    for i, human in enumerate(keypoints):
        tags = []
        sum_tags[i][1] = 0
        sum_tags[i][0] = 0
        for j in range(tag_maps.shape[0]):
            if human[j*3+2] != 0:
                tags.append(tag_maps[j,human[j*3],human[j*3+1]])
                sum_tags[i][0] += tag_maps[j,human[j*3],human[j*3+1]]
                sum_tags[i][1]+=1
        if sum_tags[i][1] == 0:
            avg_human.append(0)
        else:
            avg_human.append(sum_tags[i][0]/sum_tags[i][1])

        '''calculate each human tag loss'''
        tags = torch.Tensor(tags).to(device)
        tags = tags - avg_human[i]
        tags = tags ** 2
        loss += torch.sum(tags)
    loss = loss / num_humans
    
    ''' calculate dividing tag loss'''
    grid_x, grid_y = torch.meshgrid(avg_human,avg_human)
    diff = grid_x-grid_y
    loss_map = torch.exp(-diff**2/(2*sigma**2))
    loss += torch.sum(loss_map)/(num_humans**2)

    return loss

def SingleImage_AELoss(keypoints, tag_mapss, scope, device, sigma):
    '''
    keypoints : GT keypoints. I need to convert it
    tag_mapss : list of tag_maps.
    scope : change of the size. in Higher HRNet, basic value is [4 2]
    '''


    loss = torch.empty(len(scope),device=device)
    for i, ratio in enumerate(scope):
        loss[i] = SingleOutput_AELoss(keypoints[i],tag_mapss[i],sigma,device)
    
    return loss



def AE_Loss(keypointss, tag_mapsss, scope, device, sigma = SIGMA, IS_SUM = True):
    
    '''
    keypoints : single batch keypoints.
    tag_mapss : single batch list of list of tag maps
    sigma : sigma of the ae loss
    IS_SUM : if it's true, sum all loss from all levels and return
    (when enought GPU Mems)
    '''
    b_size = tag_mapsss[0].shape[0]
    if IS_SUM:
        losses = torch.empty((b_size),device=device)
    else:
        losses = torch.empty((b_size, len(scope)), device=device)
    

    for i in range(b_size):
        tags = [tag_mapsss[0][i], tag_mapsss[1][i]]
        temp = SingleImage_AELoss(keypointss[i], 
                                    tags, 
                                    scope, 
                                    device, 
                                    sigma)
        if IS_SUM:
            losses[i] = temp.sum() 
        else:
            losses[i] = temp
    return losses
    


if __name__ == "__main__":
    '''for debugging'''
    os.environ["CUDA_VISIBLE_DEVICES"] = str(1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    fake_IMG_WIDTH = 320
    fake_IMG_HEIGHT = 560
    fake_SCOPE = [4, 2]
    fake_BATCH = 5

    fake_keys = []
    key_src = torch.Tensor([0,1,2])

    

    
    for j in range(fake_BATCH):
        keys = []
        for k in range(2):
            fake_key = []
            for i in range(17):
                fake_key.append(torch.randint(size=(1,),low=0,high=fake_IMG_WIDTH)[0])
                fake_key.append(torch.randint(size=(1,),low=0,high=fake_IMG_HEIGHT)[0])
                fake_key.append(key_src[i%3])
            keys.append(fake_key)
        fake_keys.append(keys)
    
    fake_tags_1 = torch.randn(fake_BATCH,17,fake_IMG_WIDTH//4,fake_IMG_HEIGHT//4, device=device)
    fake_tags_2 = torch.randn(fake_BATCH,17,fake_IMG_WIDTH//2,fake_IMG_HEIGHT//2, device=device)
    fake_tags = [fake_tags_1,fake_tags_2]

    loss = AELoss(fake_keys,fake_tags,fake_SCOPE,device,1,True)
    loss.backward()
    

    
    











