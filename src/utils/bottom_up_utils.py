import torch
import torch.nn as nn

SIGMA = 1


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
    avg_human = torch.empty(num_humans,device = device)
    sum_tags = torch.zeros(2,device = device)
    for i, human in enumerate(keypoints):
        tags = []
        sum_tags[1] = 0
        sum_tags[0] = 0
        for j in range(tag_maps.shape[0]):
            if human[j*3+2] != 0:
                tags.append(tag_maps[j,human[j*3],human[j*3+1]])
                sum_tags[0] += tag_maps[j,human[j*3],human[j*3+1]]
                sum_tags[1]+=1
        if sum_tags[1] == 0:
            print("wrong keypoint input")
            assert(0)
        
        avg_tags = sum_tags[0]/sum_tags[1]
        avg_human[i] = avg_tags

        '''calculate each human tag loss'''
        tags = torch.Tensor(tags, device = device)
        tags = tags - avg_tags
        tags = tags ** 2
        loss += torch.sum(tags)
    loss = loss / num_humans
    
    ''' calculate dividing tag loss'''
    grid_x, grid_y = torch.meshgird(avg_human,avg_human,device=device)
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
    resized_keys = torch.zeros_like(keypoints,device=device)
    resized_keys[2::3] = keypoints[2::3]
    for i, ratio in enumerate(scope):
        resized_keys[0::3] = keypoints[0::3]/scope
        resized_keys[1::3] = keypoints[1::3]/scope
        loss[i] = SingleOutput_AELoss(resized_keys,tag_mapss[i],sigma,device)
    
    return loss



def AELoss(keypointss, tag_mapsss, scope, device, sigma = SIGMA, IS_SUM = True):
    '''
    keypoints : single batch keypoints.
    tag_mapss : single batch list of list of tag maps
    sigma : sigma of the ae loss
    IS_SUM : if it's true, sum all loss from all levels and return(when enought GPU Mems)
    '''
    b_size = len(tag_mapsss)
    if IS_SUM:
        losses = torch.empty((b_size),device=device)
    else:
        losses = torch.empty((b_size, len(scope)), device=device)
    

    for i in range(b_size):
        losses[i] = SingleImage_AELoss(keypointss[i], tag_mapss[i], scope, device, sigma)
     
    return losses
    


if __name__ == "__main__":
    '''for debuggin'''
    fake_IMG_WIDTH = 320
    fake_IMG_HEIGHT = 560
    fake_SCOPE = [4, 2]
    fake_BATCH = 5

    fake_keys = []
    key_src = [0,1,2]

    for i in range(17):
        fake_keys.append(torch.randint(low=0,high=fake_IMG_WIDTH,1)[0])
        fake_keys.append(torch.randint(low=0,high=fake_IMG_HEIGT,1)[0])
        fake_keys.append(key_src[i%3])
    











