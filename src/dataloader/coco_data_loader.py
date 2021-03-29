from pycocotools.coco import COCO
import os
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as img
import math
import numpy as np
import cv2
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torch

COCO_PATH = "./coco"
VAL_INS_PATH = os.path.join(COCO_PATH,"annotations/instances_val2017.json")
VAL_KEY_PATH = os.path.join(COCO_PATH,"annotations/person_keypoints_val2017.json")
SAMPLE_PATH = "./sample"



def restore_heatmap(heatmaps_in, old_bbox,org_key, flip = False, flipped_heatmaps_in = None):
    #print("org size :",heatmaps_in.shape,"new_size:",old_bbox)
    trans = transforms.Resize((old_bbox[3]+1,old_bbox[2]+1))
    old_heatmaps = trans(heatmaps_in)
    #print(old_heatmaps.shape)
    #print(old_bbox)
    mask = org_key[2::3]

    # when using flipped input
    if flip:
        old_heatmaps += transforms.RandomHorizontalFlip(p=1.0)
        # do additional things.
        #
        #
        #
    else:
        max1, arg1 = torch.max(old_heatmaps,dim = 1)
        max2, arg2 = torch.max(max1,dim=1)

        #print(arg1)
        #print(arg2)

        pred_keypoints = np.zeros(17*3)
        for i in range(17):
            #print(max2[i])
            
            if mask[i] != 0:
                pred_keypoints[i*3]=old_bbox[0] + arg2[i]
                pred_keypoints[i*3+1]=old_bbox[1] + arg1[i,arg2[i]]
                pred_keypoints[i*3+2]= mask[i]
            else:
                pred_keypoints[i*3]=0
                pred_keypoints[i*3+1]=0
            
    
    return pred_keypoints

def safe_resize(x,width,Max_width,increment):
    # increment is size 2 array
    new_x = x
    new_width = width
    padding_size = [0,0]
    
    if x - increment[0] < 0:
        new_x = x-increment[0]
        padding_size[0] = increment[0] - x
    else:
        new_x = x-increment[0]
    
    if x+width+increment[1]<Max_width:
        new_width = width + increment[0]+increment[1]
    else:
        padding_size[1] = (x+width+increment[1])-Max_width
        new_width = width + increment[0]+increment[1]
    
    return new_x, new_width, padding_size

'''need to implement'''
def increase_img(bbox,tot_pad):
    return 0

class COCO_DataLoader(Dataset):
    def __init__(self, train, in_config):
        #make sure images are located in directories that the label files mention
        self.config = in_config.DATA
        self.THEME = in_config.THEME
        self.COCO_PATH = in_config.PATH.COCO_PATH
        self.SAMPLE_PATH = in_config.PATH.SAMPLE
        self.RESIZED = in_config.PATH.RESIZED
        self.train = train
        self.check_heatmaps = in_config.DATA.CHECK_HEATMAP
        self.flag = True
        if train:
            coco = COCO("""train path""")
        else:
            VAL_KEY_PATH = os.path.join(self.COCO_PATH,in_config.PATH.COCO_VAL_KEY_PATH)
            if not os.path.isfile(VAL_KEY_PATH):
                print("Invalid validataion key path {}".format(VAL_KEY_PATH))
                assert(0)
            coco = COCO(VAL_KEY_PATH)

            VAL_INS_PATH = os.path.join(self.COCO_PATH,in_config.PATH.COCO_VAL_INS_PATH)
            if not os.path.isfile(VAL_INS_PATH):
                print("Invalid validation ins path {}".format(VAL_INS_PATH))
                assert(0)
            coco_bb = COCO(VAL_INS_PATH)

        
        self.transforms_image = transforms.Compose(
            [
                transforms.Normalize((0.485,0.456,0.406),(0.229,0.224,0.225)),
                transforms.Resize((self.config.HEIGHT,self.config.WIDTH))
            ]
        )
        
        self.transforms_heatmaps = transforms.Compose(
            [
                transforms.Resize((self.config.HEIGHT//self.config.IN_OUT_RATIO,self.config.WIDTH//self.config.IN_OUT_RATIO))
            ]
        )

        #get categories - for pose catIds are only person=0
        catIds = coco.getCatIds(catNms=['person'])
        #get imageIDs for all associated categories
        imgIds = coco.getImgIds(catIds=catIds)

        # I save coco to use showAnns() function. 
        # If it's able to substitue it, this code should be removed
        self.coco = coco
        self.data = []

        for i in imgIds:
            #get person annotation IDs -> to retrieve from cocotools
            annIds = coco.getAnnIds(imgIds=i, catIds=catIds, iscrowd=False)
            #get annotations
            anns = coco.loadAnns(annIds)
            bb_anns = coco_bb.loadAnns(annIds)
            img_info = coco_bb.loadImgs(i)
            height = img_info[0]['height']
            width = img_info[0]['width']

            #easy to exlude unwanted examples for training here
            # with this method, all instances will be treated in different model.
            # it will make it easy to make heat-map, but hard to train all instances at once.
            # However still it has benefit that don't need Bbox regression..

            # I will train it separately in Top-down method, so no problem to implement baseline model.
            for i,data in enumerate(anns):
                if data['num_keypoints']==0:
                    continue
                data['bbox']=bb_anns[i]['bbox']
                data['height']=height
                data['width']=width
                data['is_loaded']=False
                self.data.append(data)

    def __len__(self):
        return self.config.NUM_TOT_DATA
        #return len(self.data)

    def __getitem__(self, id):
        annotations = self.data[id]
        img_id = annotations['image_id']

        #load and modify info here
        # define file name (12 zeros)
        img_id = str(img_id)
        img_id = img_id.zfill(12)
        if self.train:
            img_name = "images/train2017/"+img_id+".jpg"
        else:
            img_name = "images/val2017/"+img_id+".jpg"
        img_file_name = os.path.join(self.COCO_PATH,img_name)
        
        #org_image = img.imread(img_file_name)
        org_image = Image.open(img_file_name).convert('RGB')
        org_image = np.array(org_image)/255
        

        new_bbox, tot_pad = self.image_crop_resize(id)

        new_x = new_bbox[0]+tot_pad[0][0]
        new_y = new_bbox[1]+tot_pad[1][0]
        
        ratio = self.config.HEIGHT/new_bbox[3]
        # print(ratio,self.config.HEIGHT/new_bbox[3],self.config.WIDTH/new_bbox[3],self.config.HEIGHT/new_bbox[2] )
        
        #---------------------------#
        #    cropping + resizing    #
        pad_image = cv2.copyMakeBorder(org_image,tot_pad[1][0],tot_pad[1][1],tot_pad[0][0],tot_pad[0][1],cv2.BORDER_CONSTANT,value=[0,0,0])
        re_image = pad_image[new_y:new_y+new_bbox[3],new_x:new_x+new_bbox[2],:]

        if self.config.LARGE_HEATMAP:
            heatmaps = self.Large_heatmap_generation(id,ratio)
            re_heatmaps = np.zeros((1,re_image.shape[0],re_image.shape[1]))
            for heatmap in heatmaps:
                pad_heat_map = cv2.copyMakeBorder(heatmap,tot_pad[1][0],tot_pad[1][1],tot_pad[0][0],tot_pad[0][1],cv2.BORDER_CONSTANT,value=0)
                re_heatmap = pad_heat_map[new_y:new_y+new_bbox[3],new_x:new_x+new_bbox[2]]
                re_heatmaps = np.append(re_heatmaps,[re_heatmap],axis=0)
                re_heatmaps[0] +=re_heatmap
            re_heatmaps = self.transforms_heatmaps(torch.from_numpy(re_heatmaps))
        else:
            re_heatmaps = self.Small_heatmap_generation(id,new_bbox)
        
        #print(re_image)
        # I need to change the channel dimension
        re_image = np.transpose(re_image,[2,0,1])

        # print("return size",re_image.shape)
        # Apply transforms
        re_image = self.transforms_image(torch.from_numpy(re_image))
        
        if (not annotations['is_loaded']) and self.config.SAVE_RESIZED:
            path = os.path.join(self.SAMPLE_PATH,self.RESIZED)
            Path(path).mkdir(exist_ok=True)
            # test
            plt.clf()
            plot_img = re_image.numpy()
            plot_img = np.transpose(plot_img,[1,2,0])
            plt.imshow(plot_img)
            # self.coco.showAnns([annotations],True)  
            plt.savefig(os.path.join(path,(self.THEME+"_resized_img_"+img_id+"heatmaps"+".jpg")))
            # test
            plt.clf()
            plt.imshow(re_heatmaps[0].numpy())
            plt.savefig(os.path.join(path,(self.THEME+"_resized_heat_"+img_id+"heatmaps"+".jpg")))
            #print(re_heatmaps.shape)
            self.data[id]['is_loaded'] = True
        
        re_heatmaps = re_heatmaps[1:,:,:]
        
        if (self.check_heatmaps>0) and annotations['num_keypoints'] > 8:
            self.show_heatmaps(id,re_heatmaps,re_image)
            self.check_heatmaps -= 1
        
        return re_image, re_heatmaps, np.array(new_bbox), id, np.array(annotations['keypoints']), annotations['num_keypoints']
    
    def save_key_img(self,id,is_key = False, key=None, save_img = False):
        annotations = self.data[id]
        img_id = annotations['image_id']
    
        
        img_id = str(img_id)                                # define file name (12 zeros)
        img_id = img_id.zfill(12)               
        if self.train:
            img_name = "images/train2017/"+img_id+".jpg"
        else:
            img_name = "images/val2017/"+img_id+".jpg"
        img_file_name = os.path.join(self.COCO_PATH,img_name)
        
        image = img.imread(img_file_name)
        plt.clf()
        if is_key:
            plt.subplot(2,1,1)
        plt.axis("off")
        plt.imshow(image)
        self.coco.showAnns([annotations],True)

        if is_key:
            #print("------------")
            #for i in range(17):
            #    print("true : ",annotations['keypoints'][i*3:i*3+3])
            #    print("pred : ",np.int32(key[i*3:i*3+2]),", conf : ",key[i*3+2])

            annotations['keypoints'] = np.int32(key).tolist()
            plt.subplot(2,1,2)
            plt.axis("off")
            plt.imshow(image)
            self.coco.showAnns([annotations],True)

        inst_id = annotations['id']

        if is_key and save_img:
            Path(os.path.join(SAMPLE_PATH,"pred")).mkdir(exist_ok=True) 
            plt.savefig(os.path.join(os.path.join(SAMPLE_PATH,"pred"),(self.THEME+img_id+"_"+str(inst_id)+".jpg")))
            
        elif save_img:
            Path(os.path.join(SAMPLE_PATH,"train")).mkdir(exist_ok=True)
            plt.savefig(os.path.join(os.path.join(SAMPLE_PATH,"train"),(self.THEME+img_id+"_"+str(inst_id)+".jpg")))
        
        if is_key:
            req_anno = {}
            req_anno['image_id'] = annotations['image_id']
            req_anno['category_id'] = annotations['category_id']
            req_anno['keypoints'] = key.tolist()
            req_anno['id'] = inst_id
            annotations = req_anno
        
        


        return annotations
    
    def image_crop_resize(self,id):
        img_height = self.data[id]['height']
        img_width = self.data[id]['width']
        bbox = self.data[id]['bbox']

        # I need to check the scale of the bbox.
        # First, I need to make it clear (the bbox coordinates)
        new_bbox = []
        new_bbox.append(math.floor(bbox[0]))
        new_bbox.append(math.floor(bbox[1]))
        new_x = math.ceil(bbox[0]+bbox[2])
        if new_x<img_width:
            new_bbox.append(new_x-new_bbox[0])
        else:
            new_bbox.append(new_x-new_bbox[0]-1)

        new_y = math.ceil(bbox[1]+bbox[3])
        if new_y<img_height:
            new_bbox.append(new_y-new_bbox[1])
        else:
            new_bbox.append(new_y-new_bbox[1]-1)
       

        tot_pad = [[0,0],[0,0]]
        # Check the bbox ratio and resize it. 
        # i want 3:4 size.
        if (new_bbox[2]+1)/self.config.IMG_RATIO[0] > (new_bbox[3]+1)/self.config.IMG_RATIO[1]:
            #when I need to increase in height.
            rem = (new_bbox[2]+1)%self.config.IMG_RATIO[0]
            if rem !=0:
                rem = self.config.IMG_RATIO[0]-rem
                new_bbox[0],new_bbox[2],tot_pad[0] = safe_resize(new_bbox[0],new_bbox[2],img_width,(rem//2,rem//2+rem%2))
            if (new_bbox[2]+1)%self.config.IMG_RATIO[0] != 0:
                print("error_width")
                assert(0)
            req_incre = self.config.IMG_RATIO[1]*(new_bbox[2]+1)//self.config.IMG_RATIO[0] - (new_bbox[3]+1)
            new_bbox[1],new_bbox[3],tot_pad[1] = safe_resize(new_bbox[1],new_bbox[3],img_height,(req_incre//2,req_incre//2+req_incre%2))                

        elif (new_bbox[2]+1)/self.config.IMG_RATIO[0] < (new_bbox[3]+1)/self.config.IMG_RATIO[1]:
            #whn I need to increase in width
            rem = (new_bbox[3]+1)%self.config.IMG_RATIO[1]
            if rem !=0:
                rem = self.config.IMG_RATIO[1]-rem
                new_bbox[1],new_bbox[3],tot_pad[1] = safe_resize(new_bbox[1],new_bbox[3],img_height,(rem//2,rem//2+rem%2))
            if (new_bbox[3]+1)%self.config.IMG_RATIO[1] != 0:
                print("error_height")
                print(new_bbox[3],bbox[3],rem)
                assert(0)
            req_incre = self.config.IMG_RATIO[0]*(new_bbox[3]+1)//self.config.IMG_RATIO[1] - (new_bbox[2]+1)
            new_bbox[0],new_bbox[2],tot_pad[0] = safe_resize(new_bbox[0],new_bbox[2],img_width,(req_incre//2,req_incre//2+req_incre%2))  
        

        # now increase the image 15% larger.
        # I skipped it. When I add scaling data augmentation, it will be add.

        return new_bbox, tot_pad
        
    def Large_heatmap_generation(self,id,ratio):
        anns = self.data[id]
        n_key = anns['num_keypoints']
        keypoints = anns['keypoints']

        img_width = anns['width']
        img_height = anns['height']

        x = np.arange(0,img_width-1,1)
        y = np.arange(0,img_height-1,1)

        result = np.zeros((1,img_height,img_width),dtype=float)

        
        tot_v = result[0]
        
        for i in range(len(keypoints)//3):
            if keypoints[i*3+2] != 0:
                key_x = keypoints[i*3]
                key_y = keypoints[i*3+1]
                #print(i*3+2,":",key_x,",",key_y,",",keypoints[i+2])

                x_i = np.arange(-key_x,img_width-key_x,1,dtype=float)
                y_i = np.arange(-key_y,img_height-key_y,1,dtype=float)

                x,y = np.meshgrid(x_i,y_i,indexing='xy')
                d = np.sqrt(x*x + y*y)*ratio*ratio/(self.config.SIGMA*self.config.SIGMA*2)
                # A = 1/(2*math.pi*self.config.SIGMA*self.config.SIGMA)
                A = 1
                v = A * np.exp(-d)
                
                result = np.append(result,[v],axis=0)
                tot_v += v
            else :
                v = np.zeros((1,img_height,img_width),dtype=float)
                result = np.append(result,v,axis=0)
        
        #print(len(tot_v),len(tot_v[0]))

        # for 2d plot
        '''
        plt.figure(2)
        plt.imshow(tot_v)
        plt.show()
        '''

        # for 3d plot
        '''
        x_axis = np.arange(0,img_width,1)
        y_axis = np.arange(0,img_height,1)
        fig = plt.figure(3)
        ax = plt.axes(projection = "3d")
        ax.contour3D(x_axis,y_axis,tot_v,5,cmap='binary')
        plt.show()
        '''

        result = np.delete(result,0,0)

        return result

    def Small_heatmap_generation(self,id, new_bbox):
        anns = self.data[id]
        n_key = anns['num_keypoints']
        keypoints = anns['keypoints']

        ratio = self.config.HEIGHT/(new_bbox[3]*self.config.IN_OUT_RATIO)

        key_x = keypoints[0::3]
        key_y = keypoints[1::3]
        key_label = keypoints[2::3]
        for i in range(len(keypoints)//3):
            key_x[i] = math.ceil((key_x[i] - new_bbox[0])*ratio)
            key_y[i] = math.ceil((key_y[i] - new_bbox[1])*ratio)

        result = np.zeros((1,self.config.HEIGHT//self.config.IN_OUT_RATIO,self.config.WIDTH//self.config.IN_OUT_RATIO),dtype=float)
        
        for i in range(len(keypoints)//3):
            if keypoints[i*3+2] != 0:
            
                # print(i*3+2,":",key_x[i],",",key_y[i],",",key_label[i])

                x_i = np.arange(-key_x[i],self.config.WIDTH//self.config.IN_OUT_RATIO-key_x[i],1,dtype=float)
                y_i = np.arange(-key_y[i],self.config.HEIGHT//self.config.IN_OUT_RATIO-key_y[i],1,dtype=float)

                x,y = np.meshgrid(x_i,y_i,indexing='xy')
                d = np.sqrt(x*x + y*y)/(self.config.SIGMA*self.config.SIGMA*2)
                # A = 1/(2*math.pi*self.config.SIGMA*self.config.SIGMA)
                A = 1
                v = A * np.exp(-d)
                
                result = np.append(result,[v],axis=0)
                result[0] += v
            else :
                v = np.zeros((1,self.config.HEIGHT//self.config.IN_OUT_RATIO,self.config.WIDTH//self.config.IN_OUT_RATIO),dtype=float)
                result = np.append(result,v,axis=0)
        return torch.from_numpy(result)

    def show_heatmaps(self,id,heatmaps,re_image,test=False):
        annotations = self.data[id]
        img_id = annotations['image_id']

        if heatmaps.shape[0] != 17:
            print("wrong heatmap size %d"%heatmaps.shape[0])
            assert(0)
    
        img_id = str(img_id)                                # define file name (12 zeros)
        img_id = img_id.zfill(12)               
        if self.train:
            img_name = "images/train2017/"+img_id+".jpg"
        else:
            img_name = "images/val2017/"+img_id+".jpg"
        img_file_name = os.path.join(self.COCO_PATH,img_name)
        
        image = img.imread(img_file_name)
        
        plt.figure()
        plt.clf()
        plt.subplot(5,4,1)
        plt.imshow(image)

        plt.subplot(5,4,2)
        plt.imshow(image)
        self.coco.showAnns([annotations],True)

        plt.subplot(5,4,3)
        plot_img = re_image.numpy()
        plot_img = np.transpose(plot_img,[1,2,0])
        plt.imshow(plot_img)

        max_val = np.max(heatmaps.numpy())
        print(max_val)
        for i in range(17):
            plt.subplot(5,4,4+i)
            plt.imshow(heatmaps[i]/(max_val+0.0001))
        if test:
            plt.savefig(os.path.join(self.SAMPLE_PATH,self.THEME+str(id)+"check_heatmaps_out.jpg"))
        else:
            plt.savefig(os.path.join(self.SAMPLE_PATH,self.THEME+str(id)+"check_heatmaps.jpg"))

        if self.flag and annotations['num_keypoints']>8 and test:
            for i in range(17):
                plt.clf()
                plt.imshow(heatmaps[i]/max_val)
                plt.savefig(os.path.join(self.SAMPLE_PATH,str(i)+str(id)+"heats.jpg"))
            self.flag = False

    def get_imgIds(self):
        img_ids = []
        for i in range(self.config.NUM_TOT_DATA):
            annotations = self.data[i]
            img_id = annotations['image_id']
            img_ids.append(img_id)
        
        img_ids = set(img_ids)
        return img_ids
        
# debug section
if __name__ == "__main__":
    data = DataLoader(False)
    print(data.__len__())
    
    
    '''
    for i in range(20):
        outim, heatmaps, keys = data.__getitem__(100+i)
        print("out dim :",outim.shape, ", heatmaps :",heatmaps.shape)
    '''
    print(data.data.shape)

    im_out, heatmaps, old_bbox, img_ids = data.__getitem__(121)
    
    
    
    re_keypoints = restore_heatmap(heatmaps, old_bbox)

    #print(keypoints)
    #print(re_keypoints)
    #print(keypoints-re_keypoints)
    data.save_key_img(121,True,re_keypoints,True)


    