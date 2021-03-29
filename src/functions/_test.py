import torch
import os
import json
from pathlib import Path
from torch.autograd import Variable
from ..dataloader import coco_data_loader


def _test(dataset,batch_size,criterion,model,M_PATH,PATH,TITLE,config,device):
    dataloader = torch.utils.data.DataLoader(dataset = dataset, batch_size = batch_size, shuffle = False)
    # load model
    print(M_PATH)
    with torch.no_grad():
        checkpoint = torch.load(M_PATH)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        n_keys = dataset.__len__()
        tot_data = []
        tot_loss = 0
        tot_OKS = 0
        save_predict_heatmap = config.SAVE_HEATMAP
        for i, data in enumerate(dataloader):
            imgs, heatmaps, old_bboxs, ids , keypoints, n_keys = data
            #print(old_bboxs)

            
            imgs, heatmaps = Variable(imgs).float(), Variable(heatmaps).float()
            imgs.to(device)
            heatmaps.to(device)


            p_heatmaps = model(imgs)
            loss = criterion(heatmaps,p_heatmaps)
            loss = loss.sum(axis = (2,3))
            

            for j in range(ids.shape[0]):
                re_keypoints = coco_data_loader.restore_heatmap(p_heatmaps[j], old_bboxs[j],keypoints[j])
                if config.SAVE_PREDICTED and j<config.SAVE_IMG_PER_BATCH:
                    re_anns = dataset.save_key_img(ids[j],True,re_keypoints,True)
                else:
                    re_anns = dataset.save_key_img(ids[j],True,re_keypoints)
                if save_predict_heatmap > 0 and n_keys[j]>8:
                    print("called")
                    dataset.show_heatmaps(ids[j],p_heatmaps[j],imgs[j],True)
                    save_predict_heatmap -= 1
                
                re_anns['score'] = - loss[j].sum().item()
                tot_data.append(re_anns)
            
            tot_loss += loss.sum(axis = (0,1))

            print("step %d, loss : %f" %(i,loss.sum()))
        tot_loss = tot_loss/n_keys


        # check the path exists.
        Path(PATH.RESULT_PATH).mkdir(exist_ok=True)
        f_name = os.path.join(PATH.RESULT_PATH,PATH.MODEL)
        Path(f_name).mkdir(exist_ok=True)
        f_name = os.path.join(f_name,PATH.PRED_PATH)
        Path(f_name).mkdir(exist_ok=True)
        n_file = len(os.listdir(f_name))
        file_path = os.path.join(f_name,PATH.PRED_NAME%(TITLE,n_file+1))
        
        with open(file_path,"w") as res_file:
            json.dump(tot_data,res_file)

   