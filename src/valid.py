import torch
import os
from .dataloader.coco_data_loader import COCO_DataLoader
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

def valid(config):

    val_dataset = COCO_DataLoader(config.VAL.IS_TRAIN,config)

    annType = ['segm','bbox','keypoints']
    annType = annType[config.TYPE]      #specify type here
    prefix = 'person_keypoints' if annType=='keypoints' else 'instances'

    # GT means Ground Truth
    cocoGT = val_dataset.coco

    # Get result File from the folder.
    res_path = os.path.join(config.PATH.RESULT_PATH,config.PATH.MODEL)
    if not os.path.isdir(res_path):
        print("Invalid result path")
        assert(0)
    res_path = os.path.join(res_path,config.PATH.PRED_PATH)
    if not os.path.isdir(res_path):
        print("Invalid result path")
        assert(0)
    resFile = os.path.join(res_path,config.PATH.PRED_NAME%(config.THEME,config.VAL.RES_FILE))
    if not os.path.isfile(resFile):
        print("Invalid result file")
        assert(0)
    cocoDT = cocoGT.loadRes(resFile)

    # Get the Img Ids
    imgIds = sorted(val_dataset.get_imgIds())
    # print(imgIds)

    # Evaluation
    cocoEval = COCOeval(cocoGT,cocoDT,annType)
    cocoEval.params.imgIds = imgIds
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()