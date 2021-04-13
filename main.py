
from src.test import test
from src.train import train
from src.valid import valid
from src.utils.config import config, gen_config
from src.utils.arg_parser import input_argparser

from pathlib import Path
import os
import torch

SETTING_PATH = "./test_setting"

def main():
    Path(SETTING_PATH).mkdir(exist_ok=True)

    args = input_argparser(SETTING_PATH)


    cuda_sets = ""
    for i in range(len(config.GPU_NUM)-1):
        cuda_sets = cuda_sets + str(config.GPU_NUM[i]) + ","
    cuda_sets = cuda_sets + str(config.GPU_NUM[len(config.GPU_NUM)-1])
    os.environ["CUDA_VISIBLE_DEVICES"] = cuda_sets
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # from config file.
    if config.IS_TRAIN:
        train(config,device)
    if config.IS_TEST:
        test(config,device)
    if config.IS_VALID:
        valid(config)
    
    
def init_config():
    # to make sample setting file, I used it
    Path(SETTING_PATH).mkdir(exist_ok=True)
    cfg_file = os.path.join(SETTING_PATH,"sample.yaml")
    gen_config(cfg_file)





if __name__ == '__main__':
    
    main()