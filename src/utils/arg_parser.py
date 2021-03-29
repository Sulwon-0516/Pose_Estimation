import argparse
import os
from .config import update_config

# The code is from official baseline implementation.
# ref : https://github.com/microsoft/human-pose-estimation.pytorch/blob/master/pose_estimation/train.py
def input_argparser(cfg_path):

    parser = argparse.ArgumentParser(description="Setting the basic option")

    # load config file
    parser.add_argument('--cfg',help = "config file name",required=True, type = str)
    args, rest = parser.parse_known_args()

    cfg_file = os.path.join(cfg_path,args.cfg)
    if not os.path.isfile(cfg_file):
        print("There isn't file {}".format(cfg_file))
        assert(0)
    update_config(cfg_file)

    
    parser.add_argument('--gpus',
                        help='gpus',
                        type=str)
    parser.add_argument('--workers',
                        help='num of dataloader workers',
                        type=int)

    args = parser.parse_args()

    return args