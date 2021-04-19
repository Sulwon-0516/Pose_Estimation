from easydict import EasyDict as edict
import os
import yaml

config = edict()

config.GPU_NUM = [0, 1]                  # use GPU? 
config.THEME = "trial"              # Name of the training
config.IS_TRAIN = True              # Ask whether train this time
config.IS_TEST = True               # Ask whether test
config.IS_VALID = True              # Aske whether valid
config.MODEL = "baseline"           # It's model name
config.TYPE = 2                     # 0 : segm, 1 : bbox, 2 : keypoints
config.AE = False                   # Use associative Embedding or not
config.IS_BU = False
config.IS_PARALLEL = False

# Data preprocessing configs
config.DATA = edict()
config.DATA.MINIMUM_KEY = 0        # Use # >= MINIMUM_KEY, defautl : 1
config.DATA.NUM_KEYS = 17 
config.DATA.SIGMA = 2               # Gaussain heatmap sigma
config.DATA.GAUSSIAN_SCALE = 1      # the maximum value scale of the heatmap
config.DATA.HEIGHT = 256            # Crop size
config.DATA.WIDTH = 192
config.DATA.IMG_RATIO = [3,4]
config.DATA.IS_FLIP = False         # Augmentation
config.DATA.IS_ROTATE = False
config.DATA.IS_SCALE = False

config.DATA.HALF_BODY_TRANS = True
config.DATA.HALF_JOINTS = 8
config.DATA.HALF_BODY_PROB = 1

config.DATA.ROTATE_RANGE = [-40, 40]
config.DATA.SCALE_RANGE = [0.7, 1.3]
config.DATA.IS_INCRE = False        # Img crop increment ratio
config.DATA.INCRE_RATIO = 1.0
config.DATA.NUM_TOT_DATA = 12      # Set the number of data limit for small batch test
config.DATA.IN_OUT_RATIO = 4        # the ratio between input and output width

config.DATA.SAVE_RESIZED = False    # Save resized image
config.DATA.CHECK_HEATMAP = 5       # Check the input data file.
config.DATA.LARGE_HEATMAP = False

# Train Hyper params
config.TRAIN = edict()
config.TRAIN.LOAD_PREV = False      # Load previous trained data
config.TRAIN.PREV_PATH = '%s/checkpoint'
config.TRAIN.PREV_MODEL = '100_best_loss_tensor(1.6312e-05)(109_140).pt'
config.TRAIN.LOAD_PREV_OPTIM = True
config.TRAIN.LOAD_PREV_SCHED = True

config.TRAIN.NUM_WORKER = 0
config.TRAIN.TEST_EPOCH = 0
config.TRAIN.TEST_PER_BATCH = 10
config.TRAIN.LR = 1e-3
config.TRAIN.LOSS = 'MSE'
config.TRAIN.OPTIM = 'Adam'
config.TRAIN.EPOCH = 140
config.TRAIN.IS_SCHED = True
config.TRAIN.SCHED = 'MultiStepLR'
config.TRAIN.DECAY_RATE = 0.1
config.TRAIN.MILTESTONES = [90, 120]
config.TRAIN.IS_LOSS_MASK = True
config.TRAIN.IS_SHUFFLE = False
config.TRAIN.BATCH_SIZE = 8         # from baseline paper, used 128 batch
config.TRAIN.EPOCH = 140
config.TRAIN.CHECK_FREQ = 20        # Every (freq) epoch, save the model

config.TRAIN.PRETRAIN = False
config.TRAIN.PRETRAIN_MODEL = "full_train_trial1_HRNet10777_best_model.pt"
config.TRAIN.PRETRAIN_PATH = '%s/best'
config.TRAIN.PRETRAIN_MODEL_NAME = 'HRNet'


# Test Hyper params
config.TEST = edict()
config.TEST.IS_TEST = False         # If True, Use TEST set. Else use VALID set
config.TEST.GET_RESULT = True       # If False, use the previous result
config.TEST.BATCH_SIZE = 8          # Inference batch size
config.TEST.MODEL_PATH = "result/%s/checkpoint/"
config.TEST.MODEL_FILE = "100_best_loss_tensor(1.6312e-05)(109_140).pt"
config.TEST.FLIP_ENSEMBLE = False   # If True, use both flipped and un-flipped image to make result
config.TEST.NUM_WORKER = 0

config.TEST.SAVE_PREDICTED = True   # save prediction image
config.TEST.SAVE_IMG_PER_BATCH = 2  # Images save per every batches
config.TEST.SAVE_HEATMAP = 5
config.TEST.IS_TRAIN = False        # Checking overfitting on the training set,

# Validation Hyper params
config.VAL = edict()
config.VAL.IS_TRAIN = False         # If True, evaluate the result.
config.VAL.RES_FILE = 1             # Set the number as 1

# Log 
'''It will be deprecated due to the VISDOM'''
config.LOG = edict()
config.LOG.FREQ = 1            # Save log every 10 steps
config.LOG.SAVE_LOSS = True
config.LOG.SAVE_ACC = True
config.LOG.STEP_FORMAT = "(%d,%d) %d step loss : %f"
config.LOG.EPOCH_FORMAT = "(%d,%d) epoch loss : %f"
config.LOG.PATH = "result/%s/log"
config.LOG.FILE_NAME = "%s_%s_log.txt"

# Path
config.PATH = edict()
config.PATH.RESULT_PATH = "./result"
config.PATH.PRED_PATH = "prediction"
config.PATH.PRED_NAME = "%s_result_%d.json"
config.PATH.COCO_PATH = "./coco"
config.PATH.COCO_VAL_INS_PATH = "annotations/instances_val2017.json"
config.PATH.COCO_VAL_KEY_PATH = "annotations/person_keypoints_val2017.json"
config.PATH.COCO_TRAIN_INS_PATH = "annotations/instances_train2017.json"
config.PATH.COCO_TRAIN_KEY_PATH = "annotations/person_keypoints_train2017.json"

config.PATH.MDOEL = "somethings"
config.PATH.BEST_MODEL_PATH = "best"
config.PATH.CHECKPOINT_PATH = "checkpoint"

config.PATH.BEST_FILE = "%s%d_best_model.pt"
config.PATH.CHECKPOINT_FILE = "%s%d_best_loss%f(%d_%d).pt"

config.PATH.SAMPLE = "./sample"
config.PATH.RESIZED = "resized"



# Visdom
config.VIS = edict()
config.VIS.IS_USE = True
config.VIS.IS_SAVE = False
config.VIS.IS_RESET = False
config.VIS.ENV_NAMES = ["inputs", "outputs"]              # List of the env_names will be used
config.VIS.STEP_FREQ = 10                               # plotting step frequency







# Debug options
config.DEBUG = edict()
config.DEBUG.CHECK_ACC = False            # Check accuracy per every epoch
config.DEBUG.CHECK_ACC_CYC = 20           # Test on validation set for every acc cycles.
config.DEBUG.CHECK_MODEL = False          # Check model filter params if true



def update_config(config_file):
    in_config = None
    with open(config_file) as f:
        in_config = edict(yaml.load(f))
        for k, v in in_config.items():
            if k in config:
                if isinstance(v,dict):
                    # when value is dict
                    for vk, vv in v.items():
                        if vk in config[k]:
                            config[k][vk] = vv
                        else:
                            print("{}.{} no exist in config".format(k,vk))
                            assert(0)
                else:
                    # when value is just value
                    config[k] = v
            
            else:
                print("{} no exist in config".format(k))
                assert(0)
        config['PATH']['MODEL'] = config['MODEL']

# get from the official Baseline Implementation
# ref : https://github.com/microsoft/human-pose-estimation.pytorch/blob/master/lib/core/config.py
# generate config YAML file.
def gen_config(config_file):
    cfg = dict(config)
    for k, v in cfg.items():
        if isinstance(v, edict):
            cfg[k] = dict(v)

    with open(config_file, 'w') as f:
        yaml.dump(dict(cfg), f, default_flow_style=False)



