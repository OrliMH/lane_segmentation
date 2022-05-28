
class Config(object):
    # model config
    OUTPUT_STRIDE = 16
    ASPP_OUTDIM = 256
    SHORTCUT_DIM = 48
    SHORTCUT_KERNEL = 1
    NUM_CLASSES = 8

    # train config
    EPOCHS = 10
    WEIGHT_DECAY = 1.0e-4 
    SAVE_PATH = "disk2/lane_segment/AdamW_cosine_annealing_lane_segmentation/logs"
    BASE_LR = 1e-1 
    BETA = (0.9, 0.999)
    EPS = 1e-08


    # cosine_annealing
    CYCLE_INTER = 10
    CYCLE_NUM = 3
