"""Main configuration."""


class CFG:
    DIM = (512, 512)

    NUM_WORKERS = 4
    BATCH_SIZE = 16
    EPOCHS = 6
    SEED = 1710
    LR = 3e-4

    TRAIN_IMG = "../input/shopee-product-matching/train_images"
    VAL_IMG = ""
    DEVICE = "cuda"

    MEAN = [0.485, 0.456, 0.406]
    STD = [0.229, 0.224, 0.225]

    ################################################# MODEL ####################################################################

    MODEL_NAME = "efficientnet_b0"  # efficientnet_b3 #efficientnetb5 #efficientnetb7

    SCHEDULER = "CosineAnnealingWarmRestarts"  # 'CosineAnnealingLR'
    T_0 = 3  # CosineAnnealingWarmRestarts
    min_lr = 1e-6
