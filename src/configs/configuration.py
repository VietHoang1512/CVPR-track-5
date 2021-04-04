"""Main configuration."""


class CFG:
    DIM = (224, 224)

    NUM_WORKERS = 4
    BATCH_SIZE = 16
    EPOCHS = 10
    SEED = 1710
    LR = 3e-4

    DEVICE = "cuda"

    MEAN = [0.485, 0.456, 0.406]
    STD = [0.229, 0.224, 0.225]

    ################################################# MODEL ####################################################################
    MAX_LEN = 64
    MODEL_NAME = "efficientnet-b0"  # efficientnet_b3 #efficientnetb5 #efficientnetb7
    BERT_MODEL = "roberta-base"
    SCHEDULER = "CosineAnnealingWarmRestarts"  # 'CosineAnnealingLR'

    T_0 = 3  # CosineAnnealingWarmRestarts
    min_lr = 1e-6
