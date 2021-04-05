"""Main configuration."""


class CFG:
    OUTPUT_DIR = "outputs"

    DIM = (224, 224)
    NUM_WORKERS = 0
    BATCH_SIZE = 16
    EPOCHS = 10
    SEED = 1710
    LR = 1e-5
    OUT_FEATURES = 256
    N_HIDDENS = 4
    DEVICE = "cuda"

    MEAN = [0.485, 0.456, 0.406]
    STD = [0.229, 0.224, 0.225]

    MAX_LEN = 64
    MODEL_NAME = "efficientnet-b4"
    BERT_MODEL = "roberta-base"
    SCHEDULER = "CosineAnnealingWarmRestarts"  # 'CosineAnnealingLR'

    T_0 = 3  # CosineAnnealingWarmRestarts
    min_lr = 1e-6
