# import albumentations
# import cv2
# import torch
# from albumentations.pytorch.transforms import ToTensorV2
# from torch.utils.data import Dataset


# def get_train_transforms(cfg):
#     return albumentations.Compose(
#         [
#             albumentations.HorizontalFlip(p=0.5),
#             albumentations.VerticalFlip(p=0.5),
#             albumentations.Rotate(limit=120, p=0.8),
#             albumentations.RandomBrightness(limit=(0.09, 0.6), p=0.5),
#             # albumentations.Cutout(num_holes=8, max_h_size=8, max_w_size=8, fill_value=0, always_apply=False, p=0.5),
#             # albumentations.ShiftScaleRotate(
#             #   shift_limit=0.25, scale_limit=0.1, rotate_limit=0
#             # ),
#             albumentations.Normalize(cfg.MEAN, cfg.STD, max_pixel_value=255.0, always_apply=True),
#             ToTensorV2(p=1.0),
#         ]
#     )


# def get_valid_transforms(cfg):

#     return albumentations.Compose(
#         [albumentations.Normalize(cfg.MEAN, cfg.STD, max_pixel_value=255.0, always_apply=True), ToTensorV2(p=1.0)]
#     )


# class CVPRDataset(Dataset):
#     def __init__(self, image_1, image_2, labels, dim=(512, 512), augmentation=None):
#         self.image_1 = image_1
#         self.image_2 = image_2
#         self.labels = labels
#         self.dim = dim
#         self.augmentation = augmentation

#     def __len__(self):
#         return len(self.image_1)

#     def __getitem__(self, index):
#         image = self.image_1[index]

#         image = cv2.imread(f"{TRAIN_IMG}/{image}")

#         image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#         img_1 = cv2.cvtColor(img_1, cv2.COLOR_BGR2RGB)

#         if self.dim:
#             image = cv2.resize(image, self.dim)

#         if self.augmentation:
#             augmented_0 = self.augmentation(image=image)
#             augmented_1 = self.augmentation(image=img_1)
#             image = augmented_0["image"]
#             img_1 = augmented_1["image"]

#         return image, img_1, torch.tensor(self.labels[index], dtype=torch.float32)
