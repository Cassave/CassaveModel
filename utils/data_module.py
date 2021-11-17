import pytorch_lightning as pl
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
import torchvision
from torch.utils.data import DataLoader, WeightedRandomSampler
import os
from pathlib import Path
import albumentations as A
import cv2


class Dataset(torch.utils.data.Dataset):
    def __init__(self, images, labels, trans=None, path=Path("cassava-leaf-disease-merged"), size=512):
        self.images = images
        self.labels = labels
        self.trans = trans
        self.path = path
        self.size = size

    def __len__(self):
        return len(self.images)

    def __getitem__(self, ix):
        img = cv2.imread(f"{self.path}/train/{self.images[ix]}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        if self.trans and str(self.labels[ix])!="3":
            img = self.trans(image=img)["image"]
            img = torch.from_numpy(img/255.).float().permute(2, 0, 1)
        else:
          img = torchvision.io.read_image(
              f"{self.path}/train/{self.images[ix]}").float()/255.
          transform = torchvision.transforms.CenterCrop(self.size)
          img = transform(img)

        label = torch.tensor(self.labels[ix], dtype=torch.long)
        return img, label


class DataModule(pl.LightningDataModule):
    def __init__(self,
                 path=Path("cassava-leaf-disease-merged"),
                 batch_size=64,
                 test_size=0.2,
                 random_state=42,
                 subset=0,
                 size=512,
                 train_trans=None,
                 val_trans=None,
                 sampler=None):
        super().__init__()
        self.path = path
        self.batch_size = batch_size
        self.test_size = test_size
        self.random_state = random_state
        self.subset = subset
        self.size = size
        self.train_trans = train_trans
        self.val_trans = val_trans
        self.sampler = sampler


    def setup(self, stage=None):
        merged_ds_df = pd.read_csv(self.path/"merged.csv")
        train, val = train_test_split(
            merged_ds_df,
            test_size=self.test_size,
            shuffle=True,
            stratify=merged_ds_df["label"],
            random_state=self.random_state
        )

        # Sampler
        if self.sampler:
          labels_uq, count = np.unique(train["label"], return_counts=True)
          class_weights = [sum(count)/c for c in count]
          self.sampler = WeightedRandomSampler(
              [class_weights[lbl] for lbl in train["label"]],
              len(train["label"]), replacement=True)       

        print("Ejemplos de entrenamiento: ", len(train))
        print("Ejemplos de validacion: ", len(val))

        if self.subset:
            _, train = train_test_split(
                train,
                test_size=self.subset,
                shuffle=True,
                stratify=train["label"],
                random_state=self.random_state
            )
            print("Entrenando con ", len(train), " ejemplos")

        self.train_ds = Dataset(
            train["image_id"].values,
            train["label"].values,
            trans=A.Compose([
                getattr(A, trans)(**params) for trans, params in self.train_trans.items()
            ]) if self.train_trans else None,
            path=self.path,
            size=self.size,
        )
       
        self.val_ds = Dataset(
            val["image_id"].values,
            val["label"].values,
            trans= A.Compose([
                getattr(A, trans)(**params) for trans, params in self.val_trans.items()
            ]) if self.val_trans else None,
            path=self.path,
            size=self.size)

    def train_dataloader(self):
      if self.sampler:
        return DataLoader(self.train_ds, batch_size=self.batch_size, pin_memory=True, sampler=self.sampler)
      else:
        return DataLoader(self.train_ds, batch_size=self.batch_size, pin_memory=True)
    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.batch_size, shuffle=False, pin_memory=True)
