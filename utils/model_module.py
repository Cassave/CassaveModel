import torch.nn.functional as F
from torchmetrics.functional import accuracy
import torchmetrics
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks import EarlyStopping

class ModelModule(pl.LightningModule):
    def __init__(self, config, n_classes=5):
        super().__init__()
        self.save_hyperparameters(config)
        self.resnet = torchvision.models.resnet18(pretrained=True)
        self.resnet.fc = torch.nn.Linear(self.resnet.fc.in_features, n_classes)
        #self.metric = metric

    def forward(self, x):
      return self.resnet(x)


    def training_step(self, batch, batch_idx):
      x, y = batch
      y_hat = self(x)
      loss = F.cross_entropy(y_hat, y)
      #acc = self.metric(y_hat, y)
      acc = accuracy(y_hat, y)
      self.log("train_loss", loss, prog_bar=True)
      self.log("train_acc", acc, prog_bar=True)
      return loss
    
    def validation_step(self, batch, batch_idx):
      x, y = batch
      y_hat = self(x)
      loss = F.cross_entropy(y_hat, y)
      #acc = self.metric(y_hat, y)
      acc = accuracy(y_hat, y)
      self.log("val_loss", loss, prog_bar = True)
      self.log("val_acc", acc, prog_bar=True)
      return loss

    def configure_optimizers(self):
      optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
      return optimizer