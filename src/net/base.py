import torch
import lightning as lit
from torchmetrics import Accuracy

import torch.nn as nn
import torch.nn.functional as F

from hydra.utils import instantiate

from net.cnn import ConvBlock

class Net(lit.LightningModule):
    def __init__(self,cfg
                ):  
        super(Net, self).__init__()
        self.save_hyperparameters()
        self.depth = cfg.depth 
        self.cfg = cfg

        self.embed = instantiate(cfg.embed) 
        self.features = nn.Sequential(*[instantiate(cfg.block) for _ in range(self.depth-1)])
        self.lstm_block = instantiate(cfg.rnn_block)
        self.unembed = instantiate(cfg.unembed)   
        
        self.train_acc = Accuracy(task='multiclass', num_classes=cfg.num_classes)
        self.val_acc = Accuracy(task='multiclass', num_classes=cfg.num_classes)
        self.test_acc = Accuracy(task='multiclass', num_classes=cfg.num_classes)


    def forward(self, x):
        x = self.embed(x)
        x = self.features(x) 
        x = x.permute(0, 2, 1) 
        x = self.lstm_block(x)
        x = x.mean(dim=1)
        x = self.unembed(x) 
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        self.log('train_loss', loss, prog_bar=True)
        self.log('train_acc', self.train_acc(y_hat, y), prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        self.log('val_loss', loss)
        self.log('val_acc', self.val_acc(y_hat, y), prog_bar=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        self.log('test_loss', loss)
        self.log('test_acc', self.test_acc(y_hat, y), prog_bar=True)

    def configure_optimizers(self):
        optimizer = instantiate(self.cfg.optimizer, self.parameters())
        return optimizer
    
