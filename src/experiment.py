import hydra
from omegaconf import DictConfig  
import lightning as lit
from net.base import Net
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import wandb
from lightning.pytorch.loggers import WandbLogger
from dataset.loaders import load_har, load_wisdm

import lightning.pytorch.callbacks as cb 

from omegaconf import OmegaConf

import os

@hydra.main(config_path="../cfg", config_name="base", version_base=None)
def main(cfg: DictConfig):
    lit.seed_everything(cfg.seed) 
    wandb_logger = WandbLogger(**cfg.wandb)  
    model = Net(cfg.net)

    train_loader, val_loader, test_loader = load_har(**cfg.dataset)

    trainer = lit.Trainer(logger=wandb_logger, callbacks=[
        cb.EarlyStopping(
            monitor="val_loss",
            patience=15,
            verbose=True,
            mode="min", min_delta=1e-3
        ), cb.ModelCheckpoint(
            monitor="val_acc",
             mode="max", 
             save_top_k=1, 
            filename="{epoch}-{val_acc:.4f}"),], 
             **cfg.trainer)

    print('Training...')
    trainer.fit(model, train_loader, val_loader)
    
    print('Testing...')
    best_model_path = trainer.checkpoint_callback.best_model_path
    best_model = Net.load_from_checkpoint(best_model_path, cfg=cfg.net)
    trainer.test(best_model, test_loader)
    
    hyperparams_dict = OmegaConf.to_container(cfg, resolve=True)
    hyperparams_dict["info"] = {  
        "num_params": get_num_params(model),
    }
    wandb_logger.log_hyperparams(hyperparams_dict)  

def get_num_params(module):
    """
    Returns the number of parameters in a Lightning module.
    
    Args:
        module (lightning.pytorch.LightningModule): The Lightning module to get the number of parameters for.
    
    Returns:
        int: The number of parameters in the module.
    """
    total_params = sum(p.numel() for p in module.parameters() )
    return total_params




if __name__ == "__main__":
    os.environ['HYDRA_FULL_ERROR'] = "1"
    main()