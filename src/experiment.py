import hydra
from omegaconf import DictConfig, OmegaConf
import lightning as lit
from net.base import Net
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import GroupKFold
import numpy as np
import wandb
from lightning.pytorch.loggers import WandbLogger
from dataset.loaders import load_har, load_har_features

import lightning.pytorch.callbacks as cb
import os


def get_num_params(module):
    """Returns the number of parameters in a Lightning module."""
    total_params = sum(p.numel() for p in module.parameters())
    return total_params


@hydra.main(config_path="../cfg", config_name="base", version_base=None)
def main(cfg: DictConfig):
    lit.seed_everything(cfg.seed)

    wandb_logger = WandbLogger(**cfg.wandb)

    hyperparams_dict = OmegaConf.to_container(cfg, resolve=True)
    wandb_logger.log_hyperparams(hyperparams_dict)

    train_dataset, test_loader = load_har(**cfg.dataset, load_all=True)

    k = cfg.k_folds
    groups = train_dataset.subject_ids.numpy()
    gkf = GroupKFold(n_splits=k)
    fold_accs = []
    dummy_X = np.zeros(len(train_dataset))

    for fold, (train_idx, val_idx) in enumerate(
        gkf.split(X=dummy_X, y=None, groups=groups)
    ):
        print(f"Starting Fold {fold+1}/{k}")

        train_ds = Subset(train_dataset, train_idx)
        val_ds = Subset(train_dataset, val_idx)

        train_loader = DataLoader(
            train_ds, batch_size=cfg.dataset.batch_size, shuffle=True
        )
        val_loader = DataLoader(val_ds, batch_size=cfg.dataset.batch_size)

        model = Net(cfg.net)

        callbacks = [
            cb.EarlyStopping(monitor="val_loss", patience=8, verbose=True, mode="min"),
            cb.ModelCheckpoint(
                monitor="val_acc",
                mode="max",
                filename=f"fold{fold}-{{epoch}}-{{val_acc:.4f}}",
                save_top_k=1,
            ),
        ]

        trainer = lit.Trainer(logger=wandb_logger, callbacks=callbacks, **cfg.trainer)
        trainer.fit(model, train_loader, val_loader)

        best_ckpt = trainer.checkpoint_callback.best_model_path
        best_model = Net.load_from_checkpoint(best_ckpt, cfg=cfg.net)

        val_metrics = trainer.validate(best_model, val_loader)[0]
        fold_accs.append(val_metrics["val_acc"])

        wandb_logger.log_metrics(
            {f"fold_{fold}_best_val_acc": val_metrics["val_acc"], "fold": fold}
        )

    avg_val_acc = np.mean(fold_accs)
    std_val_acc = np.std(fold_accs)

    print("\n============ K-FOLD FINISHED ============")
    print(f"Average K-fold val_acc = {avg_val_acc:.4f} ± {std_val_acc:.4f}")

    wandb_logger.log_metrics(
        {
            "kfold_avg_val_acc": avg_val_acc,
            "kfold_std_val_acc": std_val_acc,
            "kfold_avg_std_string": f"{avg_val_acc:.4f} ± {std_val_acc:.4f}",
        }
    )

    print("\n============ TRAINING FINAL MODEL ON FULL TRAIN ============")

    final_model = Net(cfg.net)

    full_train_loader = DataLoader(
        train_dataset, batch_size=cfg.dataset.batch_size, shuffle=True
    )

    final_trainer = lit.Trainer(logger=wandb_logger, **cfg.trainer)

    final_trainer.fit(final_model, full_train_loader)

    print("\n============ FINAL TEST ============")
    test_results = final_trainer.test(final_model, test_loader)
    print(test_results)

    wandb_logger.log_metrics({"final_test_acc": test_results[0]["test_acc"]})

    wandb.config.update(
        {
            "num_params": get_num_params(final_model),
        },
        allow_val_change=True,
    )

    wandb.finish()


if __name__ == "__main__":
    os.environ["HYDRA_FULL_ERROR"] = "1"
    main()
