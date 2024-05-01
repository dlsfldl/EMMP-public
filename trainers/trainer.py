import os
import time
import math

import numpy as np
import torch
from metrics import averageMeter

class BaseTrainer:
    """Trainer for a conventional iterative training of model for classification"""
    def __init__(self, optimizer, training_cfg, device):
        self.training_cfg = training_cfg
        self.device = device
        self.optimizer = optimizer

    def train(self, model, d_dataloaders, logger=None, logdir=""):
        cfg = self.training_cfg
    
        time_meter = averageMeter()
        train_loader, val_loader = (d_dataloaders["training"], d_dataloaders["validation"])
        kwargs = {'dataset_size': len(train_loader.dataset)}
        i_iter = 0
        best_val_loss = np.inf
    
        for i_epoch in range(1, cfg['n_epoch'] + 1):
            for x, w in train_loader:
                i_iter += 1
                if train_loader.dataset.augmentation:
                    x, w = train_loader.dataset.rand_aug(x, w)
                model.train()
                start_ts = time.time()
                if i_epoch < cfg['n_epoch'] * 0.2:
                    reg = False
                else:
                    reg = True
                d_train = model.train_step(x.to(self.device), w.to(self.device), optimizer=self.optimizer, reg=reg, **kwargs)
                time_meter.update(time.time() - start_ts)
                if logger is not None:
                    logger.process_iter_train(d_train)

                if i_iter % cfg.print_interval == 0:
                    if logger is not None:
                        d_train = logger.summary_train(i_iter)
                    else:
                        d_train['loss/train_loss_'] = d_train['loss']
                    print(
                        f"Epoch [{i_epoch:d}] \nIter [{i_iter:d}]\tAvg Loss: {d_train['loss/train_loss_']:.6f}\tElapsed time: {time_meter.sum:.4f}"
                    )
                    time_meter.reset()
                if i_epoch % cfg.save_epochs == 0:
                    self.save_model(model, logdir, i_epoch=i_epoch)
                model.eval()
                if i_iter % cfg.val_interval == 0:
                    # log_likelihood = eval_log_likelihood(model, train_loader, split=50)
                    # d_ll= {'log_likelihood_': log_likelihood.sum()}
                    # logger.add_val(i_iter, d_ll)
                    for x, w in val_loader:
                        d_val = model.validation_step(x.to(self.device), w.to(self.device))
                        if logger is not None:
                            logger.process_iter_val(d_val)
                    if logger is not None:
                        d_val = logger.summary_val(i_iter)
                        print(d_val['print_str'])
                    else:
                        d_val['loss/val_loss_'] = d_val['loss']
                        print(
                            f"Iter [{i_iter:d}]\tAvg Loss: {d_val['loss/val_loss_']:.6f}"
                        )
                    val_loss = d_val['loss/val_loss_']
                    best_model = val_loss < best_val_loss

                    if best_model:
                        print(f'Iter [{i_iter:d}] best model saved {val_loss:.6f} <= {best_val_loss:.6f}')
                        best_val_loss = val_loss
                        self.save_model(model, logdir, best=True)
                    
                if cfg.visualization.type == 'none' or cfg.visualization.type == []:
                    pass
                else:
                    if i_iter % cfg.visualize_interval == 0:
                        d_vis = model.visualization_step(train_loader, val_loader, device=self.device, **cfg.visualization)
                        if logger is not None:
                            logger.add_val(i_iter, d_vis)

        self.save_model(model, logdir, i_iter="last")
        return model, best_val_loss

    def save_model(self, model, logdir, best=False, i_iter=None, i_epoch=None):
        if best:
            pkl_name = "model_best.pkl"
        else:
            if i_iter is not None:
                pkl_name = f"model_iter_{i_iter}.pkl"
            else:
                pkl_name = f"model_epoch_{i_epoch}.pkl"
        state = {"epoch": i_epoch, "iter": i_iter, "model_state": model.state_dict()}
        save_path = os.path.join(logdir, pkl_name)
        torch.save(state, save_path)
        print(f"Model saved: {pkl_name}")

class LatentTrainer(BaseTrainer):
    def __init__(self, optimizer, training_cfg, device):
        super().__init__(optimizer, training_cfg, device)