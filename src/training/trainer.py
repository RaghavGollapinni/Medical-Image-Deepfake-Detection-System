import os
from typing import Dict, Optional
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from sklearn.metrics import roc_auc_score
from .losses import MultiTaskLoss, compute_pos_weights

class EarlyStopping:
    def __init__(self, patience: int = 7, min_delta: float = 0.001):
        self.patience   = patience
        self.min_delta  = min_delta
        self.counter    = 0
        self.best_score = None
        self.stop       = False

    def step(self, score: float) -> bool:
        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            print(f"[EarlyStopping] No improvement. Counter: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                print("[EarlyStopping] Triggered. Stopping training.")
                self.stop = True
        else:
            self.best_score = score
            self.counter    = 0
        return self.stop

class Trainer:
    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        config,
        device,
        max_train_batches: Optional[int] = None,
        max_val_batches: Optional[int] = None,
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device
        self.max_train_batches = max_train_batches
        self.max_val_batches = max_val_batches
        
        # Loss
        weights = config['training']['loss_weights']
        pos_weights = compute_pos_weights(config["paths"]["train_csv"]).to(device)
        forgery_pw = weights.get('forgery_pos_weight', 10.0)
        self.criterion = MultiTaskLoss(
            alpha=weights.get('alpha', 1.0),
            beta=weights.get('beta', 2.0),
            gamma=weights.get('gamma', 0.3),
            pos_weight=pos_weights,
            forgery_pos_weight=forgery_pw
        ).to(device)
        
        # We will configure optimizer learning rates per phase dynamically
        # Start with Phase 1 LR
        self.lr = config['training']['phase1']['learning_rate']
        
        # Logging config (required for scheduler & best metric)
        self.log_cfg = config.get("training", {}).get("logging", {})
        self.save_best_by = self.log_cfg.get("save_best_by", "forgery_auc")
        self.best_metric = -float('inf') if "auc" in self.save_best_by else float('inf')
        self.save_dir = config['paths'].get('checkpoints', 'checkpoints/')
        os.makedirs(self.save_dir, exist_ok=True)
        
        opt_name = config['training'].get('optimizer', {}).get('name', 'adamw').lower()
        wd = config['training'].get('optimizer', {}).get('weight_decay', 1e-4)
        if opt_name == 'adamw':
            self.optimizer = optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=wd)
        else:
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=wd)
            
        sched_cfg = config.get('training', {}).get('scheduler', {})
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode="max" if "auc" in self.save_best_by else "min",
            factor=sched_cfg.get('factor', 0.5),
            patience=sched_cfg.get('patience', 3),
            min_lr=float(sched_cfg.get('min_lr', 1e-7))
        )
            
        use_mixed_precision = (
            config['training'].get('mixed_precision', True)
            and str(device).startswith("cuda")
        )
        self.scaler = torch.amp.GradScaler('cuda') if use_mixed_precision else None
        self.use_autocast = use_mixed_precision
        
        # Early Stopping (Decoupled)
        self.es_disease = EarlyStopping(patience=10, min_delta=0.001)
        self.es_forgery = EarlyStopping(patience=5, min_delta=0.001)
        
        # Logging backends
        self.writer = None
        self.wandb = None

        logs_dir = config.get("paths", {}).get("logs", "outputs/logs")
        os.makedirs(logs_dir, exist_ok=True)

        if self.log_cfg.get("tensorboard", True):
            self.writer = SummaryWriter(log_dir=logs_dir)

        if self.log_cfg.get("wandb", False):
            try:
                import tempfile
                wandb_tmp = os.path.join(logs_dir, "wandb_tmp")
                os.makedirs(wandb_tmp, exist_ok=True)
                tempfile.tempdir = wandb_tmp
                os.environ["TMP"] = wandb_tmp
                os.environ["TEMP"] = wandb_tmp
                os.environ.setdefault("TMPDIR", wandb_tmp)

                import wandb  # type: ignore
                run_mode = os.environ.get("WANDB_MODE", "offline")
                wandb.init(
                    project=self.log_cfg.get("wandb_project", "medical-deepfake-detector"),
                    config=config,
                    mode=run_mode,
                    reinit=True,
                )
                self.wandb = wandb
            except Exception as exc:
                print(f"[Trainer] W&B init skipped: {exc}")

    def set_phase(self, epoch):
        c_train = self.config['training']
        p1_epochs = c_train['phase1']['epochs']
        p2_epochs = c_train['phase2']['epochs']
        p3_epochs = c_train['phase3']['epochs']
        
        if epoch <= p1_epochs:
            phase = 'phase1'
        elif epoch <= (p1_epochs + p2_epochs):
            phase = 'phase2'
        else:
            phase = 'phase3'
        self.current_phase = phase
            
        # Update backbone frozen status
        freeze_backbone = c_train[phase]['freeze_backbone']
        for param in self.model.backbone.parameters():
            param.requires_grad = not freeze_backbone
            
        # Update learning rate
        new_lr = c_train[phase]['learning_rate']
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = new_lr
            
        print(f"\n--- Epoch {epoch}: {phase.upper()} | LR: {new_lr} | Backbone Frozen: {freeze_backbone} ---")

    def _zero_metrics(self) -> Dict[str, float]:
        return {
            "total_loss": 0.0,
            "disease_loss": 0.0,
            "forgery_loss": 0.0,
            "localization_loss": 0.0,
        }

    def train_epoch(self):
        self.model.train()
        metrics_sum = self._zero_metrics()
        steps = 0
        
        pbar = tqdm(self.train_loader, desc="Training")
        for batch_idx, batch in enumerate(pbar):
            if self.max_train_batches is not None and batch_idx >= self.max_train_batches:
                break

            images = batch['image'].to(self.device)
            targets = {
                'disease_labels': batch['disease'].to(self.device),
                'forgery_labels': batch['forgery'].to(self.device),
                'localization_masks': batch['mask'].to(self.device)
            }
            
            # Difficulty Curriculum (Fix 5)
            intensities = batch['metadata']['manipulation_intensity'].to(self.device)
            if self.current_phase == 'phase1':
                # Obvious fakes: > 0.10, or Authentic (relaxed from 0.20)
                valid_mask = (intensities > 0.10) | (targets['forgery_labels'] == 0)
            elif self.current_phase == 'phase2':
                # Medium fakes: > 0.05, or Authentic (relaxed from 0.10)
                valid_mask = (intensities > 0.05) | (targets['forgery_labels'] == 0)
            else:
                # Phase 3: all fakes
                valid_mask = torch.ones_like(targets['forgery_labels'], dtype=torch.bool)
                
            if not valid_mask.any():
                continue
                
            images = images[valid_mask]
            targets['disease_labels'] = targets['disease_labels'][valid_mask]
            targets['forgery_labels'] = targets['forgery_labels'][valid_mask]
            targets['localization_masks'] = targets['localization_masks'][valid_mask]
            
            self.optimizer.zero_grad()
            
            if self.scaler:
                with torch.amp.autocast('cuda'):
                    preds = self.model(images)
                    loss_dict = self.criterion(preds, targets)
                    loss = loss_dict['total_loss']
                
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                preds = self.model(images)
                loss_dict = self.criterion(preds, targets)
                loss = loss_dict['total_loss']
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
                
            steps += 1
            for k in metrics_sum.keys():
                metrics_sum[k] += float(loss_dict[k].item())

            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'd': f"{loss_dict['disease_loss'].item():.4f}",
                'f': f"{loss_dict['forgery_loss'].item():.4f}",
                'l': f"{loss_dict['localization_loss'].item():.4f}",
            })
            
        if steps == 0:
            raise RuntimeError("No train batches were processed.")
        return {k: v / steps for k, v in metrics_sum.items()}

    @torch.no_grad()
    def validate_epoch(self):
        self.model.eval()
        metrics_sum = self._zero_metrics()
        steps = 0
        all_forgery_probs = []
        all_forgery_targets = []
        all_disease_probs = []
        all_disease_targets = []
        
        pbar = tqdm(self.val_loader, desc="Validation")
        for batch_idx, batch in enumerate(pbar):
            if self.max_val_batches is not None and batch_idx >= self.max_val_batches:
                break

            images = batch['image'].to(self.device)
            targets = {
                'disease_labels': batch['disease'].to(self.device),
                'forgery_labels': batch['forgery'].to(self.device),
                'localization_masks': batch['mask'].to(self.device)
            }
            
            if self.use_autocast:
                with torch.amp.autocast('cuda'):
                    preds = self.model(images)
                    loss_dict = self.criterion(preds, targets)
                    loss = loss_dict['total_loss']
            else:
                preds = self.model(images)
                loss_dict = self.criterion(preds, targets)
                loss = loss_dict['total_loss']
                
            steps += 1
            for k in metrics_sum.keys():
                metrics_sum[k] += float(loss_dict[k].item())

            probs = torch.sigmoid(preds['forgery_logits']).squeeze(-1).detach().cpu().numpy()
            targets_np = targets['forgery_labels'].detach().cpu().numpy()
            all_forgery_probs.append(probs)
            all_forgery_targets.append(targets_np)
            
            d_probs = torch.sigmoid(preds['disease_logits']).detach().cpu().numpy()
            d_targets_np = targets['disease_labels'].detach().cpu().numpy()
            all_disease_probs.append(d_probs)
            all_disease_targets.append(d_targets_np)

        if steps == 0:
            raise RuntimeError("No validation batches were processed.")
        metrics = {k: v / steps for k, v in metrics_sum.items()}

        try:
            y_prob = np.concatenate(all_forgery_probs, axis=0)
            y_true = np.concatenate(all_forgery_targets, axis=0)
            if len(np.unique(y_true)) > 1:
                metrics["forgery_auc"] = float(roc_auc_score(y_true, y_prob))
            else:
                metrics["forgery_auc"] = float("nan")
        except Exception:
            metrics["forgery_auc"] = float("nan")
            
        try:
            d_prob = np.concatenate(all_disease_probs, axis=0)
            d_true = np.concatenate(all_disease_targets, axis=0)
            metrics["disease_auc"] = float(roc_auc_score(d_true, d_prob, average="macro"))
        except Exception:
            metrics["disease_auc"] = float("nan")

        return metrics

    def _log_epoch(self, epoch, train_metrics, val_metrics):
        if self.writer is not None:
            for k, v in train_metrics.items():
                self.writer.add_scalar(f"train/{k}", v, epoch)
            for k, v in val_metrics.items():
                self.writer.add_scalar(f"val/{k}", v, epoch)

        if self.wandb is not None:
            payload = {"epoch": epoch}
            payload.update({f"train/{k}": v for k, v in train_metrics.items()})
            payload.update({f"val/{k}": v for k, v in val_metrics.items()})
            self.wandb.log(payload)

    def fit(self, start_epoch: int = 1, end_epoch: Optional[int] = None):
        from src.evaluation.metrics import plot_training_curves
        total_epochs = self.config['training']['total_epochs']
        if end_epoch is None:
            end_epoch = total_epochs
        history = []
        train_losses = []
        val_losses = []
        val_aucs = []
        for epoch in range(start_epoch, end_epoch + 1):
            self.set_phase(epoch)
            
            train_metrics = self.train_epoch()
            val_metrics = self.validate_epoch()
            self._log_epoch(epoch, train_metrics, val_metrics)
            
            print(
                f"Epoch {epoch}: "
                f"Train total={train_metrics['total_loss']:.4f} "
                f"(d={train_metrics['disease_loss']:.4f}, "
                f"f={train_metrics['forgery_loss']:.4f}, "
                f"l={train_metrics['localization_loss']:.4f}) | "
                f"Val total={val_metrics['total_loss']:.4f} "
                f"| Val forgery_auc={val_metrics.get('forgery_auc', float('nan')):.4f}"
            )
            
            if self.save_best_by in val_metrics and np.isfinite(val_metrics[self.save_best_by]):
                metric = float(val_metrics[self.save_best_by])
                is_better = metric > self.best_metric
            else:
                metric = float(val_metrics['total_loss'])
                is_better = metric < self.best_metric

            train_losses.append(train_metrics['total_loss'])
            val_losses.append(val_metrics['total_loss'])
            val_aucs.append(val_metrics.get('forgery_auc', 0.0))

            self.scheduler.step(metric)

            if is_better:
                self.best_metric = metric
                if self.save_best_by == "forgery_auc":
                    save_name = "best_forgery_auc.pt"
                else:
                    save_name = f"best_{self.save_best_by}.pt"
                save_path = os.path.join(self.save_dir, save_name)
                torch.save(self.model.state_dict(), save_path)
                print(f"Saved new best model to {save_path}")

            d_auc = float(val_metrics.get('disease_auc', 0.0))
            f_auc = float(val_metrics.get('forgery_auc', 0.0))
            stop_d = self.es_disease.step(d_auc)
            stop_f = self.es_forgery.step(f_auc)
            
            if stop_d and stop_f:
                print("[Trainer] Both Early Stoppings triggered. Halting.")
                break

            history.append({
                "epoch": epoch,
                "train": train_metrics,
                "val": val_metrics,
            })

        if self.writer is not None:
            self.writer.flush()
            self.writer.close()
        if self.wandb is not None:
            try:
                self.wandb.finish()
            except Exception:
                pass
                
        # Save training curves plot
        reports_dir = self.config['paths'].get('reports', 'outputs/reports/')
        plot_training_curves(train_losses, val_losses, val_aucs, save_path=os.path.join(reports_dir, 'training_curves.png'))

        return history
