"""
2411252359
cr yasaisen
next add config to init
"""


import torch
from tqdm import tqdm
import torch.nn as nn
from datetime import datetime
import json
import os


class RewardModelTask():
    def __init__(self, model, train_loader, val_loader, cfg):
        self.cfg = cfg
        self.num_epochs = int(cfg['task'].get("num_epochs"))
        self.device = cfg['task'].get("device")
        self.max_patience = int(cfg['task'].get("max_patience"))
        self.lr = float(cfg['task'].get("lr"))
        self.max_lr = float(cfg['task'].get("max_lr"))
        self.pct_start = float(cfg['task'].get("pct_start"))
        self.weight_decay = float(cfg['task'].get("weight_decay"))
        self.output_dir = cfg['task'].get("output_dir")

        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.nowtime = datetime.now().strftime("%y%m%d%H%M")

        self.save_path = os.path.join(self.output_dir, self.nowtime)
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

        self.optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=self.lr,
            weight_decay=self.weight_decay
        )
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=self.max_lr,
            epochs=self.num_epochs,
            steps_per_epoch=len(train_loader),
            pct_start=self.pct_start ,
            anneal_strategy='cos'
        )
        self.criterion = nn.BCEWithLogitsLoss()

        self.best_val_accuracy = 0.0
        self.best_epoch = 0
        self.patience = 0
        self.train_loss_list = []
        self.train_accuracy_list = []
        self.val_loss_list = []
        self.val_accuracy_list = []

        self.model = self.model.to(self.device)

    def train_model(self):
        for epoch in range(self.num_epochs):
            self.model.train()
            running_loss = 0.0
            correct_predictions = 0
            total_predictions = 0
            
            pbar = tqdm(self.train_loader, desc=f'Epoch {epoch+1}/{self.num_epochs} - Training')
            for batch in pbar:
                context_ids = batch['context_ids'].to(self.device)
                context_mask = batch['context_mask'].to(self.device)
                better_ids = batch['better_ids'].to(self.device)
                better_mask = batch['better_mask'].to(self.device)
                worse_ids = batch['worse_ids'].to(self.device)
                worse_mask = batch['worse_mask'].to(self.device)
                
                self.optimizer.zero_grad()
                
                reward_better, reward_worse = self.model(
                    context_ids, context_mask,
                    better_ids, better_mask,
                    worse_ids, worse_mask
                )
                
                reward_diff = reward_better - reward_worse
                labels = torch.ones_like(reward_diff)
                loss = self.criterion(reward_diff, labels)
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
                self.scheduler.step()
                
                running_loss += loss.item() * context_ids.size(0)
                

                predictions = (reward_diff > 0).float()
                correct_predictions += (predictions == labels).sum().item()
                total_predictions += labels.size(0)
            
            train_loss = running_loss / len(self.train_loader.dataset)
            train_accuracy = correct_predictions / total_predictions
            self.train_loss_list.append(train_loss)
            self.train_accuracy_list.append(train_accuracy)
            
            self.model.eval()
            val_running_loss = 0.0
            val_correct_predictions = 0
            val_total_predictions = 0
            
            with torch.no_grad():
                pbar = tqdm(self.val_loader, desc=f'Epoch {epoch+1}/{self.num_epochs} - Validation')
                for batch in pbar:
                    context_ids = batch['context_ids'].to(self.device)
                    context_mask = batch['context_mask'].to(self.device)
                    better_ids = batch['better_ids'].to(self.device)
                    better_mask = batch['better_mask'].to(self.device)
                    worse_ids = batch['worse_ids'].to(self.device)
                    worse_mask = batch['worse_mask'].to(self.device)
                    
                    reward_better, reward_worse = self.model(
                        context_ids, context_mask,
                        better_ids, better_mask,
                        worse_ids, worse_mask
                    )
                    
                    reward_diff = reward_better - reward_worse
                    labels = torch.ones_like(reward_diff)
                    loss = self.criterion(reward_diff, labels)
                    
                    val_running_loss += loss.item() * context_ids.size(0)
                    
                    predictions = (reward_diff > 0).float()
                    val_correct_predictions += (predictions == labels).sum().item()
                    val_total_predictions += labels.size(0)
            
            val_loss = val_running_loss / len(self.val_loader.dataset)
            val_accuracy = val_correct_predictions / val_total_predictions
            self.val_loss_list.append(val_loss)
            self.val_accuracy_list.append(val_accuracy)
            
            current_lr = self.scheduler.get_last_lr()[0]
            
            print(f'Epoch [{epoch+1}/{self.num_epochs}]')
            print(f'Train Loss: {train_loss:.5f}, Train Accuracy: {train_accuracy:.5f}')
            print(f'Val Loss: {val_loss:.5f}, Val Accuracy: {val_accuracy:.5f}')
            print(f'Best Val Accuracy: {self.best_val_accuracy:.5f} (Epoch {self.best_epoch+1})')
            print(f'Learning Rate: {current_lr:.6f}')
            
            if val_accuracy >= self.best_val_accuracy:
                self.best_val_accuracy = val_accuracy
                self.best_epoch = epoch
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'best_val_accuracy': self.best_val_accuracy,
                }, os.path.join(self.save_path, 'best_model_' + self.nowtime + '.pt'))
                self.patience = 0
            else:
                self.patience += 1
                
            if self.patience >= self.max_patience:
                print(f'Early stopping triggered after epoch {epoch+1}')
                break
        
        result_dict = {
            'config': self.cfg,
            'train_loss_list': self.train_loss_list, 
            'train_accuracy_list': self.train_accuracy_list, 
            'val_loss_list': self.val_loss_list, 
            'val_accuracy_list': self.val_accuracy_list
            }

        with open(os.path.join(self.save_path, 'result_dict_' + self.nowtime + '.json'), 'w', encoding='utf-8') as file:
            json.dump(result_dict, file, ensure_ascii=False, indent=4)

        # return self.model, self.train_loss_list, self.train_accuracy_list, self.val_loss_list, self.val_accuracy_list