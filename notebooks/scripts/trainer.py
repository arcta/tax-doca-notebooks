#!/usr/bin/env python
# coding: utf-8

#------------------------------------------------
# utilities collection serving notebooks
#------------------------------------------------

import os
import re
import json
import torch
import numpy as np
import pandas as pd

from PIL import Image, ImageOps
from matplotlib import pyplot as plt
#from matplotlib import patches
from pathlib import Path
from typing import Callable
from einops import rearrange, repeat

from torch import nn, Tensor
from torchvision import transforms, models
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from torch.optim import SGD, AdamW
from torch.cuda.amp import GradScaler
from torchmetrics import Dice, JaccardIndex
from torchsummary import summary
from datetime import datetime
from time import time


# notebooks path
ROOT = f'{os.environ["HOME"]}/notebooks'
SAVE_PATH = f'{ROOT}/models/checkpoint'

#torch._dynamo.config.verbose = True
#torch.autograd.set_detect_anomaly(True)

#plt.style.use('ggplot')

torch.cuda.empty_cache()
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'



class DiceLoss(nn.Module):
    def __init__(self, num_classes: int):
        super(DiceLoss, self).__init__()
        self.num_classes = num_classes

    def forward(self, X, Y):
        # unsqueeze classes
        Y = nn.functional.one_hot(Y, self.num_classes)
        # align axes
        X = rearrange(X, 'b c h w -> b h w c')
        # compute class weight
        W = torch.zeros((self.num_classes,))
        W = 1. / (torch.sum(Y, (0, 1, 2)) ** 2 + 1e-9)        
        # compute weighted cross and union sums over b h w
        cross = X * Y
        cross = W * torch.sum(cross, (0, 1, 2))
        cross = torch.sum(cross)
        union = Y + X
        union = W * torch.sum(union, (0, 1, 2))
        union = torch.sum(union)
        return 1. - 2. * (cross + 1e-9)/(union + 1e-9)
    
    
class FocalLoss(nn.Module):
    def __init__(self, num_classes: int, alpha: float = 1., gamma: float = 2.):
        super(FocalLoss, self).__init__()
        self.num_classes = num_classes
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, X, Y):
        # unsqueeze classes
        Y = nn.functional.one_hot(Y, self.num_classes)
        # align axes
        X = rearrange(X, 'b c h w -> b h w c')
        # flatten all but batch
        X = torch.flatten(X, start_dim=1)
        Y = torch.flatten(Y, start_dim=1)
        # prediction
        p = torch.sigmoid(X)
        p = torch.where(Y > 0, self.alpha * p, 1 - p)
        # loss
        pt = -torch.log(torch.clamp(p, 1e-4, 1 - 1e-4))
        loss = pt * ((1 - p) ** self.gamma)
        return self.num_classes * loss.mean()
    
    
class HydraLoss(nn.Module):
    """
    Construct combined loss with trainable weights:
    https://arxiv.org/abs/1705.07115
    """
    def __init__(self, criteria: list):
        super().__init__()
        self.criteria = criteria
        self.log_vars = nn.Parameter(torch.zeros((len(criteria))))

    def forward(self, preds, targets):
        losses = []
        for i, criterion in enumerate(self.criteria):
            loss = criterion(preds[i], targets[i])
            losses.append(torch.exp(-self.log_vars[i]) * loss + self.log_vars[i])
        return torch.sum(torch.stack(losses))


class Trainer:
    def __init__(self, model: nn.Module, dataset: Dataset, view_size: int, 
                       criterion: nn.Module, optimizer: nn.Module, metrics: dict,
                       multi_x: bool = False, multi_y: bool = False, tag: str = '',
                       window: int = 10, tensorboard_dir: str = None, autocast: bool = False):
        
        self.model = model
        self.dataset = dataset
        self.view_size = view_size
        self.criterion = criterion
        self.optimizer = optimizer
        self.metrics = metrics
        self.multi_x = multi_x
        self.multi_y = multi_y
        self.tag = tag
        self.window = window
        self.autocast = autocast
        self.scaler = GradScaler()
        self.writer = None if tensorboard_dir is None else SummaryWriter(tensorboard_dir)
        self.checkpoints = []
        self.track = 0
        self.loss_history = { 'training':[], 'validation':[] }
        self.metrics_history = { task:{ m:[] for m in metrics[task] } for task in metrics } if self.multi_y else \
                               { m:[] for m in metrics }
        
    def training_step(self, inputs, target):            
        # pass forward
        with torch.cuda.amp.autocast(enabled=self.autocast):
            preds = self.model(*inputs) if self.multi_x else self.model(inputs)
            loss = self.criterion(preds, target)
        # backp-prop
        self.optimizer.zero_grad()
        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()
        return float(loss.item())
    
    def train(self, samples: list, batch_size: int):
        self.model.train()
        history = []
        for i, source in enumerate(samples):
            # create loader for this sample random views
            loader = DataLoader(self.dataset(source, self.view_size, max_samples=batch_size),
                                batch_size=batch_size, shuffle=False)
            for b, (X, Y) in enumerate(loader):
                inputs = [x.to(DEVICE) for x in X] if self.multi_x else X.to(DEVICE)
                target = [y.to(DEVICE) for y in Y] if self.multi_y else Y.to(DEVICE)
                history.append(self.training_step(inputs, target))
                if self.writer:
                    self.writer.add_scalar('training loss', history[-1], self.track + i * len(loader) + b)
            print(f'Training:   {(i + 1)/len(samples):.2%}   loss: {sum(history)/len(history):.4f}          ', end='\r')
        print('')
        # releasing gpu memory
        torch.cuda.empty_cache()
        # loss history
        return history

    def validate(self, samples: list, batch_size: int):
        self.model.eval()
        metrics = self.metrics
        results = { task:{ m:[] for m in metrics[task] } for task in metrics } if self.multi_y else \
                  { m:[] for m in metrics }
        history = []
        with torch.no_grad():
            for i, source in enumerate(samples):
                loader = DataLoader(self.dataset(source, self.view_size, max_samples=batch_size),
                                    batch_size=batch_size, shuffle=False)
                for b, (X, Y) in enumerate(loader):
                    inputs = [x.to(DEVICE) for x in X] if self.multi_x else X.to(DEVICE)
                    target = [y.to(DEVICE) for y in Y] if self.multi_y else Y.to(DEVICE)
                    # get predictions and calculate loss
                    preds = self.model(*inputs) if self.multi_x else self.model(inputs)
                    loss = self.criterion(preds, target)
                    history.append(float(loss.item()))
                    if self.writer:
                        self.writer.add_scalar('validation loss', loss, self.track + i * len(loader) + b)
                    # compute metrics
                    if self.multi_y:
                        for k, task in enumerate(metrics):
                            for metric, calc in metrics[task].items():
                                results[task][metric].append(calc(preds[k], target[k]).detach().cpu().numpy().astype(float))
                    else:
                        for metric, calc in metrics.items():
                            results[metric].append(calc(preds, target).detach().cpu().numpy().astype(float))
                print(f'Validation: {(i + 1)/len(samples):.2%}   loss: {sum(history)/len(history):.4f}          ', end='\r')
            print('')
        # loss and metrics history
        return results, history

    
    def run(self, train_samples: list, test_samples: list,
                  batch_size: int, num_epochs: int = 1, validation_steps: int = 1, offset: int = 0,
                  silent: bool = False):
        
        n = len(train_samples)//validation_steps + 1
        for epoch in range(offset, offset + num_epochs):
            np.random.shuffle(train_samples)
            if not silent: print(f'\t\tEpoch: {epoch + 1}\n'
                                  '====================================================')
            for i in range(validation_steps):
                self.track = i * n
                train_loss = self.train(train_samples[i * n:(i + 1) * n], batch_size)
                self.loss_history['training'] += train_loss
                txt = f' {i + 1}/{validation_steps}' if validation_steps > 1 else '----'
                print(f'------------------------------------------------{txt}')
                results, validation_loss = self.validate(test_samples, batch_size)
                self.loss_history['validation'] += validation_loss
                if self.multi_y:
                    for k, task in enumerate(self.metrics):
                        for metric, calc in self.metrics[task].items():
                            if metric == 'confmat':
                                results[task][metric] = np.sum(np.array(results[task][metric]), axis=0).astype(float).tolist()
                                self.metrics_history[task][metric] = results[task][metric]
                            else:
                                mean, std = np.mean(results[task][metric]), np.std(results[task][metric])
                                self.metrics_history[task][metric].append((mean, std))
                                results[task][metric] = np.round(mean, 4).astype(float)
                else:
                    for metric in self.metrics:
                        if metric == 'confmat':
                            results[metric] = np.sum(np.array(results[metric]), axis=0).astype(float).tolist()
                            self.metrics_history[metric] = results[metric]
                        else:
                            mean, std = np.mean(results[metric]), np.std(results[metric])
                            self.metrics_history[metric].append((mean, std))
                            results[metric] = np.round(mean, 4).astype(float)
                # save progress
                key = f'{datetime.now().isoformat()}-{self.tag}-{epoch + 1}:{self.track + n}'
                torch.save(self.model.state_dict(), f'{SAVE_PATH}/{key}.pt')
                with open(f'{SAVE_PATH}/{key}.json','w') as output:
                    json.dump({'loss':self.loss_history, 'metrics':self.metrics_history, 'results':results}, output)
                self.checkpoints.append(f'{SAVE_PATH}/{key}.pt')
                if not silent: print('====================================================')
        # averaged final test
        return results
    
    
    def plot_history():
        #plt.plot(x, y, 'k-')
        #plt.fill_between(x, y-error, y+error)
        pass    
    

def plot_history(loss_history, metrics_history, multi_x: bool = False, multi_y: bool = False,
                 offset: int = 0, markers: str = 'os^dp*v><h'):
    """
    show actual and running average for loss and metrics history
    """
    training_loss = loss_history['training'][offset:]
    validation_loss = loss_history['validation']
    results = metrics_history
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    # plot actual loss history
    ax[0].plot(training_loss, color='teal', alpha=0.2)
    ax[0].plot(np.linspace(0, len(training_loss), len(validation_loss)), validation_loss, color='magenta', alpha=0.2)
    # plot loss rolling mean
    series, window = pd.Series(training_loss), len(training_loss)//10
    y = series.rolling(window=window).mean().iloc[window:]
    ax[0].plot(series.rolling(window=window).mean().iloc[window:], color='teal', label='training')
    ax[0].axhline(y=np.mean(y.iloc[-len(y)//4:]), linestyle=':', color='teal')
    offset = window
    series, window = pd.Series(validation_loss), len(validation_loss)//10
    y = series.rolling(window=window).mean().iloc[window:]
    x = np.linspace(offset, len(training_loss), len(validation_loss) - window)
    ax[0].plot(x, y, color='magenta', label='validation')
    ax[0].axhline(y=np.mean(y.iloc[-len(y)//4:]), linestyle=':', color='magenta')
    ax[0].legend(frameon=False)
    ax[0].set_title('Loss history')
    if len(results) == 0:
        ax[1].axis('off')
        plt.show()
        return
    # plot validation metrics history
    if multi_y:
        c = 0
        for task in results:
            for i, metric in enumerate([r for r in results[task]]):
                if metric == 'confmat': continue
                y = results[task][metric]
                n = len(training_loss)//len(y)
                x = [i * n for i in range(1, len(y) + 1)]
                ax[1].fill_between(x, [m - s for m, s in y], [m + s for m, s in y], color=f'C{c}', alpha=0.1)
                ax[1].plot(x, [m for m, s in y], color=f'C{c}', marker=markers[c], label=f'{task} {metric}')
                ax[1].axhline(y=y[-1][0], linestyle=':', color=f'C{c}')
                c += 1
    else: # single task metrics
        for i, metric in enumerate([r for r in results]):
            if metric == 'confmat': continue
            y = results[metric]
            n = len(training_loss)//len(y)
            x = [i * n for i in range(1, len(y) + 1)]
            ax[1].fill_between(x, [m - s for m, s in y], [m + s for m, s in y], color=f'C{i}', alpha=0.1)
            ax[1].plot(x, [m for m, s in y], color=f'C{i}', marker=markers[i], label=metric)
            ax[1].axhline(y=y[-1][0], linestyle=':', color=f'C{i}')
    ax[1].legend(bbox_to_anchor=(1, 1), frameon=False)
    ax[1].set_title('Evaluation history')
    plt.show()
    
    
def plot_confmat(matrix, class_labels: list = None, ticks: list = [], title: str = 'Confusion Matrix', size: int = 4):
    d = len(matrix[0])
    assert class_labels is None or d == len(class_labels)
    fig, ax = plt.subplots(figsize=(size, size))
    # sum over batches
    ax.imshow(matrix/np.max(matrix), cmap='coolwarm')
    total = np.sum(matrix)
    if class_labels is None:
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)
    else:
        for i in range(d):
            for j in range(d):
                if i == j or matrix[i,j]/total >= 0.01:
                    ax.text(j, i, f'{matrix[i,j]/total:.0%}',
                            ha='center', va='center', color='black', fontsize=8)
        ax.set_xticks(range(d))
        ax.set_xticklabels(class_labels)
        ax.set_yticks(range(d))
        ax.set_yticklabels(class_labels)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    plt.show()

    
def show_model_output(model, batch, index: int = None, size: tuple = (11, 11)):
    model.eval()
    with torch.no_grad():
        c = batch[0][0].shape[0]
        fig, ax = plt.subplots(1, c, figsize=size)
        for X, Y in batch:
            P = model(X.to(DEVICE))
            P = P[index or 0].cpu() if type(P) == tuple else P.cpu()
            for i in range(c):
                ax[i].imshow(P[i,:].squeeze().numpy(), 'gray')
                ax[i].axis('off')
        index = '' if index is None else f' [{index}]'
        ax[0].set_title(f'Model{index}:  {P.shape[1:]}', fontsize=10, ha='left', x=0)
        plt.show()
    
    
class MultiTrainer:
    """
    to compare architectures or for ablation studies --
    train several models in paralell using the same data
    """
    def __init__(self, dataset: Dataset, models: list, view_size: int,
                       criterions: list, optimizers: list, metrics: list,
                       multi_x: bool = False, multi_y: bool = False,
                       tags: list = None):
        
        self.multi_x = multi_x
        self.multi_y = multi_y
        self.results = []
        self.tags = tags or list(range(1, len(models) + 1))
        self.trainers = [Trainer(models[i], dataset, view_size, criterions[i], optimizers[i], metrics[i],
                                 multi_x=multi_x, multi_y=multi_y, tag=str(self.tags[i])) for i in range(len(models))]
        

    def plot_history(self, offset: int = 0):
        for k in range(len(self.trainers)):
            print(f'\n-----------------------------------------------------------------------------------------------------')
            print(f'  Model-{self.tags[k]}', end='  ')
            if self.multi_y:
                for task, result in self.results[k].items():
                    outcome = '  '.join([f'{m}: {v:.4f}' for m, v in result.items() if m != 'confmat'])
                    print(f'\n  {task}: {outcome}', end='  ')
                print('')
            else:
                print('  '.join([f'{m}: {v:.4f}' for m, v in self.results[k].items() if m != 'confmat']))
            plot_history(self.loss_history[k], self.metrics_history[k], self.multi_x, self.multi_y)
            
            
    def plot_compare(self):
        fig, ax = plt.subplots(1, 2, figsize=(10, 4))
        for i, x in enumerate(['training','validation']):
            for k, tag in enumerate(self.tags):
                series, window = pd.Series(self.loss_history[k][x]), len(self.loss_history[k][x])//10
                history = series.rolling(window=window).mean().iloc[window:]
                linestyle = 'solid' if k < 10 else 'dashdot'
                label = tag if tag != '' else 'base'
                ax[i].plot(history, linestyle=linestyle, label=label)
            ax[i].set_title(f'Comparative {x} loss history', fontsize=10)
            ax[i].set_xticks([])
        ax[1].legend(bbox_to_anchor=(1, 1), frameon=False)
        plt.show()
            
        
    def run(self, train_samples: list, test_samples: list,
                  batch_size: int, num_epochs: int = 1, validation_steps: int = 4, offset: int = 0):
        
        self.loss_history = [{ 'training':[], 'validation':[] } for _ in range(len(self.trainers))]
        self.metrics_history = [{ task:{ m:[] for m in trainer.metrics[task] } for task in trainer.metrics }
                                  for trainer in self.trainers] if self.multi_y else \
                               [{ m:[] for m in trainer.metrics } for trainer in self.trainers]
        
        n = len(train_samples)//validation_steps + 1
        for epoch in range(offset, offset + num_epochs):
            np.random.shuffle(train_samples)
            print(f'\tEpoch: {epoch + 1}')
            for i in range(validation_steps):
                results = []
                print(f'================================================ {i + 1}/{validation_steps}')
                for k, trainer in enumerate(self.trainers):
                    results.append(trainer.run(train_samples[i * n:(i + 1) * n], test_samples, batch_size,
                                               offset=epoch, silent=True))
                    for key in trainer.loss_history:
                        self.loss_history[k][key] = trainer.loss_history[key]
                    
                    for key in trainer.metrics_history:
                        if type(trainer.metrics_history[key]) == dict:
                            for m in trainer.metrics_history[key]:
                                self.metrics_history[k][key][m] = trainer.metrics_history[key][m]
                        else:
                            self.metrics_history[k][key] = trainer.metrics_history[key]
                    
                    if k < len(self.trainers) - 1:
                        print(f'------------------------------------------ '
                              f'{self.tags[k + 1]:<5} {k + 2}/{len(self.trainers)}')
            self.results = results
            print(f'====================================================')    
        # list of final tests
        return results


    def save(self, path):
        with open(f'{path}.json','w') as output:
            json.dump({'loss':self.loss_history, 'metrics':self.metrics_history, 'tags':self.tags}, output)
        
    