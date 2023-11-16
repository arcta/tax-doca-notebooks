#!/usr/bin/env python
# coding: utf-8

import os
import re
import cv2
import torch
import pandas as pd
import numpy as np
import pytesseract as pts
import matplotlib as mpl

from PIL import Image, ImageOps
from matplotlib import pyplot as plt
from matplotlib import patches
from pathlib import Path
from IPython.display import display, clear_output

from torch import nn, Tensor
from torchvision import transforms, models
from torch.utils.data import DataLoader, Dataset
from torch.optim import SGD, AdamW
from torch.cuda.amp import GradScaler
from torchmetrics import F1Score, Precision, Recall, ConfusionMatrix, Accuracy, AUROC
from torchsummary import summary
from fitz import fitz
from tqdm import tqdm
from time import time

from .prep import detect_skew, get_bg_value, img_rotate, img_load, fit_straight
from .parse import extract_lines
from .simulate import generate_sample


# notebooks path
ROOT = f'{os.environ["HOME"]}/notebooks'

#torch._dynamo.config.verbose = True
torch.cuda.empty_cache()
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

LATENT_DIM = 512


class GrayResNet(models.resnet.ResNet):
    def __init__(self, block, layers, num_classes):
        self.inplanes = 64
        super(GrayResNet, self).__init__(block, layers)
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)


class ClassificationHead(nn.Sequential):
    def __init__(self, emb_size, num_classes, activation=None):
        layers = [nn.LayerNorm(emb_size),
                  nn.ReLU(),
                  nn.Linear(emb_size, num_classes)]
        if activation:
            layers.append(activation)
        super(ClassificationHead, self).__init__(*layers)


class LayoutBaseline(nn.Sequential):
    def __init__(self, encoder, head):
        super(LayoutBaseline, self).__init__(encoder, head)


def normalize(image: np.array) -> np.array:
    output = cv2.GaussianBlur(image, (5, 5), 1)
    return 1 - cv2.normalize(output, None, 0, 1, cv2.NORM_MINMAX)


def generate_data(samples, image_size: int, save_path: str, num_variations: int = 1,
                  light: float = 0.3, noise: float = 0.3):
    
    save_path = save_path.strip('/')
    for filename in os.listdir(save_path):
        os.unlink(f'{save_path}/{filename}')

    labels, skipped = [], 0
    print('---------------------------------------------------------------------------------------------------')
    for i, source in enumerate(samples):
        source = source.split('/').pop()[:-4] # file name without extension
        # textual content
        columns = ['left','top','right','bottom','block-type','text']
        try:
            content = pd.read_csv(f'./data/info/{source}.csv.gz')[columns]
        except FileNotFoundError:
            continue
        content = len(content[(content['block-type'] == 'word')&(content['text'] != '.')])
        # extract outlines if any (to make it useful dpi should be 150 minimum)
        outlines = extract_lines(img_load(f'./data/images/{source}.png'), units=10)
        outlines = np.mean(outlines[100:-100,100:-100])
        
        for n in range(num_variations):
            # hash in case we want to merge
            suffix = np.random.randint(1000, 9999)
            # generate a low-resolution noisy view of a filled-in form along with data for the labels
            image, info, inputs = generate_sample(source, dpi=150, light=light, noise=noise)
            # correct skew and distortion
            angle = detect_skew(image, max_angle=45)
            if abs(angle) > 10: # default setting is in range [-9, 9]
                # detection failure: doesn't suit for training
                skipped += 1
                continue
            image = img_rotate(image, angle, fill=get_bg_value(image))
            image = fit_straight(image)
            # scale and invert
            size = tuple((np.array(image.shape) * image_size/min(image.shape)).astype(int))[::-1]
            image = cv2.bitwise_not(cv2.resize(image, size, interpolation=cv2.INTER_AREA))
            path = f'{save_path}/{source}-{n}{suffix}.png'
            cv2.imwrite(path, image)
            labels.append({
                'source': source,
                'path': path,
                'content': content,
                'outlines': outlines,
                'inputs': len(inputs),
                'orientation': info['orient'],
                'skew': info['skew'],
                'corrected': angle,
            })
            print(f'generating data: {(i * num_variations + n)/len(samples)/num_variations:.2%}', end='\r')
            pd.DataFrame.from_dict(labels).to_csv(f'{save_path}/labels.csv.gz', index=False, compression='gzip')
    pd.DataFrame.from_dict(labels).to_csv(f'{save_path}/labels.csv.gz', index=False, compression='gzip')
    print(f'generated data: skipped {skipped/len(samples):.2%} samples due to correction failure')
    print('---------------------------------------------------------------------------------------------------')


def plot_history(loss_history, metrics_history, multi_x: bool = False, multi_y: bool = False):
    """
    show actual and running average for loss and metrics history
    """
    training_loss = loss_history['training']
    validation_loss = loss_history['validation']
    results = metrics_history
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    # plot actual loss history
    ax[0].plot(training_loss, color='teal', alpha=0.3)
    ax[0].plot(np.linspace(0, len(training_loss), len(validation_loss)), validation_loss, color='magenta', alpha=0.3)
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
    # plot validation metrics history
    if multi_y:
        c = 0
        for task in results:
            for i, metric in enumerate([r for r in results[task]]):
                if metric == 'confmat': continue
                y = results[task][metric]
                n = len(training_loss)//len(y)
                x = [i * n for i in range(1, len(y) + 1)]
                ax[1].fill_between(x, [m - s for m, s in y], [m + s for m, s in y], color=f'C{c}', alpha=0.2)
                ax[1].plot(x, [m for m, s in y], color=f'C{c}', marker='s', label=f'{task} {metric}')
                ax[1].axhline(y=y[-1][0], linestyle=':', color=f'C{c}')
                c += 1
    else: # single task metrics
        for i, metric in enumerate([r for r in results]):
            if metric == 'confmat': continue
            y = results[metric]
            n = len(training_loss)//len(y)
            x = [i * n for i in range(1, len(y) + 1)]
            ax[1].fill_between(x, [m - s for m, s in y], [m + s for m, s in y], color=f'C{i}', alpha=0.2)
            ax[1].plot(x, [m for m, s in y], color=f'C{i}', marker='s', label=metric)
            ax[1].axhline(y=y[-1][0], linestyle=':', color=f'C{i}')
    ax[1].legend(bbox_to_anchor=(1, 1), frameon=False)
    ax[1].set_title('Evaluation history')
    plt.show()
    
    
def plot_confmat(matrix, class_labels: list, title: str = 'Confusion Matrix', size: int = 4):
    d = len(class_labels)
    assert d == len(matrix[0])
    fig, ax = plt.subplots(figsize=(size, size))
    # sum over batches
    ax.imshow(matrix/np.max(matrix), cmap='cool')
    total = np.sum(matrix)
    for i in range(d):
        for j in range(d):
            if i == j or matrix[i,j]/total >= 0.01:
                ax.text(i, j, f'{matrix[i,j]/total:.0%}',
                        ha='center', va='center', color='black', fontsize=8)
    ax.set_xticks(range(d))
    ax.set_xticklabels(class_labels)
    ax.set_yticks(range(d))
    ax.set_yticklabels(class_labels)
    ax.set_title(title, fontsize=10)
    plt.show()
    
    
class Trainer:
    def __init__(self, model: nn.Module, criterion: nn.Module, optimizer: nn.Module, metrics: dict,
                       multi_x: bool = False, multi_y: bool = False):
        
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.metrics = metrics
        self.multi_x = multi_x
        self.multi_y = multi_y
        self.scaler = GradScaler()
        
    def training_step(self, inputs, target):            
        # pass forward
        with torch.cuda.amp.autocast():
            preds = self.model(*inputs) if self.multi_x else self.model(inputs)
            loss = self.criterion(preds, target)
        # backp-prop
        self.optimizer.zero_grad()
        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()
        return loss.item()
    
    def train(self, loader):
        self.model.train()
        history, step = [], 1
        for X, Y in loader:
            inputs = [x.to(DEVICE) for x in X] if self.multi_x else X.to(DEVICE)
            target = [y.to(DEVICE) for y in Y] if self.multi_y else Y.to(DEVICE)
            history.append(self.training_step(inputs, target))
            print(f'Training:   {step/len(loader):.2%}:  loss: {sum(history)/len(history):.4f}', end='\r')
            step += 1
        print('')
        # releasing gpu memory
        torch.cuda.empty_cache()
        # loss history
        return history

    def validate(self, loader):
        self.model.eval()
        metrics = self.metrics
        results = { task:{ m:[] for m in metrics[task] } for task in metrics } if self.multi_y else \
                  { m:[] for m in metrics }
        history, step = [], 1
        with torch.no_grad():
            for X, Y in loader:
                inputs = [x.to(DEVICE) for x in X] if self.multi_x else X.to(DEVICE)
                target = [y.to(DEVICE) for y in Y] if self.multi_y else Y.to(DEVICE)
                # get predictions and calculate loss
                preds = self.model(*inputs) if self.multi_x else self.model(inputs)
                loss = self.criterion(preds, target)
                history.append(loss.item())
                # compute metrics
                if self.multi_y:
                    for k, task in enumerate(metrics):
                        for metric, calc in metrics[task].items():
                            results[task][metric].append(calc(preds[k], target[k]).detach().cpu().numpy())
                else:
                    for metric, calc in metrics.items():
                        results[metric].append(calc(preds, target).detach().cpu().numpy())
                print(f'Validation: {step/len(loader):.2%}:  loss: {sum(history)/len(history):.4f}', end='\r')
                step += 1
            print('')
        # loss and metrics history
        return results, history
        
    def run(self, train_dataloader: DataLoader, test_dataloader: DataLoader,
            num_epochs: int = 1, silent: bool = False):
        
        metrics = self.metrics
        self.loss_history = { 'training':[], 'validation':[] }
        self.metrics_history = { task:{ m:[] for m in metrics[task] } for task in metrics } if self.multi_y else \
                               { m:[] for m in metrics }
        
        for epoch in range(num_epochs):
            if not silent: print(f'\t\tEpoch: {epoch + 1}\n'
                                 '====================================================')
            train_loss = self.train(train_dataloader)
            self.loss_history['training'] += train_loss
            results, validation_loss = self.validate(test_dataloader)
            self.loss_history['validation'] += validation_loss
            if self.multi_y:
                for k, task in enumerate(metrics):
                    for metric, calc in metrics[task].items():
                        if metric == 'confmat':
                            self.metrics_history[task][metric] = results[task][metric]
                        else:
                            mean, std = np.mean(results[task][metric]), np.std(results[task][metric])
                            self.metrics_history[task][metric].append((mean, std))
                            results[task][metric] = np.round(mean, 4)
            else:
                for metric in metrics:
                    if metric == 'confmat':
                        self.metrics_history[metric] = results[metric]
                    else:
                        mean, std = np.mean(results[metric]), np.std(results[metric])
                        self.metrics_history[metric].append((mean, std))
                        results[metric] = np.round(mean, 4)
            if not silent: print(f'====================================================')
        # averaged final test
        return results
    
    