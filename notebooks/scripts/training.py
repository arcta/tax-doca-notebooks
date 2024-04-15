#!/usr/bin/env python
# coding: utf-8

#--------------------------------------------------------
# VAE contrast training https://github.com/arcta/ae-lego
#--------------------------------------------------------

import os
import re
import json
import torch
import numpy as np
import pandas as pd

from matplotlib import pyplot as plt
from typing import Optional, Callable

from torch import nn, Tensor
from torchvision import transforms, models
from torch.utils.tensorboard import SummaryWriter
from torch.optim import SGD, AdamW
from torch.cuda.amp import GradScaler
from torchsummary import summary

from .aelego import *
from .dataset import *

torch.cuda.empty_cache()
scaler = GradScaler()


def get_classifier(latent_dim, num_classes):
    hidden_dim = max(latent_dim, num_classes)
    return nn.Sequential(nn.Linear(latent_dim, hidden_dim),
                         nn.ReLU(),
                         nn.InstanceNorm1d(hidden_dim), #, affine=True), # small batch
                         nn.Linear(max(latent_dim, num_classes), num_classes))


class MultiVAE(nn.Module):
    """
    Use the same visual backbone encoder with multiple conjoined VAE models
    """
    tags = ['concept','target','context']
    
    def __init__(self, encoder: nn.Module, latent_dim: int,
                       num_classes: int = None, semantic: nn.Module = None,
                       align: bool = False, neighbors: bool = False):
        super().__init__()
        keys = []
        # models
        for tag in self.tags:
            mlp = None if num_classes is None else get_classifier(latent_dim, num_classes)
            vae = VAE(encoder, get_decoder(), latent_dim, mlp=mlp)
            setattr(self, tag, vae)
            keys += [f'{tag}-{key}' for key in vae.keys]                    
        # consensus classifier(s)
        self.semantic = semantic
        if not semantic is None:
            keys += ['semantic']
        # discriminator loop
        if align:
            keys += ['align']
        # context: try to classify neighbors
        if num_classes and neighbors:
            self.neighbors = nn.ModuleList([get_classifier(latent_dim, num_classes) for _ in ['left','right']])
            keys += ['left','right']
        else:
            self.neighbors = None              
        # list all outputs
        self.keys = keys
        
    def forward(self, X):
        outputs, z = [], []
        for i, tag in enumerate(self.tags):
            out = getattr(self, tag)(X[i])
            outputs += list(out)
            z.append(out[1])
        if 'semantic' in self.keys:
            outputs.append(self.semantic(outputs))
        if 'align' in self.keys:
            outputs.append(self.concept.decoder(z[self.tags.index('target')])[0])
        if self.neighbors is not None:
            outputs += [side(z[self.tags.index('context')]) for side in self.neighbors]
        return outputs
    
    
class MultitaskLoss(nn.Module):
    tags = ['concept','target','context'] # model-components
    order = ['label','semantic','left','right'] # labels
    
    def __init__(self, keys: list, config: dict,
                       label_align: nn.Module, semantic_align: nn.Module,
                       weight: dict, trainable: bool = False):
        super().__init__()
        self.keys = keys
        keys = ['rec','z','mean','log-var','tau','z-context','label'] # all VAE outputs
        for tag in self.tags:
            setattr(self, tag, VAELoss(keys, config[tag], align=label_align))
        self.semantic = semantic_align
        self.align = ReconstructionLoss()
        self.neighbor = label_align #nn.CrossEntropyLoss()
        # consensus classifiers weight
        weight = [weight[k] if k in weight else 0. for k in ['semantic','align','neighbors']]
        self.weight = nn.Parameter(torch.Tensor(weight)) if trainable else torch.Tensor(weight)
        self.trainable = trainable
        
    def forward(self, outputs, targets, labels):
        loss, metrics = [], {}
        # encoders-align
        for i, tag in enumerate(['concept','target','context']):
            r, l = self.keys.index(f'{tag}-rec'), self.keys.index(f'{tag}-label')
            tag_loss, tag_metrics = getattr(self, tag)(outputs[r:l + 1], targets[i], labels[0])
            metrics = {**metrics, **{f'{tag}-{k}':v for k,v in tag_metrics.items()}}
            loss.append(tag_loss)
        # consensus classifiers
        semantic_loss = self.semantic(outputs[self.keys.index('semantic')], labels[1])
        loss.append(semantic_loss * torch.exp(self.weight[0]))
        metrics['semantic'] = semantic_loss.item()
        # discriminator loop
        align_loss = self.align(outputs[self.keys.index('align')], targets[self.tags.index('concept')]).squeeze()
        loss.append(align_loss * torch.exp(self.weight[1]))
        metrics['align'] = align_loss.item()
        # context neighbors
        for i, side in enumerate(['left','right']):
            side_loss = self.neighbor(outputs[self.keys.index(side)], labels[self.order.index(side)])
            loss.append(side_loss * torch.exp(self.weight[2]))
            metrics[side] = side_loss.item()
        loss = torch.sum(torch.stack(loss))
        if self.trainable:
            # balance component-model vs. consensus plus regularization
            loss += torch.sum(self.weight.pow(4))
        return loss, metrics
    
    
class ContextClassifier(nn.Module):
    """
    Classify based on selective opinion 
    """
    def __init__(self, keys: list, latent_dim: int, semantic_dim: int):
        super().__init__()
        self.keys = keys
        self.semantic = get_classifier(latent_dim, semantic_dim)
        
    def forward(self, outputs):
        context = outputs[self.keys.index('context-z')]
        return self.semantic(context)    


class ReconstructionLoss(nn.Module):
    """
    value depends on batch size and image-size
    """
    def __init__(self, loss: nn.Module = nn.MSELoss(), weight: float = 0., trainable: bool = False):
        super().__init__()
        self.loss = loss
        weight = torch.Tensor([weight]).to(DEVICE)
        self.weight = nn.Parameter(weight) if trainable else weight
        
    def forward(self, x, y):
        return self.loss(x.squeeze(), y.squeeze()) * torch.exp(self.weight)
    
    
class KLDGaussianLoss(nn.Module):
    """
    assuming diagonal Gaussian prior and posterior
    """
    def __init__(self, reduction: str = 'mean', weight: float = 0., trainable: bool = False):
        super().__init__()
        self.reduction = reduction
        weight = torch.Tensor([weight]).to(DEVICE)
        self.weight = nn.Parameter(weight) if trainable else weight
        
    def forward(self, mean, logvar):
        loss = (mean.pow(2) + logvar.exp() - logvar - 1.) * 0.5
        if self.reduction == 'mean':
            return torch.sum(loss, axis=1).mean() * torch.exp(self.weight)
        return torch.sum(loss) * torch.exp(self.weight)
    
    
class KLDGumbelLoss(nn.Module):
    """
    uniform prior 1/#categories for all categories
    """
    def __init__(self, categorical_dim: int, reduction: str = 'mean', weight: float = 0., trainable: bool = False):
        super().__init__()
        self.dim = categorical_dim
        self.reduction = reduction
        weight = torch.Tensor([weight]).to(DEVICE)
        self.weight = nn.Parameter(weight) if trainable else weight
        
    def forward(self, proba):
        if self.reduction == 'mean':
            return torch.sum(proba * torch.log(proba * self.dim + 1e-11), dim=1).mean() * torch.exp(self.weight)
        return torch.sum(proba * torch.log(proba * self.dim + 1e-11)) * torch.exp(self.weight)
    
    
class ContrastLoss(nn.Module):
    """
    the input x is a batch of representation vectors --
    focus on difference: make covariance look more like identity
    """
    def __init__(self, weight: float = 0., trainable: bool = False):
        super().__init__()
        weight = torch.Tensor([weight]).to(DEVICE)
        self.weight = nn.Parameter(weight) if trainable else weight

    def forward(self, x):
        b, d = x.size()
        C = torch.abs(x @ x.T)
        return (1. - torch.trace(C)/torch.sum(C)) * np.log(d + b) * torch.exp(self.weight)
    
    
class AlignLoss(nn.Module):
    """
    cosine similarity between `perceived` and `hinted`
    """
    def __init__(self, weight: float = 0., trainable: bool = False):
        super().__init__()
        weight = torch.Tensor([weight]).to(DEVICE)
        self.weight = nn.Parameter(weight) if trainable else weight

    def forward(self, x, y):
        xnorm, ynorm = torch.sum(x ** 2, dim=1), torch.sum(y ** 2, dim=1)
        if torch.any(xnorm) and torch.any(ynorm):
            norm = (xnorm * ynorm) ** 0.5
            return 1. - torch.mean(torch.sum((torch.tanh(x) * y) ** 2, dim=1)/norm) * torch.exp(self.weight)
        return torch.Tensor([1.])
    
    
class MultiMLPLoss(nn.Module):
    """
    list of classifiers to train together
    """
    def __init__(self, losses=[], weights=[]):
        super().__init__()
        assert len(losses) == len(weights)
        self.loss = nn.ModuleList(losses)
        self.weight = torch.Tensor(weights)
        
    def forward(self, X, Y):
        loss = torch.Tensor([l(x, y) * w for x, y, l, w in zip(X, Y, self.loss, self.weight)]).to(DEVICE)
        return torch.sum(loss)
    
    
class VAELoss(nn.Module):
    KEYS = ['Reconstruction','KLD','Contrast','Align']
    
    def __init__(self, keys: list, config: dict, semantic_dim: int = 0, align: nn.Module = AlignLoss(),
                       trainable: bool = False):
        super().__init__()
        # outputs ids
        self.keys = keys
        self.semantic_dim = semantic_dim
        # initialize losses
        self.metrics = {        
            'Reconstruction': ReconstructionLoss(),
            'KLD': KLDGaussianLoss(),
            'Contrast': ContrastLoss(),
            'Align': align,
        }
        # build configuration
        init, order, weight = [], [], 1.
        for key in config:
            if not key in self.KEYS:
                raise Exception(f'Unknown key: {key}')
            if key == 'Reconstruction': # first priority non-trainable static weight
                weight = config[key]
            else: # regularizers with trainable if configured weights
                init.append(config[key])
                order.append(key)
        self.mixer = torch.Tensor(init)
        if trainable:
            self.mixer = nn.Parameter(self.mixer)
        self.order = order
        self.reconstruction_weight = weight
        self.trainable = trainable
                
    def forward(self, outputs, targets, labels=None):
        losses = {}
        # calculate component losses
        losses['Reconstruction'] = self.metrics['Reconstruction'](outputs[self.keys.index('rec')], targets)
        losses['KLD'] = self.metrics['KLD'](outputs[self.keys.index('mean')], outputs[self.keys.index('log-var')])
        losses['Contrast'] = self.metrics['Contrast'](outputs[self.keys.index('z')])
        if outputs[self.keys.index('label')] is not None and labels is not None:
            losses['Align'] = torch.unsqueeze(self.metrics['Align'](outputs[self.keys.index('label')], labels), 0)
        elif self.semantic_dim > 0:
            losses['Align'] = self.metrics['Align'](outputs[self.keys.index('z')], outputs[self.keys.index('z-context')])
        # calculate total loss
        rec = [losses['Reconstruction'] * np.exp(self.reconstruction_weight)]
        # use only those included in config
        mix = [losses[k] for k in self.order]
        # balance reconstruction vs. others plus regularization
        loss = torch.sum(torch.stack(rec + [torch.exp(f) * v for f, v in zip(self.mixer, mix)]))
        loss += torch.sum(self.mixer.pow(4))
        return loss, {k:float(v.item()) for k, v in losses.items()}
    
    
class DVAELoss(VAELoss):
    def __init__(self, keys: list, config: dict, categorical_dim: int, **kwargs):
        super().__init__(keys, config, **kwargs)
        self.metrics['KLD'] = KLDGumbelLoss(categorical_dim)
    
    def forward(self, outputs, targets, labels=None):
        losses = {}
        # calculate all component losses
        losses['Reconstruction'] = self.metrics['Reconstruction'](outputs[self.keys.index('rec')], targets)
        losses['KLD'] = self.metrics['KLD'](outputs[self.keys.index('q')])
        losses['Contrast'] = self.metrics['Contrast'](outputs[self.keys.index('p')])
        if outputs[self.keys.index('label')] is not None and labels is not None:
            losses['Align'] = torch.unsqueeze(self.metrics['Align'](outputs[self.keys.index('label')], labels), 0)
        elif self.semantic_dim > 0:            
            q = torch.mean(outputs[self.keys.index('q')].view(-1, LATENT_DIM, CATEGORICAL_DIM), dim=-1)
            pc = torch.mean(outputs[self.keys.index('p-context')].view(-1, LATENT_DIM, CATEGORICAL_DIM), dim=-1)            
            losses['Align'] = self.metrics['Align'](q, pc)
        losses['Temperature'] = self.metrics['Temperature'](outputs[self.keys.index('tau')])
        # calculate total loss
        rec = [losses['Reconstruction'] * np.exp(self.reconstruction_weight)]
        # use only those included in config
        mix = [losses[k] for k in self.order]
        # balance reconstruction vs. others plus regularization
        loss = torch.sum(torch.stack(rec + [torch.exp(f) * v for f, v in zip(self.mixer, mix)]))
        loss += torch.sum(self.mixer.pow(4))
        return loss, {k:float(v.item()) for k, v in losses.items()}


def training_step(model, optimizer, criterion, X, Y, labels):            
    # pass forward
    with torch.cuda.amp.autocast(enabled=False):
        preds = model([x.to(DEVICE) for x in X])
        loss, metrics = criterion(preds, [y.to(DEVICE) for y in Y], [l.to(DEVICE) for l in labels])
        if torch.isnan(loss):
            raise Exception('Loss went baNaNas...')
    # backp-prop
    optimizer.zero_grad()
    scaler.scale(loss).backward()    
    #scaler.unscale_(optimizer)
    #nn.utils.clip_grad_norm_(model.parameters(), 10.)
    scaler.step(optimizer)
    scaler.update()
    metrics['Loss'] = float(loss.item())
    return metrics


def train(model, dataset, criterion, optimizer, batch_size):
    model.train()
    criterion.train()
    history = []
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    for X, Y, labels in loader:
        metrics = training_step(model, optimizer, criterion, X, Y, labels)
        history.append(metrics)
    torch.cuda.empty_cache()
    return history


@torch.no_grad()
def validate(model, dataset, criterion, batch_size):
    model.eval()
    criterion.eval()
    history = []
    loader = DataLoader(dataset, batch_size=batch_size)
    for X, Y, labels in loader:
        preds = model([x.to(DEVICE) for x in X])
        loss, metrics = criterion(preds, [y.to(DEVICE) for y in Y], [l.to(DEVICE) for l in labels])
        metrics['Loss'] = float(loss.item())
        history.append(metrics)
    return history
    
    
def plot_results(path, window=100, save=False):
    history = {'train':[],'test':[]}
    for path in sorted([str(x) for x in Path('./output').glob(f'{path}*.json')],
                       key=lambda x:x if x.endswith('feedback.json') else int(x.split('-')[-1][:-len('.json')])):
        with open(path) as source:
            epoch = json.load(source)
            history['train'] += epoch['train']
            history['test'] += epoch['test']
    
    train_history = pd.DataFrame.from_dict(history['train'])
    test_history = pd.DataFrame.from_dict(history['test'])
    n = len(train_history.columns)
    fig, ax = plt.subplots(n, 1, figsize=(8, 3 * n))
    for i, key in enumerate(train_history.columns):        
        ax[i].plot(train_history[key].rolling(window).mean().dropna(), label='train')
        ax[i].plot(test_history[key].rolling(window).mean().dropna(), label='test')
        ax[i].set_title(key)
        ax[i].legend()
    plt.tight_layout()
    if save: plt.savefig(f"./{path.replace('.json','.png')}")
    

def get_rates(confmat):
    n = len(INDEX)
    pairs = [[a, b, confmat[a,b]] for a, b in np.transpose(confmat[:n,:n].nonzero())]
    pairs = pd.DataFrame(pairs, columns=['char-1','char-2','count'])
    pairs = pairs.merge(pairs.groupby('char-1').sum()['count'].rename('total').reset_index(), on='char-1')
    pairs['%rate'] = 100 * pairs['count']/pairs['total']
    pairs = pairs.loc[pairs['count'] > 1]
    pairs['char-1'] = pairs['char-1'].apply(lambda x:INDEX[x])
    pairs['char-2'] = pairs['char-2'].apply(lambda x:INDEX[x])
    return (pairs[pairs['char-1']==pairs['char-2']].set_index(['char-1','char-2']),
            pairs[pairs['char-1']!=pairs['char-2']].set_index(['char-1','char-2']))

