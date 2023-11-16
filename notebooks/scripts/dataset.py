#!/usr/bin/env python
# coding: utf-8

#------------------------------------------------
# datasets used in notebooks
#------------------------------------------------

import os
import re
import torch
import numpy as np
import pandas as pd

from PIL import Image, ImageOps
from matplotlib import pyplot as plt
from pathlib import Path
from typing import Callable
from einops import rearrange, repeat

from torch import nn, Tensor
from torchvision import transforms, models
from torch.utils.data import DataLoader, Dataset

from . import simulate as sim
from . import parse, render


# notebooks path
ROOT = f'{os.environ["HOME"]}/notebooks'

ORDER = ['void','line','input','word']
SEGMENTATION_WEIGHT = [0.02, 0.74, 0.21, 0.03]
DENOISING_WEIGHT = [0.07, 0.93]


# random images (not docs -- no text) for out-of-class examples
NONDOCS = [str(x) for x in Path(f'{ROOT}/data/unsplash').glob('*.jpg')]
# fraction of non-docs to show if needed
CONTRAST = 0.


# common preprocessing
class NormalizeView:
    """
    map to [0,1] and put channels first
    """
    def __call__(self, X):
        low, high = np.min(X), np.max(X)
        X = (X - low).astype(float)
        if high > low:
            X /= (high - low)
        if len(X.shape) == 3:
            h, w, c = X.shape
            return torch.Tensor(X).view(c, h, w)
        return torch.Tensor(X).unsqueeze(0)
    
    
def rolling_window(a, w):
    a = np.array(a)
    shape = a.shape[:-1] + (a.shape[-1] - w + 1, w)
    strides = a.strides + (a.strides[-1],)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)


def make_noisy_sample(view):
    h, w = view.shape
    noise = sim.generate_noise(max(w, h), light_bias=0.5, noise_strength=0.5)[:h,:w]
    return (1 - (view/255. * (0.5 + noise * 0.5))) * 255


def add_occlusion_mask(view: np.array) -> np.array:
    """
    add few random gradient stripes across the view
    """
    view = view.squeeze()
    view_size = max(view.shape)
    if np.random.rand() < 0.5:
        for x in np.random.randint(20, 200, np.random.randint(0, 5)):
            d = np.random.randint(10, 20)
            mask = np.array([list(range(d, 0, -1)) + list(range(d))]).astype(float)/d
            view[:,x - d:x + d] *= np.repeat(mask, view_size, axis=0)
        return view

    for x in np.random.randint(20, 200, np.random.randint(0, 5)):
        d = np.random.randint(10, 20)
        mask = np.array([list(range(d, 0, -1)) + list(range(d))]).astype(float)/d
        view[x - d:x + d,:] *= np.repeat(mask, view_size, axis=0).T
    return view


def make_negative_sample(view_size: int):
    """
    generate random non-document view
    """
    sample = 255 - np.array(ImageOps.grayscale(Image.open(np.random.choice(NONDOCS))))
    nav = render.AgentView(sample, view_size, bias=np.random.randint(100))
    center = (np.array(sample.shape) * (1 - np.random.rand(2))).astype(int)
    rotation = np.random.randint(0, 360)
    zoom = np.random.rand() * 2 - 2
    return nav.render(center, rotation, zoom)


def get_layout_labels() -> pd.DataFrame:
    """
    baselines: simple-layout dataset generated data
    """
    labels = pd.read_csv(f'{ROOT}/data/layout-baseline/labels.csv.gz')
    labels['type'] = (labels['inputs'] > 4).astype(int)
    return labels.loc[:,['orientation','type','path']]


def set_layout_labels(test_samples: list) -> pd.DataFrame:
    """
    mark training/validation subsets
    """
    labels = pd.read_csv(f'{ROOT}/data/layout-baseline/labels.csv.gz')
    labels['test'] = 0
    labels.loc[labels['source'].isin([x.split('/').pop()[:-4] for x in test_samples]),'test'] = 1
    labels['type'] = (labels['inputs'] > 4).astype(int)
    return labels.loc[:,['orientation','type','path','test']]


class CenterCrop:
    def __init__(self, view_size: int):
        self.view_size = view_size
        
    def __call__(self, X):
        X = X.squeeze()
        assert len(X.shape) == 2
        h, w = X.shape
        h, w = (h - self.view_size)//2, (w - self.view_size)//2
        return X[h:h + self.view_size,w:w + self.view_size]


class SimpleLayoutDataset(Dataset):
    """
    pages top-views with some non-docs for contrast if `nondocs_fracthion` > 0
    """
    def __init__(self, view_size: int, samples: pd.DataFrame, nondocs_fracthion: float = 0., batch_size: int = 32):
        self.view_size = view_size
        self.samples = samples
        order = list(range(len(samples))) + [-1] * int(len(samples) * nondocs_fracthion)
        self.order = np.random.choice(order, len(order), replace=False)
        self.transform = CenterCrop(view_size)

    def __len__(self):
        return len(self.order)

    def __getitem__(self, idx):
        if self.order[idx] == -1:
            X = NormalizeView()(make_negative_sample(self.view_size))
            return X, (torch.tensor(0).long(), torch.tensor(0).long())
        source = self.samples.iloc[self.order[idx]].to_dict()
        X = NormalizeView()(self.transform(np.array(ImageOps.grayscale(Image.open(source['path'])))))
        Y1 = torch.tensor(source['orientation']//90 + 1).long()
        Y2 = torch.tensor(source['type'] + 1).long()
        return X, (Y1, Y2)


class RandomViewDataset(Dataset):
    """
    use a single document noisy variation to create a batch of random view-ports
    make new data loader for each doc rather than reload resources for each view
    this scenario makes training very sensitive to the bad samples with bigger batches
    """
    def __init__(self, source: str, view_size: int, max_samples: int = 64, threshold: float = 0.25):
        self.view_size = view_size
        self.max_samples = max_samples
        self.threshold = threshold
        # load source image
        orig = np.array(ImageOps.grayscale(Image.open(f'{ROOT}/data/images/{source}')))
        view = make_noisy_sample(orig)
        # define renderers for all
        self.view = render.AgentView((view).astype(np.uint8), view_size, bias=np.random.randint(100))
        self.target = render.AgentView(255. - orig, view_size)
        # define image preprocesing
        self.transform = NormalizeView()

    def __len__(self):
        return self.max_samples
    
    def random_viewport(self):
        # pan: anywhere within the page-view bounding box
        center = (np.array(self.view.space.center) * (0.25 + np.random.rand() * 1.5)).astype(int)
        rotation = np.random.randint(0, 360)
        zoom = np.random.rand() * 4 - 3.5
        return center, rotation, zoom
    
    def __getitem__(self, idx):
        # once a while we need a negative sample
        if np.random.rand() < CONTRAST:
            X = self.transform(make_negative_sample(self.view_size))
            Y = torch.Tensor(np.zeros((self.view_size, self.view_size))).long()
            return X, Y
        # generate random viewport
        center, rotation, zoom = self.random_viewport()
        std = 0
        while std < 10: # make sure there's something to see
            center, rotation, zoom = self.random_viewport()
            view = self.view.render(center, rotation, zoom)
            std = np.std(view)
        # render corresponding views
        X = self.transform(view)
        # initialize masks channels
        target = self.transform(self.target.render(center, rotation, zoom))
        # sqrt here to make subtle lines pass the threshold
        Y = (target >= self.threshold).squeeze().long()
        return X, Y


class TopViewDataset(Dataset):
    """
    render full-page view
    """
    def __init__(self, view_size: int, samples: list, labels: list, contrast: float = 0.):
        self.view_size = view_size
        # add non-docs for contrast if needed
        n, c = int(len(samples) * contrast), max(labels)
        self.samples = list(samples) + ['?'] * n
        self.labels = labels
        self.non_doc_class = c + 1
        self.transform = NormalizeView()
        self.labels = labels

    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        source = self.samples[idx]

        if source == '?': # non-doc sample for contrast
            X = self.transform(make_negative_sample(self.view_size))
            Y = self.non_doc_class
            return X, Y
        
        # load source image
        view = np.array(ImageOps.grayscale(Image.open(f'{ROOT}/data/images/{source}.png')))
        # renderer full-page view
        view = render.AgentView((255. - view).astype(np.uint8), self.view_size).top()
        X = self.transform(view)
        Y = self.labels[idx]
        return X, Y
    
    
def get_alignment_weight(alignment_threshold=0, aligned_fraction=0.2, unknown_fraction=CONTRAST):
    """
    estimate MultitaskPretrainingDataset alignment task class-weight given chosen configuration
    """
    x = []
    for _ in range(10000):
        if np.random.rand() < unknown_fraction:
            x.append(0)
        else:
            r = np.random.choice([0, 90, 180, 270]) if np.random.rand() < aligned_fraction else \
                np.random.randint(0, 360)
            d = r % 90
            x.append(int(min(d, 90 - d) <= alignment_threshold) + 1)

    w = pd.Series(x)
    w = w.sum() - w.groupby(w).size()
    return list(np.round(list(w/w.sum()), 2))


def prep_batch(samples: list, dataset: Dataset, batch_size: int, view_size: int):
    """
    pregenerated batch for static data
    """
    source = np.random.choice(samples)
    loader = DataLoader(dataset(source, view_size, max_samples=batch_size), batch_size=batch_size)
    return list(loader)


def show_inputs(batch, size=(11, 11)):
    c = batch[0][0].shape[0]
    fig, ax = plt.subplots(1, c, figsize=size)
    for X, Y in batch:
        for i in range(c):
            ax[i].imshow(X[i,:].squeeze().numpy(), 'gray')
            ax[i].axis('off')
    ax[0].set_title(f'Input:  {X.shape[1:]}', fontsize=10, ha='left', x=0)
    plt.show()
    
    
def show_targets(batch, size=(11, 11)):    
    c = batch[0][0].shape[0]
    fig, ax = plt.subplots(1, c, figsize=size)
    for X, Y in batch:
        for i in range(c):
            ax[i].imshow(Y[i,:].squeeze().numpy(), 'gray')
            ax[i].axis('off')
    ax[0].set_title(f'Target:  {Y.shape[1:]}', fontsize=10, ha='left', x=0)
    plt.show()
    
    