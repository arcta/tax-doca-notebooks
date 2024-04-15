#!/usr/bin/env python
# coding: utf-8

import torch
import numpy as np
import pandas as pd

from typing import Optional, Callable
from torch import nn, Tensor
from torch.utils.data import DataLoader, Dataset

from .extract import *


labels = pd.read_csv('./data/training-labels.csv')
index = pd.read_csv('./data/training-index.csv')
trainset = index[index['test']==0]['source'].tolist()
testset  = index[index['test']==1]['source'].tolist()

INDEX  = labels[(labels['label'].str.len()==1)&(labels['count'] > 1)]['label'].tolist()
WEIGHT = labels.set_index('label').loc[INDEX,'weight'].tolist()

NUM_CLASSES = 156

#EXTRAS = [s for s in '‹›«»"|_' if s not in INDEX]
INDEX = [' '] + INDEX #+ EXTRAS
WEIGHT = [round(1./len(trainset + testset), 4)] + WEIGHT #+ [1.] * len(EXTRAS)
INDEX += [''] * (NUM_CLASSES - len(INDEX))
WEIGHT += [1e-6] * (NUM_CLASSES - len(WEIGHT))


def show_inputs(X, title='Input', size=(10, 10)):
    n = len(X)
    fig, ax = plt.subplots(1, n, figsize=size)
    for i in range(n):
        ax[i].imshow(X[i,:].squeeze().numpy(), 'gray')
        ax[i].axis('off')
    ax[0].set_title(f'{title}:  {X.shape[1:]}', fontsize=10, ha='left', x=0)
    plt.show()
    
    
def show_targets(Y, size=(10, 10)):
    show_inputs(Y, title='Target')
    
    
def square_clip(image, x, y, w, h):
    if y < 0:
        y, h = 0, h + y
    clip = image[y:y + h,x:x + w]
    s = max(clip.shape)
    square = np.zeros((s, s))
    try:
        square[(s - h)//2:(s - h)//2 + clip.shape[0],(s - w)//2:(s - w)//2 + clip.shape[1]] = clip
    except ValueError:
        return np.zeros((s, s))
    return square


class TokenDataset(Dataset):
    def __init__(self, source, labels, proba_threshold=0.8, path='training', amp=1, min_count=1, dim=32):
        self.dim = (dim, dim)
        self.amp = amp
        self.labels = labels
        data = pd.read_csv(f'./data/{path}/{source}.csv.gz')
        filters = (data['label'].isin(self.labels))&(data['proba'] > proba_threshold)
        self.data = data.loc[filters,:]
        index = data[filters].groupby('label').size()
        # having at least `min_count` samples of each
        self.index = np.random.permutation(index[index >= min_count].index.tolist())
        self.image = 255 - load_image(source)
        
    def __len__(self):
        return len(self.index) * self.amp
    
    def clip(self, x, y, w, h):        
        return square_clip(self.image, x, y, w, h)/255.
    
    def encode(self, label):
        if str(label) in self.labels:
            return self.labels.index(str(label))
        return self.labels.index('') # not known yet
    
    def get_values(self, idx):
        columns = BOX + ['label','left-side','right-side']
        values = self.data[self.data['label']==self.index[idx//self.amp]].sample()[columns].values[0]
        box, label, neighbors = values[:4], values[4], values[5:]
        return box, label, neighbors

    def __getitem__(self, idx):
        box, label, neighbors = self.get_values(idx)
        x, y, w, h = box
        target = self.clip(x - 1, y - 2, w + 2, h + 4)
        X = torch.Tensor(cv2.resize(target, self.dim, cv2.INTER_AREA)).unsqueeze(0)
        Y = torch.Tensor([self.encode(label)]).long()
        return X, Y


class VanillaDataset(TokenDataset):
    def __init__(self, source, labels, amp=1, min_count=1, dim=32, rec_dim=22, transform=0.9, empty=True):
        super().__init__(source, labels, amp=amp, min_count=min_count, dim=dim)
        self.rec = (rec_dim, rec_dim)
        self.transform_threshold = transform
        self.empty = empty
        
    def transform(self, clip, sigma):
        if np.random.rand() > self.transform_threshold:
            return clip
        h, w = clip.shape
        box = np.array([[1, 1],[h - 1, 1],[h - 1, w - 1],[1, w - 1]])
        skew = box + np.random.normal(0, sigma, (4, 2))
        matrix = cv2.getPerspectiveTransform(box.astype(np.float32), skew.astype(np.float32))
        return cv2.warpPerspective(clip, matrix, (h, w),flags=cv2.INTER_LINEAR)
    
    def __getitem__(self, idx):
        if self.empty and np.random.rand() < 1./len(self.index):
            # blank space
            return torch.zeros(self.dim).unsqueeze(0), torch.zeros(self.rec), 0
        box, label, neighbors = self.get_values(idx)
        x, y, w, h = box
        view = self.transform(self.clip(x, y, w, h), h//10)
        X = torch.Tensor(cv2.resize(view, self.dim, cv2.INTER_AREA)).unsqueeze(0)
        Y = torch.Tensor(cv2.resize(view, self.rec, cv2.INTER_AREA))
        return X, Y, self.encode(label)
    
    
class ConceptDataset(VanillaDataset):
    def __getitem__(self, idx):
        box, label, neighbors = self.get_values(idx)
        x, y, w, h = box
        concept = self.clip(x, y, w, h)
        view = self.transform(concept, h//10)
        X = torch.Tensor(cv2.resize(view, self.dim, cv2.INTER_AREA)).unsqueeze(0)
        Y = torch.Tensor(cv2.resize(concept, self.rec, cv2.INTER_AREA))
        return X, Y, self.encode(label)
    
    
class FocusDataset(TokenDataset):
    def __init__(self, source, labels, rec_dim=22, transform=0.9, empty=True, **kwargs):
        super().__init__(source, labels, **kwargs)
        self.rec = (rec_dim, rec_dim)
        self.transform_threshold = transform
        self.empty = empty
        
    def transform(self, views, sigma, margins=[0, 0]):
        if np.random.rand() > self.transform_threshold:
            return views
        h, w = views[0].shape
        output = []
        skew = np.random.normal(0, sigma, (4, 2))
        for view, m in zip(views, margins):
            box = np.array([[m, m],[h - m, m],[h - m, w - m],[m, w - m]])
            matrix = cv2.getPerspectiveTransform(box.astype(np.float32), (box + skew).astype(np.float32))
            output.append(cv2.warpPerspective(view, matrix, (h, w),flags=cv2.INTER_LINEAR))
        return output
    
    def __getitem__(self, idx):
        if self.empty and np.random.rand() < 1./len(self.index):
            # blank space
            return torch.zeros(self.dim).unsqueeze(0), torch.zeros(self.rec), 0
        box, label, neighbors = self.get_values(idx)
        x, y, w, h = box
        target = self.clip(x - 1, y - 2, w + 2, h + 4)
        d = h//4
        s = max(w + d * 2, h + d)
        dx, dy = (s - w)//2, (s - h)//2
        context = self.clip(x - dx, y - dy, s, s)
        target, context = self.transform((target, context), 4)
        X = torch.Tensor(cv2.resize(context, self.dim, cv2.INTER_AREA)).unsqueeze(0)
        Y = torch.Tensor(cv2.resize(target, self.rec, cv2.INTER_AREA))
        return X, Y, self.encode(label)
    
    
class ContextDataset(FocusDataset):    
    def classify(self, neighbors):
        left, right = neighbors
        if left != ' ' and right != ' ':
            return 0 # mid-word
        if left == ' ' and right != ' ':
            return 1 # head
        if left != ' ' and right == ' ':
            return 2 # tail
        return 3 # stand-alone
    
    def get_raw(self, idx):
        if self.empty and np.random.rand() < 1./len(self.index):
            # blank space
            return np.zeros(self.dim), np.zeros(self.rec), 0, 4
        box, label, neighbors = self.get_values(idx)
        x, y, w, h = box
        d = 2 * h//3
        s = max(w + d * 2, h + d)
        dx, dy = (s - w)//2, (s - h)//2
        view = self.clip(x - dx, y - dy, s, s)
        context = view.copy()
        t, b, l, r = (s - h)//2 - 1, (s + h)//2 + 2, (s - w)//2 - 1, (s + w)//2 + 2
        context[t:b,l:r] = 0
        state = self.classify(neighbors)
        return view, context, self.encode(label), state    
    
    def __getitem__(self, idx):
        view, context, label, state = self.get_raw(idx)
        view = cv2.resize(view, self.dim, cv2.INTER_AREA)
        context = cv2.resize(context, self.rec, cv2.INTER_AREA)
        X = torch.Tensor(cv2.resize(view, self.dim, cv2.INTER_AREA)).unsqueeze(0)
        Y = torch.Tensor(cv2.resize(context, self.rec, cv2.INTER_AREA))
        return view, context, label, state
    
    
class PretrainingDataset(ContextDataset):
    options = [0, cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_180, cv2.ROTATE_90_COUNTERCLOCKWISE]
    
    def spoil(self, view):
        partial = view.copy()
        if np.random.rand() < 0.3:
            return partial
        s = view.shape[0]
        d = np.random.randint(s//8, s//2)
        a = np.random.randint(0, s - d)
        options = [(slice(None, None), slice(a, a + d)),(slice(a, a + d), slice(None, None))]
        o = np.random.choice(len(options))
        r, c = options[o]
        partial[r,c] *= np.random.normal(0, 1, size=(s, d) if o == 0 else (d, s))
        return np.clip(partial, 0, 1)
    
    def __getitem__(self, idx):
        # get view, context and labels
        V, C, label, state = super().get_raw(idx)
        if state == 4:
            X, Y = [self.spoil(V + np.random.rand(*V.shape) * 1e-4) for _ in range(3)], [C] * 3
            if np.random.rand() > 0.5:
                X = [1. - x for x in X]
            return [torch.Tensor(x).unsqueeze(0) for x in X], [torch.Tensor(y) for y in Y], [0, 0]
        
        box, _, neighbors = self.get_values(idx)
        x, y, w, h = box
        X, Y = [], []
        # char
        orig = self.clip(x - 1, y - 2, w + 2, h + 4)
        view = self.transform((orig,), h//10)[0]
        X.append(cv2.resize(self.spoil(view), self.dim, cv2.INTER_AREA))
        Y.append(cv2.resize(view, self.rec, cv2.INTER_AREA))
        # focus
        d = h//4
        s = max(w + d * 2, h + d)
        dx, dy = (s - w)//2, (s - h)//2
        orig = self.clip(x - dx, y - dy, s, s)
        view = self.transform((orig,), 4)[0]
        X.append(cv2.resize(self.spoil(view), self.dim, cv2.INTER_AREA))
        Y.append(cv2.resize(view, self.rec, cv2.INTER_AREA))
        # context
        X.append(cv2.resize(self.spoil(V), self.dim, cv2.INTER_AREA))
        Y.append(cv2.resize(V, self.rec, cv2.INTER_AREA))
        # rotation
        orientation = np.random.choice(len(self.options))
        if orientation != 0:
            angle = self.options[orientation]
            X = [cv2.rotate(x, angle) for x in X]
            Y = [torch.Tensor(cv2.rotate(y, angle)) for y in Y]
        X, Y = [torch.Tensor(x).unsqueeze(0) for x in X], [torch.Tensor(y) for y in Y]
        if np.random.rand() > 0.5:
            return [1. - x for x in X], Y, [label, orientation]
        return X, Y, [label, orientation]
    
    
class MultitaskDataset(ContextDataset):
    def spoil(self, view):
        partial = view.copy()
        if np.random.rand() < 0.3:
            return partial
        s = view.shape[0]
        d = np.random.randint(s//8, s//2)
        a = np.random.randint(0, s - d)
        options = [(slice(None, None), slice(a, a + d)),(slice(a, a + d), slice(None, None))]
        o = np.random.choice(len(options))
        r, c = options[o]
        partial[r,c] *= np.random.normal(0, 1, size=(s, d) if o == 0 else (d, s))
        return np.clip(partial, 0, 1)
    
    def __getitem__(self, idx):
        # get view, context and labels
        V, C, label, semantic = super().get_raw(idx)
        if semantic == 4:
            X, Y = [self.spoil(V + np.random.rand(*V.shape) * 0.001) for _ in range(3)], [C] * 3
            if np.random.rand() > 0.5:
                X = [1. - x for x in X]
            return [torch.Tensor(x).unsqueeze(0) for x in X], [torch.Tensor(y) for y in Y], [0, 4, 0, 0]
        
        box, _, neighbors = self.get_values(idx)
        x, y, w, h = box
        X, Y = [], []
        # concept
        concept = self.clip(x - 1, y - 2, w + 2, h + 4)
        view = self.transform((concept,), h//10)[0]
        X.append(cv2.resize(self.spoil(view), self.dim, cv2.INTER_AREA))
        Y.append(cv2.resize(concept, self.rec, cv2.INTER_AREA))
        # focus
        d = h//4
        s = max(w + d * 2, h + d)
        dx, dy = (s - w)//2, (s - h)//2
        view = self.clip(x - dx, y - dy, s, s)
        focus, view = self.transform((concept, view), 4)
        X.append(cv2.resize(self.spoil(view), self.dim, cv2.INTER_AREA))
        Y.append(cv2.resize(focus, self.rec, cv2.INTER_AREA))
        # context
        left, right = [self.encode(x) for x in neighbors]
        X.append(cv2.resize(self.spoil(V), self.dim, cv2.INTER_AREA))
        Y.append(cv2.resize(C, self.rec, cv2.INTER_AREA))
        X, Y = [torch.Tensor(x).unsqueeze(0) for x in X], [torch.Tensor(y) for y in Y]
        if np.random.rand() > 0.5:
            return [1. - x for x in X], Y, [label, semantic, left, right]
        return X, Y, [label, semantic, left, right]

