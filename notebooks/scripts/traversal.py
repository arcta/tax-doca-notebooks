#!/usr/bin/env python
# coding: utf-8

#------------------------------------------------
# VAE contrast training
#------------------------------------------------

import torch
import numpy as np
import pandas as pd

from matplotlib import pyplot as plt
from typing import Optional, Callable
from scipy.stats import mode

from .training import *


torch.cuda.empty_cache()
scaler = GradScaler()



class EncoderClassifier(nn.Module):
    def __init__(self, encoder: nn.Module, mlp: nn.Module):
        super().__init__()
        self.encoder = encoder
        self.mlp = mlp
        
    def forward(self, x):
        return self.mlp(self.encoder(x)[0])
    
    
class ConceptDecoder(nn.Module):
    def __init__(self, encoder: nn.Module, decoder: nn.Module):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        
    def forward(self, x):
        return self.decoder(self.encoder(x)[0])
    
    
class Reader(nn.Module):
    """
    Use encoders with their MLPs, and optional orientation detector
    """
    def __init__(self, model: MultiVAE, orientation: EncoderClassifier = None, neighbors: list = None):
        super().__init__()
        # assemble encoder-classifier pipelines
        self.orientation = orientation
        for tag in model.tags:
            m = getattr(model, tag)
            setattr(self, tag, EncoderClassifier(m.encoder, m.mlp))
        self.tags = model.tags
        # add semantic classifier
        self.semantic = EncoderClassifier(model.context.encoder, model.semantic.semantic)
        self.keys = model.tags + ['semantic']        
        # add orientation detector
        self.orientation = orientation
        if self.orientation is not None:
            self.keys.append('orientation')
        # add neighbors detection
        self.neighbors = None
        if neighbors is not None:
            self.neighbors = nn.ModuleList([EncoderClassifier(model.context.encoder, side) for side in neighbors])
            self.keys += ['left','right']
        
    def forward(self, X):
        output = []
        for i, tag in enumerate(self.tags):
            output.append(getattr(self, tag)(X[i]))
        output.append(self.semantic(X[2]))
        if self.orientation is not None:
            output.append(self.orientation(X[2]))
        if self.neighbors is not None:
            output += [side(X[2]) for side in self.neighbors]
        return output
    
    
def get_readers(orientation=True, neighbors=True, source='online'):
    """
    Load trained model and unpack it into separate components
    """
    latent_dim = 64
    semantic_dim = 5
    num_classes = 156
    
    if orientation:
        detector = MultiVAE(get_encoder(), latent_dim, 4).to(DEVICE)
        detector.load_state_dict(torch.load(f'./models/VAE-{latent_dim}.pt'))
        orientation = EncoderClassifier(detector.context.encoder, detector.context.mlp)
    
    keys = [f'{tag}-{key}' for tag in ['concept','target','context']
                           for key in ['rec', 'z', 'mean', 'log-var', 'tau', 'z-context', 'label']]
    semantic = ContextClassifier(keys, latent_dim, semantic_dim)
    model = MultiVAE(get_encoder(), latent_dim, num_classes, semantic=semantic, neighbors=neighbors).to(DEVICE)
    model.load_state_dict(torch.load(f'./models/VAE-multi-{latent_dim}-{num_classes}-{semantic_dim}-{source}.pt'))
    return Reader(model, orientation, model.neighbors)


class ReaderDataset(Dataset):
    def __init__(self, image, tokens, dim=32):
        self.d = dim
        self.tokens = tokens
        self.image = 255 - image
        
    def __len__(self):
        return len(self.tokens)
    
    @staticmethod
    def get_input_view(image, x, y, w, h, tag):
        if tag == 'concept':
            return square_clip(image, x - 1, y - 2, w + 2, h + 4)/255.
        d = h//4 if tag == 'target' else 2 * h//3
        s = max(w + d * 2, h + d)
        dx, dy = (s - w)//2, (s - h)//2
        return square_clip(image, x - dx, y - dy, s, s)/255.
    
    @staticmethod
    def transform(view, d):
        return torch.Tensor(cv2.resize(view, (d, d), cv2.INTER_AREA)).view(1, d, d)

    def __getitem__(self, idx):
        x, y, w, h = self.tokens.iloc[idx][BOX].values
        return [self.transform(self.get_input_view(self.image, x, y, w, h, tag), self.d)
                for tag in ['concept','target','context']]

    
def get_consensus(r):
    """
    Take the one with most votes, otherwise `context` has priority
    """
    v = r.values
    m = mode(v[:3])
    if m.count > 1:
        return INDEX[m.mode]
    if v[3] in v[:3]:
        return INDEX[v[3]]
    return ''


@torch.no_grad()
def batch_read(reader, image, tokens, batch_size=128):
    assert len(tokens) > 0
    
    reader.eval()

    Y, P = [], []
    # identify tokens with models consensus
    for X in DataLoader(ReaderDataset(image, tokens), batch_size=batch_size, shuffle=False):
        output = reader([x.to(DEVICE) for x in X])
        P.append(torch.t(torch.stack([torch.amax(torch.softmax(o, dim=1), dim=1)
                                      for o in output])).cpu().numpy())
        Y.append(torch.t(torch.stack([torch.argmax(torch.softmax(o, dim=1), dim=1)
                                      for o in output])).cpu().numpy())

    P = pd.DataFrame(np.vstack(P), columns=reader.keys)
    Y = pd.DataFrame(np.vstack(Y), columns=reader.keys)
    assert len(Y) == len(tokens)
    
    tokens['proba'] = P.iloc[:,:3].mean(axis=1)
    tokens['label'] = Y.apply(get_consensus, axis=1)
    
    for c in ['concept','target','context','left','right']:
        Y[c] = Y[c].apply(lambda x:INDEX[x])
    Y['semantic'] = Y['semantic'].apply(lambda x:'mhtse'[x])
    Y['orientation'] = Y['orientation'].apply(lambda x:[0, 90, 180, 270][x])
    # reader stats for analysis
    return Y, P


def aggregate_text(tokens, words, lines, labels, proba, proba_threshold=0):
    """
    Do not rely on extractor but go by visual: add spaces if detected
    """
    lines['text'] = None
    words['text'] = None
    indexer = ['word-index','line-index']
    stats, word, text = [], [], []        
    last_wi, last_li = tokens.iloc[0][indexer].values
    for i in range(len(tokens)):
        V, Y, P = tokens.iloc[i], labels.iloc[i], proba.iloc[i]
        x, y, w, h = V[BOX].values
        # layout hierarchy indexers
        wi, li = V[indexer].values
        # classifications
        if Y['semantic'] == 'e' and P['semantic'] >= proba_threshold:
            continue
        
        label = '☹' if V['proba'] < proba_threshold else str(V['label'])
        
        # extractor hit `end-word`
        if wi != last_wi:
            # aggregate as extracted
            words.loc[last_wi,'text'] = ''.join(word)
            word = []
            last_wi = wi
            
        # extractor hit `end-line`
        if li != last_li:
            # aggregate as detected and remove redundant spaces
            lines.loc[last_li,'text'] = ' '.join((''.join(text)).split())
            text = []
            last_li = li

        # add space if precedes
        if Y['semantic'] in 'hs' and P['semantic'] >= proba_threshold:
            text.append(' ')
        #elif Y['left'] == ' ' and P['left'] >= proba_threshold:
        #    text.append('_')
        
        word.append(label)
        text.append(label)
        
        # add space if follows
        if Y['semantic'] in 'ts' and P['semantic'] >= proba_threshold:
            text.append(' ')
        #elif Y['right'] == ' ' and P['right'] >= proba_threshold:
        #    text.append('_')
        
    lines.loc[last_li,'text'] = ' '.join((''.join(text)).split())
    
    
def get_form(source):
    _, inputs = load_info(source)
    return inputs[inputs['field_type_string'].isin(['Text','CheckBox','RadioButton'])][['field_label','field_type_string']]


def scan(reader, tag, image, box, cw, lh, proba_threshold=0.95, dim=32):
    """
    Scan through the wider context window with the `context` model component
    """
    model = getattr(reader, tag)    
    Y, P = [], []
    x, y, w, h = box
    x1 = x - cw//2
    y1, h1 = y, min(h, lh + lh//2)
    while y1 + h1 < y + lh + lh//2:
        while x1 <= x + w + cw//2:
            X = ReaderDataset.transform(ReaderDataset.get_input_view(image, x1, y, cw, h, tag), dim)
            proba = torch.softmax(model(X.view(1, 1, dim, dim).to(DEVICE)), dim=1).cpu().numpy().squeeze()
            index = np.argmax(proba)
            if proba[index] >= proba_threshold:
                Y.append(INDEX[index])
                P.append(proba[index])
                if w <= cw:
                    return Y, P
                x1 += cw
            x1 += cw//4
        y1 += lh
    return Y, P

