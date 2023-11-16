#!/usr/bin/env python
# coding: utf-8

import os
import re
import numpy as np
import pandas as pd
import matplotlib as mpl

from PIL import Image, ImageOps
from matplotlib import pyplot as plt
from matplotlib import patches
from IPython.display import display, clear_output
from elasticsearch import Elasticsearch

# local scripts lib
from . import simulate as sim
from . import parse


INDEX = 'doc-pages'
BOX = ['top','left','bottom','right']
# priority order: next overrides previous
ORDER = ['void','word','input','line']


eclient = Elasticsearch(
    hosts=[os.environ['ELASTIC_URI']],
    basic_auth=('elastic', os.environ['ELASTIC_PASSWORD']),
    verify_certs=False
)

def get_page_content(doc, page, size=10000):
    query = {'bool': {'must': [{'match': {'doc_id': doc }}, {'match': {'page_id': page }}]}}
    sort = [{'top': {'order': 'asc'}}, {'left': {'order': 'asc'}}]
    return [x['_source'] for x in eclient.search(index=INDEX, query=query, sort=sort, size=size)['hits']['hits']]


class Renderer:
    def __init__(self, channels: int = 1, bias: int = 0):
        self.bias = bias   # background
        self.channels = channels
    
    def render(self, radius: int) -> np.array:
        # square bounding box for a circle reception around a view center
        radius = max(0, radius)
        size = 2 * radius
        view = (np.zeros((size, size, self.channels)) + self.bias).astype(np.uint8)
        return view


class ImageSpace(Renderer):
    def __init__(self, image: np.array, bias: int = 0):
        if len(image.shape) == 2: # make sure channels are set
            image = image[..., np.newaxis]
        super(ImageSpace, self).__init__(channels=image.shape[2], bias=bias)
        self.image = image[:,:,:]
        self.H, self.W, self.C = image.shape
        # relative units to measure transitions
        self.norm = np.array([self.H, self.W, 180, 1]).astype(np.float32)
    
    @property
    def center(self):
        return self.H//2, self.W//2
    
    # clip of background image which appears in the view ( could be nothing -- plain background )
    def render(self, center: tuple, radius: int) -> np.array:
        view = super().render(radius) # plain background
        dh, dw = center
        # image-matrix indices
        t, l = max(dh - radius, 0), max(dw - radius, 0)
        if t >= self.H or l >= self.W:
            return view
        b, r = min(dh + radius, self.H), min(dw + radius, self.W)
        if b <= t or r <= l:
            return view        
        self._loc = (t, b, l, r)
        clip = self.image[t:b,l:r,:]
        # view-matrix indices
        t, l = abs(min(dh - radius, 0)), abs(min(dw - radius, 0)) # top, left
        b, r = t + clip.shape[0], l + clip.shape[1]  # bottom, right
        view[t:b,l:r,:] = clip[:,:,:]
        return view


class AgentView:    
    def __init__(self, image: np.array, dim: int, bias: int = 0, zoom_range: tuple = (-5, 1)):
        # sets the exploration space
        self.space = ImageSpace(image, bias=bias) #np.quantile(image, 0.25)
        # sets the agent's view resolution
        self.dim = dim
        self.channels = self.space.C
        # limit zoom to avoid overload
        self.zoom_range = tuple(sorted(zoom_range))
        # initialised the agent's state tracking: default top-left corner
        self._state = np.zeros((4,))
    
    def set_state(self, center: tuple, rotation: float = 0., zoom: float = 0.) -> np.array:
        if zoom > self.zoom_range[1] or zoom < self.zoom_range[0]:
            #print(f'WARNING: zoom value is clipped to range {self.zoom_range} relative to the page top-view')
            zoom = np.clip(zoom, *self.zoom_range)
        rotation = self.get_angle(rotation)
        self._state = np.array(list(center) + [rotation, zoom])
        return self.render(center, rotation, zoom)
                        
    @property
    def state(self):
        return self._state[:]
    
    @property
    def loc(self):
        return self._state[:2]/self.space.norm[:2]
        
    @staticmethod
    def get_angle(rotation: float = 0.) -> float:
        """
        map into [0, 360] interval
        """
        rotation = rotation % 360
        if rotation >= 0:
            return rotation
        return 360 + rotation
    
    def get_size(self, zoom: float = 0.) -> int:
        """
        size of the actual image clip in the view
        """
        if zoom == 0.:
            return self.dim
        return int(np.round(self.dim * (2 ** -zoom)))
    
    def get_zoom(self) -> tuple:
        """
        bird's-eye view zoom-value
        """
        return -np.log2(max(self.space.W, self.space.H) / self.dim)
    
    def top(self):
        """
        render bird's-eye view: fit the whole image into a view-frame
        """
        return self.set_state(self.space.center, 0, self.get_zoom())
    
    def isin(self) -> bool:
        """
        at least some part of the doc is present in the view vs nothing at all
        """
        radius = self.get_size(self._state[-1])
        distance = np.linalg.norm(np.array(self.space.center) - self._state[:2])
        if distance - radius > max(self.space.center):
            return False
        return abs(self._state[0] - self.space.center[0]) < radius + self.space.center[0] and \
                abs(self._state[1] - self.space.center[1]) < radius + self.space.center[1]
        
    
    def render(self, center: tuple, rotation: float = 0., zoom: float = 0.) -> np.array:
        """
        generate a view relative to the page origin (absolute)
        """
        if zoom > self.zoom_range[1] or zoom < self.zoom_range[0]:
            print(f'Zoom value is clipped to range {self.zoom_range} relative to the page origin')
            zoom = np.clip(zoom, *self.zoom_range)
        size = self.get_size(zoom)
        radius = size//2
        h, w = center
        angle = self.get_angle(rotation)
        if angle == 0:
            view = self.space.render(center, radius)           
        else: # rotate
            # make room for rotation without expand
            diag = int(np.ceil(radius * np.sqrt(2)))
            view = self.space.render(center, diag)
            view = np.array(Image.fromarray(view.squeeze()).rotate(angle, expand=False))
            # crop margins back to `size`
            view = view[ diag - radius : radius - diag, diag - radius : radius - diag]
        if size == self.dim:
            return view
        # resample to match view dim
        view = np.array(Image.fromarray(view.squeeze().astype(np.uint8)).resize((self.dim, self.dim)))
        return view.reshape((self.dim, self.dim, self.channels))
    
    def translate_action(self, action):
        """
        action defined by 4 values between -1 and 1 -- relative and normalized
        translate (denormalize) it into absolute units -- w/h pixels and rotation degrees
        """
        value = action * self.space.norm
        value[:2] = np.round(value[:2])
        value[2] = self.get_angle(value[2])
        return value
    
    def transform(self, action):
        """
        reset the state by action value and generate resulting view
        """
        action = np.clip(action, -1., 1.)
        state = self.state + self.translate_action(action)
        return self.set_state(state[:2].astype(int), state[2], state[3])


def video(nav, sequence):
    """
    show animated visual for a sequence of actions
    """
    for action in sequence:
        observation = nav.transform(action)
        img = plt.imshow(observation, 'gray')
        value = ', '.join([f'{x:.2f}' for x in nav.translate_action(action)])
        plt.title(f' Action:[ {value} ]', ha='left', x=0, fontdict={'family':'monospace','size':10})
        display(plt.gcf())
        clear_output(wait=True)


def build_mask(image: np.array, source: str) -> np.array:
    """
    create class-indices matrix (classes are xclusive)
    """
    # extract outlines
    lines = parse.extract_lines(image, units=20)
    h, w = image.shape
    mask = np.zeros((h, w, len(ORDER)))            
    mask[:,:,ORDER.index('line')] = lines
    # load page data
    doc = source.split('/').pop()[:-4].split('-')
    doc, page = '-'.join(doc[:-1]), int(doc[-1])
    data = get_page_content(doc, page)
    if len(data) == 0:
        return None #mask
    data = pd.DataFrame.from_dict(data)
    data.loc[:,BOX] = np.round(data[BOX] * min((h, w))).astype(int)
    for t,l,b,r in data[data['block_type']=='word'][BOX].values:
        # reduce space between neighbor words
        mask[int(t):int(b), int(l) - 1:int(r) + 2, ORDER.index('word')] = 1
    for t,l,b,r in data[data['block_type']=='input'][BOX].values:
        # increase space between neighbor inputs
        mask[int(t) + 3:int(b) - 2, int(l) + 1:int(r), ORDER.index('input')] = 1
    # convert to indices
    return np.argmax(mask, axis=(2))

