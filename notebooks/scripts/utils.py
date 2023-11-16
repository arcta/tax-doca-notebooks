#!/usr/bin/env python
# coding: utf-8

import os
import cv2
import numpy as np
import pandas as pd
import pytesseract as ts

from pathlib import Path
from matplotlib import pyplot as plt
from matplotlib import patches


LAYOUT = ['left','top','width','height']
# notebooks path
ROOT = f'{os.environ["HOME"]}/notebooks'


class Blocks:
    """
    Find connected components of non-zero values in a matrix of non-negative integers
    """
    def __init__(self, words, size):
        """
        Calculate connected components and return pix-map
        for quick attribution by the top-left corner
        """
        rows, num_bins = size
        grid = quantize_layout(words.loc[~words['text'].str.strip().isin(['|','_',''])], num_bins)
        self.matrix = self.build(grid[LAYOUT], size)
        self.text = {(i,j):0 for i in range(rows) for j in range(num_bins) if self.matrix[i,j] == 1}
        self.count = 0
        for start in self.text:
            if self.text[start] == 0:
                self.count += 1
                self.search(start)
        
    def build(self, grid, size):
        """
        low-dimensional text presence map:
        cells without text in them set to 0, cells with text set to 1
        """
        matrix = np.zeros(size)
        for x, y, w, h in grid.values:
            matrix[y : y + h + 1, x : x + w + 2] = 1                
        return matrix
    
    def search(self, start):
        stack = [start]
        while len(stack) > 0:
            i,j = stack.pop()
            self.text[(i,j)] = self.count
            self.matrix[i,j] = self.count
            stack += [(a,b) for a,b in [(i-1,j),(i,j-1),(i+1,j),(i,j+1)]
                      if (a,b) in self.text and self.text[(a,b)] == 0]



def get_failed() -> dict:
    docs = {}
    with open(f'{ROOT}/data/prep.log') as log:
        lines = log.read().split('\n')
        for i in range(len(lines)):
            if lines[i].startswith('INFO data/images/') and lines[i].endswith('failed...'):
                cls, name, page = lines[i].split()[1][:-4].split('-')
                docs[name] = docs.get(name, {'class':1 if cls == 'cnd' else 0, 'pages':[]})
                docs[name]['pages'].append(page)
    return docs


def get_docs(path: str = 'orig') -> dict:
    docs = {}
    for path in [str(x) for x in Path(f'{ROOT}/data/{path}').glob('*.png')]:
        source = path.split('/').pop()
        cls, name, page = source[:-4].split('-')
        docs[name] = docs.get(name, {'class':1 if cls == 'cnd' else 0, 'size':0, 'pages':[]})
        docs[name]['size'] = max(docs[name]['size'], int(page))
        docs[name]['pages'].append(source)
    return docs
    
    
def img_show(imgarr, title='', hide_ticks=True, figsize=None) -> None:
    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(imgarr, cmap='gray')
    ax.set_title(title)
    if hide_ticks is True:
        ax.set_xticks([])
        ax.set_yticks([])
    plt.show()
    
    
def add_outlines(ax, boxes, edgecolor='white', facecolor='none'):
    for box in boxes:
        x, y, w, h = box
        ax.add_patch(patches.Rectangle((x, y), w, h, linewidth=1, edgecolor=edgecolor, facecolor=facecolor))
        
        
def add_text(ax, boxes, color='white', fontsize='xx-small'):
    for box in boxes:
        x, y, w, h, text = box
        ax.text(x + w/2, y + h/2, text, verticalalignment='center', horizontalalignment='center', color=color, fontsize=fontsize)
    

def extract_outlines(imgarr, kernel_size=5, min_height=0):
    dilated = cv2.dilate(imgarr, np.ones((kernel_size, kernel_size), np.uint8), iterations=1)
    contours,_ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    boxes = sorted([cv2.boundingRect(cnt) for cnt in contours], key=lambda t:(t[2] * t[3]), reverse=True)[1:]
    return pd.DataFrame([[x, y, w, h] for x, y, w, h in boxes if h >= min_height], columns=LAYOUT)


def remove_outlines(imgarr):
    output = imgarr.copy()
    mask = 255 - imgarr
    mask = cv2.medianBlur(mask, 11)
    mask = cv2.erode(mask, np.ones((5, 5)))
    mask = cv2.threshold(mask, 190, 255, cv2.THRESH_BINARY)[1]
    mask = cv2.GaussianBlur(mask, (5, 0), 11)
    mask = cv2.threshold(mask, 110, 255, cv2.THRESH_BINARY)[1]
    mask = cv2.erode(mask, np.ones((11, 11)))
    output = imgarr.copy()
    output[np.where(mask != 0)] = 0
    return output


def quantize_layout(data, num_bins):
    """
    snap layout data to the low-dimensional grid
    """
    return (data.loc[:,LAYOUT] * num_bins).round().astype(int)


def value_norm(img):
    """
    downsampled original image inversed standardized and minimaxed into [0,1]
    """
    value = (img - np.mean(img))/np.std(img)
    value -= np.min(value)
    value /= np.max(value)
    return value


def set_value_func(img):
    """
    use normalized down-sampled original image to estimate the box value
    """
    value = value_norm(img)    
    def get_box_value(box):
        x, y, w, h = box
        return np.mean(value[y : y + h + 1, x : x + w + 1])
    return get_box_value


def extract_layout(imgarr: np.array) -> pd.DataFrame:
    """
    get OpenCV contours normalized on the page width (preserve aspect-ratio)
    """
    height, width = imgarr.shape
    layout = extract_outlines(imgarr)
    layout /= width
    return layout
    
    
def extract_words(imgarr: np.array, lang: str = 'eng', mode: int = 3) -> pd.DataFrame:
    """
    get words (tesseracts level 5) with bounding boxes normalized on the page width (preserve aspect-ratio)
    """
    height, width = imgarr.shape
    try:
        data = pd.DataFrame.from_dict(ts.image_to_data(imgarr, lang=lang, config=f'--psm {mode}',
                                                       output_type=ts.Output.DICT))
    except Exception as e:
        print(f'Failed text extraction: {e}')
        return pd.DataFrame()
    else:
        data = data.loc[(data['level'] == 5)&(data['text'].str.strip() != '')]
        # we are not going to use tesseract layout detection
        data = data.drop(['level','block_num','par_num','line_num','word_num'], axis=1)
        data.loc[:,LAYOUT] /= width
        return data
            
        