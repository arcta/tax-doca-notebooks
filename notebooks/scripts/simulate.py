#!/usr/bin/env python
# coding: utf-8

import os
import sys
import cv2
import numpy as np
import pandas as pd
import pytesseract as ts

from PIL import Image, ImageOps
from pathlib import Path
from fitz import fitz

from .prep import img_rotate


# target resolution
DPI = 200
# meta-data to collect
INFO = ['font-size','italic','bold','color','cos','sin']
# interactiv widgets info parced from pdf
WIDGETS = ['CheckBox','RadioButton','Text','ComboBox']
WIDGET_DATA = ['field_name','field_label','field_display','field_type','field_type_string','button_caption',
               'text_fontsize','text_maxlen','text_format','border_width']
# notebooks path
ROOT = f'{os.environ["HOME"]}/notebooks'
# corners TL, BL, BR, TR in the order of 0, 90, 180, 270 rotation labels
ORDER = [(0,0),(1,0),(1,1),(0,1)]



def fake_number_input(widget: fitz.Widget, l) -> str:
    """
    generate random numeric input like: 2,456.58
    """
    value = f'{np.random.rand() * (10 ** l):,.2f}'
    try:
        widget.field_value = value
        widget.update()
    except:
        #print('Failure: fake_number_input')
        return widget.field_value
    return value


def fake_multiline_text_input(widget: fitz.Widget) -> str:
    """
    generate free-style textual content
    we only care here about presense of specific characters, order does not matter
    """
    chars = list('QWERTYUI OPAS DFGHJKLZ XCVBNM qwer tyuiop asdfghjkl zxcvbnm 012 345 6789,-') # seed
    w, h = widget.rect.x1 - widget.rect.x0, widget.rect.y1 - widget.rect.y0
    # estimate line length and the number of lines
    f = widget.text_fontsize or 8.
    l, n = int(w/(min(h, f * 2))), int(np.ceil(h/f/2))
    if n == 1 and (l < 10 or widget.field_value == '0'):
        # most probably numeric, we do not care much if it is not
        return fake_number_input(widget, l)
    length = np.random.randint(l + 1, (l + 1) * n + 1)
    value = ''.join(np.random.choice(chars, length, replace=True))
    try:
        widget.field_value = value
        widget.update()
    except:
        #print('Failure: fake_multiline_text_input')
        return widget.field_value
    return value
    
    
def fake_sized_text_input(widget: fitz.Widget) -> str:
    """
    generate fixed-size sequence (not a word, some code) each charater in its own box
    """
    chars = list('1234567890QWERTYUIOPASDFGHJKLZXCVBNM')
    value = ''.join(np.random.choice(chars, widget.text_maxlen, replace=True))
    try:
        widget.field_value = value
        widget.update()
    except:
        #print('Failure: fake_sized_text_input')
        return widget.field_value
    return value


def fake_radio_button_input(widget: fitz.Widget) -> str:
    """
    try to set radio-butt value: may fail not that important here
    """
    value = np.random.choice(['Off','On'])
    try:
        widget.field_value = value
        widget.update()        
    except:
        #print('Failure: fake_radio_button_input')
        return widget.field_value        
    return value


def fake_checkbox_input(widget: fitz.Widget) -> str:
    """
    leave some (25%) unchecked
    """
    value = np.random.choice([1, 1, 1, 0])
    try:
        widget.field_value = value
        widget.update()
    except:
        #print('Failure: fake_checkbox_input')
        return widget.field_value
    return value
    
    
def fake_select_input(widget: fitz.Widget) -> str:
    """
    set random value from the options
    """
    try:
        index = np.random.randint(len(widget.choice_values) - 1)
        value = widget.choice_values[index][1]
        widget.field_value = value
        widget.update()
    except:
        #print('Failure: fake_select_input')
        return widget.field_value
    return value


def parse_input_info(widget: fitz.Widget, scale: int, value: str) -> dict:
    """
    make sure the value is set as expected
    compile labels for generated input
    """
    assert widget.field_value == value
    d = { k:widget.__dict__[k] for k in WIDGET_DATA }
    d['left'] = widget.rect.x0 / scale
    d['top'] = widget.rect.y0 / scale
    d['right'] = widget.rect.x1 / scale
    d['bottom'] = widget.rect.y1 / scale
    d['value'] = widget.field_value
    return d


def layout_matrix(content: pd.DataFrame, scale: int = 128) -> np.array:
    """
    build a low-resolution text-presence map
    """
    value = (content[['left','top','right','bottom']] * 128).astype(int)
    size = tuple(value[['bottom','right']].max())
    matrix = np.zeros(size)
    for l,t,r,b in value.values:
        matrix[t:b,l:r] = 1
    return matrix    


def get_value(box: tuple, matrix: np.array) -> float:
    """
    estimate textual content density in the box specified by x0,y0,x2,y2 coords
    given the page-content-map matrix
    """
    l, t, r, b = box
    if t == b or l == r:
        return 0
    return np.mean(matrix[t:b,l:r])

    
def fill_in_blanks(page: fitz.Page, dpi: int = 200, matrix: np.array = None) -> tuple:
    """
    extract the widgets from the Page object and simulate the input
    return image with filled-in values and the labels-info
    """
    widgets = page.widgets()
    info = []
    pix = page.get_pixmap()
    scale = min(pix.width, pix.height)
    for widget in widgets:
        if widget.field_display != 0 and matrix is not None:
            # usually means the conditional block of inputs:
            # if you selected YES on any of these, do also those... 
            box = np.array([widget.rect.x0, widget.rect.y0, widget.rect.x1, widget.rect.y1])
            content = get_value((box/scale * 128).astype(int), matrix)
            if content > 0: continue # there's some textual content at the same area, so skip
            # othervise: generate some inputs and make them visible
            widget.field_display = 0
            widget.update()
        
        if widget.field_type_string == 'Text':
            if widget.text_maxlen > 0:
                value = fake_sized_text_input(widget)
            else: # free-style
                value = fake_multiline_text_input(widget)
            info.append(parse_input_info(widget, scale, value))
        elif widget.field_type_string == 'CheckBox':
            value = fake_checkbox_input(widget)
            info.append(parse_input_info(widget, scale, value))
        elif widget.field_type_string == 'RadioButton':
            value = fake_radio_button_input(widget)
            info.append(parse_input_info(widget, scale, value))
        elif widget.field_type_string == 'ComboBox':
            value = fake_select_input(widget)
            info.append(parse_input_info(widget, scale, value))
            
    # generate requested resolution image
    pix = page.get_pixmap(dpi=dpi)
    image = np.array(ImageOps.grayscale(Image.frombytes('RGB', [pix.width, pix.height], pix.samples)))
    # return generated image along with information on simulated inputs for the labels
    return image, info


def generate_noise(dim: int, light_bias: float = 0.5, noise_strength: float = 0.1, scale: int = 1) -> np.array:
    """
    generate square block filled with random noisy gradients
    light-bias: overall level of light -- "white" cannot go darker than that
    noise-strength: controls random uniform jitter 
    """
    x, y = np.random.rand(2)
    x = np.linspace(x, x + scale * np.random.rand(), dim)
    y = np.linspace(y, y + scale * np.random.rand(), dim)
    W = np.random.rand(np.random.randint(2, 5), 2) - 0.5
    W = W / W.sum(axis=0)
    A = (np.random.rand(W.shape[0], 2) - 0.5) * np.random.rand() * 20
    B = (np.random.rand(W.shape[0], 2) - np.random.rand()) * 2 * np.pi    
    X = np.tile(A[:,0], (dim, 1)) * np.tile(x, (A.shape[0], 1)).T
    X = np.sin(X + np.tile(B[:,0], (dim, 1)))
    X = (np.dot(X, W[:,0]) + 1.) * 0.5
    Y = np.tile(A[:,1], (dim, 1)) * np.tile(y, (A.shape[0], 1)).T
    Y = np.sin(Y + np.tile(B[:,1], (dim, 1)))
    Y = (np.dot(Y, W[:,1]) + 1.) * 0.5
    view = np.sum(np.array(np.meshgrid(X, Y)), axis=0)
    view -= np.min(view)
    view /= np.max(view)
    view = cv2.GaussianBlur(view, (11, 11), 0)
    view = light_bias + (1 - light_bias) * view
    return view * (1 - 0.1 * noise_strength) + (np.random.normal(0, 1, (dim, dim)) * 0.1 * noise_strength)
        
        
def random_transform(image: np.array, max_skew: int = 9,
                     light: float = 0.5, noise: float = None, perspective: bool = None) -> np.array:
    """
    apply warp, rotation, and add some noise to grayscale image
    """
    assert len(image.shape) == 2
    
    info = { 'skew':0, 'orient':0, 'warp':None, 'noise':None }    
    bg = np.random.randint(60, 240)
    
    size = tuple((np.array(image.shape) * 1.1).astype(int))
    transformed = np.ones(size) * bg
    # make room for some warp
    size = (np.array(transformed.shape) - np.array(image.shape))//2
    d = max(size)
    transformed[size[0]:size[0] + image.shape[0],size[1]:size[1] + image.shape[1]] = image[:,:]
    # random resize
    size = tuple((np.array(transformed.shape) * (0.75 + np.random.rand() * 0.25))[::-1].astype(int))
    transformed = cv2.resize(transformed, size, interpolation=cv2.INTER_AREA)
    info['size'] = size

    if perspective: # generate slight warp
        h, w = transformed.shape
        base = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]])
        a, b = np.random.randint(5, d, 2)
        points = np.float32([[a, b],[b, -a],[-a, -b],[-b, a]])
        points = np.float32((base + np.random.rand(4, 2) * points).astype(int))
        matrix = cv2.getPerspectiveTransform(points, base)
        transformed = cv2.warpPerspective(transformed, matrix, (w, h), borderMode=cv2.BORDER_REPLICATE)
        info['warp'] = points

    angle = 0
    # apply random skew and orientation flip
    if np.random.rand() > 0.95:
        angle = np.random.randint(max_skew * 2) - max_skew
        info['skew'] = angle
    if np.random.rand() > 0.25:
        orient = np.random.choice([90, 180, 270])
        angle += orient
        info['orient'] = orient
    transformed = img_rotate(transformed, angle, fill=bg)
    if noise is None:
        return transformed.astype(np.uint8), info
    
    # add noise
    h, w = transformed.shape
    info['noise'] = noise
    info['light'] = light
    noise = generate_noise(max(h, w), light_bias=light, noise_strength=noise)[:h,:w]
    transformed *= (0.5 + noise * 0.5)
    return transformed.astype(np.uint8), info


def generate_empty_sample(image_size: int, noise: float = 0.5) -> np.array:
    """
    generate empty view with noise
    """
    amp = np.random.rand()
    noise = generate_noise(image_size, strength=noise)
    image = np.ones((image_size, image_size)) * 255
    image *= (1 - amp + noise * amp)
    return image.astype(np.uint8)


def generate_sample(source: str, dpi: int = 200,
                    max_skew: int = 10, light: float = 0.5, noise: float = 0.5, perspective: bool = True) -> tuple:
    """
    generate a noisy scan-similar view along with info on applied transform and the content
    """
    index = int(source.split('-').pop())
    doc = '-'.join(source.split('-')[:-1])
    with fitz.open(f'./data/forms/{doc}.pdf') as doc:
        page = doc.load_page(index)
        image, inputs = fill_in_blanks(page, dpi=dpi)
    image, info = random_transform(image, max_skew=max_skew, light=light, noise=noise,
                                   perspective=perspective)
    return image, info, inputs


