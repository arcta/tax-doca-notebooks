#!/usr/bin/env python
# coding: utf-8

import os
import sys
import cv2
import math
import logging
import numpy as np
import pandas as pd
import pytesseract as ts

from PIL import Image
from pathlib import Path


# target image size 2550 or 300 dpi
SCALE = 2048
# notebooks path
ROOT = f'{os.environ["HOME"]}/notebooks'
# corners TL, BL, BR, TR in the order of 0, 90, 180, 270 rotation labels
ORDER = [(0,0),(1,0),(1,1),(0,1)]


#logging.basicConfig(filename=f'{ROOT}/data/prep.log', filemode='a', format='%(levelname)s %(message)s', level=logging.INFO)


def img_load(path: str) -> np.array:
    """
    Load image data into numpy array 2d matrix -- we only need gray-scale

    Args:
        path (str): source file path

    Returns:
        np.array: 2d matrix
    """
    return cv2.imread(path, cv2.IMREAD_GRAYSCALE)


def img_upscale(imgarr: np.array, scale: int = SCALE) -> tuple:
    """
    Resize if low dpi: should be 300 pixel per inch

    Args:
        imgarr (np.array): 2d matrix (grayscale image)
        scale (int, optional): target size -- defaults to 2048

    Returns:
        tuple: resized image (2d matrix), resize factor (float)
    """
    height, width = imgarr.shape
    factor = float(scale) / width
    if factor > 1:
        size = int(factor * width), int(factor * height)
        return cv2.resize(imgarr, size, interpolation=cv2.INTER_AREA), factor        
    return imgarr, factor


def img_rotate(imgarr: np.array, angle: float, fill: int = None) -> np.array:
    """
    Rotate the image preserving all content (without cropping corners):
    may result in a bigger size

    Args:
        imgarr (np.array): 2d matrix
        angle (float): rotation angle in degrees (0-360)

    Returns:
        np.array: 2d matrix
    """
    height, width = imgarr.shape
    center = (width // 2, height // 2)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    
    # expand to accomodate corners
    radians = math.radians(angle)
    sin, cos = math.sin(radians), math.cos(radians)
    w = int((height * abs(sin)) + (width * abs(cos)))
    h = int((height * abs(cos)) + (width * abs(sin)))
    matrix[0, 2] += ((w/2) - center[0])
    matrix[1, 2] += ((h/2) - center[1])
    
    # rotate and fillup
    if fill is not None:
        return cv2.warpAffine(imgarr, matrix, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=fill)    
    # rotate and fillup with margins backgroung
    return cv2.warpAffine(imgarr, matrix, (w, h), borderMode=cv2.BORDER_REPLICATE)


def img_normalize(imgarr: np.array) -> np.array:
    """
    resampling, denoising, and range-normalization pixel intensity values
    """
    output = imgarr.copy()    
    output, factor = img_upscale(output)
    #output = cv2.fastNlMeansDenoising(output, None, 3, 7, 99)
    #output = cv2.normalize(output, None, 0, 1, cv2.NORM_MINMAX) * 255
    if factor > 1: # low-resolution
        kernel = np.array([[0, -1, 0], [-1, 5,-1], [0, -1, 0]])
        for _ in range(3):
            output = cv2.GaussianBlur(output,(5, 5), 0)
            output = cv2.filter2D(output, ddepth=-1, kernel=kernel)            
    else: # high resolution
        output = cv2.erode(output, np.ones((2, 2), np.uint8), iterations=1)
    # binarize
    return cv2.threshold(output, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]


def detect_skew(gray: np.array, max_angle: int = 45, base_size: int = 256) -> int:
    """
    detect skew based on text; does not detect orientation
    """    
    def calc_score(arr, angle):
        """
        this could be done with cv2 and scipy.ndimage
        PIL, however, delivers the fastest and better result
        """
        data = Image.fromarray(arr).rotate(angle)
        histogram = np.sum(data, axis=1, dtype=float)
        score = np.sum((histogram[1:] - histogram[:-1]) ** 2, dtype=float)
        return histogram, score

    # sample original image down for faster execution
    factor = base_size/max(gray.shape)
    size = tuple((np.array(gray.shape) * factor).astype(int))[::-1]
    test = cv2.resize(gray, size, interpolation=cv2.INTER_AREA)
    test = cv2.threshold(test, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1] 

    # score outcomes in the expected range
    scores = []
    angles = np.arange(-max_angle, max_angle + 1, 1)
    for angle in angles:
        histogram, score = calc_score(test, angle)
        scores.append(score)
    # return the angle with best score
    return angles[scores.index(max(scores))]


def img_deskew(imgarr: np.array, max_angle: int = 45) -> tuple:
    """
    Deskew detects up to 45deg, so we still have to handle cardinal flips
    Tesseract-osd only handles cardinal flips (90, 180, 270)
        
    Correction is driven by text presence and orientation:
    will fail if no text available

    Args:
        imgarr (np.array): 2d matrix

    Returns:
        tuple: 2d matrix (fixed image), detected angle (skew)
    """
    angle = detect_skew(imgarr, max_angle=max_angle)
    if angle is None:
        logging.info('failed deskew')
        return None, None
    
    if angle != 0:
        aligned = img_rotate(imgarr, angle)
    else:
        aligned = imgarr.copy()
    try:
        osd = ts.image_to_osd(aligned, output_type=ts.Output.DICT,
                              config='-l osd --psm 0 -c min_characters_to_try=10')
    except Exception as e:
        logging.info(f'failed osd: {e}')
        return None, -int(round(angle))

    if osd['rotate'] != 0:
        flip = -osd['rotate']
        aligned = img_rotate(aligned, flip)
        return aligned, -int(round(angle + flip))
    
    return aligned, -int(round(angle))


def img_crop_margins(imgarr: np.array, pad: int = 10) -> np.array:
    """
    remove extra white space from the sides:
    we need to be able to locate header/footer areas
    """
    imgarr = cv2.normalize(imgarr, np.zeros(imgarr.shape), 0, 255, cv2.NORM_MINMAX)
    output = 255 * (imgarr >= 128).astype(np.uint8)
    output = cv2.morphologyEx(output, cv2.MORPH_OPEN, np.ones(2, dtype=np.uint8))
    coords = cv2.findNonZero(output)
    x, y, w, h = cv2.boundingRect(coords)
    H, W = imgarr.shape
    return imgarr[max(0, y - pad):min(H + 1, y + h + pad), max(0, x - pad):min(W + 1, x + w + pad)]


def get_bg_value(image: np.array) -> int:
    """
    estimate background color value
    """
    h, w = image.shape
    h, w = h >> 3, w >> 3 # downscale
    output = cv2.resize(image, (w, h), interpolation=cv2.INTER_AREA)
    test = list(output[0,:]) + list(output[:,0]) + list(output[-1,:]) + list(output[:,-1])
    return np.mean(test)


def fit_straight(image: np.array, size_threshold: float = 0.5) -> np.array:
    """
    straighten up and crop based on top countour
    parameters are optimized for the resolution 200dpi+
    """
    d = 2
    h, w = image.shape
    # downscale for speed and consistency
    h, w = h >> d, w >> d
    output = cv2.resize(image, (w, h), interpolation=cv2.INTER_AREA)
    # compute background
    bg = np.mean(list(output[0,:]) + list(output[:,0]) + list(output[-1,:]) + list(output[:,-1]))
    corners = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]])
    output = cv2.GaussianBlur(output, (3, 3), 9)
    output = cv2.erode(output, np.ones((5, 5), np.uint8), iterations=1)
    output = cv2.adaptiveThreshold(output, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, -1)
    # extract countour covering most area
    contours,_ = cv2.findContours(output, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour = max(contours, key=cv2.contourArea)
    if cv2.contourArea(contour) > 0.5 * h * w:
        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.01 * perimeter, True)
        if len(approx) == 4: # got rect -- map points order to corners
            input_points = approx.squeeze()
            norm = sorted(input_points.flatten())[3]
            order = (input_points - norm > 0).astype(int)            
            output_keys = {tuple((x > 0).astype(int)):x for x in corners}
            output_points = [output_keys[tuple(key)] for key in order]
        else: # use bounding box to approximate rect
            X, Y, W, H = cv2.boundingRect(contour)
            input_points = [[X, Y], [X + W - 1, Y], [X + W - 1, Y + H - 1], [X, Y + H - 1]]
            output_points = corners
        # compute transformation
        matrix = cv2.getPerspectiveTransform((np.array(input_points) << d).astype(np.float32),
                                             (np.array(output_points) << d).astype(np.float32))
        return cv2.warpPerspective(image, matrix, (w << d, h << d), flags=cv2.INTER_LINEAR,
                                   borderMode=cv2.BORDER_CONSTANT, borderValue=bg)
    # failed transform -- return original
    return image


### data augmentation func ###
        
        
def generate_noise(dim: int, light_bias: float = 0.5, noise_strength: float = 0.1, scale: int = 1) -> np.array:
    """
    generate square block filled with random noisy gradients
    light-bias: overall level of light -- "white" cannot go darker than that
    noise-strength: controls random uniform jitter 
    """
    x, y = np.random.rand(2)
    x = np.linspace(x, x + scale * np.random.rand(), dim)
    y = np.linspace(y, y + scale * np.random.rand(), dim)
    view = np.empty((dim, dim))
    W = np.random.rand(np.random.randint(2, 5), 2) - 0.5
    W = W / W.sum(axis=0)
    A = (np.random.rand(W.shape[0], 2) - 0.5) * np.random.rand() * 20
    B = (np.random.rand(W.shape[0], 2) - np.random.rand()) * 2 * np.pi    
    X = np.tile(A[:,0], (dim, 1)) * np.tile(x, (A.shape[0], 1)).T
    X = np.sin(X + np.tile(B[:,0], (dim, 1)))
    X = (np.dot(X, W[:,0]) + 1.) * 0.5
    Y = np.tile(A[:,1], (dim, 1)) * np.tile(y, (A.shape[0], 1)).T
    Y =np.sin(Y + np.tile(B[:,1], (dim, 1)))
    Y = (np.dot(Y, W[:,1]) + 1.) * 0.5
    view[:,:] = np.sum(np.array(np.meshgrid(X, Y)), axis=0)
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
    """generate empty view with noise"""
    amp = np.random.rand()
    noise = generate_noise(image_size, strength=noise)
    image = np.ones((image_size, image_size)) * 255
    image *= (1 - amp + noise * amp)
    return image.astype(np.uint8)


def simulate_quadrant_views(samples: list, test: list, image_size: int,
                            quadrants: pd.DataFrame, num_variations: int = 1) -> None:    
    save_path = './data/layout-baseline'
    # clearup previous proc
    for x in Path(save_path).glob('*.png'): x.unlink()

    labels = []
    slices = [slice(None, image_size), slice(-image_size, None)]
    # qudrants transition order with rotation
    order = ORDER * 2
    failure = 0
    for k in range(num_variations):
        for n, source in enumerate(samples):
            # load image along with pdf data and quadrants data; and make a noisy variation
            image, info = random_transform(img_load(source), max_skew=5, noise=0.5, perspective=False)
            extra = quadrants[quadrants['source']==source]
            
            # try to correct  the view
            angle = detect_skew(image, max_angle=10)
            if abs(angle - info['skew']) > 5:
                failure += 1
                continue
            image = img_rotate(image, angle)
            image = fit_straight(image)

            # scale down to one of two options and invert
            scale = image_size * 4 if np.random.rand() > 0.5 else image_size * 2
            size = tuple((np.array(image.shape) * scale/min(image.shape)).astype(int))[::-1]
            layout = cv2.bitwise_not(cv2.resize(image, size, interpolation=cv2.INTER_AREA))

            path = f"{save_path}/{source[:-4].split('/').pop()}"
            in_test = int(source in test)
            # get corners views with labels
            for i,(r,c) in enumerate(order[4:]):
                pos = order[i + 4 - info['orient']//90] # translate rotation to retrieve the proper quadrant
                cv2.imwrite(f'{path}-{i}-{r}{c}.png', layout[slices[r],slices[c]])
                x1, x2, x3 = extra.loc[extra['quadrant']==str(pos),['content','lines','words']].values[0]
                labels.append({
                    'path':f'{path}-{i}-{r}{c}.png',
                    'quadrant':pos,
                    'content':x1,
                    'lines':x2,
                    'words':x3,
                    'scale':scale,
                    'orient':info['orient'],
                    'skew':info['skew'],
                    'corrected':angle,
                    'test':in_test,
                })

            print(f'generating data: {(k * num_variations + n)/len(samples)/num_variations:.2%}', end='\r')
    print(f'skipped {failure/(len(labels) + failure):.2%} due to correction failure')
    pd.DataFrame.from_dict(labels).to_csv(f'{save_path}/labels.csv.gz', index=False, compression='gzip')



if __name__ == '__main__':
    
    if len(sys.argv) == 3:        
        # preprocess clean images which came from converting pdf
        _, source, output = sys.argv
        print(f'processing {source} saving as {output}')
        imgarr = img_load(source)
        imgarr = img_normalize(imgarr)
        imgarr, angle = img_deskew(imgarr)
        if not imgarr is None:
            imgarr = img_crop_margins(imgarr)
            cv2.imwrite(output, imgarr)
        else:
            logging.info(f'{source} failed...')
