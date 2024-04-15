#!/usr/bin/env python
# coding: utf-8

# # OCR: basic data processing

#import os
import re
import cv2
#import json
import string
import pandas as pd
import numpy as np

from PIL import Image
from matplotlib import pyplot as plt
from matplotlib import patches
from pathlib import Path
from sklearn.neighbors import KDTree
from unidecode import unidecode


# const
CHAR = 1
CELL = 2
HLINE = 3
VLINE = 4
BLOCK = 5
INPUT = 6
SPACE = 7

BOX = ['left','top','width','height']



def resample(image, f):
    if f == 1:
        return image
    h, w = image.shape[:2]
    return cv2.resize(image, (max(int(round(w/f)), 1), max(int(round(h/f)), 1)), cv2.INTER_AREA)
    

def binarize(image, kernel):
    image = cv2.erode(image, np.ones(kernel, np.uint8), iterations=1)
    return cv2.threshold(image, 0, 255, cv2.THRESH_BINARY)[1]


def sharpen(image):
    return cv2.filter2D(image, -1, np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]]))


def load_image(source, kernel=(2, 2)):
    return binarize(cv2.imread(f'./data/images/{source}.png', cv2.IMREAD_GRAYSCALE), kernel=kernel)


def load_info(source):
    content = pd.read_csv(f'./data/content/{source}.csv.gz')
    if content.loc[0,'num-inputs'] == 0:
        return content, None
    inputs = pd.read_csv(f'./data/inputs/{source}.csv.gz')
    return content, inputs


def visual_check(output, visual, figsize=(8, 8)):
    if not visual: return
    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(output, 'gray')
    plt.show()


def extract_lines(image, scale=(20, 10)):
    dv, dh = image.shape[0]//scale[0], image.shape[1]//scale[1]
    output = cv2.erode(image, np.ones((5, 5), np.uint8), iterations=1)
    output = cv2.bitwise_not(sharpen(output))
    output = cv2.adaptiveThreshold(output, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, -1)
    # create images to hold extracted lines
    v, h = np.copy(output), np.copy(output)
    vs = cv2.getStructuringElement(cv2.MORPH_RECT, (1, dv//2))
    hs = cv2.getStructuringElement(cv2.MORPH_RECT, (dh, 1))
    v, h = cv2.erode(v, vs), cv2.erode(h, hs)
    v, h = cv2.dilate(v, vs), cv2.dilate(h, hs)
    return v, h


def make_convex(grid, kernel=(7, 7)):
    convex = grid.copy()
    indices = np.array(grid.nonzero())
    if len(indices[1]) == 0:
        return convex
    x0, y0 = np.min(indices, axis=1)
    x1, y1 = np.max(indices, axis=1)
    convex[x0:x0 + kernel[0], y0:y1] = 255
    convex[x1 - kernel[0]:x1, y0:y1] = 255
    convex[x0:x1, y0:y0 + kernel[1]] = 255
    convex[x0:x1, y1 - kernel[1]:y1] = 255
    return convex


def extract_cells(grid, f=2, kernel=(5, 5), convex=False, visual=False):
    if grid.shape[0] < 4 * f or grid.shape[1] < 4 * f:
        return []
    output = 255 - (make_convex(grid, kernel=kernel) if convex else grid)
    output = resample(output, f)
    output = cv2.erode(output, np.ones(kernel, np.uint8), iterations=1)
    output = cv2.threshold(output, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    visual_check(output, visual)
    c, h = cv2.findContours(output, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # return bounding-box and parent-ref
    #return [np.append((np.array(cv2.boundingRect(c[i])) * f), h[0][i][-2:]) for i in range(len(c))]
    return [np.array(cv2.boundingRect(c)) * f for c in c]


def filter_boxes(boxes, shape, scale=13, reverse=True):
    """
    keep only one level hierarchy; either parents or the children
    """
    if len(boxes) < 2:
        return boxes
    order = np.array(sorted(boxes, key=lambda b:b[2]*b[3], reverse=reverse))
    matrix = np.zeros((shape[0]//scale, shape[1]//scale))
    x, y, w, h = order[0][:4]//scale
    matrix[y:y + h,x:x + w] = 1
    filtered = [order[0]]
    for b in np.array(order)[1:]:
        x, y, w, h = b[:4]//scale
        # drop overlap
        if np.sum(matrix[y:y + h,x:x + w]) > 0:
            continue
        matrix[y:y + h,x:x + w] = 1
        filtered.append(b)
    return filtered


def extract_blocks(mask, f=4, kernel=(11, 11), visual=False):
    """
    extract high-level structural elements
    from the original (whight background) image fragment
    """
    if mask.shape[0] < 4 * f or mask.shape[1] < 4 * f:
        return np.array([])
    output = cv2.erode(mask, np.ones(kernel, np.uint8), iterations=1)
    output = resample(output, f)
    output = cv2.erode(output, np.ones(kernel, np.uint8), iterations=1)
    output = cv2.threshold(output, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    visual_check(output, visual)
    contours,_ = cv2.findContours(output, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return np.array([cv2.boundingRect(c) for c in contours]) * f


def block_order(blocks):
    if len(blocks) <  2:
        return blocks
    order, visited = list(), set()
    point = (0, 0)
    for n in range(len(blocks)):
        d,i = block_search.query([point], k=len(blocks))
        for k in range(len(blocks)):
            if not i[0][k] in visited:
                order.append(blocks[i[0][k]])
                point = (order[-1][0], order[-1][1] + order[-1][3])
                visited.add(i[0][k])
                break
    return order


def extract_textlines(mask, f=3, kernel=(1, 11), threshold=20, visual=False):
    if mask.shape[0] < 4 * f or mask.shape[1] < 4 * f:
        return np.array([])
    output = resample(mask, f)
    output = cv2.erode(output, np.ones(kernel, np.uint8), iterations=1)
    output = cv2.threshold(output, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    visual_check(output, visual)
    c, h = cv2.findContours(output, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    C = [cv2.boundingRect(c[i]) for i in range(len(c))]
    return np.array([box for box in C if box[-1] > threshold/f]) * f


def extract_words(mask, f=1, kernel=(1, 11), threshold=100, visual=False):
    if mask.shape[0] < 4 * f or mask.shape[1] < 4 * f:
        return np.array([])
    #output = 255 - mask if np.median(mask) == 0 else mask # catch inverce boxes
    output = resample(mask, f)
    output = cv2.erode(output, np.ones(kernel, np.uint8), iterations=1)
    output = cv2.threshold(output, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    visual_check(output, visual)
    contours, _ = cv2.findContours(output, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return np.array([cv2.boundingRect(c) for c in contours if cv2.contourArea(c) > threshold]) * f


#def extract_tokens(mask, f=0.1, kernel=(17, 1), visual=False): upsampling sometimes makes sense
def extract_tokens(mask, f=1, kernel=(11, 1), visual=False):
    if mask.shape[0] < 4 * f or mask.shape[1] < 4 * f:
        return []
    #output = 255 - mask if np.median(mask) == 0 else mask # catch inverce boxes
    output = cv2.erode(mask, np.ones(kernel, np.uint8), iterations=1)
    output = cv2.threshold(output, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    visual_check(output, visual, figsize=(4, 4))
    c, h = cv2.findContours(output, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return np.array([cv2.boundingRect(c[i]) for i in range(len(c))])


def get_grid(image, lt=20, m=4):
    # to catch vertical ticks the scale should be more detailed (30, 50)
    V, H = extract_lines(image)
    grid = V + H
    cells = extract_cells(grid, f=2, kernel=(5, 5), convex=True)
    # exclude lines add margin
    cells = [(x - m, y - m, w + m * 2, h + m * 2) for x, y, w, h in cells if w >= 20 and h >= lt]
    # remove parents
    if len(cells) > 0:
        cells = filter_boxes(cells, image.shape, reverse=False)
        cells = pd.DataFrame(cells, columns=BOX).sort_values(['top','left']).reset_index(drop=True)
        cells['type'] = CELL
    else:
        cells = pd.DataFrame([], columns=BOX)

    V, H = extract_lines(image, scale=(20, 10))
    vlines = pd.DataFrame(extract_cells(V, convex=False), columns=BOX)
    if len(vlines) > 0:
        vlines['type'] = VLINE

    hlines = pd.DataFrame(extract_cells(H, convex=False), columns=BOX)
    if len(hlines) > 0:
        hlines['type'] = HLINE
        
    # remove all line from the image for the further text extraction
    # mask will also serve as a reading map
    mask = image.copy()
    mask[V.nonzero()] = 255
    mask[H.nonzero()] = 255        
    return mask, vlines, hlines, cells


def read_line(mask, box, bi, li, wi, lh):
    X, Y, W, H = box
    # height adjusted for failed lines
    C = [[X + x, Y, w, H if H < lh * 1.25 else h + y, bi, li]
         for x, y, w, h in extract_words(mask[Y:Y + H,X:X + W])]
    # in the line: words only ordered by `left` (x)
    words = sorted(C, key=lambda c:c[0])        
    tokens = []
    for i, (X, Y, W, H, bi, li) in enumerate(words, wi):
        C = [[X + x, Y, w, H, bi, li, i] for x, y, w, h in extract_tokens(mask[Y:Y + H,X:X + W])]
        # in the word: tokens only ordered by `left` (x)
        tokens += sorted(C, key=lambda c:c[0])
        # remove processed word from the map
        mask[Y - 1:Y + H + 2,X:X + W] = 255
    return words, tokens


def extract_layout(image, lh=40, stop=3):
    """
    Extract layout features and visual tokens
    """    
    mask, vlines, hlines, cells = get_grid(image)
    
    lines, words, tokens = [], [], []
    
    if len(cells) > 0:
        for bi in cells.index:
            X, Y, W, H = cells.loc[bi, BOX].values
            ln = [[X + x - 1, Y + y - 1, w + 1, h + 1]
                  for x, y, w, h in extract_textlines(mask[Y:Y + H,X:X + W])]
            # in the block: lines only ordered by `top` (y)
            for li, box in enumerate(sorted(ln, key=lambda l:l[1]), len(lines)):
                wrd, tkn = read_line(mask, box, bi, li, len(words), lh)
                words += wrd
                tokens += tkn
                lines.append(box + [bi])

    # extract text blocks from the mask residual
    boxes = pd.DataFrame()
    blocks = extract_blocks(mask)
    while len(blocks) > 0 and stop > 0:
        # make sure we do not lose any start from the smallest
        blocks = sorted(blocks, key=lambda b:b[2] * b[3], reverse=True)
        for bi, (X, Y, W, H) in enumerate(blocks, len(cells) + len(boxes)):
            ln = [[X + x - 1, Y + y - 1, w + 1, h + 1]
                  for x, y, w, h in extract_textlines(mask[Y:Y + H,X:X + W])]
            # in the block: lines only ordered by `top` (y)
            for li, box in enumerate(sorted(ln, key=lambda l:l[1]), len(lines)):
                wrd, tkn = read_line(mask, box, bi, li, len(words), lh)
                words += wrd
                tokens += tkn
                lines.append(list(box) + [bi])

        blocks = pd.DataFrame(blocks, columns=BOX)
        blocks['type'] = BLOCK
        boxes = pd.concat([boxes, blocks])
        blocks = extract_blocks(mask)
        stop -= 1
    
    boxes = pd.concat([cells, boxes, vlines, hlines], ignore_index=True)

    # mask should be blank at this point
    success = len((255 - mask).nonzero()) < 10
    
    tokens = pd.DataFrame(tokens, columns=BOX + ['block-index','line-index','word-index'])
    tokens = tokens.drop_duplicates(subset=BOX, keep='last')
    tokens['type'] = CHAR
    
    lines = pd.DataFrame(lines, columns=BOX + ['block-index'])
    words = pd.DataFrame(words, columns=BOX + ['block-index','line-index'])
    return tokens, words, lines, boxes, success


def run_checkup(tokens, image):
    H, W = image.shape
    result = np.ones((H//4, W//4))
    order = 0
    c = tokens.columns.tolist().index('index')
    for i in range(len(tokens)):        
        x, y, w, h, t = tokens[BOX + ['type']].values[i,:]        
        if t == CHAR:
            if result[y//4 + 1,x//4 + 1] == 0:
                tokens.iloc[i,c] = -1
                continue
            result[y//4:(y + h)//4 + 1,x//4:(x + w)//4 + 1] = 0
        tokens.iloc[i,c] = order
        order += 1
    return result


# rough relative char width estimation
def map_char_width(estimated_line_height):
    scale = cv2.getFontScaleFromHeight(cv2.FONT_HERSHEY_SIMPLEX, estimated_line_height, 1)
    (w, h), baseline = cv2.getTextSize('test', cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
    #print(f'Estimated fontsize: {estimated_line_height}  scale: {scale:.2f}  baseline-height: {baseline}')
    return { c:cv2.getTextSize(c, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)[0][0]/estimated_line_height
             for c in list(string.printable)[:-5]} # tail dropped: \t\n\r\x0b\x0c


def plot_info(source):
    image = load_image(source)
    info = pd.read_csv(f'./data/content/{source}.csv.gz')
    inputs = pd.read_csv(f'./data/inputs/{source}.csv.gz') if info['num-inputs'].iloc[0] > 0 else None

    fig, ax = plt.subplots(figsize=(12, 12))
    colors = {'word':'gold','block':'cyan','image':'magenta','input':'crimson'}
    for label, color in colors.items():
        ax.plot([100], [100], color=color, label=label)

    ax.imshow(image, 'gray')
    columns = ['left','top','right','bottom','scale','block-type','text']
    for row in info[columns].values:
        x, y, x1, y1, s = row[:5].astype(float)
        c = colors.get(row[5], 'cyan')
        w, h = int((x1 - x) * s), int((y1 - y) * s)
        x, y = x * s, y * s
        d = ['word','line','block'].index(row[5]) * 2
        ax.add_patch(patches.Rectangle((x - d, y - d), w + d, h + d,
                                           linewidth=1, edgecolor=c, facecolor='none'))
    if inputs is not None:
        columns = ['left','top','right','bottom','field_value','field_type_string','field_display']    
        for row in inputs[columns].values:
            x, y, x1, y1 = row[:4].astype(float)
            w, h = int((x1 - x) * s), int((y1 - y) * s)
            x, y = x * s, y * s
            ax.add_patch(patches.Rectangle((x, y), w, h, linewidth=1,
                                               edgecolor=colors['input'], facecolor='none'))

    ax.set_title(f'Layout extracted from PDF')
    ax.legend(bbox_to_anchor=(1, 0), ncol=5, frameon=False)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.show()
    
    
def get_words(content):
    scale = int(content.loc[0,'scale'])
    columns = ['text','left','top','right','bottom','sin']
    words = content.loc[(content['block-type']=='word')&(~content['text'].isna()), columns]
    words['rotation'] = words['sin'].apply(lambda x:0 if np.isnan(x) else np.rad2deg(np.arcsin(1)))
    words.loc[:,['left','top','right','bottom']] = (words[['left','top','right','bottom']] * scale).round()
    return words.loc[:,['left','top','right','bottom','rotation','text']]

    
def get_chars(words, size_map):
    """
    based on generic font aspect ratio
    estimate individual characters bounding boxes
    within the given word bounding box
    """
    chars = words[BOX + ['index']].copy()
    chars['label'] = words['text'].apply(list)
    chars = chars.explode('label', ignore_index=True)
    
    def calc_width(r):
        if unidecode(r['label']) in size_map:
            return size_map[unidecode(r['label'])] * r['height']
        return r['width']
    chars['char-width'] = chars.apply(calc_width, axis=1)
    
    A = chars[['char-width','index']].groupby('index').sum()
    B = chars[['width','index']].groupby('index').mean()
    f = pd.Series((B.values/A.values).flatten(), index=A.index).rename('factor').reset_index()
    chars = chars.merge(f, on=('index'))
    chars = chars.loc[(chars['width'] > 0)&(chars['height'] > 0)]
    chars['char-width'] = (chars['char-width'] * chars['factor']).astype(int)
    
    chars = chars.loc[:,['left','top','char-width','height','label','index']]
    chars.columns = BOX + ['label','word-index']
    shift = chars.groupby('word-index')['width'].cumsum().values.tolist()
    chars['left'] += shift - chars['width']    
    chars['type'] = CHAR
    return chars


def load_labels(source: str, line_norm : int):
    content, inputs = load_info(source)

    scale = int(content.loc[0, 'scale'])
    rotation = content.loc[0, 'rotation']

    words = get_words(content)
    # filter out wrohg orientation
    words = words.loc[words['rotation']==0,:]
    words['width'] = words['right'] - words['left']
    words['height'] = words['bottom'] - words['top']
    columns = BOX + ['text','rotation','right']
    words = words.loc[:,columns].reset_index()
            
    size_map = map_char_width(line_norm)    
    chars = get_chars(words, size_map)
    chars['level'] = (chars['top'] + chars['height']//2).astype(int)
    chars['order'] = (chars['left'] + chars['width']//2).astype(int)
    chars['index'] = list(range(len(chars)))
    chars = chars.set_index(['level','order'])

    words = words.set_index('index')
    if inputs is None:
        return words, chars, None
                      
    columns = ['field_type_string','field_value','field_display','left','top','right','bottom']
    inputs = inputs.loc[:, columns]
    inputs.loc[:,['left','top','right','bottom']] *= scale
    inputs['width'] = inputs['right'] - inputs['left']
    inputs['height'] = inputs['bottom'] - inputs['top']
    inputs = inputs.loc[(inputs['width'] > 0)&(inputs['height'] > 0)]
    inputs['level'] = (inputs['top'] + inputs['height']//2).astype(int)
    inputs['order'] = (inputs['left'] + inputs['width']//2).astype(int)
    inputs['index'] = list(range(len(chars), len(chars) + len(inputs)))
    inputs['label'] = inputs['field_type_string']
    inputs['type'] = INPUT
    inputs = inputs.drop(['field_type_string','right','bottom'], axis=1).set_index(['level','order'])
    inputs['word-index'] = -1
    chars = pd.concat([chars, inputs[inputs['label'].isin(['CheckBox','RadioButton'])][chars.columns]])
    return words, chars, inputs
    

def no_match(tok, obj):
    x, y, w, h = tok[BOX].values
    x1, y1, w1, h1 = obj[BOX].values
    # boxes do not overlap
    if (x > x1 + w1 or x1 > x + w) or (y > y1 + h1 or y1 > y + h):
        return True
    # boxes are way too different in size
    if h > 2 * h1 or w > 2 * w1 or w < w1//3:
        return True
    return False


def estimate_proba(tok, obj):
    # pdf box is bigger, so we take token coverage as estimate
    if no_match(tok, obj):
        return 0.
    x, y, w, h = tok[BOX].values
    x1, y1, w1, h1 = obj[BOX].values
    # rough estimate based on the overlap area
    return ((min(x1 + w1, x + w) - max(x1, x)) * (min(y1 + h1, y + h) - max(y1, y)))/(w * h)


def get_neighbors(index, chars):
    if index is None:
        return None, None
    # neighbors in the same word <wi> if any
    left, right = ' ', ' ' # space for nothing
    # chars are ordered by line-index, word-index, left
    if index > 0 and chars.iloc[index - 1]['word-index'] == chars.iloc[index]['word-index']:
        left = chars.iloc[index - 1]['label']
    if index < len(chars) - 1 and chars.iloc[index + 1]['word-index'] == chars.iloc[index]['word-index']:
        right = chars.iloc[index + 1]['label']
    return left, right


def run_labeling(tokens, image, words, chars, inputs):
    char_search = KDTree(list(chars.index))
    input_search = KDTree(list(inputs.index)) if inputs is not None else None
    
    output = []
    for i in range(len(tokens)):
        tok, obj = tokens.iloc[i], None
        l, o = tokens.index[i]
        x, y, w, h, t = tok[BOX + ['type']].values

        if t == CHAR:
            d, j = char_search.query([(l, o)], k=1)
            index = j[0][0]
            obj = chars.iloc[index]

        elif input_search is not None:
            d, j = input_search.query([(l, o)], k=1)
            index = j[0][0]
            obj = inputs.iloc[index]
            
        keys = ['type','label','proba','index','left-side','right-side']
        if obj is None:
            # Failed to find match...
            output.append({key:None for key in keys})
        else:
            out = {}
            out['proba'] = estimate_proba(tok, obj)
            out['left-side'], out['right-side'] = get_neighbors(index, chars)
            output.append({key:out[key] if key in out else obj[key] for key in keys})
    # return as data-frame
    return pd.DataFrame.from_dict(output)


def get_block_text(tokens, block_index):
    chars = tokens[(tokens['block-index']==block_index)&(tokens['type']==CHAR)]
    # aggregate words
    text = chars.groupby(['line-index','word-index']).sum()['label']
    # aggregate lines
    text = text.apply(lambda x:f'{x} ').groupby(['line-index']).sum()
    # aggregate block
    return text.apply(lambda x:f'{x}\n').sum()


def get_ctegory(label):
    if label in string.digits: # digit
        return 0
    label = unidecode(label)
    if label in string.ascii_letters: # letter
        if label in string.ascii_letters[:26]:
            return 1  # lower-case
        return 2  # upper-case
    return 3 # symbol


# in our corput those looks identical
def stem_label(s):
    if  s is None or pd.isnull(s):
        return
    s = str(s)
    if s in '–—−':
        return '-'
    if s in '×✗':
        return "x"
    if s in ('CheckBox','RadioButton'):
        return '☐'
    if s == '”':
        return '"'
    if s == '’':
        return "'"
    if s == '●':
        return "•"
    if s == '►':
        return "▶"     
    if s == '◄':
        return "◀"     
    return s


def main(source):    
    image = load_image(source)
    if np.min(image) == 255:
        return
    objects = extract_layout(image)
    if objects is None:
        return
    
    tokens, words, lines, boxes, success = objects    
    # remove duplicates
    tokens['index'] = None
    run_checkup(tokens, image)
    tokens = tokens.loc[tokens['index']!=-1,:].drop('index', axis=1)    
    # set center-point reference for matching label
    tokens['level'] = (tokens['top'] + tokens['height']//2).astype(int)
    tokens['order'] = (tokens['left'] + tokens['width']//2).astype(int)
    boxes.loc[:,['block-index','line-index','word-index']] = -1
    boxes['level'] = (boxes['top'] + boxes['height']//2).astype(int)
    boxes['order'] = (boxes['left'] + boxes['width']//2).astype(int)
    # lift center-point for underscore-lines to match the input space
    LN = int(tokens['height'].median())
    boxes.loc[boxes['type'] == HLINE,'level'] -= LN//2
    tokens = pd.concat([tokens, boxes[tokens.columns]]).set_index(['level','order'])
    # match labels
    words, chars, inputs = load_labels(source, LN)    
    result = run_labeling(tokens, image, words, chars, inputs)
    tokens.loc[:, result.columns] = result.values
    return tokens, success


if __name__ == '__main__':
    
    labels, result = [], []
    for i, source in enumerate(samples):
        output = main(source)
        if output is None:
            continue
        tokens, success = output
        tokens['label'] = tokens['label'].apply(stem_label)
        # keep only those with reliable label
        proba = tokens['proba'].mean()
        tokens = tokens.loc[tokens['proba'] > proba * 1.25]
        # save for training
        tokens.to_csv(f'./data/extract/{source}.csv.gz', index=False, compression='gzip')
        # get in-doc counts
        count = tokens.groupby('label').size()
        # append to adjusted-in-all
        labels += count.index.tolist()
        print(f"{i:>4} {source:<20} {len(tokens):<5} {proba:.2%}")
        result.append({'source':  source,
                       'samples': len(tokens),
                       'labels':  len(count),
                       'proba':   proba,
                       'success': success})

    labels = pd.DataFrame(Counter(labels).most_common(), columns=['label','count'])
    # separate `char`-level tokens (including CheckBox and Radio)
    labels['len'] = labels['label'].str.len()
    labels = labels.sort_values(['len','count'], ascending=[True, False]).drop('len', axis=1)
    # calculate label weight for training
    labels['weight'] = (1./labels['count']).round(4).values
    index = pd.DataFrame.from_dict(result)
    # set train-test split
    index['test'] = (np.random.rand(len(index)) > 0.9).astype(int)

    labels.to_csv('./data/training-labels.csv', index=False)
    index.to_csv('./data/source-index.csv', index=False)
    
    index['filter'] = index.apply(lambda d:int(d['proba'] > 0.6 and d['labels'] >= 16), axis=1)
    index = index.loc[index['filter'] > 0].drop('filter', axis=1)
    index.to_csv('./data/training-index.csv', index=False)

