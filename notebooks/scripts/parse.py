#!/usr/bin/env python
# coding: utf-8

import os
import fitz
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from PIL import Image
from pathlib import Path
from fitz.fitz import Page
from matplotlib import pyplot as plt
from matplotlib import patches


# target resolution
DPI = 300
# layout data to collect
INFO = ['font-size','italic','bold','cos','sin']
# layout formats
PORT = ['left','top','right','bottom']
LAND = ['top','left','bottom','right']
# interactive forms input data to collect
WIDGETS = ['CheckBox','ComboBox','RadioButton','Text']
WIDGET_DATA = ['field_name','field_value','field_label','field_display','field_type','field_type_string',
               'text_fontsize','text_maxlen','text_format']


# Widget props:

#      'border_color',
#      'border_style',
#      'border_width',
#      'border_dashes',
#      'choice_values',
#      'rb_parent',
#      'field_name',
#      'field_label',
#      'field_value',
#      'field_flags',
#      'field_display',
#      'field_type',
#      'field_type_string',
#      'fill_color',
#      'button_caption',
#      'is_signed',
#      'text_color',
#      'text_font',
#      'text_fontsize',
#      'text_maxlen',
#      'text_format',
#      '_text_da',
#      'script',
#      'script_stroke',
#      'script_format',
#      'script_change',
#      'script_calc',
#      'script_blur',
#      'script_focus',
#      'rect',
#      'xref',
#      'parent',
#      '_annot'


def parse_layout(df: pd.DataFrame, angle: int, width: int, height: int):
    if angle == 90:
        df['left'] = width - df['left']
        df['right'] = width - df['right']
    elif angle == 270:
        df['top'] = height - df['top']
        df['bottom'] = height - df['bottom']


def parse_widgets(page: Page) -> pd.DataFrame:
    """
    get data-inputs information from interactive pdf
    ATT: it sometimes difficult to filter out pdf-utils from user-filled inputs 
    """
    data, count = [], 0
    for widget in page.widgets():
        d = { k:widget.__dict__[k] for k in WIDGET_DATA }
        if type(d['field_label']) == str:
            # some contain \r !?
            d['field_label'] = d['field_label'].encode('unicode-escape').decode('ascii').replace('\\r','')
        d['left'] = int(widget.rect.x0)
        d['top'] = int(widget.rect.y0)
        d['right'] = int(widget.rect.x1)
        d['bottom'] = int(widget.rect.y1)
        data.append(d)
        count += 1
        
    if len(data) == 0:
        return
        
    df = pd.DataFrame.from_dict(data)
    df = df.loc[df[PORT + ['field_type_string']].dropna().index]
    if len(df) == 0:
        # we only interested in the user input
        return

    df['field_value'] = df['field_value'].fillna('').str.strip()
    return df


def parse_lines(page: Page) -> pd.DataFrame:
    """
    get layout info from line-blocks
    there are two types: text and image
    ATT: not the same aggregation level as mode `block` this one is rather `line` then `block`
    mode `dict` provides more layout data, but might be hard to merge with mode `word` output
    sin,cos gives the line orientation in relation to the main content
    """
    columns = PORT + ['text','block-num','block-type'] + INFO
    data = []
    lines = page.get_text('dict')
    for block in lines['blocks']:
        num = block['number']
        if block['type'] == 0:
            # text-line: bbox, text, block-num, block-type, font-size, italic, bold, cos, sin
            for line in block['lines']:
                wmode = line['wmode']
                cos, sin = line['dir']
                for span in line['spans']:
                    italic = span['flags'] & 2**1
                    bold = span['flags'] & 2**4
                    bbox = list(span['bbox'])
                    text = span['text'] #.encode('unicode-escape').decode('ascii')
                    fsize = span['size']

                data.append(bbox + [text, num, 'line', fsize, italic, bold, cos, sin])
                
    return pd.DataFrame(data, columns=columns)


def parse_info(df: pd.DataFrame, scale: int, width: float, height: int, rotation: float) -> pd.DataFrame:
    """
    convert PyMuPDF notation to a page view-port orientation and
    scale bounding boxes in min width/height units (preserving aspect-ratio)
    """
    aspect = width / height
    
    angle = 0
    if len(df[~df['sin'].isna()]) > 0:
        sin, cos = df[~df['sin'].isna()][['sin','cos']].mode().round().iloc[0].to_list()
        if sin == -1 and cos == 0 and rotation == 0:
            angle = 90
            aspect = 1 / aspect
        elif sin == 1 and cos == 0 and rotation == 0:
            angle = 270
            aspect = 1 / aspect
            
    df['aspect-ratio'] = aspect
    df['rotation'] = angle
    # ATT: this is different from `scale` arg which comes from fixed fitz setting
    df['scale'] = min(width, height)
    df[PORT] /= scale
    return df


def parse_content(page: Page) -> pd.DataFrame:
    # extract words only
    columns = (PORT if page.rotation == 0 else LAND) + ['text','block-num','line-num','word-num']
    words = pd.DataFrame(page.get_text('words'), columns=columns)[columns[:-2]]
    words.loc[:,'text'] = words.loc[:,'text'].fillna('').astype(str).str.strip()
    #words.loc[:,'text'] = words.loc[:,'text'].apply(lambda s:s.encode('unicode-escape').decode('ascii'))
    words = words.loc[words['text']!='']
    words['block-type'] = 'word'
    words.loc[:,INFO] = None
    
    # extract layout info: see INFO columns
    lines = parse_lines(page)[words.columns]

    # extract multi-line blocks
    blocks = pd.DataFrame(page.get_text('blocks'), columns=words.columns[:-len(INFO)])
    blocks['block-type'] = blocks['block-type'].apply(lambda x:'image' if x == 1 else 'block')
    blocks.loc[blocks['block-type']=='image','text'] = None
    blocks.loc[:,INFO] = None
            
    df = pd.concat([blocks, lines, words])
    df = df.drop('block-num', axis=1)
                
    n = len(df)
    # filter out invalid bboxes
    df[df[PORT].map(lambda x: isinstance(x, (int, float))).all(axis=1)]
    df.loc[:,PORT] = df.loc[:,PORT].astype(float)
    df = df.loc[(df['right'] > df['left'])&(df['bottom'] > df['top'])]
    if len(df) < n:
        print(f'{source}-{page.number} invalid bbox: {1 - len(df)/n:.2%}')
        return            
    return df    


def extract_pages(source: str, dpi: int = None) -> int:
    """
    convert pages to images with specified resolution (dpi) save as png
    content (text, images, layout) data save in csv
    """
    if not os.path.isfile(f'./data/forms/{source}.pdf'):
        # handle FileNotFoundError
        return
    with fitz.open(f'./data/forms/{source}.pdf') as doc:
        if doc.page_count > 50: return
        num_inputs = 0
        for page in doc:
            pix = page.get_pixmap()
            scale = min(pix.width, pix.height) # not the same as width&height
            # save image
            path = f'{source}-{page.number}'
            page.get_pixmap(dpi=dpi).save(f'./data/images/{path}.png')
            image = Image.open(f'./data/images/{path}.png')
            width, height = image.size
            angle = int(round(page.rotation))
            
            # interactive pdf has inputs info
            widgets = parse_widgets(page)
            if widgets is not None:
                parse_layout(widgets, angle, pix.width, pix.height)
                widgets['page'] = page.number
                widgets.loc[:,PORT] /= min(pix.width, pix.height)
                widgets['source'] = source
                # save widgets
                widgets.to_csv(f'./data/inputs/{path}.csv.gz', index=False,
                               encoding='utf-8', escapechar='\\', compression='gzip')
                num_inputs += len(widgets)
                
            # textual content
            content = parse_content(page)
            if content is not None:
                parse_layout(content, angle, pix.width, pix.height)
                content['num-inputs'] = 0 if widgets is None else len(widgets)
                content = parse_info(content, scale, width, height, page.rotation).reset_index(drop=True)
                content['num-pages'] = doc.page_count
                content['page'] = page.number
                content['source'] = source                            
                # save content
                content.to_csv(f'./data/content/{path}.csv.gz', index=False,
                               encoding='utf-8', escapechar='\\', compression='gzip')
        # return doc info
        return doc.page_count, num_inputs
    

if __name__ == '__main__':
    
    # clear previuos runs
    for x in Path(f'./data/images').glob('*.png'): x.unlink()
    for x in Path(f'./data/content').glob('*.csv.gz'): x.unlink()
    for x in Path(f'./data/inputs').glob('*.csv.gz'): x.unlink()
        
    forms = [str(x)[len('data/forms/'):-len('.pdf')] for x in Path('./data/forms').glob('*.pdf')]
    info = []
    for i, name in enumerate(forms, 1):
        num_pages, num_inputs = extract_pages(name, DPI)
        info.append((name, num_pages, num_inputs))
        print(f'done: {i/len(forms):.2%}', end='\r')
    # save summary
    pd.DataFrame(info, columns=['source','num-pages','num-inputs'])\
      .set_index('source').to_csv('./data/doc-summary.csv.gz',compression='gzip')

