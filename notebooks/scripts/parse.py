#!/usr/bin/env python
# coding: utf-8

import os
import cv2
import fitz
import json
import pandas as pd
import numpy as np

from PIL import Image, ImageOps
from matplotlib import pyplot as plt
from matplotlib import patches
from pathlib import Path
from IPython.display import display, Javascript, HTML
from fitz.fitz import Page


# target resolution
DPI = 200
# filter out instructional booklets
MAX_PAGES = 35
# layout data to collect
INFO = ['font-size','italic','bold','cos','sin']
# layout formats
PORT = ['left','top','right','bottom']
LAND = ['top','left','bottom','right']
# interactive forms input data to collect
WIDGETS = ['CheckBox','RadioButton','ComboBox','Text']
WIDGET_DATA = ['field_name','field_label','field_display','field_type','field_type_string',
               'text_fontsize','text_maxlen','text_format','border_width']
# notebooks path
ROOT = f'{os.environ["HOME"]}/notebooks'
# view qadrants TL, BL, BR, TR in the order of 0, 90, 180, 270 rotation labels
ORDER = [(0,0),(1,0),(1,1),(0,1)]


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


def parse_widgets(page: Page, width: int, height: int) -> pd.DataFrame:
    """
    get data-inputs information from interactive pdf
    ATT: it sometimes difficult to filter out pdf-utils from user-filled inputs 
    """
    data = []
    for field in page.widgets():
        d = { k:field.__dict__[k] for k in WIDGET_DATA }
        d['left'] = field.rect.x0
        d['top'] = field.rect.y0
        d['right'] = field.rect.x1
        d['bottom'] = field.rect.y1
        data.append(d)
        
    if len(data) == 0:
        return
    
    data = pd.DataFrame.from_dict(data)
    data = data.loc[data[PORT + ['field_type_string']].dropna().index]
    if len(data[(data['field_type_string'].isin(WIDGETS))&(data['field_display']==0)]) == 0:
        # we only interested in the user input
        return
                    
    if page.rotation == 90:
        data['left'] = width - data['left']
        data['right'] = width - data['right']
    elif page.rotation == 270:
        data['top'] = height - data['top']
        data['bottom'] = height - data['bottom']
                        
    data['page'] = page.number
    data.loc[:,PORT] /= min(width, height)            
    return data


def parse_lines(page: Page) -> pd.DataFrame:
    """
    get layout info from line-blocks
    there are two types: text and image
    ATT: not the same aggregation level as mode `block` this on is rather `line` then `block`
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
                    d = bbox + [span['text'], num, 'line'] + [span['size'], italic, bold, cos, sin]
                data.append(d)
    return pd.DataFrame(data, columns=columns)


def extract_pages(source: str, dpi: int = None, max_pages: int = MAX_PAGES) -> int:
    """
    convert pages to images with specified resolution (dpi) save as png
    content (text, images, layout) data save in csv
    """
    if not os.path.isfile(f'{ROOT}/data/forms/{source}.pdf'):
        return
    with fitz.open(f'{ROOT}/data/forms/{source}.pdf') as doc:
        if doc.page_count > max_pages: # skip if too many pages
            print(f'skipping {source}: {doc.page_count} pages...')
            return doc.page_count

        for page in doc:
            pix = page.get_pixmap()
            scale = min(pix.width, pix.height) # not the same as width&height
            path = f'{ROOT}/data/images/{source}-{page.number}.png'
            page.get_pixmap(dpi=dpi).save(path)
            image = Image.open(path)
            width, height = image.size
            
            # interactive pdf has data-inputs info
            widgets = parse_widgets(page, pix.width, pix.height)

            # extract words only
            columns = (PORT if page.rotation == 0 else LAND) + ['text','block-num','line-num','word-num']
            words = pd.DataFrame(page.get_text('words'), columns=columns)[columns[:-2]]
            words.loc[:,'text'] = words.loc[:,'text'].fillna('').astype(str).str.strip()
            words = words.loc[words['text']!='']
            # filter out Acrobat-Reader failure
            #if page.number == 0 and doc.page_count == 1:
            #    text = '\n'.join(words['text'].to_list())
            #    if text.find('The document you are trying to load requires Adobe Reader') != -1 \
            #        or text.find('Please wait') != -1:
            #        print(f'{source} Acrobat-Reader failure...')
            #        return            
            # to merge in the layout info
            words['block-type'] = 'word'
            words.loc[:,INFO] = None
            # extract layout info: see INFO columns
            lines = parse_lines(page)[words.columns] #[PORT + ['text','block-num','block-type'] + INFO]

            # extract high level (multi-line) aggregation blocks
            blocks = pd.DataFrame(page.get_text('blocks'), columns=words.columns[:-len(INFO)])
            blocks['block-type'] = blocks['block-type'].apply(lambda x:'image' if x == 1 else 'block')
            blocks.loc[blocks['block-type']=='image','text'] = None
            blocks.loc[:,INFO] = None
            
            df = pd.concat([blocks, lines, words])
            df = df.drop('block-num', axis=1)
            
            if page.rotation == 90:
                df['left'] = pix.width - df['left']
                df['right'] = pix.width - df['right']
            elif page.rotation == 270:
                df['top'] = pix.height - df['top']
                df['bottom'] = pix.height - df['bottom']
                
            n = len(df)
            # filter out invalid bboxes
            df[df[PORT].applymap(lambda x: isinstance(x, (int, float))).all(axis=1)]
            df.loc[:,PORT] = df.loc[:,PORT].astype(float)
            df = df.loc[(df['right'] > df['left'])&(df['bottom'] > df['top'])]
            if len(df) < n:
                print(f'{source}-{page.number} invalid bbox: {1 - len(df)/n:.2%}')
                #os.unlink(f'{ROOT}/data/forms/{source}.pdf')
                return
            
            df = parse_info(df, scale, width, height, page.rotation).reset_index(drop=True)
            # filter out line-level duplicates brought by appending blocks (some blocks are lines)
            #boxes = df.sort_values('block-type', ascending=False)
            #boxes = boxes.loc[boxes['block-type'].isin(['block','line']), PORT]
            #boxes.loc[:,PORT] = np.round(boxes.loc[:,PORT] * 1000).astype(int)
            #n = len(boxes)
            #boxes = boxes.drop_duplicates(keep='first')
            #drop = df[(df['block-type']=='block')&(~df.index.isin(boxes.index))].index
            #if len(drop) < n - len(boxes):
            #    print(f'duplicate blocks: dropped {len(drop)} of {n - len(boxes)} ...')
            #df = df.loc[~df.index.isin(drop)]
            
            df['page'] = page.number
            df['num-pages'] = doc.page_count
            df['source'] = source
            df.to_csv(f'{ROOT}/data/info/{source}-{page.number}.csv.gz', index=False, compression='gzip')
            if widgets is not None:
                widgets['source'] = source
                widgets.to_csv(f'{ROOT}/data/inputs/{source}-{page.number}.csv.gz', index=False, compression='gzip')
        # numer of pages
        return doc.page_count
    

def extract(df: pd.DataFrame, dpi: int = 300) -> float:
    """
    run extraction for all pdfs in the data-frame and update it with extracted page count
    """
    for i in df.index:
        d = df.loc[i].to_dict()
        if type(d['file']) == str:
            name = d['file'][:-4] # remove .pdf extension
            try:
                count = extract_pages(name, dpi=dpi)
            except Exception as e:
                print(f'failed extract {name}...', e)
            else:
                df.loc[i,'pages'] = count
                df.to_csv(f'{ROOT}/data/forms.tmp.csv', index=False)
            print(f'done: {i/len(df):.2%}', end='\r')
    return len(df[~df['pages'].isna()])/len(df)

    
def extract_boxes(image: np.array, kernel_size: int = 5, min_height: int = 0) -> pd.DataFrame:
    """
    use cv2 to get bounding boxes of all "visible" artifacts:
    logos, checkboxes etc. might be missing from the non-interactive PDF extracted info
    ATT: the setting is not universal and sensitive to the targeted dataset
    """
    output = image
    output = cv2.bitwise_not(output)
    output = cv2.GaussianBlur(output, (5, 5), 0)
    output = cv2.adaptiveThreshold(output, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 5, -2)    
    output = cv2.dilate(output, cv2.getStructuringElement(cv2.MORPH_RECT, (9, 5)), iterations=2)
    output = cv2.dilate(output, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 9)), iterations=2)
    contours,_ = cv2.findContours(output, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    boxes = sorted([cv2.boundingRect(cnt) for cnt in contours], key=lambda x:x[2]*x[3], reverse=True)[1:]
    boxes = [[x, y, w, h] for x, y, w, h in boxes if h >= min_height]
    return pd.DataFrame(boxes, columns=['left','top','width','height'])


def extract_lines(image: np.array, units: int = 10) -> tuple:
    """
    extract long vertical and horizontal lines if any
    length threshold specified by relative units: fraction of the image size
    return as separate images
    """
    dv, dh = image.shape[0]//units, image.shape[1]//units
    layout = cv2.bitwise_not(image)
    layout = cv2.adaptiveThreshold(layout, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, -1)
    # create images to hold extracted lines
    v, h = np.copy(layout), np.copy(layout)    
    # create vertical and horizontal structure elements
    vs = cv2.getStructuringElement(cv2.MORPH_RECT, (1, dv//2))
    hs = cv2.getStructuringElement(cv2.MORPH_RECT, (dh, 1))
    # apply morphology operations
    v, h = cv2.erode(v, vs), cv2.erode(h, hs)
    v, h = cv2.dilate(v, vs), cv2.dilate(h, hs)
    # return original and extracted
    layout, v, h = (v + h > 0).astype(int), None, None
    return layout


def extract_quadrants(pages: pd.DataFrame) -> pd.DataFrame:
    """
    simple variance-based assessment
    """
    quadrants = []
    for source in pages['source']:
        # load image
        image = cv2.imread(f'data/images/{source}.png', cv2.IMREAD_GRAYSCALE)
        if image is None: continue

        # load pdf-info
        info = pd.read_csv(f'data/info/{source}.csv.gz')[['top','left']]

        # resize for results consistency
        size = tuple((np.array(image.shape) * 256/min(image.shape)).astype(int))[::-1]
        image = cv2.resize(image, size, interpolation=cv2.INTER_AREA)
        h, w = image.shape
        d = min(h, w)//2
        slices = [slice(None, d), slice(-d, None)]

        info = info[info > 0].dropna() * 256
        words = { (0,0):len(info[(info['top'] < 128)&(info['left'] < 128)]),
                  (1,0):len(info[(info['top'] > h - 128)&(info['left'] < 128)]),
                  (1,1):len(info[(info['top'] > h - 128)&(info['left'] > w - 128)]),
                  (0,1):len(info[(info['top'] < 128)&(info['left'] > w - 128)]) }        
        
        # extract layout features
        layout, lines = extract_lines(image, units=5)        
        for r,c in ORDER:
            quadrants.append({
                'quadrant':(r,c),
                'content':np.mean(layout[slices[r],slices[c]]),
                'lines':np.mean(lines[slices[r],slices[c]]),
                'words':words[r,c],
                'source':source
            })           
    return pd.DataFrame.from_dict(quadrants)


def crop_margins(image: np.array, pad: int = 0, norm: float = 1.5) -> np.array:
    """
    Remove colored margins from the sides (scanned images often have):
    sometimes we need extract text from the header/footer areas only
    this way we can normalize location
    """
    test = cv2.GaussianBlur(image, (13, 13), 0)
    vs = np.std(test, axis=0)
    vm = np.mean(test, axis=0)
    hs = np.std(test, axis=1)
    hm = np.mean(test, axis=1)
    # construct criterion
    v = abs((vm[1:] - vm[:-1]) * (vs[1:] - vs[:-1]) / norm).astype(int)
    h = abs((hm[1:] - hm[:-1]) * (hs[1:] - hs[:-1]) / norm).astype(int)
    t = 1
    while v[t] == 0: t += 1
    b = -2
    while v[b] == 0: b -= 1
    l = 1
    while h[l] == 0: l += 1
    r = -2
    while h[r] == 0: r -= 1
    # return clip with some minimal padding
    return image[max(0, t - pad):b - 1 + pad,max(0, l - pad):r - 1 + pad]


def normalize(image: np.array, ):
    """
    preprocess high-res noisy image
    """
    output = cv2.GaussianBlur(image, (3, 3), 1)
    output = cv2.erode(output, np.ones((2, 2), np.uint8), iterations=1)
    return cv2.adaptiveThreshold(output, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 5, -1)


def get_stats() -> pd.DataFrame:
    """
    extract page-level stats potentially useful for labels
    """
    inputs = {'CheckBox':'check-box','RadioButton':'radio-button','Text':'text-input'}
    data = []
    # extractd page images
    images = [str(x).split('/').pop()[:-4] for x in Path(f'{ROOT}/data/images').glob('*.png')]
    widgets = set(str(x).split('/').pop()[:-7] for x in Path(f'{ROOT}/data/inputs').glob('*.csv.gz'))
    for source in images:
        # get text-content stats
        try:
            D = pd.read_csv(f'{ROOT}/data/info/{source}.csv.gz')
        except FileNotFoundError:
            # parsing info failed -- remove image
            os.unlink(f'{ROOT}/data/images/{source}.png')
        else:
            rec = D[D['block-type']=='line'].median(numeric_only=True)
            rec['source'] = source
            rec['word-count'] = len(D[D['block-type']=='word'])
            rec['num-lines'] = len(D[D['block-type']=='lines'])
            rec['num-blocks'] = len(D[D['block-type']=='block'])
            rec['num-images'] = len(D[D['block-type']=='image'])
            for x in inputs.values():
                rec[x] = None
            if source in widgets:
                # get form-input content stats
                try:
                    D = pd.read_csv(f'{ROOT}/data/inputs/{source}.csv.gz').groupby('field_type_string').size()
                except: #ParserError
                    os.unlink(f'{ROOT}/data/inputs/{source}.csv.gz')
                    for x in inputs.values():
                        rec[x] = -1
                else:
                    for k,v in inputs.items():
                        if k in D.index:
                            rec[v] = D.loc[k]
                        else:
                            rec[v] = 0
            data.append(rec)
    return pd.DataFrame.from_records(data).drop('text', axis=1)


def show_examples(sample: list) -> None:
    for name in sample:
        img = Image.open(f'{ROOT}/data/images/{name}.png')
        plt.imshow(img)
        plt.title(name)
        plt.show()
        
        
def build_taxonomy_tree(data: pd.Series, name: str) -> dict:
    """
    parse a set of alpha-numeric patterns into a dict-datastruct for a d3.sunburst diagram
    """
    # initialize
    root = {'name':name,'children':[]}
    tmp = data.apply(lambda x:x[0:1])
    tmp = tmp.groupby(tmp).size()
    index = set(tmp.index)
    root['children'] = [{'name':x,'size':int(tmp.loc[x])} for x in tmp.index]
    lookup = {c['name']:c for c in root['children']}
    
    # build
    for i in range(2,50):
        tmp = data.apply(lambda x:x[0:i])
        tmp = tmp.groupby(tmp).size()
        diff = set(tmp.index).difference(index)
        if len(diff) == 0:
            break
        for x in diff:
            node = lookup[x[:-1]]
            _ = node.pop('size', None)
            node['children'] = node.get('children', [])
            child = {'name':x,'size':int(tmp.loc[x])}
            node['children'].append(child)
            lookup[x] = child
        index = index.union(diff)
        
    # compress
    def collapce_long_path(r):
        if not 'children' in r:
            return r
        for x in r['children']:
            if 'children' in x:
                for i in range(len(x['children'])):
                    c = x['children'][i]
                    while 'children' in c and len(c['children']) == 1:
                        x['children'][i] = c['children'][0].copy()
                        c = c['children'][0].copy()
            x = collapce_long_path(x)
        return r
    
    root = collapce_long_path(root)               
    return root


def iframe(path: str, height: int) -> str:
    return f'''
<iframe src="{path}" width="960" height="{height}"
        sandbox="allow-same-origin allow-scripts allow-forms"
        style="overflow:hidden;border:0 none">
</iframe>'''


def interactive_chart(root: dict, output_file: str = 'sunburst.html') -> None:
    """
    embed dynamic visual into a notebook cell and save it as html for standalone view
    """
    with open(f'{ROOT}/assets/sunburst.html.txt','r') as source:
        html = source.read().replace('### data ###', json.dumps(root))

    with open(output_file,'w') as output:
        output.write(html)

    display(HTML(iframe(output_file, 700)))    



if __name__ == '__main__':
    
    # delete all previous IRS data
    for x in Path(f'{ROOT}/data/images/').glob('irs-*.png'): x.unlink()
    for x in Path(f'{ROOT}/data/info/').glob('irs-*.csv.gz'): x.unlink()
    for x in Path(f'{ROOT}/data/inputs/').glob('irs-*.csv.gz'): x.unlink()
    # get collected metadata
    irs = pd.read_csv(f'{ROOT}/data/irs-forms.csv')
    irs.loc[~irs['sub'].isna(),'sub'] = irs.loc[~irs['sub'].isna(),'sub'].astype(int).astype(str)
    irs = irs.fillna('').astype(str)

    # init/clearup
    irs['pages'] = 0
    # run extraction
    print('IRS forms:')
    done = extract(irs, dpi=DPI)
    print(f'extracted {done:.2%}')
    irs.to_csv(f'{ROOT}/data/irs-forms.tmp.csv', index=False)

    # same as above for canadian forms
    for x in Path(f'{ROOT}/data/images/').glob('cnd-*.png'): x.unlink()
    for x in Path(f'{ROOT}/data/info/').glob('cnd-*.csv.gz'): x.unlink()
    for x in Path(f'{ROOT}/data/inputs/').glob('cnd-*.csv.gz'): x.unlink()
    cnd = pd.read_csv(f'{ROOT}/data/cnd-forms.csv')
    cnd['pages'] = 0
    print('Canada forms:')
    done = extract(cnd, dpi=DPI)
    print(f'extracted {done:.2%}')
    cnd.to_csv(f'{ROOT}/data/cnd-forms.tmp.csv', index=False)

    # same as above for Quebek forms
    for x in Path(f'{ROOT}/data/images/').glob('que-*.png'): x.unlink()
    for x in Path(f'{ROOT}/data/info/').glob('que-*.csv.gz'): x.unlink()
    for x in Path(f'{ROOT}/data/inputs/').glob('que-*.csv.gz'): x.unlink()
    que = pd.read_csv(f'{ROOT}/data/que-forms.csv')
    que['pages'] = 0
    print('Quebec forms:')
    done = extract(que, dpi=DPI)
    print(f'extracted {done:.2%}')
    que.to_csv(f'{ROOT}/data/que-forms.tmp.csv', index=False)

    # set origin label
    irs['orig'] = 0
    cnd['orig'] = 1
    que['orig'] = 2
    # make combined dataset
    columns = ['orig','type','sub','ext','desc','lang','pages','file']
    data = pd.concat([irs[columns], cnd[columns], que[columns]], ignore_index=True)
    # remove missing and extra-long
    data = data.loc[(~data['file'].isna())&(~data['pages'].isna())]
    data = data.loc[data['pages'] <= MAX_PAGES].fillna('')
    # keep base filename as common source for reference
    data['file'] = data['file'].apply(lambda x:x[:-4]) # remove .pdf extension
    data.to_csv(f'{ROOT}/data/forms.csv.gz', index=False, compression='gzip')
    
    # get page-level stats
    data = get_stats()    
    columns = ['left', 'top', 'right', 'bottom',                           # bounding-box stats
               'font-size', 'italic', 'bold', 'cos', 'sin',                # text units average props
               'word-count', 'num-blocks', 'num-images', 'aspect-ratio',   # page layout hints
               'rotation', 'scale', 'page', 'num-pages', 'source',
               'check-box', 'radio-button', 'text-input']
    data[columns].to_csv(f'{ROOT}/data/page-summary.csv.gz', index=False, compression='gzip')
    
    # make train-test split
    np.random.seed(1)
    samples = [str(x).split('/').pop()[:-4] for x in Path(f'{ROOT}/data/images').glob('*.png')]
    train = np.random.choice(samples, int(0.9 * len(samples)), replace=False)
    test = list(set(samples).difference(set(train)))
    # save class to name translation
    with open(f'{ROOT}/data/train-test.json','w') as output:
        output.write(json.dumps({'train':list(train), 'test':list(test)}))
    # remove temporary
    for x in Path(f'{ROOT}/data/').glob('*.tmp.csv'): x.unlink()
    