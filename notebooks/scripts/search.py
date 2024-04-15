#!/usr/bin/env python
# coding: utf-8


import re
import os
import json
import numpy as np
import pandas as pd

from pathlib import Path
from unidecode import unidecode
from torch.cuda import is_available
from elasticsearch import Elasticsearch
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance
from sentence_transformers import SentenceTransformer, util


PAGE_INDEX = 'doc-pages'

SETTINGS = {
    "analysis": {
        "analyzer": {
            "custom_analyzer": {
                "type": "custom",
                "tokenizer": "whitespace",
                "filter": ["lowercase"]
            }
        },
    }
}


MAPPINGS = {
    "properties": {
        "content": {
            "type": "text",
            "analyzer": "custom_analyzer"
        },
        "block_type": {
            "type": "keyword"
        },
        "font_size": {
            "type": "byte"
        },
        "display": {
            "type": "keyword"
        },
        "page_id": {
            "type": "keyword"
        },
        "doc_id": {
            "type": "keyword"
        },
        "taxonomy_id": {
            "type": "keyword"
        },
        "orig": {
            "type": "keyword"
        },
        "lang": {
            "type": "keyword"
        },
        "left": {
            "type": "float"
        },
        "top": {
            "type": "float"
        },
        "right": {
            "type": "float"
        },
        "bottom": {
            "type": "float"
        },
    }
}


ENCODER_MODEL = 'distiluse-base-multilingual-cased-v1'
DEVICE = 'cuda' if is_available() else 'cpu'
semantic_encoder = SentenceTransformer(ENCODER_MODEL, device=DEVICE)
semantic_encoder.max_seq_length = 200


def data_filter(data):
    """
    keep non-empty texts which are not `code` (like xml)
    """
    return data.loc[(~data['text'].isna())&(data['text']!='')&(~data['text'].astype(str).str.startswith('<'))]


def normalize(text: str):
    """
    normalize whitespace and unicode, keep casing and punctuation
    remove `......` and `. . . . .` from anywhere
    strip some non-word characters ▶ from the start/end
    """
    text = re.sub(r'\s+', ' ', unidecode(text).replace('*', '')).strip()
    text = re.sub(r'\.{2,}', '', text)
    text = re.sub(r'\s[\s.]{3,}\s', ' ', text)
    
    # remove hyphen between digit and letter (for taxonomy pattern matching)
    text = re.sub(r'(?<=[A-Z])-(?=\d)', '', re.sub(r'(?<=\d)-(?=[A-Z])', '', text))
    
    return unidecode(re.sub('^[^a-zA-Z0-9\(\$]*|[^a-zA-Z0-9\)]*$', '', text)).strip(' .')


def parse_info(data, doc) -> dict:    
    display = { 0:'h', 1:'v' } # horizontal and vertical
    return {
        'content': normalize(str(data['text'])),
        'block_type': 'text',
        
        # useful for identifying page title
        'font_size': int(round(data['font-size'])) if data['font-size'] > 0 else None,
        
        # sin and cos define text-block orientataion: some could be different from the main content
        'display': '?' if np.isnan(data['sin']) else display.get(data['sin'], 'd'),
        
        # spacial layout
        'left': data['left'],
        'top': data['top'],
        'right': data['right'],
        'bottom': data['bottom'],
        
        # page props
        'page_id': data['page'],
        'doc_id': doc['file'],
        'taxonomy_id': doc['taxonomy'],
        'taxonomy_ext': doc['ext'],
        'lang': doc['lang'],
        'orig': doc['orig'],
    }


def parse_inputs(data, doc) -> dict:
    return {
        'content': f"TYPE: {data['field_type_string']} NAME: {data['field_name']} LABEL: {data['field_label']}",
        'block_type': 'input',
        'font_size': int(round(data['text_fontsize'])) if data['text_fontsize'] > 0 else None,
        'display': '?' if np.isnan(data['field_display']) else str(data['field_display']),
        'left': data['left'],
        'top': data['top'],
        'right': data['right'],
        'bottom': data['bottom'],
        'page_id': data['page'],
        'doc_id': doc['file'],
        'taxonomy_id': doc['taxonomy'],
        'taxonomy_ext': doc['ext'],
        'lang': doc['lang'],
        'orig': doc['orig'],
    }   


def data_to_ndjson(data, doc, parse) -> list:
    """
    Generator for text-blocks data ingest
    """
    for d in data.to_dict('records'):
        record = parse(d, doc)
        if record['content'] == '': # skip empty
            continue
        taxonomy_ext = re.sub(r'\W+', '', doc['ext']).upper()
        ID = f"{doc['file']}-{d['page']}-{d['block']}".upper()
        yield json.dumps({'index':{'_index':PAGE_INDEX,'_id':ID}})
        yield json.dumps({ x:record[x] for x in MAPPINGS['properties'] if not x.startswith('ml.') })


class SemanticSearch:
    def __init__(self):
        self.collection = PAGE_INDEX
        # initialize encoder model
        self.model = SentenceTransformer(ENCODER_MODEL, device=DEVICE)
        # initialize Qdrant client
        self.client = QdrantClient(host=os.environ['QDRANT_HOST'], port=6333)

    def find(self, text: str, num: int = 5):
        # convert text query into vector
        vector = self.model.encode(text).tolist()
        # search for closest vectors in the collection
        results = self.client.search(
            collection_name=self.collection,
            query_vector=vector,
            query_filter=None,
            limit=num,
        )
        # return payload of closest matches
        return sorted([hit.payload for hit in results], key=lambda x:x['orig'])
    
    
class InputSearch:
    def __init__(self):
        self.collection = PAGE_INDEX
        self.client = Elasticsearch(
            hosts=[os.environ['ELASTIC_URI']],
            basic_auth=('elastic', os.environ['ELASTIC_PASSWORD']),
            verify_certs=False)
        
    def get_docs(self, taxonomy):
        query = {'bool': {'must': [{'match': {'taxonomy_id': taxonomy}}, {'match': {'block_type': 'input'}}]}}
        aggs = {'docs': {'terms': {'field': 'doc_id'}}}
        return self.client.search(index=PAGE_INDEX, query=query, aggs=aggs)['aggregations']['docs']
            
    def find_inputs(self, doc, input_type=None, size=100):
        must = [{'match': {'doc_id': doc}}, {'match': {'block_type': 'input'}}]
        if input_type is not None:
            must.append({'match_phrase': {'content': {'query': input_type, 'slop': 5 }}})
        query = {'bool': {'must': must }}
        sort = [{'page_id': {'order': 'asc'}}, {'top': {'order': 'asc'}}, {'left': {'order': 'asc'}}]
        return self.client.search(index=PAGE_INDEX, query=query, sort=sort, size=size)['hits']['hits']
            
    def find_parent_block(self, input_data, size=100):
        """
        get a text-block which contains this input
        """
        query = {
            'bool': {
                'must': [{'match': {'doc_id': input_data['_source']['doc_id'] }},
                         {'match': {'page_id': input_data['_source']['page_id'] }},
                         {'match': {'block_type': 'block'}}],
                #'filter': [{'shape': {'box': {'shape': hit['_source']['box'], 'relation': 'contains'}}}]            
                'filter': [{'range': {'top': {'lte': input_data['_source']['top']}}},
                           {'range': {'bottom': {'gte': input_data['_source']['bottom']}}},
                           {'range': {'left': {'lte': input_data['_source']['left']}}},
                           {'range': {'right': {'gte': input_data['_source']['right']}}}]
            }}
        sort = [{'top': {'order': 'asc'}}, {'left': {'order': 'asc'}}]
        return self.client.search(index=PAGE_INDEX, query=query, sort=sort, size=size)['hits']['hits']

    def find_nearest_text(self, input_data, size=3):
        """
        get text-blocks which are at the same level of above:
        inputs usually on the same line or right under
        """
        query = {
            'bool': {
                'must': [{'match': {'doc_id': input_data['_source']['doc_id'] }},
                         {'match': {'page_id': input_data['_source']['page_id'] }},
                         {'match': {'block_type': 'word'}}],
                'filter': [{'range': {'top': {'lte': input_data['_source']['bottom']}}}]
            }}
        sort = [{'top': {'order': 'desc'}}, {'left': {'order': 'asc'}}]
        return self.client.search(index=PAGE_INDEX, query=query, sort=sort, size=size)['hits']['hits']


