#!/usr/bin/env python
# coding: utf-8

# 
# https://dylancastillo.co/ai-search-engine-fastapi-qdrant-chatgpt/
# 
# https://blog.metarank.ai/from-zero-to-semantic-search-embedding-model-592e16d94b61
# 
# 
# In such a scenario, you would want to use a technique called `domain adaptation`. Essentially, this entails that you take your general-purpose language model that was trained on general text and continue training it on data from your domain of interest. Again, I won't get into the nitty-gritty of how this all works: check out our upcoming more technical blogpost for that.
# 
# Drawback of semantic search: it can be quite slow and compute-intensive. The solution is `Retrieve & Rerank`. Essentially, the idea here is to use a (fast and inexpensive) lexical search algorithm (such as TF-IDF or BM25 as discussed earlier) to filter out the irrelevant documents in your database so that you arrive at a smaller subset of `candidate documents` that could all potentially be relevant. This is the `retrieve` step. Then you use your semantic search engine to search only within these `candidate documents` rather than within all documents in your database. This is the `rerank` step. Graphically, the flow will look something like the graph below.
# 
# As for now, sparse (ELSER) and dense (E5) retrieval methods are pretty close in quality, there is no clear winner in this area yet.
# 
# 
# 
# Embeddings versus sparse retrieval
# Embeddings are one of many ways of doing search. The good old BM25 is still a strong baseline, and there are a couple of new “sparse” retrieval methods like SPLADEv2 and ColBERT — combining strengths of term search and neural networks.
# 
# In the table below, we tried to aggregate all the publicly available BEIR scores across the following sources:
# 
# The MTEB results repository contains all the raw scores used in the leaderboard. It also includes scores for OpenAI commercial embeddings, including the newest ada-2 model.
# ArXiv preprints for ColBERT, SPLADEv2, and BEIR papers.
# Vendor blogposts on Vespa’s ColBERT implementation, and Elastic’s ELSER.
# 
# BEIR NDCG@10 scores for SOTA models. An image by author.
# If you compare this table with the BEIR one published only two years ago, you may notice that BM25 was considered a strong baseline — but BM25 is no longer a clear winner in 2023.
# 
# Another observation is that sparse (e.g., ELSER and SPLADEv2) and dense (E5) retrieval methods are pretty close in quality. So there is no clear winner in this area, but seeing so much competition is great.
# 
# Author’s personal opinion on sparse vs. dense retrieval methods debate:
# 
# Dense retrieval is more future-proof. Upgrading from SBERT to E5 is 10 lines of code, yielding massive improvements in retrieval quality. And your vector search engine stays the same, no extra engineering needed.
# Sparse retrieval hallucinates less and can handle exact and keyword matches. And NDCG@10 is not the only measure of search quality.
# 
# There is only a 0.08 dot difference between love and hate from the OpenAI point of view. Image by author.
# But the debate is still ongoing and we will keep you posted on latest updates.
# 
# The hidden cost of large models
# There is a common wisdom that the larger the model, the better is its retrieval quality. It can be clearly seen from the MTEB leaderboard, but is misses the important and practical characteristic of how easy & cheap is to serve these models.
# 
# 
# Embedding model should be served both online and offline. Image by author.
# In practice you need to run the embedding model twice:
# 
# Offline during the indexing stage. It’s a batch job requiring high throughput, but not sensitive to latency.
# Online to embed the search query, on each search request.
# Small models like SBERT and E5 can be easily run on CPU within a reasonable latency budget, but if you go beyond 500M params (which is the case for SGPT), you cannot avoid using a GPU. And GPUs are expensive these days.
# 
# ONNX Inference latency for different document lengths for SBERT/E5 models in milliseconds. CPU: 16-core Ryzen7/2600. GPU: RTX3060Ti. Image by author.
# As you can see from the table:
# 
# There is linear dependency between number of params and latency, both on CPU and GPU. Large models like SGPT-1.3B should have estimated latency of 200ms for short 4-word queries, which is often way too much for customer-facing workloads.
# There is a latency/costs/quality trade-off. Your semantic search is fast, cheap, precise — choose any two.
# 
# BM25 is not that easy to outperform in search quality. Considering that it requires zero tuning and you can spawn an Elasticsearch/OpenSearch cluster in 3 minutes — it is still practical to rely on this in 2023.
# A lot of recent development is happening in the area of retrieval both in sparse and dense worlds. SGPT and E5 are less than 1 year old, and SPLADE & ELSER are even younger.
# There is no single winner between sparse/dense approaches, but the IR industry converged into a single benchmarking harness to measure retrieval quality with MTEB/BEIR suite.

# Designing a (Tiny) Search Engine with ChatGPT
# Before you get started, you should understand the overall approach you'll take to build your AI search engine. There are three parts to it:
# 
# Extraction: This part consists of extracting the data that you want users to be able to search. In this case, that means parsing Meditations. I won't go into detail about this because it is very project-specific. The parsed data is available in the repository.
# Indexing: This entails indexing the extracted data so that it can be accessed later when running searches. In this case, you'll use a semantic search approach, which means you'll search the data based on its meaning rather than keywords. That is, if you search for "How can I be happy?" you should get passages from Meditations that discuss happiness or feeling good, not just those that contain the exact words from the query.
# Search: This consists of a backend service that processes the user's query, vectorizes it, finds vectors in the index that are the most similar to it, and then calls OpenAI's API to generate a summarized answer for the user.
# Here's a visual representation of how the parts of the application you'll build in this tutorial fit together:
# 

import re
import os
import json
import numpy as np
import pandas as pd

from time import time
from pathlib import Path
from unidecode import unidecode
from torch.cuda import is_available
from elasticsearch import Elasticsearch
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance
from sentence_transformers import SentenceTransformer, util


INDEX = 'doc-pages'

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
        # for range-queries experiment
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
        yield json.dumps({'index':{'_index':INDEX,'_id':ID}})
        yield json.dumps({ x:record[x] for x in MAPPINGS['properties'] if not x.startswith('ml.') })


class SemanticSearch:
    def __init__(self):
        self.collection = INDEX
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
        self.collection = INDEX
        self.client = Elasticsearch(
            hosts=[os.environ['ELASTIC_URI']],
            basic_auth=('elastic', os.environ['ELASTIC_PASSWORD']),
            verify_certs=False)
        
    def get_docs(self, taxonomy):
        query = {'bool': {'must': [{'match': {'taxonomy_id': taxonomy}}, {'match': {'block_type': 'input'}}]}}
        aggs = {'docs': {'terms': {'field': 'doc_id'}}}
        return self.client.search(index=INDEX, query=query, aggs=aggs)['aggregations']['docs']
            
    def find_inputs(self, doc, input_type=None, size=100):
        must = [{'match': {'doc_id': doc}}, {'match': {'block_type': 'input'}}]
        if input_type is not None:
            must.append({'match_phrase': {'content': {'query': input_type, 'slop': 5 }}})
        query = {'bool': {'must': must }}
        sort = [{'page_id': {'order': 'asc'}}, {'top': {'order': 'asc'}}, {'left': {'order': 'asc'}}]
        return self.client.search(index=INDEX, query=query, sort=sort, size=size)['hits']['hits']
            
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
        return self.client.search(index=INDEX, query=query, sort=sort, size=size)['hits']['hits']

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
        return self.client.search(index=INDEX, query=query, sort=sort, size=size)['hits']['hits']


