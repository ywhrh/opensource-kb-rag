"""Selectable local BM25Okapi or scalable Milvus built-in sparse BM25."""
from functools import lru_cache
import re
import numpy as np
from rank_bm25 import BM25Okapi
from ingestion.vector_store_writer import client, iter_chunks, FIELDS


def tokenize(text):
    # Preserve error codes and dotted API names; add component tokens for mixed queries.
    terms = re.findall(r'[a-zA-Z0-9_]+(?:[.:-][a-zA-Z0-9_]+)*', text.lower())
    return terms + [piece for term in terms if re.search(r'[_.:-]', term) for piece in re.split(r'[_.:-]', term) if piece]


@lru_cache(maxsize=2)
def corpus(collection_name):
    # ponytail: BM25Okapi holds the corpus in RAM and scans it per query; use Milvus sparse BM25 at million-chunk scale.
    rows = list(iter_chunks(collection_name))
    if not rows:
        raise ValueError('Indexed collection is empty')
    return rows, BM25Okapi([tokenize(row['text']) or ['__empty__'] for row in rows])


def bm25_search(query, collection_name, config):
    settings = config.bm25_search
    if settings.model_name == 'Milvus Built-in Sparse BM25':
        hits = client().search(collection_name=collection_name, data=[query], anns_field='sparse',
                               limit=settings.top_k, output_fields=FIELDS,
                               search_params={'metric_type': 'BM25', 'params': {}})[0]
        return [dict(hit['entity'], score=float(hit['distance']), bm25_score=float(hit['distance']), score_type='bm25') for hit in hits]
    rows, index = corpus(collection_name)
    scores = index.get_scores(tokenize(query))
    positions = np.argsort(-scores, kind='stable')[:settings.top_k]
    return [dict(rows[int(i)], score=float(scores[i]), bm25_score=float(scores[i]), score_type='bm25') for i in positions]
