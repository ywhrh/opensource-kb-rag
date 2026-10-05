"""Dense ANN retrieval with correct ordering for similarity and L2 distance."""
from core.profiling import timed
from core.models import embed
from ingestion.vector_store_writer import client, FIELDS


def vector_search(query, collection_name, config, credentials=None, timings=None):
    with timed(timings, 'embedding_ms'):
        vector = embed([query], config, credentials, query=True)[0].tolist()
    settings = config.vector_search
    params = {'ef': settings.hnsw_params.ef} if settings.index_type == 'HNSW' else {'nprobe': settings.ivf_params.nprobe}
    with timed(timings, 'vector_db_search_ms'):
        hits = client().search(collection_name=collection_name, data=[vector], anns_field='dense',
                               limit=settings.top_k, output_fields=FIELDS,
                               search_params={'metric_type': settings.metric_type, 'params': params})[0]
    rows = []
    for hit in hits:
        raw = float(hit['distance'])
        score = -raw if settings.metric_type == 'L2' else raw
        rows.append(dict(hit['entity'], score=score, vector_score=raw, score_type='negative_l2' if settings.metric_type == 'L2' else settings.metric_type.lower()))
    return rows
