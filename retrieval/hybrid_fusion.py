"""Deduplicate evidence and combine normalized scores or reciprocal ranks."""

def fuse(dense, sparse, config):
    merged = {}
    for rows, weight, kind in [(dense, config.weights.vector_weight, 'vector'), (sparse, config.weights.bm25_weight, 'bm25')]:
        scores = [row['score'] for row in rows]
        low, high = (min(scores), max(scores)) if scores else (0, 0)
        for rank, row in enumerate(rows, 1):
            item = merged.setdefault(row['chunk_id'], {**row, 'score': 0.0, 'score_type': config.strategy})
            if config.strategy == 'RRF':
                contribution = 1 / (config.rrf_k + rank)
            else:
                normalized = (row['score'] - low) / (high - low) if high != low else (1.0 if len(rows) == 1 else 0.0)
                contribution = weight * normalized
            item['score'] += contribution
            for key in ['vector_score', 'bm25_score']:
                if key in row:
                    item[key] = row[key]
            item[kind + '_rank'] = rank
    return sorted(merged.values(), key=lambda row: (-row['score'], row['chunk_id']))
