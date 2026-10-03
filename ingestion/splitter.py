"""Chunk by semantic boundaries or paragraph/size rules, preserving fenced code."""
import re
import numpy as np
from ingestion.cleaner import FENCES
from core.models import embed


def text_units(text, strategy="recursive"):
    """Recursive preserves paragraph/sentence boundaries; fixed size slices prose."""
    units = []
    for part in FENCES.split(text):
        if not part.strip():
            continue
        if part.startswith(('```', '~~~')):
            units.append(part.strip())
        elif strategy == 'fixed_size':
            units.append(part.strip())
        else:
            for paragraph in re.split(r'\n\s*\n', part):
                units.extend(unit.strip() for unit in re.split(r'(?<=[.!?])\s+(?=[A-Z])', paragraph) if unit.strip())
    return units


def split_document(text, config, credentials=None):
    chunking = config.ingestion_pipeline.chunking
    fixed = chunking.fixed_size_config
    limit = min(fixed.chunk_size, config.ingestion_pipeline.cleaning.max_chunk_length)
    units = []
    for unit in text_units(text, chunking.strategy):
        if len(unit) <= limit or unit.startswith(('```', '~~~')):
            units.append(unit)
        else:
            # Oversized prose can split; code blocks remain atomic even above the soft limit.
            while len(unit) > limit:
                cut = unit.rfind(' ', 0, limit)
                cut = cut if cut > limit // 2 else limit
                units.append(unit[:cut].strip())
                unit = unit[cut:].strip()
            if unit:
                units.append(unit)
    breaks = set()
    if chunking.strategy == 'semantic' and len(units) > 1:
        semantic = chunking.semantic_config
        windows = [' '.join(units[max(0, i - semantic.buffer_size):i + semantic.buffer_size + 1]) for i in range(len(units))]
        vectors = embed(windows, config, credentials)
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        vectors = vectors / np.maximum(norms, 1e-12)
        distances = 1 - np.sum(vectors[:-1] * vectors[1:], axis=1)
        amount = semantic.breakpoint_threshold_amount
        if semantic.breakpoint_threshold_type == 'percentile':
            threshold = float(np.percentile(distances, amount))
        elif semantic.breakpoint_threshold_type == 'standard_deviation':
            threshold = float(distances.mean() + amount * distances.std())
        else:
            q1, q3 = np.percentile(distances, [25, 75])
            threshold = float(q3 + amount * (q3 - q1))
        breaks = {i + 1 for i, distance in enumerate(distances) if distance > threshold}
    buffer, size, section, buffer_section = [], 0, 'Overview', 'Overview'
    for index, unit in enumerate(units):
        heading = re.match(r'^#{1,6}\s+(.+)', unit)
        boundary = index in breaks or (heading and bool(buffer)) or size + len(unit) + 2 > limit
        if buffer and boundary:
            content = '\n\n'.join(buffer)
            yield {'text': content, 'section_title': buffer_section, 'oversized': len(content) > limit}
            overlap = []
            if chunking.strategy != 'semantic' and not heading:
                overlap_size = 0
                for previous in reversed(buffer):
                    if previous.startswith(('```', '~~~')) or overlap_size + len(previous) > fixed.chunk_overlap:
                        break
                    overlap.insert(0, previous)
                    overlap_size += len(previous) + 2
            buffer = overlap
            size = sum(len(item) + 2 for item in buffer)
        if heading:
            section = heading.group(1).strip()
        if not buffer:
            buffer_section = section
        buffer.append(unit)
        size += len(unit) + 2
    if buffer:
        content = '\n\n'.join(buffer)
        yield {'text': content, 'section_title': buffer_section, 'oversized': len(content) > limit}
