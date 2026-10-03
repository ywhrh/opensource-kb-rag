"""Keep original evidence text separate from model-facing title/question enrichment."""
import hashlib
import re
from core.models import hypothetical_questions

ENTITY_PATTERNS = {
    'API': r'\b(?:GET|POST|PUT|PATCH|DELETE)\s+/[\w/{}.:?-]+|\b[A-Za-z_]\w*\.[A-Za-z_]\w*\(',
    'VERSION': r'\bv?\d+\.\d+(?:\.\d+)?\b',
    'CODE': r'\b(?:ERR_[A-Z0-9_]+|[A-Z][A-Z0-9]+_[A-Z0-9_]+|HTTP\s+[1-5]\d{2})\b',
}


def enrich(chunk, document, config, credentials=None):
    settings = config.ingestion_pipeline.metadata_enrichment
    text = chunk['text']
    metadata = {'source': document.source, 'doc_title': document.title, 'content_hash': document.content_hash,
                'oversized_code_block': chunk['oversized']}
    if settings.extract_entities:
        for entity_type in settings.entity_types:
            metadata[entity_type] = list(dict.fromkeys(re.findall(ENTITY_PATTERNS[entity_type], text)))[:50]
    language = re.search(r'```([a-zA-Z0-9_+-]+)', text)
    if language:
        metadata['programming_language'] = language.group(1)
    embedding_text = f'{document.title}\n{chunk["section_title"]}\n\n{text}' if settings.inject_doc_title else text
    if settings.generate_hypothetical_questions:
        questions = hypothetical_questions(text, config, credentials)
        metadata['hypothetical_questions'] = questions
        embedding_text += '\n\nQuestions answered by this passage:\n' + questions
    chunk_id = document.doc_id + '_' + hashlib.sha256(text.encode()).hexdigest()[:16]
    return {'chunk_id': chunk_id, 'doc_id': document.doc_id, 'section_title': chunk['section_title'],
            'text': text, 'embedding_text': embedding_text, 'metadata': metadata}
