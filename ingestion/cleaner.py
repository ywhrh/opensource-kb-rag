"""Clean prose while preserving fenced code; deduplicate chunks with MinHash."""
from html.parser import HTMLParser
import re

from datasketch import MinHash, MinHashLSH
from langdetect import DetectorFactory, detect, LangDetectException

DetectorFactory.seed = 0
FENCES = re.compile(r'(```[^\n]*\n.*?(?:\n```[^\n]*(?:\n|$)|$)|~~~[^\n]*\n.*?(?:\n~~~[^\n]*(?:\n|$)|$))', re.S)


class TextOnly(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.text = []
        self.hidden = 0

    def handle_starttag(self, tag, attrs):
        if tag in {'script', 'style'}:
            self.hidden += 1
        if tag in {'p', 'br', 'div', 'li', 'h1', 'h2', 'h3', 'pre'}:
            self.text.append('\n')

    def handle_endtag(self, tag):
        if tag in {'script', 'style'}:
            self.hidden = max(0, self.hidden - 1)

    def handle_data(self, text):
        if not self.hidden:
            self.text.append(text)


def clean_text(text, config):
    parts = FENCES.split(text.replace('\r\n', '\n').replace('\r', '\n').replace('\x00', ''))
    cleaned = []
    for part in parts:
        if part.startswith(('```', '~~~')):
            cleaned.append(part)
            continue
        if config.remove_html_tags and re.search(r'</?[a-zA-Z][^>]*>', part):
            parser = TextOnly()
            parser.feed(part)
            part = ''.join(parser.text)
        cleaned.append(re.sub(r'\n[ \t]*\n(?:[ \t]*\n)+', '\n\n', part))
    return ''.join(cleaned).strip()


def permitted_language(text, languages):
    if not languages:
        return True
    prose = FENCES.sub('', text)
    if len(re.findall(r'[A-Za-z]+', prose)) < 10:
        return True  # Code-only and short technical fragments are not reliable language samples.
    try:
        return detect(prose) in languages
    except LangDetectException:
        return True


class Deduplicator:
    def __init__(self):
        self.index = MinHashLSH(threshold=0.9, num_perm=128)
        self.position = 0

    def duplicate(self, text):
        words = re.findall(r'\w+', text.lower())
        shingles = [' '.join(words[i:i + 5]) for i in range(max(1, len(words) - 4))]
        signature = MinHash(num_perm=128)
        for shingle in shingles:
            signature.update(shingle.encode())
        if self.index.query(signature):
            return True
        self.index.insert(str(self.position), signature)
        self.position += 1
        return False
