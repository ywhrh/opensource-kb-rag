"""Local provider keys, stored separately from configuration and run snapshots."""
import json
import os
import tempfile
from pathlib import Path

from core.config_loader import ROOT, config_lock

KEY_FILE = ROOT / '.secrets' / 'api_keys.json'
PROVIDERS = {'openai', 'cohere'}


def load_keys():
    if not KEY_FILE.exists():
        return {}
    values = json.loads(KEY_FILE.read_text())
    if not isinstance(values, dict) or any(key not in PROVIDERS or not isinstance(value, str) for key, value in values.items()):
        raise ValueError('Invalid local API key file')
    return values


def save_key(provider, value):
    """Atomically replace a private file; blank input removes that provider key."""
    if provider not in PROVIDERS:
        raise ValueError('Unsupported provider')
    with config_lock():
        values = load_keys()
        if value.strip():
            values[provider] = value.strip()
        else:
            values.pop(provider, None)
        KEY_FILE.parent.mkdir(mode=0o700, exist_ok=True)
        KEY_FILE.parent.chmod(0o700)
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(mode='w', dir=KEY_FILE.parent, delete=False) as file:
                temporary = Path(file.name)
                os.fchmod(file.fileno(), 0o600)
                json.dump(values, file)
                file.flush()
                os.fsync(file.fileno())
            temporary.replace(KEY_FILE)
        finally:
            if temporary and temporary.exists():
                temporary.unlink()
