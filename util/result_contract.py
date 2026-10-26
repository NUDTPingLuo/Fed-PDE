"""Reject stale result reuse when scientific settings have changed."""
import json
from pathlib import Path


def _scientific_config(value):
    ignored = {'resume', 'checkpoint', 'output', 'output_dir', 'output_prefix',
               'device', 'training_device', 'log_every'}
    if isinstance(value, dict):
        return {k: _scientific_config(v) for k, v in value.items() if k not in ignored}
    if isinstance(value, (list, tuple)):
        return [_scientific_config(v) for v in value]
    return value


def matching_metadata(path, cfg, public=False):
    try:
        metadata = json.loads(Path(path).read_text(encoding='utf-8'))
        recorded = metadata['benchmark']['config'] if public else metadata['config']
        return _scientific_config(recorded) == _scientific_config(cfg)
    except (OSError, ValueError, KeyError):
        return False
