"""
phrase_loader.py — Cached DB loader for named phrase lists.

All phrase/keyword lists used in the analysis pipeline are stored in the
`phrase_config.PhraseList` database table. This module provides a thin
in-process TTL cache (5 min per worker process) so the DB is not queried on
every call analysis, while still allowing real-time updates from the Django
admin without a code deploy.

Usage:
    import phrase_loader as _pl

    _MY_LIST_DEFAULT = ["fallback", "values"]

    def some_function():
        phrases = _pl.get("my_list_name", _MY_LIST_DEFAULT)
        for p in phrases:
            ...

Falls back to `default` (the hardcoded constant) if the DB row is missing or
the database is unreachable — the pipeline never crashes due to a DB issue.
"""

import time
import logging

logger = logging.getLogger(__name__)

_cache: dict = {}   # {name: {"data": ..., "ts": float}}
_TTL = 300          # seconds — re-fetches from DB after 5 min per worker process


def get(name: str, default=None):
    """
    Return the phrase list/dict stored under `name`.

    Uses an in-process TTL cache so the DB is queried at most once every
    5 minutes per Celery worker — not on every analysis call.
    Falls back to `default` (the hardcoded constant) if the DB is unreachable
    or the row does not exist.
    """
    now = time.monotonic()
    hit = _cache.get(name)
    if hit and (now - hit["ts"]) < _TTL:
        return hit["data"]

    try:
        from phrase_config.models import PhraseList
        obj = PhraseList.objects.get(name=name, is_active=True)
        _cache[name] = {"data": obj.data, "ts": now}
        return obj.data
    except Exception as exc:
        logger.debug("phrase_loader: DB miss for %r (%s) — using default", name, exc)
        if hit:
            return hit["data"]   # stale cache is better than nothing
        return default if default is not None else []


def invalidate(name: str = None):
    """
    Force the next call to re-fetch from the DB.
    Pass None to clear all cached entries; pass a name to clear one entry.
    """
    global _cache
    if name is None:
        _cache = {}
    else:
        _cache = {k: v for k, v in _cache.items() if k != name}
