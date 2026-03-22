# src/platforms/__init__.py
"""
Platform-specific discovery and download helpers.

Each module exposes:
- discover_candidates(settings) -> list[dict]
- download_audio(...) for that platform (where implemented)
"""
