# Defines structured models (with Pydantic) to parse and validate base.yaml.

from pydantic import BaseModel
from pathlib import Path
import yaml


class AudioSettings(BaseModel):
    min_seconds: int = 6
    raw_dir: str = "data/raw_audio"
    max_download_seconds: int = 1200


class TranscriptionSettings(BaseModel):
    transcripts_dir: str = "data/transcripts"

    # Fast model for screening (chunks)
    screening_model_size: str = "medium"

    # Heavy model for final transcription
    full_model_size: str = "large-v2"

    device: str = "cpu"
    compute_type: str = "int8"


class YouTubeSettings(BaseModel):
    api_key: str = ""
    region_code: str = "LB"
    relevance_language: str = "ar"
    search_queries: list[str] = []
    channels: list[str] = []


class TikTokSettings(BaseModel):
    hashtags: list[str] = []
    users: list[str] = []


class InstagramSettings(BaseModel):
    users: list[str] = []


class FacebookSettings(BaseModel):
    pages: list[str] = []
    keywords: list[str] = []


class PodcastSettings(BaseModel):
    rss_feeds: list[str] = []
    keywords: list[str] = []


class WeakLabelsSettings(BaseModel):
    """Trusted sources for metadata-based weak supervision."""
    trusted_lebanese_channels: list[str] = []
    trusted_lebanese_feeds: list[str] = []
    trusted_lebanese_tiktok_users: list[str] = []
    trusted_non_lebanese_channels: list[str] = []
    trusted_non_lebanese_feeds: list[str] = []
    trusted_non_lebanese_tiktok_users: list[str] = []


class PlatformsSettings(BaseModel):
    youtube: YouTubeSettings = YouTubeSettings()
    tiktok: TikTokSettings = TikTokSettings()
    instagram: InstagramSettings = InstagramSettings()
    facebook: FacebookSettings = FacebookSettings()
    podcasts: PodcastSettings = PodcastSettings()


class Settings(BaseModel):
    db_url: str
    audio: AudioSettings
    transcription: TranscriptionSettings
    platforms: PlatformsSettings = PlatformsSettings()
    weak_labels: WeakLabelsSettings = WeakLabelsSettings()

    @classmethod
    def load(cls, path: str = "config/base.yaml") -> "Settings":
        p = Path(path)
        data = yaml.safe_load(p.read_text(encoding="utf-8"))
        return cls(**data)
