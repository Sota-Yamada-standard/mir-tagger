"""
アニソン判定アナライザー

MyAnimeList (Jikan API) のデータベースと照合して、
楽曲がアニメソングかどうかを高精度で判定する。

このアナライザーは音声解析ではなく、メタデータ（曲名、アーティスト名）
をデータベースと照合することで判定を行う。
"""

import numpy as np
from typing import Dict, Any, Optional
from pathlib import Path
from src.core.base_analyzer import BaseAnalyzer, AnalyzerRegistry

# アニソンDBモジュール
try:
    from src.utils.anisong_db import AnisongMatcher
    ANISONG_DB_AVAILABLE = True
except ImportError:
    ANISONG_DB_AVAILABLE = False


@AnalyzerRegistry.register
class AnisongAnalyzer(BaseAnalyzer):
    """
    アニソン判定アナライザー
    
    MyAnimeListのデータベースと照合して、楽曲がアニメに関連するかを判定。
    音声解析は行わず、曲名・アーティスト名のマッチングで高精度判定。
    
    タグ:
    - [Anime:Title] - アニメ作品名
    - [Genre:anime] - アニソンである場合（高信頼度）
    """
    
    name = "anisong"
    description = "Anime song detection via MAL database matching"
    tag_prefix = "Anime"
    
    # 信頼度閾値
    CONFIDENCE_THRESHOLD = 0.7
    
    def __init__(self, cache_dir: str = None):
        """
        Args:
            cache_dir: キャッシュDBのディレクトリ（Noneでデフォルト）
        """
        self._matcher = None
        self._cache_dir = Path(cache_dir) if cache_dir else None
        self._metadata = {}  # 外部から設定されるメタデータ
    
    @property
    def matcher(self) -> 'AnisongMatcher':
        """マッチャーの遅延初期化"""
        if self._matcher is None:
            if not ANISONG_DB_AVAILABLE:
                raise RuntimeError("AnisongDB module not available")
            self._matcher = AnisongMatcher(cache_dir=self._cache_dir)
        return self._matcher
    
    def set_metadata(self, title: str = None, artist: str = None):
        """
        外部からメタデータを設定
        
        Args:
            title: 曲名
            artist: アーティスト名
        """
        self._metadata = {
            'title': title,
            'artist': artist
        }
    
    def analyze(self, audio: np.ndarray, sample_rate: int) -> Dict[str, Any]:
        """
        アニソン判定を実行
        
        注意: このアナライザーは音声データを使用せず、
        事前に set_metadata() で設定されたメタデータを使用する。
        音声データが渡されるのはBaseAnalyzerのインターフェース互換のため。
        
        Returns:
            {
                'is_anime_song': bool,
                'anime_title': str or None,
                'anime_id': int or None,
                'theme_type': str or None,  # 'opening', 'ending', 'insert'
                'confidence': float,
                'db_available': bool
            }
        """
        result = {
            'is_anime_song': False,
            'anime_title': None,
            'anime_id': None,
            'theme_type': None,
            'song_title': None,
            'matched_artist': None,
            'confidence': 0.0,
            'db_available': ANISONG_DB_AVAILABLE
        }
        
        if not ANISONG_DB_AVAILABLE:
            return result
        
        title = self._metadata.get('title')
        artist = self._metadata.get('artist')
        
        if not title:
            return result
        
        try:
            is_anime, match_info = self.matcher.is_anime_song(title, artist)
            
            if is_anime and match_info:
                confidence = match_info.get('confidence', 0.0)
                
                # 信頼度が閾値以上の場合のみ採用
                if confidence >= self.CONFIDENCE_THRESHOLD:
                    result['is_anime_song'] = True
                    result['anime_title'] = match_info.get('anime_title')
                    result['anime_id'] = match_info.get('anime_id')
                    result['theme_type'] = match_info.get('theme_type')
                    result['song_title'] = match_info.get('song_title')
                    result['matched_artist'] = match_info.get('artist')
                    result['confidence'] = confidence
                    result['value'] = match_info.get('anime_title', '')
        except Exception as e:
            result['error'] = str(e)
        
        return result
    
    def to_tag(self, result: Dict[str, Any]) -> str:
        """タグ文字列を生成"""
        tags = []
        
        if result.get('is_anime_song'):
            # [Genre:anime] タグ（高信頼度のアニソン判定）
            tags.append("[Genre:anime]")
            
            # [Anime:タイトル] タグ（作品名）
            anime_title = result.get('anime_title')
            if anime_title:
                # タグに使えない文字をエスケープ
                safe_title = anime_title.replace('[', '(').replace(']', ')')
                tags.append(f"[{self.tag_prefix}:{safe_title}]")
            
            # 主題歌タイプ（オプション）
            theme_type = result.get('theme_type')
            if theme_type and theme_type != 'unknown':
                type_map = {
                    'opening': 'OP',
                    'ending': 'ED',
                    'insert': 'Insert'
                }
                type_label = type_map.get(theme_type, theme_type)
                tags.append(f"[AnimeTheme:{type_label}]")
        
        return ''.join(tags)


def check_cache_status():
    """キャッシュの状態を確認"""
    if not ANISONG_DB_AVAILABLE:
        print("❌ AnisongDB module not available")
        return
    
    matcher = AnisongMatcher()
    stats = matcher.cache.get_stats()
    
    print("📊 Anisong Cache Status:")
    print(f"   Songs:     {stats['songs']:,}")
    print(f"   Anime:     {stats['anime']:,}")
    print(f"   DB Size:   {stats['db_size_mb']} MB")
    
    last_update = matcher.cache.get_meta('last_full_update')
    if last_update:
        print(f"   Last Update: {last_update}")
    
    return stats


if __name__ == "__main__":
    check_cache_status()
