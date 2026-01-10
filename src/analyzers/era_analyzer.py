"""
年代（Era）解析アナライザー

音楽ファイルのリリース年を取得し、年代タグを生成する。
まずファイルのメタデータを参照し、オプションでMusicBrainz検索も可能。
"""

import numpy as np
from typing import Dict, Any, Optional
from pathlib import Path
from src.core.base_analyzer import BaseAnalyzer, AnalyzerRegistry

# MusicBrainz検索（オプション）
try:
    import musicbrainzngs
    musicbrainzngs.set_useragent("MIR-Tagger", "1.0", "https://github.com/")
    MUSICBRAINZ_AVAILABLE = True
except ImportError:
    MUSICBRAINZ_AVAILABLE = False


@AnalyzerRegistry.register
class EraAnalyzer(BaseAnalyzer):
    """
    年代解析アナライザー
    
    ファイルのメタデータからリリース年を取得し、年代タグを生成。
    
    タグ:
    - [Era:1980s] - 1980年代
    - [Era:1990s] - 1990年代
    - [Era:2000s] - 2000年代
    - [Era:2010s] - 2010年代
    - [Era:2020s] - 2020年代
    - [Year:2008] - 正確なリリース年（オプション）
    """
    
    name = "era"
    description = "Release year/era detection from metadata"
    tag_prefix = "Era"
    
    # MusicBrainz検索を有効にするか
    USE_MUSICBRAINZ = False  # デフォルトはOFF（ネットワーク依存を避ける）
    
    # 正確な年もタグに含めるか
    INCLUDE_EXACT_YEAR = True
    
    def __init__(self, use_musicbrainz: bool = False, include_exact_year: bool = True):
        """
        Args:
            use_musicbrainz: MusicBrainz検索を有効にするか
            include_exact_year: 正確な年もタグに出力するか
        """
        self.use_musicbrainz = use_musicbrainz and MUSICBRAINZ_AVAILABLE
        self.include_exact_year = include_exact_year
        self._metadata = {}  # 外部から設定されるメタデータ
    
    def set_metadata(self, title: str = None, artist: str = None, 
                     album: str = None, year: int = None, file_path: str = None):
        """
        外部からメタデータを設定
        
        Args:
            title: 曲名
            artist: アーティスト名
            album: アルバム名
            year: リリース年（既知の場合）
            file_path: ファイルパス（メタデータ読み取り用）
        """
        self._metadata = {
            'title': title,
            'artist': artist,
            'album': album,
            'year': year,
            'file_path': file_path
        }
    
    def _get_year_from_file(self, file_path: str) -> Optional[int]:
        """ファイルのID3/MP4タグからリリース年を取得"""
        try:
            from mutagen import File
            audio = File(file_path)
            
            if not audio or not audio.tags:
                return None
            
            tags = audio.tags
            
            # ID3 (MP3)
            if hasattr(tags, 'getall'):
                # TDRC: Recording time (ID3v2.4)
                if 'TDRC' in tags:
                    year_str = str(tags['TDRC'])[:4]
                    if year_str.isdigit():
                        return int(year_str)
                # TYER: Year (ID3v2.3)
                if 'TYER' in tags:
                    year_str = str(tags['TYER'])[:4]
                    if year_str.isdigit():
                        return int(year_str)
                # TORY: Original release year
                if 'TORY' in tags:
                    year_str = str(tags['TORY'])[:4]
                    if year_str.isdigit():
                        return int(year_str)
            
            # MP4/M4A
            if isinstance(tags, dict):
                if '©day' in tags:
                    year_str = str(tags['©day'][0])[:4]
                    if year_str.isdigit():
                        return int(year_str)
            
            return None
            
        except Exception:
            return None
    
    def _get_year_from_musicbrainz(self, title: str, artist: str) -> Optional[int]:
        """MusicBrainzからリリース年を検索"""
        if not self.use_musicbrainz or not title:
            return None
        
        try:
            # 録音を検索
            query = f'recording:"{title}"'
            if artist:
                query += f' AND artist:"{artist}"'
            
            result = musicbrainzngs.search_recordings(query=query, limit=5)
            
            recordings = result.get('recording-list', [])
            if not recordings:
                return None
            
            # 最初のリリース年を取得
            for rec in recordings:
                releases = rec.get('release-list', [])
                for rel in releases:
                    date = rel.get('date', '')
                    if date and len(date) >= 4:
                        year_str = date[:4]
                        if year_str.isdigit():
                            return int(year_str)
            
            return None
            
        except Exception:
            return None
    
    def _year_to_era(self, year: int) -> str:
        """年を年代文字列に変換"""
        if year < 1950:
            return "Pre-1950s"
        elif year < 1960:
            return "1950s"
        elif year < 1970:
            return "1960s"
        elif year < 1980:
            return "1970s"
        elif year < 1990:
            return "1980s"
        elif year < 2000:
            return "1990s"
        elif year < 2010:
            return "2000s"
        elif year < 2020:
            return "2010s"
        else:
            return "2020s"
    
    def analyze(self, audio: np.ndarray, sample_rate: int) -> Dict[str, Any]:
        """
        年代を解析
        
        注意: このアナライザーは音声データを使用せず、
        事前に set_metadata() で設定されたメタデータを使用する。
        """
        result = {
            'year': None,
            'era': None,
            'source': None,  # 'metadata', 'musicbrainz', 'unknown'
            'confidence': 0.0
        }
        
        # 1. 既に年が設定されている場合
        if self._metadata.get('year'):
            year = self._metadata['year']
            result['year'] = year
            result['era'] = self._year_to_era(year)
            result['source'] = 'metadata'
            result['confidence'] = 1.0
            return result
        
        # 2. ファイルのメタデータから取得
        file_path = self._metadata.get('file_path')
        if file_path:
            year = self._get_year_from_file(file_path)
            if year:
                result['year'] = year
                result['era'] = self._year_to_era(year)
                result['source'] = 'file_metadata'
                result['confidence'] = 1.0
                return result
        
        # 3. MusicBrainz検索（オプション）
        if self.use_musicbrainz:
            title = self._metadata.get('title')
            artist = self._metadata.get('artist')
            year = self._get_year_from_musicbrainz(title, artist)
            if year:
                result['year'] = year
                result['era'] = self._year_to_era(year)
                result['source'] = 'musicbrainz'
                result['confidence'] = 0.9  # 外部検索は少し信頼度を下げる
                return result
        
        # 年代不明
        result['source'] = 'unknown'
        return result
    
    def to_tag(self, result: Dict[str, Any]) -> str:
        """タグ文字列を生成"""
        tags = []
        
        era = result.get('era')
        if era:
            tags.append(f"[{self.tag_prefix}:{era}]")
        
        # 正確な年も出力（オプション）
        if self.include_exact_year:
            year = result.get('year')
            if year:
                tags.append(f"[Year:{year}]")
        
        return ''.join(tags)


def check_musicbrainz_availability():
    """MusicBrainzの利用可能性を確認"""
    if MUSICBRAINZ_AVAILABLE:
        print("✅ MusicBrainz API is available")
    else:
        print("❌ MusicBrainz API is not available")
        print("   Install with: pip install musicbrainzngs")


if __name__ == "__main__":
    check_musicbrainz_availability()
