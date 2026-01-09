"""
rekordbox XMLパーサー
プレイリスト・再生履歴から優先処理対象を抽出
"""
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Set, Optional
from urllib.parse import unquote
import os


class RekordboxParser:
    """
    rekordbox XMLファイルをパースし、優先処理対象のファイルパスを抽出
    
    使用方法:
        parser = RekordboxParser('rekordbox.xml')
        priority_files = parser.get_priority_files()
    """
    
    def __init__(self, xml_path: str):
        """
        Args:
            xml_path: rekordbox XMLファイルのパス
        """
        self.xml_path = Path(xml_path)
        self.tree = ET.parse(self.xml_path)
        self.root = self.tree.getroot()
        
        # トラック情報をキャッシュ
        self._tracks: Dict[str, Dict] = {}
        self._load_tracks()
    
    def _load_tracks(self):
        """COLLECTIONからトラック情報を読み込み"""
        collection = self.root.find('.//COLLECTION')
        if collection is None:
            return
        
        for track in collection.findall('TRACK'):
            track_id = track.attrib.get('TrackID', '')
            if track_id:
                self._tracks[track_id] = {
                    'id': track_id,
                    'name': track.attrib.get('Name', ''),
                    'artist': track.attrib.get('Artist', ''),
                    'album': track.attrib.get('Album', ''),
                    'location': self._decode_location(track.attrib.get('Location', '')),
                    'play_count': int(track.attrib.get('PlayCount', 0)),
                    'rating': int(track.attrib.get('Rating', 0)),
                    'date_added': track.attrib.get('DateAdded', ''),
                }
    
    def _decode_location(self, location: str) -> str:
        """
        rekordboxのLocationをファイルパスに変換
        例: file://localhost/Users/... → /Users/...
        """
        if not location:
            return ''
        
        # file://localhost/ を除去
        if location.startswith('file://localhost'):
            location = location[16:]  # len('file://localhost') = 16
        elif location.startswith('file://'):
            location = location[7:]
        
        # URLエンコードをデコード
        location = unquote(location)
        
        return location
    
    def get_all_tracks(self) -> List[Dict]:
        """全トラック情報を取得"""
        return list(self._tracks.values())
    
    def get_playlist_track_ids(self) -> Set[str]:
        """プレイリストに入っているトラックIDを取得"""
        playlist_ids = set()
        
        playlists = self.root.find('.//PLAYLISTS')
        if playlists is None:
            return playlist_ids
        
        # Type="1" がプレイリスト（Type="0" はフォルダ）
        for node in playlists.findall('.//NODE[@Type="1"]'):
            for track_ref in node.findall('TRACK'):
                key = track_ref.attrib.get('Key', '')
                if key:
                    playlist_ids.add(key)
        
        return playlist_ids
    
    def get_played_track_ids(self, min_play_count: int = 1) -> Set[str]:
        """再生履歴のあるトラックIDを取得"""
        played_ids = set()
        
        for track_id, track in self._tracks.items():
            if track['play_count'] >= min_play_count:
                played_ids.add(track_id)
        
        return played_ids
    
    def get_rated_track_ids(self, min_rating: int = 1) -> Set[str]:
        """レーティングのあるトラックIDを取得"""
        rated_ids = set()
        
        for track_id, track in self._tracks.items():
            if track['rating'] >= min_rating:
                rated_ids.add(track_id)
        
        return rated_ids
    
    def get_priority_files(
        self,
        include_playlists: bool = True,
        include_played: bool = True,
        include_rated: bool = False,
        min_play_count: int = 1,
        min_rating: int = 1
    ) -> List[str]:
        """
        優先処理対象のファイルパスを取得
        
        Args:
            include_playlists: プレイリストに入っているトラックを含める
            include_played: 再生履歴のあるトラックを含める
            include_rated: レーティングのあるトラックを含める
            min_play_count: 最小再生回数
            min_rating: 最小レーティング
        
        Returns:
            ファイルパスのリスト
        """
        priority_ids = set()
        
        if include_playlists:
            priority_ids.update(self.get_playlist_track_ids())
        
        if include_played:
            priority_ids.update(self.get_played_track_ids(min_play_count))
        
        if include_rated:
            priority_ids.update(self.get_rated_track_ids(min_rating))
        
        # ファイルパスに変換
        files = []
        for track_id in priority_ids:
            track = self._tracks.get(track_id)
            if track and track['location']:
                # ファイルが存在するか確認
                if os.path.exists(track['location']):
                    files.append(track['location'])
        
        return sorted(files)
    
    def get_remaining_files(self, priority_files: List[str]) -> List[str]:
        """
        優先処理対象以外のファイルパスを取得
        
        Args:
            priority_files: 優先処理対象のファイルパスリスト
        
        Returns:
            残りのファイルパスのリスト
        """
        priority_set = set(priority_files)
        
        remaining = []
        for track in self._tracks.values():
            if track['location'] and track['location'] not in priority_set:
                if os.path.exists(track['location']):
                    remaining.append(track['location'])
        
        return sorted(remaining)
    
    def get_stats(self) -> Dict:
        """統計情報を取得"""
        playlist_ids = self.get_playlist_track_ids()
        played_ids = self.get_played_track_ids()
        rated_ids = self.get_rated_track_ids()
        
        priority_ids = playlist_ids | played_ids
        
        return {
            'total_tracks': len(self._tracks),
            'playlist_tracks': len(playlist_ids),
            'played_tracks': len(played_ids),
            'rated_tracks': len(rated_ids),
            'priority_tracks': len(priority_ids),
            'remaining_tracks': len(self._tracks) - len(priority_ids),
        }


def main():
    """テスト用"""
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python rekordbox_parser.py <rekordbox.xml>")
        sys.exit(1)
    
    parser = RekordboxParser(sys.argv[1])
    stats = parser.get_stats()
    
    print("=== rekordbox XML Statistics ===")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    print("\n=== Priority Files (first 5) ===")
    priority_files = parser.get_priority_files()
    for f in priority_files[:5]:
        print(f"  {f}")


if __name__ == '__main__':
    main()
