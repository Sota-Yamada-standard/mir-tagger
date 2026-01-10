"""
アニソンデータベース照合モジュール

MyAnimeList (Jikan API) を使用してアニソン判定を行う。
ローカルキャッシュ（SQLite）で高速照合を実現。
"""

import sqlite3
import json
import time
import re
import unicodedata
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
import urllib.request
import urllib.error
import urllib.parse

# Jikan API設定
JIKAN_BASE_URL = "https://api.jikan.moe/v4"
JIKAN_RATE_LIMIT = 0.34  # 3 requests/sec → 0.34秒間隔

# キャッシュ設定
DEFAULT_CACHE_DIR = Path.home() / ".mir-tagger" / "cache"
CACHE_DB_NAME = "anisong_cache.db"
CACHE_EXPIRY_DAYS = 30  # キャッシュ有効期限


class JikanClient:
    """Jikan API (MyAnimeList) クライアント"""
    
    def __init__(self):
        self._last_request_time = 0.0
    
    def _rate_limit(self):
        """レート制限を守る"""
        elapsed = time.time() - self._last_request_time
        if elapsed < JIKAN_RATE_LIMIT:
            time.sleep(JIKAN_RATE_LIMIT - elapsed)
        self._last_request_time = time.time()
    
    def _request(self, endpoint: str, params: Dict = None) -> Optional[Dict]:
        """APIリクエストを実行"""
        self._rate_limit()
        
        url = f"{JIKAN_BASE_URL}/{endpoint}"
        if params:
            query = urllib.parse.urlencode(params)
            url = f"{url}?{query}"
        
        try:
            req = urllib.request.Request(url, headers={
                'User-Agent': 'MIR-Tagger/1.0 (https://github.com/)'
            })
            with urllib.request.urlopen(req, timeout=10) as response:
                return json.loads(response.read().decode('utf-8'))
        except urllib.error.HTTPError as e:
            if e.code == 429:  # Too Many Requests
                print(f"⚠️ Rate limited, waiting 1 second...")
                time.sleep(1)
                return self._request(endpoint, params)
            print(f"❌ HTTP Error {e.code}: {e.reason}")
            return None
        except Exception as e:
            print(f"❌ Request failed: {e}")
            return None
    
    def get_anime_themes(self, anime_id: int) -> Optional[Dict]:
        """アニメの主題歌情報を取得"""
        result = self._request(f"anime/{anime_id}/themes")
        if result and 'data' in result:
            return result['data']
        return None
    
    def get_anime_full(self, anime_id: int) -> Optional[Dict]:
        """アニメの全情報を取得（themes含む）"""
        result = self._request(f"anime/{anime_id}/full")
        if result and 'data' in result:
            return result['data']
        return None
    
    def search_anime(self, query: str, limit: int = 10) -> List[Dict]:
        """アニメを検索"""
        result = self._request("anime", {"q": query, "limit": limit})
        if result and 'data' in result:
            return result['data']
        return []
    
    def get_anime_list(self, page: int = 1, limit: int = 25) -> Tuple[List[Dict], bool]:
        """
        アニメ一覧を取得（ページネーション）
        
        Returns:
            (anime_list, has_next_page)
        """
        result = self._request("anime", {"page": page, "limit": limit, "order_by": "mal_id"})
        if result and 'data' in result:
            has_next = result.get('pagination', {}).get('has_next_page', False)
            return result['data'], has_next
        return [], False


class AnisongCache:
    """アニソンデータのローカルキャッシュ（SQLite）"""
    
    def __init__(self, cache_dir: Path = None):
        self.cache_dir = cache_dir or DEFAULT_CACHE_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = self.cache_dir / CACHE_DB_NAME
        self._init_db()
    
    def _init_db(self):
        """データベース初期化"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS songs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    title TEXT NOT NULL,
                    title_normalized TEXT NOT NULL,
                    artist TEXT,
                    artist_normalized TEXT,
                    anime_id INTEGER NOT NULL,
                    anime_title TEXT NOT NULL,
                    anime_title_normalized TEXT NOT NULL,
                    theme_type TEXT,  -- 'opening', 'ending', 'insert'
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS anime (
                    mal_id INTEGER PRIMARY KEY,
                    title TEXT NOT NULL,
                    title_english TEXT,
                    title_japanese TEXT,
                    year INTEGER,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS cache_meta (
                    key TEXT PRIMARY KEY,
                    value TEXT,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            # インデックス作成
            conn.execute("CREATE INDEX IF NOT EXISTS idx_songs_title ON songs(title_normalized)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_songs_artist ON songs(artist_normalized)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_songs_anime ON songs(anime_title_normalized)")
            conn.commit()
    
    @staticmethod
    def normalize_text(text: str) -> str:
        """テキストを正規化（照合用）"""
        if not text:
            return ""
        # Unicode正規化
        text = unicodedata.normalize('NFKC', text)
        # 小文字化
        text = text.lower()
        # 記号・スペース除去
        text = re.sub(r'[^\w]', '', text)
        return text
    
    def add_song(self, title: str, artist: str, anime_id: int, 
                 anime_title: str, theme_type: str = None):
        """楽曲をキャッシュに追加"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO songs (title, title_normalized, artist, artist_normalized,
                                   anime_id, anime_title, anime_title_normalized, theme_type)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                title,
                self.normalize_text(title),
                artist,
                self.normalize_text(artist),
                anime_id,
                anime_title,
                self.normalize_text(anime_title),
                theme_type
            ))
            conn.commit()
    
    def add_anime(self, mal_id: int, title: str, title_english: str = None,
                  title_japanese: str = None, year: int = None):
        """アニメをキャッシュに追加"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO anime (mal_id, title, title_english, title_japanese, year, updated_at)
                VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            """, (mal_id, title, title_english, title_japanese, year))
            conn.commit()
    
    def search_by_title(self, title: str, threshold: float = 0.8) -> List[Dict]:
        """曲名で検索（正規化マッチ）"""
        normalized = self.normalize_text(title)
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            # 完全一致
            cursor = conn.execute("""
                SELECT * FROM songs WHERE title_normalized = ?
            """, (normalized,))
            results = [dict(row) for row in cursor.fetchall()]
            
            # 部分一致（完全一致がない場合）
            if not results and len(normalized) >= 3:
                cursor = conn.execute("""
                    SELECT * FROM songs WHERE title_normalized LIKE ?
                """, (f"%{normalized}%",))
                results = [dict(row) for row in cursor.fetchall()]
        
        return results
    
    def search_by_artist(self, artist: str) -> List[Dict]:
        """アーティスト名で検索"""
        normalized = self.normalize_text(artist)
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            # 完全一致
            cursor = conn.execute("""
                SELECT * FROM songs WHERE artist_normalized = ?
            """, (normalized,))
            results = [dict(row) for row in cursor.fetchall()]
            
            # 部分一致
            if not results and len(normalized) >= 3:
                cursor = conn.execute("""
                    SELECT * FROM songs WHERE artist_normalized LIKE ?
                """, (f"%{normalized}%",))
                results = [dict(row) for row in cursor.fetchall()]
        
        return results
    
    def search_by_title_and_artist(self, title: str, artist: str) -> List[Dict]:
        """曲名とアーティスト名で検索"""
        title_norm = self.normalize_text(title)
        artist_norm = self.normalize_text(artist)
        
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            
            # 1. タイトル+アーティスト完全一致
            cursor = conn.execute("""
                SELECT * FROM songs 
                WHERE title_normalized LIKE ? AND artist_normalized LIKE ?
            """, (f"%{title_norm}%", f"%{artist_norm}%"))
            results = [dict(row) for row in cursor.fetchall()]
            
            # 2. タイトル一致（アーティストが見つからない場合）
            if not results:
                cursor = conn.execute("""
                    SELECT * FROM songs WHERE title_normalized LIKE ?
                """, (f"%{title_norm}%",))
                results = [dict(row) for row in cursor.fetchall()]
        
        return results
    
    def search_flexible(self, title: str, artist: str = None) -> List[Dict]:
        """柔軟な検索（タイトルの部分一致、括弧内のタイトルも検索）"""
        title_norm = self.normalize_text(title)
        results = []
        
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            
            # タイトル部分一致
            cursor = conn.execute("""
                SELECT * FROM songs WHERE title_normalized LIKE ?
            """, (f"%{title_norm}%",))
            results = [dict(row) for row in cursor.fetchall()]
            
            # アーティストでフィルタリング（指定された場合）
            if results and artist:
                artist_norm = self.normalize_text(artist)
                filtered = [r for r in results if artist_norm in r.get('artist_normalized', '')]
                if filtered:
                    results = filtered
        
        return results
    
    def get_stats(self) -> Dict[str, int]:
        """キャッシュ統計を取得"""
        with sqlite3.connect(self.db_path) as conn:
            song_count = conn.execute("SELECT COUNT(*) FROM songs").fetchone()[0]
            anime_count = conn.execute("SELECT COUNT(*) FROM anime").fetchone()[0]
            return {
                'songs': song_count,
                'anime': anime_count,
                'db_size_mb': round(self.db_path.stat().st_size / 1024 / 1024, 2)
            }
    
    def get_meta(self, key: str) -> Optional[str]:
        """メタデータを取得"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "SELECT value FROM cache_meta WHERE key = ?", (key,)
            )
            row = cursor.fetchone()
            return row[0] if row else None
    
    def set_meta(self, key: str, value: str):
        """メタデータを設定"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO cache_meta (key, value, updated_at)
                VALUES (?, ?, CURRENT_TIMESTAMP)
            """, (key, value))
            conn.commit()


class AnisongMatcher:
    """アニソン照合エンジン"""
    
    def __init__(self, cache_dir: Path = None):
        self.cache = AnisongCache(cache_dir)
        self.client = JikanClient()
    
    def is_anime_song(self, title: str, artist: str = None) -> Tuple[bool, Optional[Dict]]:
        """
        楽曲がアニソンかどうか判定
        
        Args:
            title: 曲名
            artist: アーティスト名（オプション）
        
        Returns:
            (is_anime_song, match_info)
            match_info: {'anime_title': str, 'anime_id': int, 'theme_type': str, 'confidence': float}
        """
        # キャッシュから検索
        if artist:
            results = self.cache.search_by_title_and_artist(title, artist)
        else:
            results = self.cache.search_by_title(title)
        
        if results:
            # 最も確実な結果を返す
            best = results[0]
            confidence = 1.0 if len(results) == 1 else 0.9
            return True, {
                'anime_title': best['anime_title'],
                'anime_id': best['anime_id'],
                'theme_type': best.get('theme_type', 'unknown'),
                'song_title': best['title'],
                'artist': best.get('artist', ''),
                'confidence': confidence,
                'match_count': len(results)
            }
        
        return False, None
    
    def build_cache_from_mal(self, max_anime: int = None, progress_callback=None):
        """
        MALからキャッシュを構築
        
        Args:
            max_anime: 取得する最大アニメ数（Noneで全件）
            progress_callback: 進捗コールバック (current, total, anime_title)
        """
        page = 1
        total_anime = 0
        total_songs = 0
        
        print("🔄 MALからアニソンデータを取得中...")
        
        while True:
            anime_list, has_next = self.client.get_anime_list(page=page, limit=25)
            
            if not anime_list:
                break
            
            for anime in anime_list:
                mal_id = anime.get('mal_id')
                anime_title = anime.get('title', '')
                
                if progress_callback:
                    progress_callback(total_anime, max_anime or '?', anime_title)
                
                # アニメ情報を保存
                self.cache.add_anime(
                    mal_id=mal_id,
                    title=anime_title,
                    title_english=anime.get('title_english'),
                    title_japanese=anime.get('title_japanese'),
                    year=anime.get('year')
                )
                
                # 主題歌情報を取得
                themes = self.client.get_anime_themes(mal_id)
                if themes:
                    # Opening themes
                    for op in themes.get('openings', []):
                        song_info = self._parse_theme_string(op)
                        if song_info:
                            self.cache.add_song(
                                title=song_info['title'],
                                artist=song_info.get('artist', ''),
                                anime_id=mal_id,
                                anime_title=anime_title,
                                theme_type='opening'
                            )
                            total_songs += 1
                    
                    # Ending themes
                    for ed in themes.get('endings', []):
                        song_info = self._parse_theme_string(ed)
                        if song_info:
                            self.cache.add_song(
                                title=song_info['title'],
                                artist=song_info.get('artist', ''),
                                anime_id=mal_id,
                                anime_title=anime_title,
                                theme_type='ending'
                            )
                            total_songs += 1
                
                total_anime += 1
                
                if max_anime and total_anime >= max_anime:
                    break
            
            if max_anime and total_anime >= max_anime:
                break
            
            if not has_next:
                break
            
            page += 1
            
            # 進捗表示
            if total_anime % 100 == 0:
                print(f"  📊 {total_anime} anime, {total_songs} songs processed")
        
        # メタデータ更新
        self.cache.set_meta('last_full_update', datetime.now().isoformat())
        self.cache.set_meta('total_anime', str(total_anime))
        self.cache.set_meta('total_songs', str(total_songs))
        
        print(f"\n✅ キャッシュ構築完了: {total_anime} anime, {total_songs} songs")
        stats = self.cache.get_stats()
        print(f"   DB size: {stats['db_size_mb']} MB")
        
        return {'anime': total_anime, 'songs': total_songs}
    
    def add_anime_by_id(self, mal_id: int) -> bool:
        """特定のアニメをMAL IDでキャッシュに追加"""
        try:
            anime_data = self.client.get_anime_full(mal_id)
            if not anime_data:
                return False
            
            anime_title = anime_data.get('title', '')
            
            # アニメ情報保存
            self.cache.add_anime(
                mal_id=mal_id,
                title=anime_title,
                title_english=anime_data.get('title_english'),
                title_japanese=anime_data.get('title_japanese'),
                year=anime_data.get('year')
            )
            
            # 主題歌情報
            song_count = 0
            themes = anime_data.get('theme', {})
            for op in themes.get('openings', []):
                song_info = self._parse_theme_string(op)
                if song_info:
                    self.cache.add_song(
                        title=song_info['title'],
                        artist=song_info.get('artist', ''),
                        anime_id=mal_id,
                        anime_title=anime_title,
                        theme_type='opening'
                    )
                    song_count += 1
            
            for ed in themes.get('endings', []):
                song_info = self._parse_theme_string(ed)
                if song_info:
                    self.cache.add_song(
                        title=song_info['title'],
                        artist=song_info.get('artist', ''),
                        anime_id=mal_id,
                        anime_title=anime_title,
                        theme_type='ending'
                    )
                    song_count += 1
            
            print(f"✅ Added: {anime_title} ({song_count} songs)")
            return True
            
        except Exception as e:
            print(f"❌ Error adding anime {mal_id}: {e}")
            return False
    
    def add_anime_by_search(self, query: str) -> List[int]:
        """アニメを検索してキャッシュに追加"""
        results = self.client.search_anime(query, limit=5)
        added_ids = []
        
        for anime in results:
            mal_id = anime.get('mal_id')
            if self.add_anime_by_id(mal_id):
                added_ids.append(mal_id)
        
        return added_ids
    
    def _parse_theme_string(self, theme_str: str) -> Optional[Dict]:
        """
        MALの主題歌文字列をパース
        例: '"Seikan Hikou (星間飛行)" by Megumi Nakajima (eps 12, 17)'
        """
        if not theme_str:
            return None
        
        # パターン1: "Title" by Artist (eps X)
        match = re.match(r'"([^"]+)"(?:\s+by\s+(.+?))?(?:\s*\(eps?\s*[\d,\s-]+\))?$', theme_str)
        if match:
            return {
                'title': match.group(1).strip(),
                'artist': match.group(2).strip() if match.group(2) else ''
            }
        
        # パターン2: 数字付き "1: "Title" by Artist"
        match = re.match(r'\d+:\s*"([^"]+)"(?:\s+by\s+(.+?))?(?:\s*\(.*\))?$', theme_str)
        if match:
            return {
                'title': match.group(1).strip(),
                'artist': match.group(2).strip() if match.group(2) else ''
            }
        
        # パターン3: シンプルに引用符内を抽出
        match = re.search(r'"([^"]+)"', theme_str)
        if match:
            title = match.group(1)
            # "by" 以降をアーティストとして抽出
            artist_match = re.search(r'by\s+(.+?)(?:\s*\(|$)', theme_str)
            return {
                'title': title,
                'artist': artist_match.group(1).strip() if artist_match else ''
            }
        
        return None


def test_search():
    """テスト検索"""
    matcher = AnisongMatcher()
    
    # キャッシュ統計
    stats = matcher.cache.get_stats()
    print(f"📊 Cache stats: {stats}")
    
    # テスト検索
    test_cases = [
        ("星間飛行", "中島愛"),
        ("星間飛行", None),
        ("ライオン", None),
        ("残酷な天使のテーゼ", None),
    ]
    
    for title, artist in test_cases:
        is_anime, info = matcher.is_anime_song(title, artist)
        print(f"\n🔍 '{title}' by '{artist or 'Unknown'}'")
        print(f"   Anime song: {is_anime}")
        if info:
            print(f"   Match: {info}")


def build_popular_anime_cache():
    """人気アニメのキャッシュを構築（推奨：初回セットアップ用）"""
    # 人気アニメのMAL ID（手動リスト）
    POPULAR_ANIME_IDS = [
        # マクロスシリーズ
        3572,   # マクロスF
        5310,   # マクロスF 劇場版
        28013,  # マクロスΔ
        # ガンダムシリーズ
        80,     # 機動戦士ガンダム
        2581,   # 機動戦士ガンダムSEED
        4224,   # 機動戦士ガンダム00
        # エヴァンゲリオン
        30,     # 新世紀エヴァンゲリオン
        2759,   # ヱヴァンゲリヲン新劇場版
        # その他人気作品
        1,      # カウボーイビバップ
        5,      # カウボーイビバップ 天国の扉
        20,     # NARUTO
        21,     # ONE PIECE
        1735,   # NARUTO -ナルト- 疾風伝
        16498,  # 進撃の巨人
        30276,  # ワンパンマン
        31964,  # BORUTO
        # アイドル・音楽系
        15051,  # ラブライブ!
        32526,  # ラブライブ!サンシャイン!!
        22319,  # アイドルマスター シンデレラガールズ
        10278,  # THE IDOLM@STER
        # 京アニ作品
        2966,   # けいおん!
        7791,   # けいおん!!
        22199,  # 涼宮ハルヒの憂鬱
        4181,   # CLANNAD
        # その他
        9253,   # Steins;Gate
        11757,  # ソードアート・オンライン
        21459,  # Fate/stay night UBW
        22535,  # Fate/Zero
        23273,  # 四月は君の嘘
        31240,  # Re:ゼロ
        37510,  # SSSS.GRIDMAN
        40748,  # 呪術廻戦
        48583,  # 推しの子
        51009,  # 葬送のフリーレン
    ]
    
    matcher = AnisongMatcher()
    print(f"🎵 人気アニメ {len(POPULAR_ANIME_IDS)} 作品のキャッシュを構築中...")
    
    total_songs = 0
    for i, mal_id in enumerate(POPULAR_ANIME_IDS):
        try:
            anime_data = matcher.client.get_anime_full(mal_id)
            if not anime_data:
                continue
            
            anime_title = anime_data.get('title', '')
            print(f"  [{i+1}/{len(POPULAR_ANIME_IDS)}] {anime_title}")
            
            # アニメ情報保存
            matcher.cache.add_anime(
                mal_id=mal_id,
                title=anime_title,
                title_english=anime_data.get('title_english'),
                title_japanese=anime_data.get('title_japanese'),
                year=anime_data.get('year')
            )
            
            # 主題歌情報
            themes = anime_data.get('theme', {})
            for op in themes.get('openings', []):
                song_info = matcher._parse_theme_string(op)
                if song_info:
                    matcher.cache.add_song(
                        title=song_info['title'],
                        artist=song_info.get('artist', ''),
                        anime_id=mal_id,
                        anime_title=anime_title,
                        theme_type='opening'
                    )
                    total_songs += 1
            
            for ed in themes.get('endings', []):
                song_info = matcher._parse_theme_string(ed)
                if song_info:
                    matcher.cache.add_song(
                        title=song_info['title'],
                        artist=song_info.get('artist', ''),
                        anime_id=mal_id,
                        anime_title=anime_title,
                        theme_type='ending'
                    )
                    total_songs += 1
                    
        except Exception as e:
            print(f"    ⚠️ Error: {e}")
    
    stats = matcher.cache.get_stats()
    print(f"\n✅ 完了: {stats['songs']} songs, {stats['anime']} anime")
    print(f"   DB size: {stats['db_size_mb']} MB")


if __name__ == "__main__":
    # テスト実行
    test_search()
