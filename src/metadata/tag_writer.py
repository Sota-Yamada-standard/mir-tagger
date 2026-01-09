"""
メタデータ（タグ）書き込みモジュール
MP3, FLAC, M4A, AIFFなど複数形式に対応
"""
from pathlib import Path
from typing import Optional, Dict, Any, List
import shutil
from datetime import datetime

from mutagen import File as MutagenFile
from mutagen.id3 import ID3, COMM, TCON, TIT1
from mutagen.flac import FLAC
from mutagen.mp4 import MP4
from mutagen.aiff import AIFF


class TagWriter:
    """
    音楽ファイルにメタデータを書き込むクラス
    
    対応形式:
    - MP3 (ID3v2)
    - FLAC (Vorbis Comments)
    - M4A/MP4 (iTunes tags)
    - AIFF (ID3v2)
    """
    
    # MIRタグを識別するためのプレフィックス
    MIR_TAG_PREFIX = "[MIR]"
    
    # Comment欄に書き込む際の説明
    COMMENT_DESCRIPTION = "MIR"
    
    def __init__(self, backup: bool = False, backup_dir: Optional[str] = None):
        """
        Args:
            backup: 書き込み前にバックアップを作成するか
            backup_dir: バックアップ先ディレクトリ（Noneの場合は元ファイルと同じ場所）
        """
        self.backup = backup
        self.backup_dir = Path(backup_dir) if backup_dir else None
    
    def _create_backup(self, file_path: Path) -> Optional[Path]:
        """ファイルのバックアップを作成"""
        if not self.backup:
            return None
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_name = f"{file_path.stem}_backup_{timestamp}{file_path.suffix}"
        
        if self.backup_dir:
            self.backup_dir.mkdir(parents=True, exist_ok=True)
            backup_path = self.backup_dir / backup_name
        else:
            backup_path = file_path.parent / backup_name
        
        shutil.copy2(file_path, backup_path)
        return backup_path
    
    def write_tags(
        self,
        file_path: str,
        tags: str,
        field: str = "comment",
        append: bool = True,
        genre: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        タグを音楽ファイルに書き込む
        
        Args:
            file_path: 音楽ファイルのパス
            tags: 書き込むタグ文字列（例: "[Key:Am][Dance:High]"）
            field: 書き込み先フィールド ("comment" or "grouping")
            append: Trueの場合、既存のコメントに追記。Falseの場合は上書き
            genre: ジャンルを設定する場合に指定
        
        Returns:
            {
                'success': True,
                'file': 'track.mp3',
                'field': 'comment',
                'tags_written': '[Key:Am][Dance:High]',
                'backup_path': '/path/to/backup' or None
            }
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # バックアップ
        backup_path = self._create_backup(path)
        
        # ファイル形式に応じて処理
        suffix = path.suffix.lower()
        
        if suffix == '.mp3':
            self._write_mp3(path, tags, field, append, genre)
        elif suffix == '.flac':
            self._write_flac(path, tags, field, append, genre)
        elif suffix in ['.m4a', '.mp4']:
            self._write_m4a(path, tags, field, append, genre)
        elif suffix in ['.aiff', '.aif']:
            self._write_aiff(path, tags, field, append, genre)
        else:
            raise ValueError(f"Unsupported file format: {suffix}")
        
        return {
            'success': True,
            'file': path.name,
            'field': field,
            'tags_written': tags,
            'backup_path': str(backup_path) if backup_path else None
        }
    
    def _write_mp3(
        self,
        path: Path,
        tags: str,
        field: str,
        append: bool,
        genre: Optional[str]
    ):
        """MP3ファイル（ID3タグ）への書き込み"""
        try:
            audio = ID3(str(path))
        except Exception:
            # ID3タグがない場合は新規作成
            audio = ID3()
        
        if field == "comment":
            # 既存のMIRコメントを探す
            existing = ""
            if append:
                for key in audio.keys():
                    if key.startswith("COMM"):
                        frame = audio[key]
                        if hasattr(frame, 'desc') and frame.desc == self.COMMENT_DESCRIPTION:
                            # MIRコメントは上書き
                            pass
                        elif hasattr(frame, 'text'):
                            # 他のコメントは保持
                            existing = str(frame.text[0]) if frame.text else ""
            
            # 新しいコメントを構築
            if append and existing and not existing.endswith(']'):
                new_comment = f"{existing} {tags}"
            elif append and existing:
                new_comment = f"{existing}{tags}"
            else:
                new_comment = tags
            
            # MIRタグ用のコメントを追加/更新
            # rekordboxは desc="" のCOMMタグを読み取るため、空にする
            audio.delall("COMM")
            audio.add(COMM(
                encoding=3,  # UTF-8
                lang='eng',
                desc='',  # rekordbox互換のため空文字
                text=new_comment
            ))
        
        elif field == "grouping":
            # Grouping (TIT1) への書き込み
            audio.delall("TIT1")
            audio.add(TIT1(encoding=3, text=tags))
        
        # ジャンル
        if genre:
            audio.delall("TCON")
            audio.add(TCON(encoding=3, text=genre))
        
        audio.save(str(path))
    
    def _write_flac(
        self,
        path: Path,
        tags: str,
        field: str,
        append: bool,
        genre: Optional[str]
    ):
        """FLACファイル（Vorbisコメント）への書き込み"""
        audio = FLAC(str(path))
        
        if field == "comment":
            tag_key = "comment"
        else:
            tag_key = "grouping"
        
        if append and tag_key in audio:
            existing = audio[tag_key][0]
            if existing and not existing.endswith(']'):
                audio[tag_key] = f"{existing} {tags}"
            else:
                audio[tag_key] = f"{existing}{tags}"
        else:
            audio[tag_key] = tags
        
        if genre:
            audio["genre"] = genre
        
        audio.save()
    
    def _write_m4a(
        self,
        path: Path,
        tags: str,
        field: str,
        append: bool,
        genre: Optional[str]
    ):
        """M4A/MP4ファイル（iTunesタグ）への書き込み"""
        audio = MP4(str(path))
        
        if field == "comment":
            tag_key = "\xa9cmt"  # Comment
        else:
            tag_key = "\xa9grp"  # Grouping
        
        if append and tag_key in audio:
            existing = audio[tag_key][0]
            if existing and not existing.endswith(']'):
                audio[tag_key] = [f"{existing} {tags}"]
            else:
                audio[tag_key] = [f"{existing}{tags}"]
        else:
            audio[tag_key] = [tags]
        
        if genre:
            audio["\xa9gen"] = [genre]
        
        audio.save()
    
    def _write_aiff(
        self,
        path: Path,
        tags: str,
        field: str,
        append: bool,
        genre: Optional[str]
    ):
        """AIFFファイル（ID3タグ）への書き込み"""
        # AIFFもID3タグを使用
        try:
            audio = AIFF(str(path))
            if audio.tags is None:
                audio.add_tags()
        except Exception:
            audio = AIFF(str(path))
            audio.add_tags()
        
        # MP3と同様の処理
        if field == "comment":
            audio.tags.delall("COMM")
            audio.tags.add(COMM(
                encoding=3,
                lang='eng',
                desc=self.COMMENT_DESCRIPTION,
                text=tags
            ))
        elif field == "grouping":
            audio.tags.delall("TIT1")
            audio.tags.add(TIT1(encoding=3, text=tags))
        
        if genre:
            audio.tags.delall("TCON")
            audio.tags.add(TCON(encoding=3, text=genre))
        
        audio.save()
    
    def read_tags(self, file_path: str) -> Dict[str, Any]:
        """
        音楽ファイルから現在のタグを読み取る
        
        Returns:
            {
                'comment': '...',
                'grouping': '...',
                'genre': '...',
                'title': '...',
                'artist': '...'
            }
        """
        path = Path(file_path)
        audio = MutagenFile(str(path))
        
        if audio is None:
            raise ValueError(f"Cannot read file: {file_path}")
        
        result = {
            'comment': None,
            'grouping': None,
            'genre': None,
            'title': None,
            'artist': None
        }
        
        suffix = path.suffix.lower()
        
        if suffix == '.mp3':
            if hasattr(audio, 'tags') and audio.tags:
                # Comment
                for key in audio.tags.keys():
                    if key.startswith("COMM"):
                        result['comment'] = str(audio.tags[key].text[0])
                        break
                # その他
                if "TIT1" in audio.tags:
                    result['grouping'] = str(audio.tags["TIT1"].text[0])
                if "TCON" in audio.tags:
                    result['genre'] = str(audio.tags["TCON"].text[0])
                if "TIT2" in audio.tags:
                    result['title'] = str(audio.tags["TIT2"].text[0])
                if "TPE1" in audio.tags:
                    result['artist'] = str(audio.tags["TPE1"].text[0])
        
        elif suffix == '.flac':
            result['comment'] = audio.get('comment', [None])[0]
            result['grouping'] = audio.get('grouping', [None])[0]
            result['genre'] = audio.get('genre', [None])[0]
            result['title'] = audio.get('title', [None])[0]
            result['artist'] = audio.get('artist', [None])[0]
        
        elif suffix in ['.m4a', '.mp4']:
            result['comment'] = audio.get('\xa9cmt', [None])[0]
            result['grouping'] = audio.get('\xa9grp', [None])[0]
            result['genre'] = audio.get('\xa9gen', [None])[0]
            result['title'] = audio.get('\xa9nam', [None])[0]
            result['artist'] = audio.get('\xa9ART', [None])[0]
        
        return result
