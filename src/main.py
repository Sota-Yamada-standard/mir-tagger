"""
MIR - Music Information Retrieval System
メインエントリポイント
"""
import json
import sys
from pathlib import Path
from typing import List, Optional, Dict, Any

# プロジェクトルートをパスに追加
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.audio_loader import AudioLoader
from src.core.base_analyzer import AnalyzerRegistry
from src.config import Config
from src.metadata.tag_writer import TagWriter

# アナライザーを自動インポート（登録）
import src.analyzers


def get_file_metadata(file_path: str) -> Dict[str, str]:
    """
    音楽ファイルからメタデータを取得
    
    Returns:
        {'title': str, 'artist': str, 'album': str}
    """
    from mutagen import File
    from mutagen.id3 import ID3
    from mutagen.mp4 import MP4
    
    metadata = {'title': None, 'artist': None, 'album': None}
    path = Path(file_path)
    
    def decode_tag_value(tag_frame):
        """ID3タグのテキストを適切にデコード"""
        if hasattr(tag_frame, 'text'):
            # 複数のテキスト値がある場合
            texts = tag_frame.text if isinstance(tag_frame.text, list) else [tag_frame.text]
            for text in texts:
                if text:
                    # 既にUnicodeならそのまま返す
                    if isinstance(text, str):
                        return text
                    # バイト列ならデコードを試みる
                    if isinstance(text, bytes):
                        for enc in ['utf-8', 'shift_jis', 'cp932', 'euc-jp', 'latin-1']:
                            try:
                                return text.decode(enc)
                            except (UnicodeDecodeError, AttributeError):
                                continue
        return str(tag_frame) if tag_frame else None
    
    try:
        audio = File(file_path)
        if audio is None:
            # ファイル名から情報を抽出
            metadata['title'] = path.stem
            return metadata
        
        # MP3 (ID3)
        if hasattr(audio, 'tags') and audio.tags:
            tags = audio.tags
            
            # ID3 tags (MP3)
            if hasattr(tags, 'getall'):
                # Title
                for tag in ['TIT2', 'TIT1']:
                    if tag in tags:
                        metadata['title'] = decode_tag_value(tags[tag])
                        break
                # Artist
                for tag in ['TPE1', 'TPE2']:
                    if tag in tags:
                        metadata['artist'] = decode_tag_value(tags[tag])
                        break
                # Album
                if 'TALB' in tags:
                    metadata['album'] = decode_tag_value(tags['TALB'])
            
            # MP4/M4A tags
            elif isinstance(tags, dict):
                if '\xa9nam' in tags:
                    val = tags['\xa9nam']
                    metadata['title'] = val[0] if isinstance(val, list) else str(val)
                if '\xa9ART' in tags:
                    val = tags['\xa9ART']
                    metadata['artist'] = val[0] if isinstance(val, list) else str(val)
                if '\xa9alb' in tags:
                    val = tags['\xa9alb']
                    metadata['album'] = val[0] if isinstance(val, list) else str(val)
        
        # ファイル名から曲名を推測（ID3タグが文字化けしている場合の対策）
        stem = path.stem
        import re
        title_match = re.match(r'^\d+[\s._-]*(.+)$', stem)
        filename_title = title_match.group(1) if title_match else stem
        
        # タイトルが文字化けしているか判定（非ASCII文字が少なすぎる、または制御文字が含まれる）
        def is_garbled(text):
            if not text:
                return True
            # 制御文字が含まれている
            if any(ord(c) < 32 and c not in '\n\r\t' for c in text):
                return True
            # ASCII+日本語以外の奇妙な文字が含まれている
            import unicodedata
            for c in text:
                cat = unicodedata.category(c)
                if cat.startswith('C') and c not in '\n\r\t':  # 制御文字
                    return True
            return False
        
        # タグが文字化けしていたらファイル名を優先
        if is_garbled(metadata['title']):
            metadata['title'] = filename_title
        
        # それでもダメならファイル名を使用
        if not metadata['title']:
            metadata['title'] = filename_title
        
    except Exception as e:
        # エラー時はファイル名をタイトルとして使用
        import re
        stem = path.stem
        title_match = re.match(r'^\d+[\s._-]*(.+)$', stem)
        metadata['title'] = title_match.group(1) if title_match else stem
    
    return metadata


def analyze_file(
    file_path: str,
    analyzer_names: Optional[List[str]] = None,
    output_format: str = "json"
) -> Dict[str, Any]:
    """
    音楽ファイルを解析する
    
    Args:
        file_path: 音楽ファイルのパス
        analyzer_names: 使用するアナライザー名のリスト（Noneで全て）
        output_format: 出力形式 ("json" or "tags")
    
    Returns:
        解析結果の辞書
    """
    # ファイル形式チェック
    if not Config.is_supported_format(file_path):
        raise ValueError(f"Unsupported file format: {file_path}")
    
    # メタデータ取得（AnisongAnalyzer用）
    file_metadata = get_file_metadata(file_path)
    
    # 音声読み込み
    loader = AudioLoader()
    audio, sr = loader.load(file_path)
    duration = len(audio) / sr
    
    # アナライザー取得
    all_analyzers = AnalyzerRegistry.get_all()
    
    if analyzer_names:
        analyzers = {name: all_analyzers[name] for name in analyzer_names if name in all_analyzers}
    else:
        analyzers = all_analyzers
    
    # 解析実行
    results = {
        'file': str(Path(file_path).name),
        'duration_sec': round(duration, 2),
        'metadata': file_metadata,
        'analysis': {}
    }
    
    tags = []
    
    for name, analyzer_class in analyzers.items():
        analyzer = analyzer_class()
        
        # メタデータを必要とするアナライザーに設定
        if hasattr(analyzer, 'set_metadata'):
            if name == 'anisong':
                analyzer.set_metadata(
                    title=file_metadata.get('title'),
                    artist=file_metadata.get('artist')
                )
            elif name == 'era':
                analyzer.set_metadata(
                    title=file_metadata.get('title'),
                    artist=file_metadata.get('artist'),
                    album=file_metadata.get('album'),
                    file_path=file_path
                )
        
        try:
            result = analyzer.analyze(audio, sr)
            results['analysis'][name] = result
            
            # タグ生成
            tag = analyzer.to_tag(result)
            if tag:
                tags.append(tag)
        except Exception as e:
            results['analysis'][name] = {'error': str(e)}
    
    results['tags'] = ''.join(tags)
    
    return results


def main():
    """CLI エントリポイント"""
    import argparse
    
    parser = argparse.ArgumentParser(description='MIR - Music Information Retrieval')
    parser.add_argument('file', nargs='?', help='Audio file to analyze')
    parser.add_argument('-a', '--analyzers', nargs='+', help='Specific analyzers to use')
    parser.add_argument('-f', '--format', choices=['json', 'tags'], default='json',
                        help='Output format')
    parser.add_argument('-l', '--list', action='store_true', help='List available analyzers')
    parser.add_argument('-w', '--write', action='store_true', 
                        help='Write tags to file (Comment field)')
    parser.add_argument('--backup', action='store_true',
                        help='Create backup before writing tags')
    parser.add_argument('--field', choices=['comment', 'grouping'], default='comment',
                        help='Field to write tags to (default: comment)')
    
    args = parser.parse_args()
    
    if args.list:
        print("Available analyzers:")
        for name, cls in AnalyzerRegistry.get_all().items():
            print(f"  - {name}: {cls.description}")
        return
    
    if not args.file:
        parser.print_help()
        return
    
    try:
        results = analyze_file(args.file, args.analyzers, args.format)
        
        if args.format == 'json':
            print(json.dumps(results, indent=2, ensure_ascii=False))
        else:
            print(results['tags'])
        
        # タグ書き込み
        if args.write:
            writer = TagWriter(backup=args.backup)
            write_result = writer.write_tags(
                file_path=args.file,
                tags=results['tags'],
                field=args.field,
                append=False
            )
            print(f"\n✅ Tags written to {args.field}: {results['tags']}", file=sys.stderr)
            if write_result['backup_path']:
                print(f"   Backup: {write_result['backup_path']}", file=sys.stderr)
    
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
