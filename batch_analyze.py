#!/usr/bin/env python3
"""
一括解析・タグ書き込みスクリプト
指定ディレクトリ内の全音楽ファイルを処理する

使用例:
  # 基本的な使い方
  python batch_analyze.py /path/to/music -w --backup

  # rekordboxプレイリスト/再生履歴から優先処理
  python batch_analyze.py --from-rekordbox rekordbox.xml -w --backup -j 6

  # 処理済みスキップ（再実行時）
  python batch_analyze.py /path/to/music -w --backup --skip-tagged -j 6

  # 残りの曲を処理
  python batch_analyze.py --from-rekordbox rekordbox.xml --remaining -w --backup -j 6
"""
import sys
import argparse
import time
from pathlib import Path
from typing import List, Optional
import multiprocessing as mp

# プロジェクトルートをパスに追加
sys.path.insert(0, str(Path(__file__).parent))

from src.main import analyze_file
from src.metadata.tag_writer import TagWriter
from src.config import Config


def is_already_tagged(file_path: str) -> bool:
    """ファイルに既にMIRタグがあるか確認"""
    try:
        from mutagen import File
        from mutagen.id3 import ID3
        from mutagen.mp4 import MP4
        
        audio = File(file_path)
        if audio is None:
            return False
        
        comment = ''
        
        # MP3
        if hasattr(audio, 'tags') and audio.tags:
            if isinstance(audio.tags, ID3):
                for tag in audio.tags:
                    if tag.startswith('COMM'):
                        comment = str(audio.tags[tag].text[0])
                        break
            # M4A
            elif isinstance(audio, MP4):
                if '©cmt' in audio.tags:
                    comment = str(audio.tags['©cmt'][0])
        
        # MIRタグが存在するか
        return '[Genre:' in comment or '[Mood:' in comment or '[Key:' in comment
    except Exception:
        return False


def process_file(file_path: str, write_tags: bool, backup: bool, field: str,
                 skip_tagged: bool = False) -> dict:
    """単一ファイルを処理"""
    import gc
    import torch
    
    try:
        # スキップチェック
        if skip_tagged and is_already_tagged(file_path):
            return {
                'file': Path(file_path).name,
                'status': 'skipped',
                'reason': 'already tagged'
            }
        
        results = analyze_file(file_path)
        
        if write_tags:
            writer = TagWriter(backup=backup)
            writer.write_tags(
                file_path=file_path,
                tags=results['tags'],
                field=field,
                append=False
            )
        
        result = {
            'file': Path(file_path).name,
            'status': 'success',
            'tags': results['tags']
        }
        
        # メモリ解放（重要）
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        gc.collect()
        
        return result
    except Exception as e:
        # エラー時もメモリ解放
        gc.collect()
        return {
            'file': Path(file_path).name,
            'status': 'error',
            'error': str(e)
        }


def find_audio_files(directory: str, recursive: bool = True) -> List[Path]:
    """ディレクトリ内の音楽ファイルを検索"""
    dir_path = Path(directory)
    files = []
    
    pattern = "**/*" if recursive else "*"
    
    for ext in Config.SUPPORTED_FORMATS:
        files.extend(dir_path.glob(f"{pattern}{ext}"))
    
    return sorted(files)


def get_files_from_rekordbox(xml_path: str, remaining: bool = False) -> List[str]:
    """rekordbox XMLから処理対象ファイルを取得"""
    from src.utils.rekordbox_parser import RekordboxParser
    
    parser = RekordboxParser(xml_path)
    priority_files = parser.get_priority_files(
        include_playlists=True,
        include_played=True,
        include_rated=False
    )
    
    if remaining:
        return parser.get_remaining_files(priority_files)
    else:
        return priority_files


def format_time(seconds: float) -> str:
    """秒数を読みやすい形式に変換"""
    if seconds < 60:
        return f"{seconds:.0f}秒"
    elif seconds < 3600:
        return f"{seconds/60:.1f}分"
    else:
        return f"{seconds/3600:.1f}時間"


def main():
    parser = argparse.ArgumentParser(
        description='MIR - Batch audio analysis and tagging',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
例:
  # 基本的な使い方
  python batch_analyze.py /path/to/music -w --backup

  # rekordboxから優先処理
  python batch_analyze.py --from-rekordbox rekordbox.xml -w --backup -j 6

  # 処理済みスキップ
  python batch_analyze.py /path/to/music -w --backup --skip-tagged
        """
    )
    
    # 入力ソース（どちらか必須）
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument('directory', nargs='?', 
                              help='Directory containing audio files')
    source_group.add_argument('--from-rekordbox', metavar='XML',
                              help='rekordbox XML file for priority processing')
    
    # rekordboxオプション
    parser.add_argument('--remaining', action='store_true',
                        help='Process remaining files (not in playlists/played)')
    parser.add_argument('--stats-only', action='store_true',
                        help='Show rekordbox statistics only')
    
    # 処理オプション
    parser.add_argument('-w', '--write', action='store_true',
                        help='Write tags to files')
    parser.add_argument('--backup', action='store_true',
                        help='Create backup before writing')
    parser.add_argument('--field', choices=['comment', 'grouping'], 
                        default='comment', help='Tag field')
    parser.add_argument('--skip-tagged', action='store_true',
                        help='Skip files that already have MIR tags')
    
    # 実行オプション
    parser.add_argument('--no-recursive', action='store_true',
                        help='Do not search subdirectories')
    parser.add_argument('-j', '--jobs', type=int, default=1,
                        help='Number of parallel jobs (default: 1)')
    parser.add_argument('--dry-run', action='store_true',
                        help='Show files to process without processing')
    
    args = parser.parse_args()
    
    # ファイルリスト取得
    if args.from_rekordbox:
        # rekordbox XMLから取得
        from src.utils.rekordbox_parser import RekordboxParser
        
        print(f"📀 Loading rekordbox XML: {args.from_rekordbox}")
        parser_rb = RekordboxParser(args.from_rekordbox)
        stats = parser_rb.get_stats()
        
        print(f"\n=== rekordbox Statistics ===")
        print(f"  Total tracks:    {stats['total_tracks']:,}")
        print(f"  In playlists:    {stats['playlist_tracks']:,}")
        print(f"  Played:          {stats['played_tracks']:,}")
        print(f"  Priority (OR):   {stats['priority_tracks']:,}")
        print(f"  Remaining:       {stats['remaining_tracks']:,}")
        
        if args.stats_only:
            return
        
        files = get_files_from_rekordbox(args.from_rekordbox, args.remaining)
        mode = "remaining" if args.remaining else "priority"
        print(f"\n📁 {mode.capitalize()} files: {len(files)}")
    else:
        # ディレクトリから検索
        files = [str(f) for f in find_audio_files(args.directory, not args.no_recursive)]
        print(f"📁 Found {len(files)} audio files in: {args.directory}")
    
    if not files:
        print("❌ No files to process")
        return
    
    # 所要時間見積もり
    estimated_time = len(files) * 11.5 / max(args.jobs, 1)  # MPS想定
    print(f"⏱️  Estimated time: {format_time(estimated_time)} ({args.jobs} jobs)")
    
    if args.dry_run:
        print("\n📋 Files to process (first 20):")
        for f in files[:20]:
            print(f"  - {Path(f).name}")
        if len(files) > 20:
            print(f"  ... and {len(files) - 20} more")
        return
    
    # 処理実行
    print(f"\n🔄 Processing... (jobs: {args.jobs})")
    print("-" * 60)
    
    start_time = time.time()
    success_count = 0
    error_count = 0
    skipped_count = 0
    
    if args.jobs == 1:
        # シングルプロセス
        for i, file_path in enumerate(files, 1):
            result = process_file(
                str(file_path), 
                args.write, 
                args.backup, 
                args.field,
                args.skip_tagged
            )
            
            if result['status'] == 'success':
                print(f"[{i}/{len(files)}] ✅ {result['file']}")
                print(f"         {result['tags']}")
                success_count += 1
            elif result['status'] == 'skipped':
                print(f"[{i}/{len(files)}] ⏭️  {result['file']} (skipped)")
                skipped_count += 1
            else:
                print(f"[{i}/{len(files)}] ❌ {result['file']}")
                print(f"         Error: {result['error']}")
                error_count += 1
    else:
        # マルチプロセス（メモリ対策: 各ワーカーは5曲処理ごとに再起動）
        # maxtasksperchildで定期的にワーカーを再起動しメモリリークを防止
        chunk_size = 5  # 5曲ごとにワーカー再起動
        
        with mp.get_context('spawn').Pool(
            processes=args.jobs,
            maxtasksperchild=chunk_size  # 各ワーカーは5曲処理後に再起動
        ) as pool:
            # 非同期でタスクを投入
            tasks = [
                (str(f), args.write, args.backup, args.field, args.skip_tagged)
                for f in files
            ]
            
            # imap_unorderedで進捗表示しながら処理
            for i, result in enumerate(pool.starmap(process_file, tasks), 1):
                if result['status'] == 'success':
                    print(f"[{i}/{len(files)}] ✅ {result['file']}")
                    print(f"         {result['tags']}")
                    success_count += 1
                elif result['status'] == 'skipped':
                    print(f"[{i}/{len(files)}] ⏭️  {result['file']} (skipped)")
                    skipped_count += 1
                else:
                    print(f"[{i}/{len(files)}] ❌ {result['file']}")
                    print(f"         Error: {result['error']}")
                    error_count += 1
    
    # サマリー
    elapsed = time.time() - start_time
    print("-" * 60)
    print(f"✅ Success: {success_count}")
    print(f"⏭️  Skipped: {skipped_count}")
    print(f"❌ Errors:  {error_count}")
    print(f"⏱️  Time:    {format_time(elapsed)}")
    
    if success_count > 0:
        avg_time = elapsed / success_count
        print(f"📊 Average: {avg_time:.1f}秒/曲")
    
    if args.write:
        print("\n💡 rekordboxで「タグを再読み込み」を実行してください")


if __name__ == '__main__':
    main()
