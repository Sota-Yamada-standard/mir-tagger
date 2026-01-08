#!/usr/bin/env python3
"""
一括解析・タグ書き込みスクリプト
指定ディレクトリ内の全音楽ファイルを処理する
"""
import sys
import argparse
from pathlib import Path
from typing import List
from concurrent.futures import ProcessPoolExecutor, as_completed

# プロジェクトルートをパスに追加
sys.path.insert(0, str(Path(__file__).parent))

from src.main import analyze_file
from src.metadata.tag_writer import TagWriter
from src.config import Config


def process_file(file_path: str, write_tags: bool, backup: bool, field: str) -> dict:
    """単一ファイルを処理"""
    try:
        results = analyze_file(file_path)
        
        if write_tags:
            writer = TagWriter(backup=backup)
            writer.write_tags(
                file_path=file_path,
                tags=results['tags'],
                field=field,
                append=False
            )
        
        return {
            'file': Path(file_path).name,
            'status': 'success',
            'tags': results['tags']
        }
    except Exception as e:
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


def main():
    parser = argparse.ArgumentParser(
        description='MIR - Batch audio analysis and tagging'
    )
    parser.add_argument('directory', help='Directory containing audio files')
    parser.add_argument('-w', '--write', action='store_true',
                        help='Write tags to files')
    parser.add_argument('--backup', action='store_true',
                        help='Create backup before writing')
    parser.add_argument('--field', choices=['comment', 'grouping'], 
                        default='comment', help='Tag field')
    parser.add_argument('--no-recursive', action='store_true',
                        help='Do not search subdirectories')
    parser.add_argument('-j', '--jobs', type=int, default=1,
                        help='Number of parallel jobs (default: 1)')
    parser.add_argument('--dry-run', action='store_true',
                        help='Show files to process without processing')
    
    args = parser.parse_args()
    
    # ファイル検索
    files = find_audio_files(args.directory, not args.no_recursive)
    
    if not files:
        print(f"❌ No audio files found in: {args.directory}")
        return
    
    print(f"📁 Found {len(files)} audio files")
    
    if args.dry_run:
        print("\n📋 Files to process:")
        for f in files:
            print(f"  - {f.name}")
        return
    
    # 処理実行
    print(f"\n🔄 Processing... (jobs: {args.jobs})")
    print("-" * 60)
    
    success_count = 0
    error_count = 0
    
    if args.jobs == 1:
        # シングルプロセス
        for i, file_path in enumerate(files, 1):
            result = process_file(
                str(file_path), 
                args.write, 
                args.backup, 
                args.field
            )
            
            status_icon = "✅" if result['status'] == 'success' else "❌"
            print(f"[{i}/{len(files)}] {status_icon} {result['file']}")
            
            if result['status'] == 'success':
                print(f"         {result['tags']}")
                success_count += 1
            else:
                print(f"         Error: {result['error']}")
                error_count += 1
    else:
        # マルチプロセス
        with ProcessPoolExecutor(max_workers=args.jobs) as executor:
            futures = {
                executor.submit(
                    process_file, str(f), args.write, args.backup, args.field
                ): f for f in files
            }
            
            for i, future in enumerate(as_completed(futures), 1):
                result = future.result()
                status_icon = "✅" if result['status'] == 'success' else "❌"
                print(f"[{i}/{len(files)}] {status_icon} {result['file']}")
                
                if result['status'] == 'success':
                    print(f"         {result['tags']}")
                    success_count += 1
                else:
                    print(f"         Error: {result['error']}")
                    error_count += 1
    
    # サマリー
    print("-" * 60)
    print(f"✅ Success: {success_count}")
    print(f"❌ Errors:  {error_count}")
    
    if args.write:
        print("\n💡 rekordboxで「情報を再読み込み」を実行してください")


if __name__ == '__main__':
    main()
