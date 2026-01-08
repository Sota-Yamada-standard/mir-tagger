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
        'analysis': {}
    }
    
    tags = []
    
    for name, analyzer_class in analyzers.items():
        analyzer = analyzer_class()
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
