#!/usr/bin/env python3
"""
解析テストスクリプト
サンプルファイルで動作確認を行う
"""
import sys
from pathlib import Path

# プロジェクトルートをパスに追加
sys.path.insert(0, str(Path(__file__).parent))

from src.main import analyze_file
from src.analyzers import list_analyzers
import json


def main():
    # サンプルファイル
    sample_file = Path(__file__).parent / "tests" / "samples" / "10 星間飛行.mp3"
    
    print("=" * 60)
    print("MIR - Music Information Retrieval テスト")
    print("=" * 60)
    
    # 利用可能なアナライザー一覧
    print("\n📋 利用可能なアナライザー:")
    for name in list_analyzers():
        print(f"  - {name}")
    
    # ファイル存在確認
    if not sample_file.exists():
        print(f"\n❌ サンプルファイルが見つかりません: {sample_file}")
        return
    
    print(f"\n🎵 解析対象: {sample_file.name}")
    print("-" * 60)
    
    # 解析実行
    try:
        results = analyze_file(str(sample_file))
        
        print(f"\n⏱️  長さ: {results['duration_sec']} 秒")
        print("\n📊 解析結果:")
        
        for analyzer_name, data in results['analysis'].items():
            print(f"\n  【{analyzer_name}】")
            if 'error' in data:
                print(f"    ❌ Error: {data['error']}")
            else:
                for key, value in data.items():
                    if key != 'value':  # value は内部用
                        print(f"    {key}: {value}")
        
        print("\n" + "-" * 60)
        print(f"🏷️  生成タグ: {results['tags']}")
        print("=" * 60)
        
        # JSON出力も保存
        output_file = Path(__file__).parent / "test_output.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\n💾 詳細結果を保存: {output_file}")
        
    except Exception as e:
        print(f"\n❌ 解析エラー: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
