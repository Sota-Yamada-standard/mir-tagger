#!/usr/bin/env python3
"""
タグ書き込みテストスクリプト
サンプルファイルを解析してタグを書き込む
"""
import sys
from pathlib import Path

# プロジェクトルートをパスに追加
sys.path.insert(0, str(Path(__file__).parent))

from src.main import analyze_file
from src.metadata.tag_writer import TagWriter


def main():
    # サンプルファイル
    sample_file = Path(__file__).parent / "tests" / "samples" / "10 星間飛行.mp3"
    
    print("=" * 60)
    print("MIR - タグ書き込みテスト")
    print("=" * 60)
    
    if not sample_file.exists():
        print(f"\n❌ サンプルファイルが見つかりません: {sample_file}")
        return
    
    # タグライター初期化（バックアップ有効）
    writer = TagWriter(backup=True)
    
    # 現在のタグを確認
    print(f"\n📄 ファイル: {sample_file.name}")
    print("\n【書き込み前のタグ】")
    current_tags = writer.read_tags(str(sample_file))
    for key, value in current_tags.items():
        print(f"  {key}: {value}")
    
    # 解析実行
    print("\n" + "-" * 60)
    print("🔍 解析中...")
    results = analyze_file(str(sample_file))
    
    print(f"\n📊 解析結果:")
    for name, data in results['analysis'].items():
        if 'error' not in data:
            print(f"  {name}: {data.get('value', data)}")
    
    print(f"\n🏷️  生成タグ: {results['tags']}")
    
    # タグを書き込み
    print("\n" + "-" * 60)
    print("✍️  タグを書き込み中...")
    
    write_result = writer.write_tags(
        file_path=str(sample_file),
        tags=results['tags'],
        field="comment",
        append=False  # 上書きモード
    )
    
    print(f"\n✅ 書き込み完了!")
    print(f"  フィールド: {write_result['field']}")
    print(f"  タグ: {write_result['tags_written']}")
    if write_result['backup_path']:
        print(f"  バックアップ: {write_result['backup_path']}")
    
    # 書き込み後のタグを確認
    print("\n【書き込み後のタグ】")
    updated_tags = writer.read_tags(str(sample_file))
    for key, value in updated_tags.items():
        print(f"  {key}: {value}")
    
    print("\n" + "=" * 60)
    print("💡 rekordboxでの確認手順:")
    print("  1. rekordboxを開く")
    print("  2. 対象トラックを右クリック")
    print("  3. 「情報を再読み込み」を選択")
    print("  4. Comment欄にタグが表示されることを確認")
    print("=" * 60)


if __name__ == '__main__':
    main()
