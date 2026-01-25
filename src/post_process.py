"""
ポストプロセッサー - 既存タグから派生タグを生成
音声解析なしで高速にタグを追加・更新する

使用例:
    # 単一ファイル
    python -m src.post_process /path/to/file.mp3 --add-club-tag
    
    # ディレクトリ（再帰的）
    python -m src.post_process /path/to/music/ --add-club-tag
    
    # ドライラン（実際には書き込まない）
    python -m src.post_process /path/to/music/ --add-club-tag --dry-run
"""
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set, Any
from dataclasses import dataclass

# プロジェクトルートをパスに追加
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.metadata.tag_writer import TagWriter
from src.config import Config


@dataclass
class ParsedTags:
    """パース済みタグを保持するデータクラス"""
    raw: str  # 元のタグ文字列
    tags: Dict[str, List[str]]  # タグ名 -> 値のリスト（複数可）
    
    def has_tag(self, name: str, value: Optional[str] = None) -> bool:
        """指定タグが存在するかチェック"""
        if name not in self.tags:
            return False
        if value is None:
            return True
        return value in self.tags[name]
    
    def get_values(self, name: str) -> List[str]:
        """タグの値リストを取得"""
        return self.tags.get(name, [])
    
    def get_value(self, name: str) -> Optional[str]:
        """タグの最初の値を取得"""
        values = self.tags.get(name, [])
        return values[0] if values else None


class TagParser:
    """タグ文字列をパースするクラス"""
    
    # タグパターン: [TagName:Value]
    TAG_PATTERN = re.compile(r'\[([^:]+):([^\]]+)\]')
    
    @classmethod
    def parse(cls, tag_string: str) -> ParsedTags:
        """
        タグ文字列をパースする
        
        Args:
            tag_string: "[Dance:High][Genre:house][Genre:techno]" のような文字列
        
        Returns:
            ParsedTags オブジェクト
        """
        if not tag_string:
            return ParsedTags(raw="", tags={})
        
        tags: Dict[str, List[str]] = {}
        
        for match in cls.TAG_PATTERN.finditer(tag_string):
            name = match.group(1)
            value = match.group(2)
            
            if name not in tags:
                tags[name] = []
            tags[name].append(value)
        
        return ParsedTags(raw=tag_string, tags=tags)
    
    @classmethod
    def add_tag(cls, tag_string: str, name: str, value: str) -> str:
        """
        タグ文字列に新しいタグを追加
        同じタグが既に存在する場合は追加しない
        
        Args:
            tag_string: 既存のタグ文字列
            name: 追加するタグ名
            value: 追加する値
        
        Returns:
            新しいタグ文字列
        """
        new_tag = f"[{name}:{value}]"
        
        # 既に存在するかチェック
        if new_tag in (tag_string or ""):
            return tag_string
        
        if tag_string:
            return f"{tag_string}{new_tag}"
        return new_tag
    
    @classmethod
    def remove_tag(cls, tag_string: str, name: str, value: Optional[str] = None) -> str:
        """
        タグ文字列からタグを削除
        
        Args:
            tag_string: 既存のタグ文字列
            name: 削除するタグ名
            value: 削除する値（Noneの場合は該当タグ名を全て削除）
        
        Returns:
            新しいタグ文字列
        """
        if not tag_string:
            return ""
        
        if value:
            # 特定の値のタグのみ削除
            pattern = re.compile(rf'\[{re.escape(name)}:{re.escape(value)}\]')
        else:
            # 該当タグ名を全て削除
            pattern = re.compile(rf'\[{re.escape(name)}:[^\]]+\]')
        
        return pattern.sub('', tag_string)


class ClubTagProcessor:
    """
    クラブミュージック判定タグを追加するプロセッサー
    
    判定条件:
    1. Genre:house, Genre:techno, Genre:disco, Genre:trance のいずれか
    2. (Genre:electronic OR Genre:electro OR Genre:dance) AND Dance:High
    3. Attr:dance AND (Energy:High OR Energy:VeryHigh) AND Dance:High
    """
    
    # 条件A: 明確にクラブ系のジャンル
    CLUB_GENRES = {'house', 'techno', 'disco', 'trance'}
    
    # 条件B: ダンサビリティと組み合わせるジャンル
    ELECTRONIC_GENRES = {'electronic', 'electro', 'dance'}
    
    TAG_NAME = "Club"
    TAG_VALUE = "Yes"
    
    def __init__(self):
        self.tag_writer = TagWriter()
    
    def is_club_music(self, parsed: ParsedTags) -> bool:
        """
        クラブミュージックかどうかを判定
        
        Returns:
            True if the track is classified as club music
        """
        genres = set(parsed.get_values('Genre'))
        dance = parsed.get_value('Dance')
        energy = parsed.get_value('Energy')
        attrs = set(parsed.get_values('Attr'))
        
        # 条件A: 明確にクラブ系ジャンル
        if genres & self.CLUB_GENRES:
            return True
        
        # 条件B: エレクトロニック系 + Dance:High
        if (genres & self.ELECTRONIC_GENRES) and dance == 'High':
            return True
        
        # 条件C: Attr:dance + 高エネルギー + 高ダンサビリティ
        if ('dance' in attrs and 
            energy in ('High', 'VeryHigh') and 
            dance == 'High'):
            return True
        
        return False
    
    def process_file(self, file_path: str, dry_run: bool = False) -> Dict[str, Any]:
        """
        単一ファイルを処理
        
        Args:
            file_path: 処理するファイルパス
            dry_run: Trueの場合、実際には書き込まない
        
        Returns:
            {
                'file': str,
                'action': 'added' | 'skipped' | 'already_has' | 'error',
                'is_club': bool,
                'tags_before': str,
                'tags_after': str,
                'error': Optional[str]
            }
        """
        result = {
            'file': str(Path(file_path).name),
            'action': 'skipped',
            'is_club': False,
            'tags_before': '',
            'tags_after': '',
            'error': None
        }
        
        try:
            # 既存タグを読み取り
            current_tags = self.tag_writer.read_tags(file_path)
            comment = current_tags.get('comment') or ''
            result['tags_before'] = comment
            
            # パース
            parsed = TagParser.parse(comment)
            
            # 既に[Club:Yes]がある場合はスキップ
            if parsed.has_tag('Club', 'Yes'):
                result['action'] = 'already_has'
                result['is_club'] = True
                result['tags_after'] = comment
                return result
            
            # クラブミュージック判定
            is_club = self.is_club_music(parsed)
            result['is_club'] = is_club
            
            if not is_club:
                result['action'] = 'skipped'
                result['tags_after'] = comment
                return result
            
            # タグを追加
            new_tags = TagParser.add_tag(comment, self.TAG_NAME, self.TAG_VALUE)
            result['tags_after'] = new_tags
            
            if not dry_run:
                # 書き込み
                self.tag_writer.write_tags(
                    file_path=file_path,
                    tags=new_tags,
                    field='comment',
                    append=False
                )
            
            result['action'] = 'added'
            return result
            
        except Exception as e:
            result['action'] = 'error'
            result['error'] = str(e)
            return result
    
    def process_directory(
        self,
        directory: str,
        dry_run: bool = False,
        recursive: bool = True,
        verbose: bool = True
    ) -> Dict[str, Any]:
        """
        ディレクトリ内の全ファイルを処理
        
        Args:
            directory: 処理するディレクトリ
            dry_run: Trueの場合、実際には書き込まない
            recursive: サブディレクトリも処理するか
            verbose: 進捗を表示するか
        
        Returns:
            {
                'total': int,
                'added': int,
                'skipped': int,
                'already_has': int,
                'errors': int,
                'files': List[Dict]
            }
        """
        path = Path(directory)
        
        # 対応する拡張子のファイルを収集
        files = []
        pattern = '**/*' if recursive else '*'
        for ext in Config.SUPPORTED_FORMATS:
            files.extend(path.glob(f"{pattern}{ext}"))
        
        summary = {
            'total': len(files),
            'added': 0,
            'skipped': 0,
            'already_has': 0,
            'errors': 0,
            'files': []
        }
        
        for i, file_path in enumerate(files, 1):
            if verbose:
                print(f"\r[{i}/{len(files)}] Processing...", end='', file=sys.stderr)
            
            result = self.process_file(str(file_path), dry_run=dry_run)
            summary['files'].append(result)
            
            if result['action'] == 'added':
                summary['added'] += 1
                if verbose:
                    print(f"\r✅ {result['file']}: [Club:Yes] added", file=sys.stderr)
            elif result['action'] == 'already_has':
                summary['already_has'] += 1
            elif result['action'] == 'error':
                summary['errors'] += 1
                if verbose:
                    print(f"\r❌ {result['file']}: {result['error']}", file=sys.stderr)
            else:
                summary['skipped'] += 1
        
        if verbose:
            print(f"\r" + " " * 50, end='\r', file=sys.stderr)
        
        return summary


def main():
    """CLI エントリポイント"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='MIR Post Processor - 既存タグから派生タグを生成'
    )
    parser.add_argument('path', help='処理するファイルまたはディレクトリ')
    parser.add_argument('--add-club-tag', action='store_true',
                        help='クラブミュージック判定タグ [Club:Yes] を追加')
    parser.add_argument('--dry-run', action='store_true',
                        help='実際には書き込まず、結果のみ表示')
    parser.add_argument('--no-recursive', action='store_true',
                        help='サブディレクトリを処理しない')
    parser.add_argument('-q', '--quiet', action='store_true',
                        help='進捗表示を抑制')
    
    args = parser.parse_args()
    
    if not args.add_club_tag:
        print("Error: --add-club-tag オプションを指定してください", file=sys.stderr)
        parser.print_help()
        sys.exit(1)
    
    path = Path(args.path)
    
    if not path.exists():
        print(f"Error: パスが存在しません: {args.path}", file=sys.stderr)
        sys.exit(1)
    
    processor = ClubTagProcessor()
    
    if path.is_file():
        # 単一ファイル処理
        result = processor.process_file(str(path), dry_run=args.dry_run)
        
        prefix = "[DRY-RUN] " if args.dry_run else ""
        
        if result['action'] == 'added':
            print(f"{prefix}✅ {result['file']}: [Club:Yes] を追加しました")
            print(f"   Before: {result['tags_before']}")
            print(f"   After:  {result['tags_after']}")
        elif result['action'] == 'already_has':
            print(f"{prefix}⏭️  {result['file']}: 既に [Club:Yes] があります")
        elif result['action'] == 'skipped':
            print(f"{prefix}⏭️  {result['file']}: クラブミュージックではありません")
        else:
            print(f"{prefix}❌ {result['file']}: エラー - {result['error']}")
    
    else:
        # ディレクトリ処理
        if args.dry_run:
            print("🔍 ドライランモード（実際には書き込みません）\n", file=sys.stderr)
        
        summary = processor.process_directory(
            str(path),
            dry_run=args.dry_run,
            recursive=not args.no_recursive,
            verbose=not args.quiet
        )
        
        print(f"\n📊 処理結果:", file=sys.stderr)
        print(f"   総ファイル数: {summary['total']}", file=sys.stderr)
        print(f"   ✅ Club:Yes 追加: {summary['added']}", file=sys.stderr)
        print(f"   ⏭️  スキップ（非クラブ）: {summary['skipped']}", file=sys.stderr)
        print(f"   ⏭️  スキップ（既存）: {summary['already_has']}", file=sys.stderr)
        print(f"   ❌ エラー: {summary['errors']}", file=sys.stderr)


if __name__ == '__main__':
    main()
