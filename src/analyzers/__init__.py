"""
アナライザーモジュール
このディレクトリに新しいアナライザーを追加すると自動的に登録される
"""
import importlib
import pkgutil
from pathlib import Path

from src.core.base_analyzer import AnalyzerRegistry

# このディレクトリ内の全モジュールを自動インポート
_package_dir = Path(__file__).parent

for _, module_name, _ in pkgutil.iter_modules([str(_package_dir)]):
    if not module_name.startswith('_'):
        importlib.import_module(f'.{module_name}', package=__name__)

# 便利なエクスポート
def get_analyzer(name: str):
    """名前からアナライザーインスタンスを取得"""
    analyzer_class = AnalyzerRegistry.get(name)
    if analyzer_class:
        return analyzer_class()
    return None

def list_analyzers():
    """利用可能なアナライザー一覧を取得"""
    return AnalyzerRegistry.list_names()

def get_all_analyzers():
    """全アナライザーのインスタンスを取得"""
    return {name: cls() for name, cls in AnalyzerRegistry.get_all().items()}
