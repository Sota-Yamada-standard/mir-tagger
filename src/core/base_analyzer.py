"""
アナライザー基底クラス
新しいアナライザーはこのクラスを継承して実装する
"""
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
import numpy as np


class BaseAnalyzer(ABC):
    """
    全アナライザーの基底クラス
    
    新しいタグ/解析機能を追加する場合:
    1. このクラスを継承
    2. name, description を設定
    3. analyze() メソッドを実装
    4. to_tag() メソッドを実装
    5. src/analyzers/ にファイルを配置（自動登録される）
    """
    
    # アナライザーの識別名（例: "key", "genre", "vocal_start"）
    name: str = "base"
    
    # アナライザーの説明
    description: str = "Base analyzer class"
    
    # タグ出力時のプレフィックス（例: "Key", "Genre"）
    tag_prefix: str = ""
    
    def __init__(self, sample_rate: int = 44100):
        self.sample_rate = sample_rate
    
    @abstractmethod
    def analyze(self, audio: np.ndarray, sample_rate: int) -> Dict[str, Any]:
        """
        音声データを解析する
        
        Args:
            audio: モノラル音声データ（numpy array）
            sample_rate: サンプルレート
            
        Returns:
            解析結果の辞書
        """
        raise NotImplementedError
    
    def to_tag(self, result: Dict[str, Any]) -> str:
        """
        解析結果をタグ文字列に変換する
        
        Args:
            result: analyze() の戻り値
            
        Returns:
            タグ文字列（例: "[Key:Am]"）
        """
        if self.tag_prefix and "value" in result:
            return f"[{self.tag_prefix}:{result['value']}]"
        return ""
    
    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}: {self.description}>"


class AnalyzerRegistry:
    """アナライザーの登録・管理を行うレジストリ"""
    
    _analyzers: Dict[str, type] = {}
    
    @classmethod
    def register(cls, analyzer_class: type) -> type:
        """
        アナライザークラスを登録する（デコレータとして使用可能）
        
        Usage:
            @AnalyzerRegistry.register
            class MyAnalyzer(BaseAnalyzer):
                ...
        """
        if hasattr(analyzer_class, 'name'):
            cls._analyzers[analyzer_class.name] = analyzer_class
        return analyzer_class
    
    @classmethod
    def get(cls, name: str) -> Optional[type]:
        """名前からアナライザークラスを取得"""
        return cls._analyzers.get(name)
    
    @classmethod
    def get_all(cls) -> Dict[str, type]:
        """登録済み全アナライザーを取得"""
        return cls._analyzers.copy()
    
    @classmethod
    def list_names(cls) -> list:
        """登録済みアナライザー名の一覧を取得"""
        return list(cls._analyzers.keys())
