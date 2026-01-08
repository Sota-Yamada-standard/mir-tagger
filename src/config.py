"""
MIRシステム設定
"""
from pathlib import Path
from typing import List, Optional
import yaml


class Config:
    """システム設定クラス"""
    
    # デフォルト設定
    DEFAULT_SAMPLE_RATE = 44100
    
    # 対応ファイル形式
    SUPPORTED_FORMATS = ['.mp3', '.wav', '.flac', '.aiff', '.m4a', '.ogg']
    
    # タグ出力形式
    TAG_FORMAT = "[{prefix}:{value}]"
    TAG_SEPARATOR = ""  # タグ間の区切り文字
    
    # 有効なアナライザー（Noneの場合は全て有効）
    ENABLED_ANALYZERS: Optional[List[str]] = None
    
    # パス設定
    PROJECT_ROOT = Path(__file__).parent.parent
    SAMPLES_DIR = PROJECT_ROOT / "tests" / "samples"
    
    @classmethod
    def is_supported_format(cls, file_path: str) -> bool:
        """対応しているファイル形式かチェック"""
        return Path(file_path).suffix.lower() in cls.SUPPORTED_FORMATS
    
    @classmethod
    def load_from_yaml(cls, yaml_path: str) -> None:
        """YAMLファイルから設定を読み込む"""
        with open(yaml_path, 'r') as f:
            config_data = yaml.safe_load(f)
        
        if config_data:
            for key, value in config_data.items():
                if hasattr(cls, key.upper()):
                    setattr(cls, key.upper(), value)
