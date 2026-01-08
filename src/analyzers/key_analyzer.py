"""
Key（調）検出アナライザー
Essentiaを使用して楽曲のキーを検出する
"""
import essentia.standard as es
import numpy as np
from typing import Dict, Any

from src.core.base_analyzer import BaseAnalyzer, AnalyzerRegistry


@AnalyzerRegistry.register
class KeyAnalyzer(BaseAnalyzer):
    """楽曲のキー（調）を検出するアナライザー"""
    
    name = "key"
    description = "Detects musical key (e.g., C major, A minor)"
    tag_prefix = "Key"
    
    # キー表記のマッピング（Camelot/Open Key対応も可能）
    KEY_NOTATION = {
        # メジャーキー（シャープ表記）
        'C major': 'C', 'C# major': 'C#', 'D major': 'D',
        'D# major': 'D#', 'E major': 'E', 'F major': 'F',
        'F# major': 'F#', 'G major': 'G', 'G# major': 'G#',
        'A major': 'A', 'A# major': 'A#', 'B major': 'B',
        # メジャーキー（フラット表記）
        'Db major': 'Db', 'Eb major': 'Eb', 'Gb major': 'Gb',
        'Ab major': 'Ab', 'Bb major': 'Bb',
        # マイナーキー（シャープ表記）
        'C minor': 'Cm', 'C# minor': 'C#m', 'D minor': 'Dm',
        'D# minor': 'D#m', 'E minor': 'Em', 'F minor': 'Fm',
        'F# minor': 'F#m', 'G minor': 'Gm', 'G# minor': 'G#m',
        'A minor': 'Am', 'A# minor': 'A#m', 'B minor': 'Bm',
        # マイナーキー（フラット表記）
        'Db minor': 'Dbm', 'Eb minor': 'Ebm', 'Gb minor': 'Gbm',
        'Ab minor': 'Abm', 'Bb minor': 'Bbm',
    }
    
    def analyze(self, audio: np.ndarray, sample_rate: int) -> Dict[str, Any]:
        """
        キーを検出する
        
        Returns:
            {
                'key': 'A',
                'scale': 'minor',
                'key_full': 'A minor',
                'strength': 0.85,
                'value': 'Am'  # タグ用の短縮形
            }
        """
        # Essentiaのキー検出を使用
        key_extractor = es.KeyExtractor()
        key, scale, strength = key_extractor(audio)
        
        # フル表記
        key_full = f"{key} {scale}"
        
        # 短縮表記
        value = self.KEY_NOTATION.get(key_full, key_full)
        
        return {
            'key': key,
            'scale': scale,
            'key_full': key_full,
            'strength': float(strength),
            'value': value
        }
    
    def to_tag(self, result: Dict[str, Any]) -> str:
        """キー情報をタグ文字列に変換"""
        return f"[{self.tag_prefix}:{result['value']}]"
