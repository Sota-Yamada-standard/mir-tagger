"""
Danceability（踊りやすさ）アナライザー
Essentiaを使用して楽曲のDanceabilityスコアを算出する
"""
import essentia.standard as es
import numpy as np
from typing import Dict, Any

from src.core.base_analyzer import BaseAnalyzer, AnalyzerRegistry


@AnalyzerRegistry.register
class DanceabilityAnalyzer(BaseAnalyzer):
    """楽曲のDanceability（踊りやすさ）を検出するアナライザー"""
    
    name = "danceability"
    description = "Calculates danceability score (0.0-3.0)"
    tag_prefix = "Dance"
    
    # スコアの閾値とラベル
    THRESHOLDS = [
        (0.5, "Low"),
        (1.0, "Medium"),
        (1.5, "High"),
        (float('inf'), "VeryHigh")
    ]
    
    def analyze(self, audio: np.ndarray, sample_rate: int) -> Dict[str, Any]:
        """
        Danceabilityを計算する
        
        Returns:
            {
                'score': 1.23,
                'label': 'High',
                'value': 'High'
            }
        """
        danceability = es.Danceability()
        score, _ = danceability(audio)
        
        # スコアからラベルを決定
        label = "Unknown"
        for threshold, lbl in self.THRESHOLDS:
            if score < threshold:
                label = lbl
                break
        
        return {
            'score': float(score),
            'label': label,
            'value': label
        }
    
    def to_tag(self, result: Dict[str, Any]) -> str:
        """Danceability情報をタグ文字列に変換"""
        return f"[{self.tag_prefix}:{result['label']}]"
