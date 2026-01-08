"""
Energy（エネルギー）アナライザー
楽曲の全体的なエネルギーレベルを分析する
"""
import essentia.standard as es
import numpy as np
from typing import Dict, Any

from src.core.base_analyzer import BaseAnalyzer, AnalyzerRegistry


@AnalyzerRegistry.register
class EnergyAnalyzer(BaseAnalyzer):
    """楽曲のエネルギーレベルを検出するアナライザー"""
    
    name = "energy"
    description = "Analyzes overall energy level"
    tag_prefix = "Energy"
    
    # エネルギーレベルのラベル（正規化後のスコアに基づく）
    LEVELS = [
        (0.2, "VeryLow"),
        (0.4, "Low"),
        (0.6, "Medium"),
        (0.8, "High"),
        (float('inf'), "VeryHigh")
    ]
    
    def analyze(self, audio: np.ndarray, sample_rate: int) -> Dict[str, Any]:
        """
        エネルギーレベルを分析する
        
        Returns:
            {
                'rms': 0.123,
                'loudness': -12.5,
                'dynamic_range': 8.5,
                'label': 'Medium',
                'value': 'Medium'
            }
        """
        # RMS（二乗平均平方根）エネルギー
        rms = es.RMS()
        rms_value = rms(audio)
        
        # ラウドネス
        loudness = es.Loudness()
        loudness_value = loudness(audio)
        
        # ダイナミックレンジ（簡易計算）
        frame_size = 2048
        hop_size = 1024
        
        framecutter = es.FrameCutter(frameSize=frame_size, hopSize=hop_size)
        windowing = es.Windowing(type='hann')
        
        frames_rms = []
        frame = framecutter(audio)
        while len(frame) == frame_size:
            windowed = windowing(frame)
            frames_rms.append(rms(windowed))
            frame = framecutter(audio)
        
        if frames_rms:
            frames_rms = np.array(frames_rms)
            # 上位10%と下位10%の差をダイナミックレンジとして計算
            high = np.percentile(frames_rms, 90)
            low = np.percentile(frames_rms, 10)
            dynamic_range = 20 * np.log10(high / max(low, 1e-10))
        else:
            dynamic_range = 0.0
        
        # RMSを0-1に正規化してラベルを決定
        # 典型的なRMS値は0.0-0.3程度
        normalized_rms = min(rms_value / 0.3, 1.0)
        
        label = "Unknown"
        for threshold, lbl in self.LEVELS:
            if normalized_rms < threshold:
                label = lbl
                break
        
        return {
            'rms': float(rms_value),
            'loudness': float(loudness_value),
            'dynamic_range': float(dynamic_range),
            'label': label,
            'value': label
        }
    
    def to_tag(self, result: Dict[str, Any]) -> str:
        """エネルギー情報をタグ文字列に変換"""
        return f"[{self.tag_prefix}:{result['label']}]"
