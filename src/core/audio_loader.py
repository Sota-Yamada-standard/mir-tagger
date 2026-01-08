"""
音声ファイル読み込みモジュール
各アナライザーで共通利用する
"""
import essentia.standard as es
from pathlib import Path
from typing import Tuple, Optional
import numpy as np


class AudioLoader:
    """音声ファイルの読み込みを担当するクラス"""
    
    DEFAULT_SAMPLE_RATE = 44100
    
    def __init__(self, sample_rate: int = DEFAULT_SAMPLE_RATE):
        self.sample_rate = sample_rate
    
    def load(self, file_path: str) -> Tuple[np.ndarray, int]:
        """
        音声ファイルを読み込む
        
        Args:
            file_path: 音声ファイルのパス
            
        Returns:
            (audio_data, sample_rate) のタプル
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Audio file not found: {file_path}")
        
        loader = es.MonoLoader(filename=str(path), sampleRate=self.sample_rate)
        audio = loader()
        
        return audio, self.sample_rate
    
    def load_stereo(self, file_path: str) -> Tuple[np.ndarray, np.ndarray, int]:
        """
        ステレオ音声ファイルを読み込む
        
        Args:
            file_path: 音声ファイルのパス
            
        Returns:
            (left_channel, right_channel, sample_rate) のタプル
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Audio file not found: {file_path}")
        
        loader = es.AudioLoader(filename=str(path))
        audio, sr, channels, md5, bit_rate, codec = loader()
        
        if audio.shape[1] >= 2:
            return audio[:, 0], audio[:, 1], int(sr)
        else:
            # モノラルの場合は同じデータを返す
            return audio[:, 0], audio[:, 0], int(sr)
    
    def get_duration(self, file_path: str) -> float:
        """
        音声ファイルの長さを取得（秒）
        """
        audio, sr = self.load(file_path)
        return len(audio) / sr
