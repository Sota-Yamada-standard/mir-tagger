"""
ムード解析アナライザー
Essentia TensorFlowのムード専用モデルを使用
"""
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Tuple
from src.core.base_analyzer import BaseAnalyzer, AnalyzerRegistry

try:
    from essentia.standard import TensorflowPredictMusiCNN, TensorflowPredict2D
    ESSENTIA_TF_AVAILABLE = True
except ImportError:
    ESSENTIA_TF_AVAILABLE = False


@AnalyzerRegistry.register
class MoodAnalyzer(BaseAnalyzer):
    """
    ムード解析アナライザー
    
    Essentiaのムード専用モデル（mood_happy, mood_sad, mood_relaxed, mood_aggressive）を使用。
    MusiCNNの特徴量を抽出し、各ムードモデルで分類。
    
    出力タグ:
    - [Mood:happy] - 明るい、楽しい
    - [Mood:sad] - 切ない、悲しい
    - [Mood:relaxed] - 穏やか、リラックス
    - [Mood:aggressive] - 激しい、攻撃的
    """
    
    name = "mood"
    description = "Mood analysis using Essentia TensorFlow models"
    tag_prefix = "Mood"
    
    # 閾値設定
    MOOD_THRESHOLD = 0.75    # 特徴的なムードの閾値
    MIN_CONFIDENCE = 0.5     # 最低限の自信度（曖昧な場合に最高スコアを採用する閾値）
    
    # モデルパス
    MODELS_DIR = Path("/Users/hanyanty/essentia_models")
    MUSICNN_MODEL = "msd-musicnn-1.pb"
    MOOD_MODELS = {
        "happy": "mood_happy-msd-musicnn-1.pb",
        "sad": "mood_sad-msd-musicnn-1.pb",
        "relaxed": "mood_relaxed-msd-musicnn-1.pb",
        "aggressive": "mood_aggressive-msd-musicnn-1.pb",
    }
    
    # モデルキャッシュ
    _musicnn_model = None
    _mood_models = {}

    @classmethod
    def clear_model_cache(cls):
        """モデルキャッシュをクリアしてメモリを解放"""
        import gc
        if cls._musicnn_model is not None:
            del cls._musicnn_model
            cls._musicnn_model = None
        for key in list(cls._mood_models.keys()):
            del cls._mood_models[key]
        cls._mood_models = {}
        gc.collect()

    def __init__(
        self,
        mood_threshold: float = MOOD_THRESHOLD,
        min_confidence: float = MIN_CONFIDENCE,
        models_dir: str = None
    ):
        """
        Args:
            mood_threshold: 特徴的なムードの閾値（これを超えたら出力）
            min_confidence: 最低限の自信度（曖昧な場合に最高スコアを採用する閾値）
            models_dir: モデルディレクトリのパス
        """
        self.mood_threshold = mood_threshold
        self.min_confidence = min_confidence
        self.models_dir = Path(models_dir) if models_dir else self.MODELS_DIR
    
    def _get_musicnn_model(self):
        """MusiCNN特徴量抽出モデルを取得"""
        if not ESSENTIA_TF_AVAILABLE:
            return None
        if MoodAnalyzer._musicnn_model is None:
            model_path = self.models_dir / self.MUSICNN_MODEL
            if model_path.exists():
                MoodAnalyzer._musicnn_model = TensorflowPredictMusiCNN(
                    graphFilename=str(model_path),
                    output='model/dense/BiasAdd'  # 特徴量レイヤー
                )
        return MoodAnalyzer._musicnn_model
    
    def _get_mood_model(self, mood_name: str):
        """ムード分類モデルを取得"""
        if not ESSENTIA_TF_AVAILABLE:
            return None
        if mood_name not in MoodAnalyzer._mood_models:
            model_file = self.MOOD_MODELS.get(mood_name)
            if model_file:
                model_path = self.models_dir / model_file
                if model_path.exists():
                    MoodAnalyzer._mood_models[mood_name] = TensorflowPredict2D(
                        graphFilename=str(model_path),
                        input='model/Placeholder',
                        output='model/Softmax'
                    )
        return MoodAnalyzer._mood_models.get(mood_name)
    
    def analyze(self, audio: np.ndarray, sample_rate: int) -> Dict[str, Any]:
        """
        ムードを解析
        
        Returns:
            {
                'moods': {'happy': 0.94, 'sad': 0.89, ...},
                'detected_moods': ['happy', 'sad', 'relaxed'],
                'primary_mood': 'relaxed',
                'value': 'relaxed'
            }
        """
        result = {
            'moods': {},
            'detected_moods': [],
            'primary_mood': None,
            'value': None
        }
        
        if not ESSENTIA_TF_AVAILABLE:
            return result
        
        # MusiCNNで特徴量抽出
        musicnn = self._get_musicnn_model()
        if musicnn is None:
            return result
        
        # 16kHzにリサンプリング（MusiCNNの要件）
        if sample_rate != 16000:
            import librosa
            audio_16k = librosa.resample(audio, orig_sr=sample_rate, target_sr=16000)
        else:
            audio_16k = audio
        
        embeddings = musicnn(audio_16k)
        
        # 各ムードモデルで分類
        mood_scores = {}
        for mood_name in self.MOOD_MODELS.keys():
            model = self._get_mood_model(mood_name)
            if model is not None:
                preds = model(embeddings)
                # Softmax出力の最初のクラス（例: happy vs non_happy の happy）
                score = float(np.mean(preds[:, 0]))
                mood_scores[mood_name] = score
        
        result['moods'] = mood_scores
        
        if not mood_scores:
            return result
        
        # 閾値ロジック
        # 1. MOOD_THRESHOLD を超えるもの全て
        detected = [m for m, score in mood_scores.items() if score > self.mood_threshold]
        
        # 2. なければ、最高スコアが MIN_CONFIDENCE 以上なら採用
        if not detected:
            top_mood, top_score = max(mood_scores.items(), key=lambda x: x[1])
            if top_score >= self.min_confidence:
                detected = [top_mood]
        
        # スコア順にソート
        detected = sorted(detected, key=lambda m: mood_scores[m], reverse=True)
        
        result['detected_moods'] = detected
        result['primary_mood'] = detected[0] if detected else None
        result['value'] = result['primary_mood']
        
        return result
    
    def to_tag(self, result: Dict[str, Any]) -> str:
        """タグ文字列を生成"""
        detected = result.get('detected_moods', [])
        
        if not detected:
            return ""
        
        tags = []
        for mood in detected:
            tags.append(f"[{self.tag_prefix}:{mood}]")
        
        return ''.join(tags)
