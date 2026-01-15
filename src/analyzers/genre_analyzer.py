"""
ジャンル分類アナライザー
PANNs + MusiCNN + Essentiaルールのマルチモデルアンサンブル
"""
import json
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
from src.core.base_analyzer import BaseAnalyzer, AnalyzerRegistry
import essentia.standard as es

# PANNsのインポート（オプショナル）
try:
    from panns_inference import AudioTagging
    import librosa
    PANNS_AVAILABLE = True
except ImportError:
    PANNS_AVAILABLE = False

# Essentia TensorFlowのインポート（オプショナル）
try:
    from essentia.standard import TensorflowPredictMusiCNN, MonoLoader
    MUSICNN_AVAILABLE = True
except ImportError:
    MUSICNN_AVAILABLE = False


@AnalyzerRegistry.register
class GenreAnalyzer(BaseAnalyzer):
    """
    音楽ジャンルを推定するマルチモデルアンサンブルアナライザー
    
    使用モデル:
    1. MusiCNN (Essentia TensorFlow) - 音楽ジャンル特化、50タグ
    2. PANNs - 汎用音声分類、527クラス
    3. Essentiaルール - カスタムジャンル（anime, j-pop等）
    """
    
    name = "genre"
    description = "Genre classification (MusiCNN + PANNs + Essentia ensemble)"
    tag_prefix = "Genre"
    
    # デフォルト設定
    DEFAULT_THRESHOLD = 0.15
    MAX_GENRES = 3
    
    # モデルの重み（アンサンブル用）
    # 合計が1.0になるよう正規化され、最終スコアは0-1の範囲
    MODEL_WEIGHTS = {
        'musicnn': 0.6,   # 音楽特化なので高め
        'panns': 0.2,     # 汎用（ジャンルスコアが低い傾向）
        'essentia': 0.2,  # ルールベース補助
    }
    
    # MusiCNNラベル（50タグ）
    MUSICNN_LABELS = [
        "rock", "pop", "alternative", "indie", "electronic",
        "female vocalists", "dance", "00s", "alternative rock", "jazz",
        "beautiful", "metal", "chillout", "male vocalists", "classic rock",
        "soul", "indie rock", "Mellow", "electronica", "80s",
        "folk", "90s", "chill", "instrumental", "punk",
        "oldies", "blues", "hard rock", "ambient", "acoustic",
        "experimental", "female vocalist", "guitar", "Hip-Hop", "70s",
        "party", "country", "easy listening", "sexy", "catchy",
        "funk", "electro", "heavy metal", "Progressive rock", "60s",
        "rnb", "indie pop", "sad", "House", "happy"
    ]
    
    # MusiCNNラベルの正規化マッピング
    MUSICNN_NORMALIZE = {
        "female vocalists": "vocal-female",
        "male vocalists": "vocal-male",
        "female vocalist": "vocal-female",
        "Hip-Hop": "hip-hop",
        "Progressive rock": "progressive-rock",
        "House": "house",
        "Mellow": "mellow",
        # 年代タグはEraAnalyzerで処理するためスキップ
        # "00s": "2000s",
        # "90s": "1990s",
        # "80s": "1980s",
        # "70s": "1970s",
        # "60s": "1960s",
    }
    
    # 年代タグ（ジャンルから除外）
    ERA_TAGS = {"00s", "90s", "80s", "70s", "60s", "2000s", "1990s", "1980s", "1970s", "1960s"}
    
    # PANNsラベルマッピング（主要なもの）
    # 属性（_で始まる）はジャンルではなく楽器/特徴として扱う
    # Singingは削除（MusiCNNのvocal-female/vocal-maleで代替）
    PANNS_GENRE_MAP = {
        "Pop music": "pop",
        "Rock music": "rock",
        "Hip hop music": "hip-hop",
        "Electronic music": "electronic",
        "Jazz": "jazz",
        "Classical music": "classical",
        "Blues": "blues",
        "Country": "country",
        "Reggae": "reggae",
        "Soul music": "soul",
        "Disco": "disco",
        "Funk": "funk",
        "Punk rock": "punk",
        "Heavy metal": "metal",
        "Techno": "techno",
        "House music": "house",
        # 楽器検出（ジャンルには出力しない、内部参照用）
        "Guitar": "_guitar",
        "Drum kit": "_drums",
        "Synthesizer": "_synth",
        "Piano": "_piano",
    }
    
    # Essentiaルールベースの追加ジャンル
    # boost値は重み(0.2)を掛けた後の値が閾値を超えるように設定
    # 
    # 注意: "anime"はAnisongAnalyzer（MAL DB照合）を使用してください
    # 音声特徴量だけでは正確なアニソン判定は困難です
    ESSENTIA_RULES = {
        "j-pop": {
            "conditions": {
                "bpm": (100, 160),
                "spectral_centroid": (1500, 4000),
                "beats_confidence_min": 2.0,
            },
            "boost": 0.8,
        },
        "eurobeat": {
            "conditions": {
                "bpm": (145, 165),
                "spectral_flatness_min": 0.15,
            },
            "boost": 0.9,
        },
        "city-pop": {
            "conditions": {
                "bpm": (95, 125),
                "spectral_centroid": (2000, 4000),
            },
            "boost": 0.8,
        },
    }

    # モデルキャッシュ（シングルトン）
    _panns_model = None
    _musicnn_model = None

    def __init__(
        self,
        threshold: float = DEFAULT_THRESHOLD,
        max_genres: int = MAX_GENRES,
        use_panns: bool = True,
        use_musicnn: bool = True,
        custom_rules: Dict = None,
        device: str = "auto",
        musicnn_model_path: str = "/Users/hanyanty/essentia_models/msd-musicnn-1.pb"
    ):
        """
        Args:
            threshold: ジャンル判定の閾値（0-1）
            max_genres: 出力する最大ジャンル数
            use_panns: PANNsモデルを使用するか
            use_musicnn: MusiCNNモデルを使用するか
            custom_rules: カスタムルール辞書
            device: PANNsの実行デバイス ("auto", "mps", "cpu")
            musicnn_model_path: MusiCNNモデルのパス
        """
        self.threshold = threshold
        self.max_genres = max_genres
        self.use_panns = use_panns and PANNS_AVAILABLE
        self.use_musicnn = use_musicnn and MUSICNN_AVAILABLE
        self.custom_rules = custom_rules or {}
        self.musicnn_model_path = musicnn_model_path
        
        # デバイス自動検出
        if device == "auto":
            import torch
            if torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
        else:
            self.device = device
    
    def _get_panns_model(self):
        """PANNsモデルを取得（遅延ロード）"""
        if GenreAnalyzer._panns_model is None and self.use_panns:
            GenreAnalyzer._panns_model = AudioTagging(
                checkpoint_path=None, 
                device=self.device
            )
        return GenreAnalyzer._panns_model
    
    def _get_musicnn_model(self):
        """MusiCNNモデルを取得（遅延ロード）"""
        if GenreAnalyzer._musicnn_model is None and self.use_musicnn:
            if Path(self.musicnn_model_path).exists():
                GenreAnalyzer._musicnn_model = TensorflowPredictMusiCNN(
                    graphFilename=self.musicnn_model_path
                )
        return GenreAnalyzer._musicnn_model
    
    def analyze(self, audio: np.ndarray, sample_rate: int) -> Dict[str, Any]:
        """
        音楽ジャンルを分析（マルチモデルアンサンブル）
        """
        genre_scores = {}
        attributes = []
        model_results = {}
        
        # 1. MusiCNNで分類（音楽特化、高精度）
        if self.use_musicnn:
            musicnn_result = self._analyze_with_musicnn(audio, sample_rate)
            model_results['musicnn'] = musicnn_result
            
            weight = self.MODEL_WEIGHTS['musicnn']
            for genre, score in musicnn_result['genres'].items():
                genre_scores[genre] = genre_scores.get(genre, 0) + score * weight
            attributes.extend(musicnn_result.get('attributes', []))
        
        # 2. PANNsで分類（汎用、楽器検出に強い）
        if self.use_panns:
            panns_result = self._analyze_with_panns(audio, sample_rate)
            model_results['panns'] = panns_result
            
            weight = self.MODEL_WEIGHTS['panns']
            for genre, score in panns_result['genres'].items():
                genre_scores[genre] = genre_scores.get(genre, 0) + score * weight
            attributes.extend(panns_result.get('attributes', []))
        
        # 3. Essentia特徴量を抽出
        features = self._extract_features(audio, sample_rate)
        
        # 4. Essentiaルールでジャンルを追加
        essentia_genres = self._apply_essentia_rules(features)
        weight = self.MODEL_WEIGHTS['essentia']
        for genre, score in essentia_genres.items():
            genre_scores[genre] = genre_scores.get(genre, 0) + score * weight
        
        # 5. カスタムルール適用
        custom_genres = self._apply_custom_rules(features)
        for genre, score in custom_genres.items():
            genre_scores[genre] = max(genre_scores.get(genre, 0), score)
        
        # 6. 閾値フィルタリングとソート
        detected_genres = [
            (genre, score) for genre, score in genre_scores.items()
            if score >= self.threshold and not genre.startswith('_')
        ]
        detected_genres.sort(key=lambda x: x[1], reverse=True)
        detected_genres = detected_genres[:self.max_genres]
        
        primary_genre = detected_genres[0][0] if detected_genres else "unknown"
        
        return {
            'genres': detected_genres,
            'primary_genre': primary_genre,
            'attributes': list(set(attributes)),
            'model_results': model_results,
            'features': features,
            'value': primary_genre
        }
    
    def _analyze_with_musicnn(
        self, 
        audio: np.ndarray, 
        sample_rate: int
    ) -> Dict[str, Any]:
        """MusiCNNで分析"""
        model = self._get_musicnn_model()
        if model is None:
            return {'genres': {}, 'attributes': [], 'raw': {}}
        
        # MusiCNNは16kHzを期待
        if sample_rate != 16000:
            audio_resampled = librosa.resample(
                audio, orig_sr=sample_rate, target_sr=16000
            )
        else:
            audio_resampled = audio
        
        # 推論
        predictions = model(audio_resampled)
        mean_preds = np.mean(predictions, axis=0)
        
        # 結果をパース
        genres = {}
        attributes = []
        raw = {}
        
        for i, score in enumerate(mean_preds):
            label = self.MUSICNN_LABELS[i]
            raw[label] = float(score)
            
            # ラベル正規化
            normalized = self.MUSICNN_NORMALIZE.get(label, label.lower())
            
            # 属性（ボーカル等）とジャンルを分離
            if normalized.startswith('vocal-'):
                if score > 0.1:
                    attributes.append(normalized)
            elif normalized in ['beautiful', 'mellow', 'chill', 'sad', 'happy', 
                               'party', 'sexy', 'catchy', 'instrumental']:
                # ムード/特徴タグ
                if score > 0.1:
                    attributes.append(normalized)
            elif label in self.ERA_TAGS or normalized in self.ERA_TAGS:
                # 年代タグはEraAnalyzerで処理するためスキップ
                pass
            else:
                # ジャンルタグ
                genres[normalized] = float(score)
        
        return {
            'genres': genres,
            'attributes': attributes,
            'raw': raw
        }
    
    def _analyze_with_panns(
        self, 
        audio: np.ndarray, 
        sample_rate: int
    ) -> Dict[str, Any]:
        """PANNsで分析"""
        model = self._get_panns_model()
        if model is None:
            return {'genres': {}, 'attributes': [], 'raw': {}}
        
        # PANNsは32kHzを期待
        if sample_rate != 32000:
            audio_resampled = librosa.resample(
                audio, orig_sr=sample_rate, target_sr=32000
            )
        else:
            audio_resampled = audio
        
        # 推論
        clipwise_output, _ = model.inference(audio_resampled[None, :])
        
        # 結果をパース
        genres = {}
        attributes = []
        raw = {}
        
        for i, score in enumerate(clipwise_output[0]):
            label = model.labels[i]
            raw[label] = float(score)
            
            if label in self.PANNS_GENRE_MAP:
                mapped = self.PANNS_GENRE_MAP[label]
                
                if mapped.startswith('_'):
                    if score > 0.1:
                        attributes.append(mapped[1:])
                else:
                    genres[mapped] = max(genres.get(mapped, 0), float(score))
        
        return {
            'genres': genres,
            'attributes': attributes,
            'raw': raw
        }
    
    def _extract_features(self, audio: np.ndarray, sample_rate: int) -> Dict[str, Any]:
        """Essentia特徴量を抽出"""
        features = {}
        
        # BPM検出
        try:
            rhythm_extractor = es.RhythmExtractor2013(method="multifeature")
            bpm, beats, beats_confidence, _, _ = rhythm_extractor(audio)
            features['bpm'] = float(bpm)
            features['beats_confidence'] = float(beats_confidence)
        except Exception:
            features['bpm'] = 120.0
            features['beats_confidence'] = 0.0
        
        # スペクトル特徴量
        try:
            frame_size = 2048
            hop_size = 1024
            
            spectrum = es.Spectrum(size=frame_size)
            spectral_centroid = es.SpectralCentroidTime(sampleRate=sample_rate)
            spectral_flatness = es.Flatness()
            
            centroids = []
            flatnesses = []
            
            for frame in es.FrameGenerator(audio, frameSize=frame_size, hopSize=hop_size):
                spec = spectrum(frame)
                centroids.append(spectral_centroid(frame))
                flatnesses.append(spectral_flatness(spec))
            
            features['spectral_centroid'] = float(np.mean(centroids)) if centroids else 2000.0
            features['spectral_flatness'] = float(np.mean(flatnesses)) if flatnesses else 0.1
            
        except Exception:
            features['spectral_centroid'] = 2000.0
            features['spectral_flatness'] = 0.1
        
        return features
    
    def _apply_essentia_rules(self, features: Dict[str, Any]) -> Dict[str, float]:
        """Essentiaルールを適用"""
        genres = {}
        
        for genre, rule in self.ESSENTIA_RULES.items():
            conditions = rule.get('conditions', {})
            boost = rule.get('boost', 0.2)
            
            if self._check_conditions(conditions, features):
                genres[genre] = boost
        
        return genres
    
    def _apply_custom_rules(self, features: Dict[str, Any]) -> Dict[str, float]:
        """カスタムルールを適用"""
        genres = {}
        
        for genre, rule in self.custom_rules.items():
            conditions = rule.get('conditions', {})
            score = rule.get('score', 0.5)
            
            if self._check_conditions(conditions, features):
                genres[genre] = score
        
        return genres
    
    def _check_conditions(self, conditions: Dict, features: Dict[str, Any]) -> bool:
        """条件をチェック"""
        for key, value in conditions.items():
            if key.endswith('_min'):
                feature_key = key[:-4]
                if features.get(feature_key, 0) < value:
                    return False
            elif key.endswith('_max'):
                feature_key = key[:-4]
                if features.get(feature_key, float('inf')) > value:
                    return False
            elif isinstance(value, tuple) and len(value) == 2:
                if not (value[0] <= features.get(key, 0) <= value[1]):
                    return False
            else:
                if features.get(key) != value:
                    return False
        
        return True
    
    def to_tag(self, result: Dict[str, Any]) -> str:
        """タグ文字列を生成"""
        tags = []
        
        # ジャンルタグ
        genres = result.get('genres', [])
        for genre, score in genres:
            if score >= self.threshold:
                tags.append(f"[{self.tag_prefix}:{genre}]")
        
        # 属性タグ（vocal-female, dance等）
        attributes = result.get('attributes', [])
        for attr in attributes:
            tags.append(f"[Attr:{attr}]")
        
        return ''.join(tags)
