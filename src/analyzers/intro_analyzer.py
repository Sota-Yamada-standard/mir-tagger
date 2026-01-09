"""
イントロ解析アナライザー
Demucs + PANNsを使用した高度な検出
- セリフで始まるか
- 歌で始まるか
- ドラムで始まるか
- 音楽開始位置
"""
import numpy as np
from typing import Dict, Any, List, Tuple
from src.core.base_analyzer import BaseAnalyzer, AnalyzerRegistry

try:
    import torch
    import librosa
    from demucs.pretrained import get_model
    from demucs.apply import apply_model
    DEMUCS_AVAILABLE = True
except ImportError:
    DEMUCS_AVAILABLE = False

try:
    from panns_inference import AudioTagging
    PANNS_AVAILABLE = True
except ImportError:
    PANNS_AVAILABLE = False


@AnalyzerRegistry.register
class IntroAnalyzer(BaseAnalyzer):
    """
    イントロ解析アナライザー
    
    検出項目:
    - セリフで始まるか（Speech Intro）
    - 歌で始まるか（Vocal/Singing Intro）
    - ドラムで始まるか（Drum Intro）
    - インストで始まるか
    - 音楽開始位置（秒）
    - イントロタイプの自動判定
    """
    
    name = "intro"
    description = "Intro analysis (speech, vocal, drums detection)"
    tag_prefix = "Intro"
    
    # 検出閾値
    SPEECH_THRESHOLD = 0.3    # Speechと判定する閾値
    MUSIC_THRESHOLD = 0.5     # Musicと判定する閾値
    SINGING_THRESHOLD = 0.1   # Singingと判定する閾値
    ENERGY_THRESHOLD = 0.02   # Demucsソース検出閾値
    
    # 分析設定
    ANALYSIS_DURATION = 15    # 分析する秒数
    CHUNK_DURATION = 1.0      # 分析チャンクサイズ（秒）
    
    # モデルキャッシュ
    _demucs_model = None
    _panns_model = None

    def __init__(
        self,
        analysis_duration: float = ANALYSIS_DURATION,
        chunk_duration: float = CHUNK_DURATION,
        device: str = "auto"
    ):
        self.analysis_duration = analysis_duration
        self.chunk_duration = chunk_duration
        
        if device == "auto":
            if DEMUCS_AVAILABLE and torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
        else:
            self.device = device
    
    def _get_demucs_model(self):
        """Demucsモデル取得"""
        if not DEMUCS_AVAILABLE:
            return None
        if IntroAnalyzer._demucs_model is None:
            IntroAnalyzer._demucs_model = get_model('htdemucs')
            IntroAnalyzer._demucs_model.eval()
            IntroAnalyzer._demucs_model = IntroAnalyzer._demucs_model.to(self.device)
        return IntroAnalyzer._demucs_model
    
    def _get_panns_model(self):
        """PANNsモデル取得"""
        if not PANNS_AVAILABLE:
            return None
        if IntroAnalyzer._panns_model is None:
            IntroAnalyzer._panns_model = AudioTagging(checkpoint_path=None, device='cpu')
        return IntroAnalyzer._panns_model
    
    def analyze(self, audio: np.ndarray, sample_rate: int) -> Dict[str, Any]:
        """
        イントロを解析
        """
        result = {
            'speech_intro': False,
            'speech_duration': 0.0,
            'vocal_intro': False,
            'singing_intro': False,
            'singing_acapella': False,  # 伴奏なしのボーカルソロ
            'drum_intro': False,
            'instrumental_intro': False,
            'music_start_time': 0.0,
            'intro_type': 'unknown',
            'timeline': [],
            'source_energy': {},
            'value': 'unknown'
        }
        
        # PANNsで時間軸分析（Speech/Music/Singing検出）
        if PANNS_AVAILABLE:
            timeline = self._analyze_timeline_panns(audio, sample_rate)
            result['timeline'] = timeline
            
            # セリフ/アカペライントロの検出
            speech_result = self._detect_speech_intro(timeline)
            result.update(speech_result)
        
        # Demucsでソース分離（ドラム/ベース/ボーカル検出）
        if DEMUCS_AVAILABLE:
            source_result = self._analyze_sources_demucs(audio, sample_rate)
            result['source_energy'] = source_result.get('source_energy', {})
            
            # ソース情報でイントロタイプを補完
            if source_result.get('drum_intro'):
                result['drum_intro'] = True
            if source_result.get('vocal_intro'):
                result['vocal_intro'] = True
        
        # イントロタイプの最終判定
        result['intro_type'] = self._determine_intro_type(result)
        result['value'] = result['intro_type']
        
        return result
    
    def _analyze_timeline_panns(
        self, 
        audio: np.ndarray, 
        sample_rate: int
    ) -> List[Dict[str, Any]]:
        """PANNsで時間軸に沿って分析"""
        model = self._get_panns_model()
        if model is None:
            return []
        
        # 32kHzにリサンプリング
        if sample_rate != 32000:
            audio_resampled = librosa.resample(audio, orig_sr=sample_rate, target_sr=32000)
            sr = 32000
        else:
            audio_resampled = audio
            sr = sample_rate
        
        # 分析範囲を制限
        max_samples = int(sr * self.analysis_duration)
        audio_resampled = audio_resampled[:max_samples]
        
        # チャンクごとに分析
        chunk_samples = int(sr * self.chunk_duration)
        timeline = []
        
        # ラベルインデックスを取得
        try:
            speech_idx = model.labels.index('Speech')
            singing_idx = model.labels.index('Singing')
            music_idx = model.labels.index('Music')
            drums_idx = model.labels.index('Drum kit')
        except ValueError:
            return []
        
        num_chunks = int(len(audio_resampled) / chunk_samples)
        
        for i in range(num_chunks):
            start = i * chunk_samples
            end = start + chunk_samples
            chunk = audio_resampled[start:end]
            
            if len(chunk) < chunk_samples // 2:
                continue
            
            # PANNs推論
            clipwise, _ = model.inference(chunk[None, :])
            
            timeline.append({
                'time_start': i * self.chunk_duration,
                'time_end': (i + 1) * self.chunk_duration,
                'speech': float(clipwise[0][speech_idx]),
                'singing': float(clipwise[0][singing_idx]),
                'music': float(clipwise[0][music_idx]),
                'drums': float(clipwise[0][drums_idx]),
            })
        
        return timeline
    
    def _detect_speech_intro(self, timeline: List[Dict]) -> Dict[str, Any]:
        """セリフイントロを検出"""
        if not timeline:
            return {
                'speech_intro': False,
                'speech_duration': 0.0,
                'singing_intro': False,
                'singing_acapella': False,
                'music_start_time': 0.0,
            }
        
        speech_intro = False
        speech_duration = 0.0
        singing_intro = False
        singing_acapella = False  # 伴奏なしのボーカルソロ
        music_start_time = 0.0
        
        first_chunk = timeline[0]
        
        # セリフ判定: Speech高い & Music低い
        if first_chunk['speech'] > self.SPEECH_THRESHOLD and first_chunk['music'] < self.MUSIC_THRESHOLD:
            speech_intro = True
            
            # セリフが続く長さを計算
            for chunk in timeline:
                if chunk['speech'] > self.SPEECH_THRESHOLD and chunk['music'] < self.MUSIC_THRESHOLD:
                    speech_duration = chunk['time_end']
                else:
                    break
            
            # 音楽開始位置を検出
            for chunk in timeline:
                if chunk['music'] > self.MUSIC_THRESHOLD:
                    music_start_time = chunk['time_start']
                    break
        
        # アカペラ判定: Singing高い & Music低い & Speech低い
        elif (first_chunk.get('singing', 0) > self.SINGING_THRESHOLD and 
              first_chunk['music'] < self.MUSIC_THRESHOLD and
              first_chunk['speech'] < self.SPEECH_THRESHOLD):
            singing_intro = True
            singing_acapella = True  # 伴奏なしのボーカルソロ
            music_start_time = 0.0
        
        # 伴奏付きボーカル判定: Singing高い & Music高い
        elif (first_chunk.get('singing', 0) > self.SINGING_THRESHOLD and 
              first_chunk['music'] >= self.MUSIC_THRESHOLD):
            singing_intro = True
            singing_acapella = False  # 伴奏あり
            music_start_time = 0.0
        
        else:
            # セリフ/歌唱なしの場合、音楽は最初から
            music_start_time = 0.0
        
        return {
            'speech_intro': speech_intro,
            'speech_duration': speech_duration,
            'singing_intro': singing_intro,
            'singing_acapella': singing_acapella,
            'music_start_time': music_start_time,
        }
    
    def _analyze_sources_demucs(
        self, 
        audio: np.ndarray, 
        sample_rate: int
    ) -> Dict[str, Any]:
        """Demucsで音源分離して分析"""
        model = self._get_demucs_model()
        if model is None:
            return {}
        
        # イントロ部分を切り出し
        intro_samples = int(sample_rate * min(10, self.analysis_duration))
        intro_audio = audio[:intro_samples]
        
        # 44100Hzにリサンプリング
        if sample_rate != 44100:
            intro_audio = librosa.resample(intro_audio, orig_sr=sample_rate, target_sr=44100)
        
        # ステレオに変換
        if intro_audio.ndim == 1:
            intro_audio = np.stack([intro_audio, intro_audio])
        
        # テンソルに変換
        waveform = torch.tensor(intro_audio, dtype=torch.float32)
        waveform = waveform.unsqueeze(0).to(self.device)
        
        # 音源分離
        with torch.no_grad():
            sources = apply_model(model, waveform, device=self.device)
        
        # 各ソースのエネルギー
        source_energy = {}
        for i, name in enumerate(model.sources):
            source = sources[0, i].cpu().numpy()
            energy = float(np.sqrt(np.mean(source ** 2)))
            source_energy[name] = energy
        
        return {
            'source_energy': source_energy,
            'vocal_intro': source_energy.get('vocals', 0) > self.ENERGY_THRESHOLD,
            'drum_intro': source_energy.get('drums', 0) > self.ENERGY_THRESHOLD,
            'bass_intro': source_energy.get('bass', 0) > self.ENERGY_THRESHOLD,
        }
    
    def _determine_intro_type(self, result: Dict[str, Any]) -> str:
        """イントロタイプを最終判定"""
        # 優先順位: セリフ > アカペラ > 伴奏付き歌唱 > ドラム > インスト
        
        # セリフで始まる
        if result.get('speech_intro'):
            return 'speech'
        
        # アカペラ（伴奏なしボーカルソロ）で始まる
        if result.get('singing_acapella'):
            return 'acapella'
        
        # 伴奏付きで歌が始まる
        if result.get('singing_intro') and not result.get('singing_acapella'):
            return 'singing'
        
        # Demucsの結果を使った判定
        if result.get('vocal_intro') and not result.get('drum_intro'):
            return 'vocal'
        
        if result.get('drum_intro') and not result.get('vocal_intro'):
            return 'drums'
        
        # 全部入ってる場合
        if result.get('vocal_intro') and result.get('drum_intro'):
            return 'full'
        
        return 'instrumental'
    
    def to_tag(self, result: Dict[str, Any]) -> str:
        """タグ文字列を生成"""
        tags = []
        
        intro_type = result.get('intro_type', 'unknown')
        
        # セリフイントロ
        if result.get('speech_intro'):
            speech_dur = result.get('speech_duration', 0)
            tags.append(f"[SpeechIntro:{speech_dur:.0f}s]")
            
            # 音楽開始位置
            music_start = result.get('music_start_time', 0)
            if music_start > 0:
                tags.append(f"[MusicAt:{music_start:.0f}s]")
        
        # イントロタイプ
        elif intro_type == 'acapella':
            tags.append("[Intro:Acapella]")  # 伴奏なしボーカルソロ
        elif intro_type == 'singing':
            tags.append("[Intro:Singing]")   # 伴奏付き歌唱開始
        elif intro_type == 'vocal':
            tags.append("[Intro:Vocal]")
        elif intro_type == 'drums':
            tags.append("[Intro:Drums]")
        elif intro_type == 'instrumental':
            tags.append("[Intro:Inst]")
        elif intro_type == 'full':
            tags.append("[Intro:Full]")
        
        return ''.join(tags)
