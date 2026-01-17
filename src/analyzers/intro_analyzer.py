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
    DRUMS_PANNS_THRESHOLD = 0.05  # PANNsでDrumsと判定する閾値
    # Demucsソース検出閾値
    ENERGY_THRESHOLD_DRUMS = 0.05   # ドラムの閾値（以前は0.02で敏感すぎた）
    ENERGY_THRESHOLD_VOCALS = 0.02  # ボーカルの閾値
    ENERGY_THRESHOLD_OTHER = 0.05   # その他楽器の閾値
    
    # ビートドロップ検出
    BEAT_DROP_LOW_THRESHOLD = 0.03   # 「低い」と判断するドラムエネルギー
    BEAT_DROP_HIGH_RATIO = 3.0       # 低い状態からこの倍以上になったらドロップ
    BEAT_DROP_MIN_QUIET_TIME = 4.0   # 最低限の静かな時間（秒）
    BEAT_DROP_ANALYSIS_DURATION = 60 # ビートドロップ検出の分析範囲（秒）
    
    # 分析設定
    ANALYSIS_DURATION = 15    # 分析する秒数
    CHUNK_DURATION = 1.0      # 分析チャンクサイズ（秒）
    
    # モデルキャッシュ
    _demucs_model = None
    _panns_model = None

    @classmethod
    def clear_model_cache(cls):
        """モデルキャッシュをクリアしてメモリを解放"""
        import gc
        if cls._demucs_model is not None:
            del cls._demucs_model
            cls._demucs_model = None
        if cls._panns_model is not None:
            del cls._panns_model
            cls._panns_model = None
        gc.collect()
        if DEMUCS_AVAILABLE and torch.backends.mps.is_available():
            torch.mps.empty_cache()

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
            'beat_drop_time': 0.0,  # ビートが入るタイミング（秒）
            'has_soft_intro': False,  # 静かなイントロがあるか
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
            
            # vocal_intro判定：DemucsとPANNsの両方で人の声を検出した場合のみ
            if source_result.get('vocal_intro'):
                # PANNsでも人の声（Speech/Singing）が検出されているか確認
                timeline = result.get('timeline', [])
                if timeline:
                    first_chunks = timeline[:5]  # 最初の5秒
                    avg_speech = sum(c.get('speech', 0) for c in first_chunks) / len(first_chunks)
                    avg_singing = sum(c.get('singing', 0) for c in first_chunks) / len(first_chunks)
                    # PANNsでSpeechまたはSingingが0.05以上なら人の声あり
                    panns_voice_detected = avg_speech > 0.05 or avg_singing > 0.05
                    result['vocal_intro'] = panns_voice_detected
                else:
                    # PANNsのデータがない場合はDemucsの結果を信頼
                    result['vocal_intro'] = True
            
            # ビートドロップ検出（静かに始まって後からビートが入るパターン）
            beat_drop_result = self._detect_beat_drop(audio, sample_rate)
            result['beat_drop_time'] = beat_drop_result.get('beat_drop_time', 0.0)
            result['has_soft_intro'] = beat_drop_result.get('has_soft_intro', False)
            result['drums_timeline'] = beat_drop_result.get('drums_timeline', [])
        
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
        
        # 各ソースの判定（閾値を個別に設定）
        drums_energy = source_energy.get('drums', 0)
        vocals_energy = source_energy.get('vocals', 0)
        other_energy = source_energy.get('other', 0)
        
        # ドラムの判定：
        # - エネルギーが閾値以上
        # - かつ、other（メロディ楽器）より相対的に高い場合のみ
        drum_intro = (
            drums_energy > self.ENERGY_THRESHOLD_DRUMS and
            drums_energy > other_energy * 0.5  # ドラムがotherの半分以上
        )
        
        return {
            'source_energy': source_energy,
            'vocal_intro': vocals_energy > self.ENERGY_THRESHOLD_VOCALS,
            'drum_intro': drum_intro,
            'bass_intro': source_energy.get('bass', 0) > self.ENERGY_THRESHOLD_DRUMS,
            'other_intro': other_energy > self.ENERGY_THRESHOLD_OTHER,
        }
    
    def _detect_beat_drop(
        self,
        audio: np.ndarray,
        sample_rate: int
    ) -> Dict[str, Any]:
        """
        ビートドロップを検出
        静かに始まって途中からビートが入るパターンを検出
        
        例：「もう恋なんてしない」のように最初は静かで、
        途中からドラムが入る曲
        
        Returns:
            beat_drop_time: ビートが入るタイミング（秒）
            has_soft_intro: 静かなイントロがあるか
            drums_timeline: ドラムエネルギーの時間変化
        """
        model = self._get_demucs_model()
        if model is None:
            return {'beat_drop_time': 0.0, 'has_soft_intro': False, 'drums_timeline': []}
        
        # 分析範囲（最初の60秒）
        analysis_samples = int(sample_rate * self.BEAT_DROP_ANALYSIS_DURATION)
        audio_segment = audio[:analysis_samples]
        
        # 44100Hzにリサンプリング
        if sample_rate != 44100:
            audio_segment = librosa.resample(audio_segment, orig_sr=sample_rate, target_sr=44100)
            sr = 44100
        else:
            sr = sample_rate
        
        # ステレオに変換
        if audio_segment.ndim == 1:
            audio_stereo = np.stack([audio_segment, audio_segment])
        else:
            audio_stereo = audio_segment
        
        # Demucsで音源分離
        waveform = torch.tensor(audio_stereo, dtype=torch.float32)
        waveform = waveform.unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            sources = apply_model(model, waveform, device=self.device)
        
        # ドラムトラックを取得
        drums_idx = model.sources.index('drums')
        drums = sources[0, drums_idx].cpu().numpy()
        
        # 4秒ごとのエネルギーを計算
        chunk_duration = 4  # 秒
        chunk_samples = int(sr * chunk_duration)
        
        drums_timeline = []
        for i in range(0, len(drums[0]), chunk_samples):
            chunk = drums[:, i:i+chunk_samples]
            if chunk.shape[1] < chunk_samples // 2:
                continue
            energy = float(np.sqrt(np.mean(chunk ** 2)))
            time_sec = i / sr
            drums_timeline.append({
                'time': time_sec,
                'energy': energy
            })
        
        if len(drums_timeline) < 2:
            return {'beat_drop_time': 0.0, 'has_soft_intro': False, 'drums_timeline': drums_timeline}
        
        # ビートドロップを検出
        # 条件：最初が低い → 途中から高くなる
        first_energy = drums_timeline[0]['energy']
        
        # 最初のエネルギーが低い場合のみ検出
        if first_energy > self.BEAT_DROP_LOW_THRESHOLD:
            return {'beat_drop_time': 0.0, 'has_soft_intro': False, 'drums_timeline': drums_timeline}
        
        # 静かな状態が続く時間を確認
        quiet_end_time = 0.0
        for entry in drums_timeline:
            if entry['energy'] < self.BEAT_DROP_LOW_THRESHOLD:
                quiet_end_time = entry['time'] + chunk_duration
            else:
                break
        
        # 最低限の静かな時間がない場合はスキップ
        if quiet_end_time < self.BEAT_DROP_MIN_QUIET_TIME:
            return {'beat_drop_time': 0.0, 'has_soft_intro': False, 'drums_timeline': drums_timeline}
        
        # ビートが入るタイミングを検出
        beat_drop_time = 0.0
        for i, entry in enumerate(drums_timeline):
            if entry['energy'] > first_energy * self.BEAT_DROP_HIGH_RATIO:
                beat_drop_time = entry['time']
                break
        
        # ビートドロップが検出された場合
        if beat_drop_time > 0:
            return {
                'beat_drop_time': beat_drop_time,
                'has_soft_intro': True,
                'drums_timeline': drums_timeline
            }
        
        return {'beat_drop_time': 0.0, 'has_soft_intro': False, 'drums_timeline': drums_timeline}
    
    def _determine_intro_type(self, result: Dict[str, Any]) -> str:
        """
        イントロタイプを最終判定
        
        PANNsとDemucsの両方の結果を考慮して判定：
        - PANNs: 時間軸でのSpeech/Singing/Music/Drums検出
        - Demucs: 音源分離によるvocals/drums/bass/other検出
        """
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
        
        # PANNsのタイムライン情報を取得
        timeline = result.get('timeline', [])
        panns_drums_detected = False
        if timeline:
            # 最初の数チャンクでドラムが検出されているか
            first_chunks = timeline[:3]
            avg_drums = sum(c.get('drums', 0) for c in first_chunks) / len(first_chunks)
            panns_drums_detected = avg_drums > self.DRUMS_PANNS_THRESHOLD
        
        # ドラム判定：DemucsとPANNsの両方で検出された場合のみ
        demucs_drum_intro = result.get('drum_intro', False)
        drum_intro = demucs_drum_intro and panns_drums_detected
        
        vocal_intro = result.get('vocal_intro', False)
        other_intro = result.get('other_intro', False)
        
        # ボーカルのみ（ドラムなし）
        if vocal_intro and not drum_intro:
            return 'vocal'
        
        # ドラムのみ（ボーカルなし）
        if drum_intro and not vocal_intro:
            return 'drums'
        
        # 全部入ってる場合
        if vocal_intro and drum_intro:
            return 'full'
        
        # other（ギター、ピアノなどのメロディ楽器）が高い場合
        if other_intro and not vocal_intro and not drum_intro:
            return 'instrumental'
        
        return 'full'  # デフォルトはfull（すべての要素が含まれる）
    
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
        
        # ビートドロップ（静かに始まって途中からビートが入る）
        # セリフイントロの場合はMusicAtタグで音楽開始位置を表しているので出力しない
        if result.get('has_soft_intro') and not result.get('speech_intro'):
            beat_drop_time = result.get('beat_drop_time', 0)
            if beat_drop_time > 0:
                tags.append(f"[BeatAt:{beat_drop_time:.0f}s]")
        
        return ''.join(tags)
