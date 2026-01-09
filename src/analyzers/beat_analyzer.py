"""
拍解析アナライザー
Downbeat検出、BPM変化検出
セリフ部分を自動スキップして音楽部分のみを分析
"""
import numpy as np
from typing import Dict, Any, List, Tuple
from src.core.base_analyzer import BaseAnalyzer, AnalyzerRegistry

try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False

try:
    from panns_inference import AudioTagging
    PANNS_AVAILABLE = True
except ImportError:
    PANNS_AVAILABLE = False


@AnalyzerRegistry.register
class BeatAnalyzer(BaseAnalyzer):
    """
    拍解析アナライザー
    
    検出項目:
    - Downbeat: 曲（音楽部分）が何拍目から始まるか（1-4）
    - BPM変化: 途中でテンポが変わるか
    - ビート信頼度: ビート検出の信頼度
    
    特徴:
    - セリフ部分を自動検出してスキップ
    - 音楽開始位置から正確にDownbeatを計算
    """
    
    name = "beat"
    description = "Beat analysis (downbeat, tempo changes, speech skip)"
    tag_prefix = "Beat"
    
    # 閾値
    BPM_CHANGE_THRESHOLD = 5.0
    SPEECH_THRESHOLD = 0.3
    MUSIC_THRESHOLD = 0.5
    
    # PANNsモデルキャッシュ
    _panns_model = None
    
    def __init__(
        self, 
        bpm_change_threshold: float = BPM_CHANGE_THRESHOLD,
        auto_skip_speech: bool = True
    ):
        """
        Args:
            bpm_change_threshold: BPM変化検出の閾値
            auto_skip_speech: セリフ部分を自動スキップするか
        """
        self.bpm_change_threshold = bpm_change_threshold
        self.auto_skip_speech = auto_skip_speech
    
    def _get_panns_model(self):
        """PANNsモデル取得"""
        if not PANNS_AVAILABLE:
            return None
        if BeatAnalyzer._panns_model is None:
            BeatAnalyzer._panns_model = AudioTagging(checkpoint_path=None, device='cpu')
        return BeatAnalyzer._panns_model
    
    def analyze(self, audio: np.ndarray, sample_rate: int) -> Dict[str, Any]:
        """
        拍を解析
        """
        if not LIBROSA_AVAILABLE:
            return self._fallback_result()
        
        # 音楽開始位置を検出（セリフスキップ）
        music_start_time = 0.0
        if self.auto_skip_speech and PANNS_AVAILABLE:
            music_start_time = self._detect_music_start(audio, sample_rate)
        
        # 音楽開始位置からオーディオを切り出し
        if music_start_time > 0:
            start_sample = int(music_start_time * sample_rate)
            audio_music = audio[start_sample:]
        else:
            audio_music = audio
        
        # librosaは22050Hzを期待
        if sample_rate != 22050:
            audio_resampled = librosa.resample(
                audio_music, orig_sr=sample_rate, target_sr=22050
            )
            sr = 22050
        else:
            audio_resampled = audio_music
            sr = sample_rate
        
        # ビート検出
        tempo, beat_frames = librosa.beat.beat_track(y=audio_resampled, sr=sr)
        beat_times = librosa.frames_to_time(beat_frames, sr=sr)
        
        # tempoの処理（配列の場合）
        tempo_val = float(tempo[0]) if hasattr(tempo, '__len__') else float(tempo)
        
        # Downbeat推定（音楽部分のみ）
        downbeat, first_beat_time, confidence = self._estimate_downbeat(
            audio_resampled, sr, beat_times
        )
        
        # BPM変化検出
        tempo_stable, tempo_changes = self._detect_tempo_changes(
            audio_resampled, sr, beat_times
        )
        
        # beat_timesを元の時間軸に戻す
        beat_times_absolute = beat_times + music_start_time
        first_beat_time_absolute = first_beat_time + music_start_time
        
        return {
            'downbeat': downbeat,
            'first_beat_time': first_beat_time,
            'first_beat_time_absolute': first_beat_time_absolute,
            'music_start_time': music_start_time,
            'tempo': tempo_val,
            'tempo_stable': tempo_stable,
            'tempo_changes': tempo_changes,
            'beat_count': len(beat_times),
            'confidence': confidence,
            'value': downbeat
        }
    
    def _detect_music_start(self, audio: np.ndarray, sample_rate: int) -> float:
        """
        PANNsを使って音楽開始位置を検出
        セリフ/スピーチ部分をスキップ
        """
        model = self._get_panns_model()
        if model is None:
            return 0.0
        
        # 32kHzにリサンプリング
        if sample_rate != 32000:
            audio_resampled = librosa.resample(audio, orig_sr=sample_rate, target_sr=32000)
            sr = 32000
        else:
            audio_resampled = audio
            sr = sample_rate
        
        # 最初の15秒を分析
        max_samples = int(sr * 15)
        audio_resampled = audio_resampled[:max_samples]
        
        # 1秒ごとに分析
        chunk_duration = 1.0
        chunk_samples = int(sr * chunk_duration)
        
        try:
            speech_idx = model.labels.index('Speech')
            music_idx = model.labels.index('Music')
        except ValueError:
            return 0.0
        
        # 最初のチャンクがセリフかどうか確認
        first_chunk = audio_resampled[:chunk_samples]
        clipwise, _ = model.inference(first_chunk[None, :])
        
        first_speech = clipwise[0][speech_idx]
        first_music = clipwise[0][music_idx]
        
        # 最初からMusicが高い場合はスキップ不要
        if first_music > self.MUSIC_THRESHOLD:
            return 0.0
        
        # セリフが検出されない場合もスキップ不要
        if first_speech < self.SPEECH_THRESHOLD:
            return 0.0
        
        # 音楽が始まる位置を探す
        num_chunks = int(len(audio_resampled) / chunk_samples)
        
        for i in range(num_chunks):
            start = i * chunk_samples
            end = start + chunk_samples
            chunk = audio_resampled[start:end]
            
            if len(chunk) < chunk_samples // 2:
                continue
            
            clipwise, _ = model.inference(chunk[None, :])
            music_score = clipwise[0][music_idx]
            
            if music_score > self.MUSIC_THRESHOLD:
                return float(i * chunk_duration)
        
        return 0.0
    
    def _estimate_downbeat(
        self, 
        audio: np.ndarray, 
        sr: int, 
        beat_times: np.ndarray
    ) -> Tuple[int, float, float]:
        """
        Downbeat（何拍目から始まるか）を推定
        
        改良版：音楽開始から最初のビートまでの時間と、
        オーディオのエネルギー立ち上がりを考慮
        """
        if len(beat_times) < 4:
            return 1, 0.0, 0.0
        
        first_beat_time = float(beat_times[0])
        
        # 平均ビート間隔
        avg_interval = np.mean(np.diff(beat_times[:min(16, len(beat_times))]))
        
        # オーディオのエネルギー立ち上がりを検出
        # 音楽が始まる正確なタイミングを特定
        onset_time = self._detect_onset(audio, sr)
        
        # onset_timeが検出できた場合、それを基準にする
        if onset_time is not None and onset_time < first_beat_time:
            # onsetから最初のビートまでの時間
            time_to_first_beat = first_beat_time - onset_time
        else:
            time_to_first_beat = first_beat_time
        
        # 何拍目から始まるか計算
        # time_to_first_beatが1ビート間隔未満なら、1拍目から始まると判定
        if time_to_first_beat < avg_interval * 0.75:
            downbeat = 1
        else:
            beats_offset = time_to_first_beat / avg_interval
            # 小数点以下を四捨五入して何拍ずれているか
            beat_offset = int(round(beats_offset))
            # 4拍で1小節なので、何拍目かを計算
            downbeat = (4 - (beat_offset % 4)) % 4
            if downbeat == 0:
                downbeat = 4
        
        # 1-4の範囲に収める
        downbeat = max(1, min(4, downbeat))
        
        # 信頼度の計算
        if len(beat_times) > 8:
            intervals = np.diff(beat_times[:16])
            interval_std = np.std(intervals)
            confidence = max(0, 1 - (interval_std / avg_interval))
        else:
            confidence = 0.5
        
        return downbeat, first_beat_time, float(confidence)
    
    def _detect_onset(self, audio: np.ndarray, sr: int) -> float:
        """
        オーディオの立ち上がり（onset）を検出
        音楽が実際に始まるタイミングを特定
        """
        try:
            # RMSエネルギーを計算
            frame_length = int(sr * 0.023)  # 23ms
            hop_length = int(sr * 0.01)     # 10ms
            
            rms = librosa.feature.rms(y=audio, frame_length=frame_length, hop_length=hop_length)[0]
            
            # 閾値（最大値の10%）を超える最初のフレームを探す
            threshold = np.max(rms) * 0.1
            onset_frames = np.where(rms > threshold)[0]
            
            if len(onset_frames) > 0:
                onset_time = float(onset_frames[0] * hop_length / sr)
                return onset_time
            
            return None
        except Exception:
            return None
    
    def _detect_tempo_changes(
        self, 
        audio: np.ndarray, 
        sr: int,
        beat_times: np.ndarray
    ) -> Tuple[bool, List[Dict]]:
        """
        BPM変化を検出
        """
        if len(beat_times) < 16:
            return True, []
        
        tempo_changes = []
        window_size = 8
        tempos = []
        
        for i in range(0, len(beat_times) - window_size, window_size // 2):
            window_beats = beat_times[i:i + window_size]
            if len(window_beats) >= 2:
                intervals = np.diff(window_beats)
                avg_interval = np.mean(intervals)
                if avg_interval > 0:
                    local_tempo = 60.0 / avg_interval
                    tempos.append({
                        'time': float(window_beats[0]),
                        'tempo': local_tempo
                    })
        
        if len(tempos) < 2:
            return True, []
        
        base_tempo = tempos[0]['tempo']
        is_stable = True
        
        for i, t in enumerate(tempos[1:], 1):
            diff = abs(t['tempo'] - base_tempo)
            if diff > self.bpm_change_threshold:
                is_stable = False
                tempo_changes.append({
                    'time': t['time'],
                    'from_tempo': round(tempos[i-1]['tempo'], 1),
                    'to_tempo': round(t['tempo'], 1),
                    'change': round(t['tempo'] - tempos[i-1]['tempo'], 1)
                })
        
        return is_stable, tempo_changes
    
    def _fallback_result(self) -> Dict[str, Any]:
        """フォールバック結果"""
        return {
            'downbeat': 1,
            'first_beat_time': 0.0,
            'first_beat_time_absolute': 0.0,
            'music_start_time': 0.0,
            'tempo': 120.0,
            'tempo_stable': True,
            'tempo_changes': [],
            'beat_count': 0,
            'confidence': 0.0,
            'value': 1
        }
    
    def to_tag(self, result: Dict[str, Any]) -> str:
        """タグ文字列を生成"""
        tags = []
        
        downbeat = result.get('downbeat', 1)
        confidence = result.get('confidence', 0)
        
        if confidence > 0.5:
            tags.append(f"[Downbeat:{downbeat}]")
        
        if not result.get('tempo_stable', True):
            tags.append("[BPMChange:Yes]")
        
        return ''.join(tags)
