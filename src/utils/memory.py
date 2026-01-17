"""
メモリ管理ユーティリティ

モデルキャッシュのクリアやメモリ解放を行う
"""
import gc
import ctypes
import sys


def clear_all_model_caches():
    """
    全アナライザーのモデルキャッシュをクリア
    
    メモリリーク対策として、N曲処理ごとに呼び出すことを推奨
    次の曲でモデルは自動的に再ロードされる
    """
    # 各アナライザーのキャッシュクリア
    try:
        from src.analyzers.intro_analyzer import IntroAnalyzer
        IntroAnalyzer.clear_model_cache()
    except ImportError:
        pass
    
    try:
        from src.analyzers.genre_analyzer import GenreAnalyzer
        GenreAnalyzer.clear_model_cache()
    except ImportError:
        pass
    
    try:
        from src.analyzers.beat_analyzer import BeatAnalyzer
        BeatAnalyzer.clear_model_cache()
    except ImportError:
        pass
    
    try:
        from src.analyzers.mood_analyzer import MoodAnalyzer
        MoodAnalyzer.clear_model_cache()
    except ImportError:
        pass
    
    # Python GC を強制実行（3世代すべて）
    gc.collect(0)  # 世代0
    gc.collect(1)  # 世代1
    gc.collect(2)  # 世代2（最も古い循環参照）
    
    # PyTorch MPS キャッシュをクリア
    try:
        import torch
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
            torch.mps.synchronize()  # 同期して確実に解放
    except (ImportError, AttributeError):
        pass
    
    # macOSでmalloc_zone_pressureを使ってメモリを解放（可能なら）
    try:
        if sys.platform == 'darwin':
            libc = ctypes.CDLL('libSystem.B.dylib')
            # malloc_zone_pressure_relief: メモリプレッシャーを緩和
            libc.malloc_zone_pressure_relief(None, 0)
    except (OSError, AttributeError):
        pass


def get_memory_usage_mb() -> float:
    """現在のプロセスのメモリ使用量を取得（MB）"""
    import os
    import subprocess
    
    try:
        result = subprocess.run(
            ['ps', '-o', 'rss=', '-p', str(os.getpid())],
            capture_output=True, 
            text=True
        )
        return int(result.stdout.strip()) / 1024
    except:
        return 0.0
