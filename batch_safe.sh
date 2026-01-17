#!/bin/bash
# メモリ安全なバッチ処理スクリプト
# 指定曲数ごとにプロセスを完全再起動してメモリをリセット
#
# 使い方:
#   ./batch_safe.sh [バッチサイズ] [最大イテレーション]
#   ./batch_safe.sh 50        # 50曲ごとに再起動
#   ./batch_safe.sh 30 200    # 30曲ごと、最大200回
#   nohup ./batch_safe.sh 50 > batch.log 2>&1 &  # バックグラウンド実行

BATCH_SIZE=${1:-50}   # デフォルト50曲ずつ（メモリ安全優先）
MAX_ITERATIONS=${2:-500}  # 最大500回（25000曲対応）

cd /Users/hanyanty/projects/mir-tagger
source venv/bin/activate

echo "========================================"
echo "  メモリ安全バッチ処理"
echo "========================================"
echo "バッチサイズ: ${BATCH_SIZE}曲"
echo "最大イテレーション: ${MAX_ITERATIONS}回"
echo "開始時刻: $(date)"
echo ""

TOTAL_SUCCESS=0
TOTAL_ERROR=0
START_TIME=$(date +%s)

for i in $(seq 1 $MAX_ITERATIONS); do
    echo ""
    echo "=== バッチ $i 開始 $(date '+%H:%M:%S') ==="
    
    # 残り未処理数を確認（mp4を除外）
    REMAINING=$(python -c "
import sys
sys.path.insert(0, '.')
from batch_analyze import get_files_from_rekordbox, is_already_tagged
files = get_files_from_rekordbox('tests/collections.xml')
untagged = [f for f in files if not is_already_tagged(str(f)) and f.endswith(('.mp3', '.m4a'))]
print(len(untagged))
" 2>/dev/null)
    
    echo "残り未処理: ${REMAINING}曲"
    
    if [ "$REMAINING" = "0" ] || [ -z "$REMAINING" ]; then
        echo ""
        echo "✅ 全ファイル処理完了!"
        break
    fi
    
    # バッチ処理（シングルプロセス、BATCH_SIZE曲で自動停止）
    RESULT=$(python -c "
import sys
sys.path.insert(0, '.')
from batch_analyze import get_files_from_rekordbox, is_already_tagged, process_file
from pathlib import Path
import time

BATCH_SIZE = ${BATCH_SIZE}

files = get_files_from_rekordbox('tests/collections.xml')
# mp4を除外（サポート外）
untagged = [f for f in files if not is_already_tagged(str(f)) and f.endswith(('.mp3', '.m4a'))][:BATCH_SIZE]

success = 0
error = 0

for i, f in enumerate(untagged, 1):
    start = time.time()
    result = process_file(str(f), True, False, 'comment')
    elapsed = time.time() - start
    
    if result['status'] == 'success':
        success += 1
        print(f'[{i}/{len(untagged)}] ✅ {Path(f).name} ({elapsed:.1f}s)')
        print(f'         {result[\"tags\"]}')
    else:
        error += 1
        err_msg = result.get('error', 'Unknown')[:60]
        print(f'[{i}/{len(untagged)}] ❌ {Path(f).name}: {err_msg}')

print(f'BATCH_RESULT:{success}:{error}')
" 2>&1 | grep -v "WARNING\|INFO\|I0000\|absl\|HTTP Error")
    
    echo "$RESULT"
    
    # 結果を抽出
    BATCH_STATS=$(echo "$RESULT" | grep "BATCH_RESULT" | tail -1)
    if [ -n "$BATCH_STATS" ]; then
        BATCH_SUCCESS=$(echo "$BATCH_STATS" | cut -d: -f2)
        BATCH_ERROR=$(echo "$BATCH_STATS" | cut -d: -f3)
        TOTAL_SUCCESS=$((TOTAL_SUCCESS + BATCH_SUCCESS))
        TOTAL_ERROR=$((TOTAL_ERROR + BATCH_ERROR))
    fi
    
    ELAPSED=$(($(date +%s) - START_TIME))
    ELAPSED_MIN=$((ELAPSED / 60))
    
    echo "--- バッチ $i 完了 (累計: 成功${TOTAL_SUCCESS}, エラー${TOTAL_ERROR}, 経過${ELAPSED_MIN}分) ---"
    echo "メモリ解放のため5秒待機..."
    sleep 5
done

END_TIME=$(date +%s)
TOTAL_TIME=$(((END_TIME - START_TIME) / 60))

echo ""
echo "========================================"
echo "  処理完了"
echo "========================================"
echo "成功: ${TOTAL_SUCCESS}曲"
echo "エラー: ${TOTAL_ERROR}曲"
echo "合計時間: ${TOTAL_TIME}分"
echo "終了時刻: $(date)"
echo ""
echo "💡 rekordboxで「タグを再読み込み」を実行してください"
