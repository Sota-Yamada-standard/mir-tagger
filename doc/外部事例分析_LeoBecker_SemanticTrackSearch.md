# 外部事例分析: Leo Becker「Semantic Track Search」

**対象**: [Instagram Reel `Db4DS7UTKpY`](https://www.instagram.com/reel/Db4DS7UTKpY/)  
**投稿者**: [@leobeckerdesign](https://www.instagram.com/leobeckerdesign)（Leo Becker）  
**調査日**: 2026-09-16  
**結論先出し**: この投稿は「曲をタグで分類する」のではなく、**音そのものをベクトルにして、ジャンルより細かい“感じ”でライブラリを切る**実装を見せている。MIR-Taggerが今持っている1曲単位の離散タグ（`[Genre:]` `[Mood:]` `[Energy:]`）の**一段上のレイヤー**であり、やりたいことの中核（セットに入れる曲を高精度で決める）と重なる。

---

## 1. 投稿の確定事実

公式 oEmbed（Instagram API）と公開ページから確認した事実。推測は後段に分ける。

| 項目 | 値 |
|------|-----|
| 投稿日 | 2026-08-10 |
| 反応 | いいね 3,657 / コメント 351 |
| アカウント | `leobeckerdesign`（認証済み、フォロワー約60.4万） |
| 肩書 | Designer and Creative Engineer / Adobe Creative Partner |
| プロフィール | Craft, Code and Culture |
| 共同制作 | キャプション末尾「Claudinho codando e eu mandando」（Claudinhoが実装、Leoが指示） |

### キャプション全文（oEmbed、原文）

```
O poder dos dados na tomada de decisões em qualquer área da vida 💚
Usei a tecnologia vetorial para analisar e organizar visualmente
minhas músicas no @rekordboxofficial , me ajudando a tomar decisões
de quais músicas poderiam entrar no meu set com base no seu
sentimento, e não apenas no estilo puro.

Stack utilizado: Python, ffmpeg, CLAP, PyTorch, transformers,
scikit-learn, UMAP, pyrekordbox, pandas, NumPy, HTML/Canvas 2D.

Claudinho codando e eu mandando 🤓
```

### 日本語訳

> あらゆる領域の意思決定におけるデータの力。ベクトル技術で自分の曲を解析し、rekordbox上で視覚的に整理した。セットに入れる曲を、純粋なスタイル（ジャンル）だけでなく、**その曲の感情／感じ**で判断できるようにした。
>
> 使用スタック: Python, ffmpeg, CLAP, PyTorch, transformers, scikit-learn, UMAP, pyrekordbox, pandas, NumPy, HTML/Canvas 2D。

### サムネイル

公開サムネは、CDJ風の深緑デバイスをオレンジ背景に置いた製品写真。液晶に大きく **`SEMANTIC` / `TRACK SEARCH`**。これは「波形を見るツール」ではなく、**意味で曲を探すプロダクト**として見せている。

---

## 2. 動画で実際にやっていること

公開Reelをフレーム単位で確認した。音声の完全書き起こしはログイン壁のため取れない。画面上のポルトガル語テロップとUIから再構成する。

### 語っている主張（テロップの流れ）

1. データの力で意思決定する
2. プレイリストを組んで演奏する場面を見せる
3. 「あるスタイル」では足りない、という問題提起
4. **Deep House には、ジャンル名よりずっと多いものがある**
5. スタイルより深い分析である
6. セットを始める曲を決めるために使う
7. ライブラリ全体を細分化してくれる
8. 「同じように見える／聞こえる曲」を見つける難しさを解く

要するに: **ジャンルタグは粗すぎる。同じ Deep House の中にも、セットに繋がる曲と繋がらない曲がある。音の近さで切れ、という話。**

### 画面に映っているもの

- **rekordbox本体**: 波形、ビートグリッド、ホットキュー、トラックリスト、デッキ
- **独自オーバーレイ**: 赤／ピンクと緑／シアンの散布図。点が1曲、近い点が似た曲。UMAPの典型的な見た目
- **開発環境**: Pythonを回しているターミナル、デュアルモニタ、DJコントローラ
- 散布図は rekordbox の標準機能ではない。キャプションの HTML/Canvas 2D と一致する

### パイプラインの推定（キャプションのスタックから一意に組める）

```
音源ファイル
  → ffmpeg でデコード
  → CLAP（transformers / PyTorch）で音声埋め込み
  → 曲ごと1ベクトルとして保存（pandas / NumPy）
  → UMAP で2次元に落とす
  → 必要なら scikit-learn でクラスタ
  → HTML/Canvas で散布図
  → pyrekordbox でライブラリ／プレイリストと接続
```

CLAP（Contrastive Language-Audio Pretraining）は、**音声と自然言語を同じ空間に置く**モデル。だからこのシステムは次の2つが同時にできる。

1. **曲同士の近傍検索**（この曲に近い曲）
2. **テキスト検索**（「暗い、ボーカル少なめ、少し憂鬱なハウス」でヒット）

サムネの `SEMANTIC TRACK SEARCH` は 2 を製品名にしたもの。

---

## 3. 他投稿の調査

プロフィールグリッド、oEmbed、GitHub、公式サイト、関連Reelを当たった。

### 音楽・rekordbox関連は、この1本だけ

| 確認先 | 結果 |
|--------|------|
| Instagram プロフィール直近（2026-08〜09） | Firefly、GPT、週次まとめ、Schema3D、広告、ローカルAI。音楽解析はなし |
| GitHub [`leobeckerdesign`](https://github.com/leobeckerdesign) | 公開10リポジトリはデザイン／Web／3D。音・rekordbox・CLAPはゼロ |
| [leobecker.com](https://www.leobecker.com) | Instagramプロフィールへリダイレクト |
| Web検索（短コード、キャプション語句、SEMANTIC TRACK SEARCH） | ソース公開・技術ブログは見つからない |

このプロジェクトは**公開されていない個人／コラボツール**。コードは真似できず、**設計思想だけが使える**。

### 参考になる隣接投稿

音楽そのものではないが、方針がMIR-Taggerと噛み合うものが2本ある。

#### A. ローカル推論を前提にする  
[Reel `Dcld8ksz_Ua`](https://www.instagram.com/reel/Dcld8ksz_Ua/)（2026-08-28）

> ローカルでAIモデルを回すのが未来。Apple M5/M6、DGX Spark、RTX 5090。IAは光やラジオと同じくらい強い技術で、長く少数の手には残らない。外食ばかりしていると、いつか家で作れるようになる。ハードウェアがコンロ。

MIR-Taggerは最初からローカル（Essentia / PANNs / Demucs / MPS）。Leoのrekordbox案件もクラウドAPIではなく、列挙スタックが全部オンプレ。**埋め込みもローカルに置く**のがこの人の一貫した方針。

#### B. 関係を「見る」ための可視化  
[Reel `DdUDG-4Trdz`](https://www.instagram.com/reel/DdUDG-4Trdz/)（Schema3D、2026-09-15）

DBスキーマをMermaid/SQLから3Dにして、テーブルの関係を目で辿る。オープンソース、ブラウザ完結。

UMAP散布図と同じ発想: **一覧やタグでは見えない近傍を、空間にして判断させる**。LeoはDJライブラリでもDBでも、同じことをしている。

それ以外の直近投稿（Firefly Boards、GPT-6 Astra、ChatGPT Images、Ads for bots、VGPU、週次まとめ）は音楽解析の実装参考にはならない。Adobe案件の「先にビジョンとワークフローを作る」は、タグ設計を先に固める今の進め方と矛盾しない、程度。

---

## 4. MIR-Taggerとの重なりとズレ

今のMIR-Taggerは**1曲を解析して離散タグをCommentに書く**。Leoは**ライブラリ全体を連続空間に置いて、近傍とテキストで切る**。

| 観点 | MIR-Tagger（現状） | Leoのシステム | 重なり |
|------|-------------------|---------------|--------|
| 対象 | ローカル音源 | ローカル音源 | 同じ |
| 出口 | rekordbox Comment / 検索 | rekordbox + 独自Canvas | 同じ現場 |
| ジャンル | MusiCNN+PANNs の離散ラベル | 「スタイルだけでは足りない」が主張 | 問題意識は同じ。解法が違う |
| ムード | happy/sad/relaxed/aggressive の4択 | CLAP空間の連続量（sentimento） | 今の4択は粗い |
| エネルギー／踊り | RMS・Danceability の段階ラベル | 埋め込みに内包（明示タグなし） | 補完関係 |
| アニソン | MAL照合（高精度、メタデータ） | 音声埋め込みでは作品名は出ない | こちらが勝つ |
| イントロ／拍 | Demucs+PANNs、Downbeat | 見えない（セット選曲の話） | 別問題。残す |
| ライブラリ横断 | ほぼ無し（曲単位） | 主機能 | **欠けている層** |
| 検索 | Comment文字列の部分一致 | 意味検索（テキスト→音） | **欠けている層** |
| 可視化 | なし | UMAP散布図 | 後回しでよい |
| rekordbox接続 | ID3 Comment + XMLパーサ | pyrekordbox（DB直） | 方針が違う |

ユーザー方針（BPMはrekordbox任せ、高精度優先、プラグイン拡張、セリフ入り曲対応）と矛盾する点はない。Leoが解いているのは、進捗メモの「今後」にある構造解析より手前、**選曲の精度**。

`[Club:Yes]` / `[Club:SuperYes]` は人手のキュレーション。Leoの近傍検索は、その自動版に近い。置き換えではなく、候補出しに使える。

---

## 5. 取り込むこと / 取り込まないこと

### 取り込む（本命）

1. **CLAP埋め込みを1曲1ベクトルでキャッシュする**  
   既存の `BaseAnalyzer` には乗せつつ、本番の価値はライブラリ横断。sidecar（`.clap.npy`）か専用インデックス。
2. **ゼロショットの `[Vibe:…]` タグ**  
   「dark / euphoric / melancholic / peak-time / warm-vocal」などをテキスト埋め込みと余弦類似度で付ける。`MoodAnalyzer` の4択を置き換えず、開いた語彙を足す。rekordboxのComment検索にそのまま載る。
3. **近傍タグまたは類似検索CLI**  
   例: `python -m src.search "暗いアニソンっぽい、セリフ始まり"`。CLAPがテキストと音を同一空間に置くので、追加学習なしで動く。
4. **ローカル完結**  
   Leoの他投稿と同じ。クラウド埋め込みAPIは使わない。

### 後回しでよい

- UMAP + HTML/Canvas の散布図（説明用には強いが、DJ現場の検索には必須ではない）
- pyrekordbox による `master.db` 直書き（今のID3 Comment + 「タグを再読み込み」より事故りやすい。プレイリスト書き戻しが必要になったら）

### 取り込まない

- 既存の Genre / Mood / Anisong / Intro / Beat をCLAPに置き換えること  
  作品名（MAL）、セリフ開始秒、Downbeat は埋め込みでは出ない
- 見た目のプロダクトUIを先に作ること

---

## 6. 推奨する次の実装（1本に絞る）

**先にやるのは UMAP 可視化ではなく、CLAP埋め込み + ゼロショット `[Vibe:]` + 意味検索CLI。**

理由:

- 投稿の製品名は可視化ではなく **Semantic Track Search**
- 今のCommentタグ運用に、新しい離散タグとしてすぐ載せられる
- Deep House問題（同じジャンルの中を切る）を、開いた語彙で直接殴れる
- 散布図はその埋め込みがあれば後から何回でも作り直せる

既存設計への載せ方:

```
src/analyzers/clap_analyzer.py     # 1曲 → embedding + vibe tags
src/index/embedding_store.py      # ライブラリ横断のベクトル保存
src/search.py                     # テキスト / 曲ID で近傍検索
```

`ClapAnalyzer` は他と同様 `@AnalyzerRegistry.register`。`to_tag()` は `[Vibe:dark][Vibe:melancholic]` のように複数可。モデルは LAION-CLAP 系（`laion/clap-htsat-unfused` または music 特化チェックポイント）を遅延ロード＋クラス変数キャッシュ。ffmpeg は既存の `audio_loader` で足りる可能性が高い。

タグ例（既存を壊さない）:

```
[Downbeat:1][Dance:High][Energy:VeryHigh][Genre:pop][Genre:anime]
[Mood:happy][Vibe:euphoric][Vibe:bright-vocal][Key:Eb]
```

`[Vibe:]` の語彙は設定ファイルで持ち、アニソン／クラブ向けに後から足せるようにする。CLAPは「セリフ多めの明るいオープニング」のような自由文も、検索側ではタグを経由せずに使える。

---

## 7. 未確認（推測で埋めない）

- 使っているCLAPの具体的チェックポイント
- UMAPの `n_neighbors` / `min_dist`
- pyrekordbox で読んでいるだけか、プレイリストを書き戻しているか
- クラスタをMy Tagに落としているか
- Claudinho の公開リポジトリ（見つからず）
- コメント欄の技術的な補足（351件あるがログイン壁）

ソースは非公開なので、再現は公開されているCLAP + UMAP + pyrekordboxの一般的な組み合わせで十分。細部のコピーはできないし、する必要もない。

---

## 8. 出典

- Instagram Reel: https://www.instagram.com/reel/Db4DS7UTKpY/
- oEmbed: `https://www.instagram.com/api/v1/oembed/?url=https://www.instagram.com/p/Db4DS7UTKpY/`
- プロフィール: https://www.instagram.com/leobeckerdesign/
- 隣接Reel（ローカルAI）: https://www.instagram.com/reel/Dcld8ksz_Ua/
- 隣接Reel（Schema3D）: https://www.instagram.com/reel/DdUDG-4Trdz/
- GitHub: https://github.com/leobeckerdesign
