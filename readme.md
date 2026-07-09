# lingbot_test

## セットアップ〜実行

```bash
# clone（submodule の lingbot-vision も一緒に取得）
git clone --recurse-submodules https://github.com/N-Haru0524/lingbot_test.git
cd lingbot_test

# 既に clone 済みで lingbot-vision/ が空の場合はこちら
git submodule update --init --recursive

# 依存関係のインストール（.venv が作られる）
uv sync

# スモークテスト（付属サンプル画像で patch token 抽出）
uv run python test_cpu.py

# ウェブカメラでライブ試験（q: 終了 / s: outputs/webcam/ に保存）
uv run python webcam_test.py --variant giant

# D405 でライブ試験（q: 終了 / s: outputs/d405/ に保存）
uv run python d405_test.py --variant giant

# 深度の精緻化・補完（LingBot-Depth 2.0。付属サンプルで試す）
uv run python depth_test.py --example 0

# D405 の生深度をライブ精緻化（q: 終了 / s: outputs/d405_depth/ に保存）
uv run python d405_depth_test.py
```

GPU（CUDA）は自動検出され、あれば bfloat16 で推論する。
`--variant` は small / base / large / giant から選択（giant が最大。初回は HF からモデルを DL）。

`d405_test.py` は **lingbot-vision と lingbot-depth の 2 モデルを同時ロード**し、
ウィンドウに **[入力 | PCA特徴 | 輪郭 | 生深度 | 精緻化深度]** を上下二段組で並べて表示する
（GPU に余裕がある前提。RTX 5090 で精緻化は数十 ms/frame）。
輪郭は patch token の特徴不連続（隣接 token の cosine 距離）から作る学習不要マップ。
`--no-refine` で深度モデルを読まず vision のみ、`--no-edges` / `--no-depth` で各パネルを省略、
`--size`（vision）/ `--refine-size`（depth）で推論解像度を調整する。

### submodule 構成

- **lingbot-vision** — RGB 専用の ViT バックボーン。PCA 特徴・輪郭の元（`test_cpu.py` / `webcam_test.py` / `d405_test.py`）。
- **lingbot-depth** — RGB + 生深度 + 内部パラメータ から深度を精緻化・補完する RGB-D モデル（`depth_test.py`）。

`depth_test.py` は付属サンプル（`lingbot-depth/examples/N`）または自前の RGB-D
（`--rgb/--depth/--intrinsics`）を入力し、`outputs/depth/` に **[RGB | 生深度 | 精緻化深度]**
パネルと `.npy` を保存する。深度エンコーダは本来 xformers（nested tensor）を要求するが、
`enable_depth_mask=False` で単一テンソル経路に通し **xformers 無し**で動かしている
（`torch==2.6` 固定を避け、既存の torch を温存するため）。

### D405 の対応解像度

実機（FW 5.12.14.100）で Color / Depth の最大は **1280×720 @ 30fps**（`d405_test.py` の既定値）。
`--width` / `--height` で下げられる（例: `640x480`, `848x480`）。

## テスト状況

lingbot-vision を Intel RealSense **D405** で動作確認する。

### 済み
- [x] D405 の接続確認（デバイス認識・ドライバ）
- [x] lingbot-vision のセットアップ（`test_cpu.py` で patch token 抽出を確認）
- [x] D405 からの映像取得テスト（Color/Depth を align して取得）
- [x] lingbot-vision の動作テスト（**D405 + giant** でライブ試験。PCA・輪郭・深度パネルを表示）
- [x] lingbot-depth のセットアップ（submodule 追加。付属サンプルで精緻化を確認）

### TODO
- [ ] lingbot-depth に **D405 の生深度**を入力して精緻化（`depth_test.py` を D405 ライブ対応に拡張）
- [ ] テスト結果の記録（保存パネルの整理・所見のまとめ）
- [ ] 推論速度の確認（CPU は重い。GPU / `--size` 調整で改善）
