"""lingbot-vision CPU スモークテスト.

GPU なしのマシンで lingbot-vision のバックボーンが動くかを確認する。
- Small(ViT-S/16) モデルを fp32 / CPU で読み込み
- 付属のサンプル画像で patch token を抽出
- 形状を表示できれば成功

submodule をリポジトリ内の相対パスで解決するので、
別マシンで clone しても（ディレクトリ構成が同じなら）そのまま動く。

使い方:
    python test_cpu.py                 # 付属 examples/example.png を使用
    python test_cpu.py path/to/img.png # 任意の画像（D405 で撮った RGB など）
"""

import sys
from pathlib import Path

# --- submodule をパスに追加（絶対パス依存を避ける） ---
REPO = Path(__file__).resolve().parent
SUBMODULE = REPO / "lingbot-vision"
if not SUBMODULE.exists():
    sys.exit(
        f"[test_cpu] submodule が見つかりません: {SUBMODULE}\n"
        "  git submodule update --init --recursive を実行してください。"
    )
sys.path.insert(0, str(SUBMODULE))


def main():
    import torch

    from lingbot_vision import (
        extract_patch_tokens,
        load_image,
        load_pretrained_backbone,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    print(f"[test_cpu] torch={torch.__version__} device={device} dtype={dtype}")
    if device == "cpu":
        print("[test_cpu] GPU なし -> CPU / fp32 で実行（初回はモデルを HF から DL）")

    # 入力画像: 引数があればそれ、なければ付属サンプル
    img_path = Path(sys.argv[1]) if len(sys.argv) > 1 else SUBMODULE / "examples" / "example.png"
    if not img_path.exists():
        sys.exit(f"[test_cpu] 画像が見つかりません: {img_path}")

    # Small モデル: CPU 向けの軽量バックボーン
    print("[test_cpu] Small(ViT-S/16) バックボーンを読み込み中 ...")
    # variant= で指定する（第1引数 repo_id_or_path に "small" を渡すと
    # HF の "small" という repo を探しに行って 401 になるので注意）。
    backbone, embed_dim = load_pretrained_backbone(variant="small", device=device, dtype=dtype)
    print(f"[test_cpu] 読み込み完了 embed_dim={embed_dim} patch_size={backbone.patch_size}")

    img_norm, img_rgb, (H, W) = load_image(
        str(img_path), size=512, patch_size=backbone.patch_size
    )
    print(f"[test_cpu] 画像読み込み: {img_path.name} ({H}x{W})")

    tokens, (h, w) = extract_patch_tokens(backbone, img_norm, device, dtype)
    print(f"[test_cpu] patch tokens: shape={tuple(tokens.shape)} grid={h}x{w}")

    print("[test_cpu] ✅ 成功: CPU で lingbot-vision が動作しました。")


if __name__ == "__main__":
    main()
