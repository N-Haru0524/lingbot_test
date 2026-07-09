"""lingbot-depth（LingBot-Depth 2.0）で生深度を精緻化・補完して可視化する.

RGB + 生深度 + カメラ内部パラメータ を入力し、精緻化した深度を推定する。
既定では submodule 付属のサンプル（lingbot-depth/examples/0）を使う。
自前データを使う場合は --rgb / --depth / --intrinsics を指定する（D405 で撮った
RGB-D など）。

出力（outputs/depth/ 配下）:
    <stem>_cmp.png    ... [RGB | 生深度 | 精緻化深度] の比較パネル
    <stem>_refined.npy ... 精緻化深度(メートル, float32)

xformers について:
    LingBot-Depth の RGBD エンコーダは、深度トークンのマスク時に可変長トークンを
    束ねるため xformers（nested tensor 注意）を要求する。ここでは推論時マスクを
    切って（enable_depth_mask=False）単一テンソル経路に通すことで、xformers 無し
    ＝素の PyTorch attention で動かす。深度補完・精緻化の推論用途では全入力深度を
    使う方が自然なので、実用上の劣化はほぼない。

submodule をリポジトリ内の相対パスで解決するので、別マシンで clone しても
（ディレクトリ構成が同じなら）そのまま動く。

使い方:
    uv run python depth_test.py                      # 付属 examples/0
    uv run python depth_test.py --example 2          # examples/2
    uv run python depth_test.py --model robbyant/lingbot-depth-postrain-dc-vitl14-v0.5
    uv run python depth_test.py --rgb a.png --depth d.png --intrinsics k.txt --depth-scale 1000
"""

import argparse
import os
import sys
from pathlib import Path

# 深度エンコーダの nested-tensor 経路は xformers を要求するので明示的に無効化し、
# enable_depth_mask=False で素の attention 経路に通す（import 前に設定する）。
os.environ.setdefault("XFORMERS_DISABLED", "1")

# --- submodule を相対パスで解決（絶対パス依存を避ける） ---
REPO = Path(__file__).resolve().parent
SUBMODULE = REPO / "lingbot-depth"
if not SUBMODULE.exists():
    sys.exit(
        f"[depth] submodule が見つかりません: {SUBMODULE}\n"
        "  git submodule update --init --recursive を実行してください。"
    )
sys.path.insert(0, str(SUBMODULE))

import cv2
import numpy as np
import torch

from mdm.model.v2 import MDMModel


def colorize_depth(depth, vmin=None, vmax=None, colormap=cv2.COLORMAP_TURBO):
    """深度(メートル)を TURBO カラーの BGR 画像にする。無効画素(<=0/非有限)は黒。"""
    valid = np.isfinite(depth) & (depth > 0)
    vmin = (depth[valid].min() if valid.any() else 0.0) if vmin is None else vmin
    vmax = (depth[valid].max() if valid.any() else 1.0) if vmax is None else vmax
    u8 = np.clip((depth - vmin) / (vmax - vmin + 1e-8) * 255, 0, 255).astype(np.uint8)
    colored = cv2.applyColorMap(u8, colormap)
    colored[~valid] = 0
    return colored, (vmin, vmax)


def resolve_inputs(args):
    """(rgb_path, depth_path, intrinsics_path, stem) を返す。"""
    if args.rgb or args.depth or args.intrinsics:
        missing = [n for n, v in [("--rgb", args.rgb), ("--depth", args.depth), ("--intrinsics", args.intrinsics)] if not v]
        if missing:
            sys.exit(f"[depth] 自前データを使うには {', '.join(missing)} も必要です。")
        return Path(args.rgb), Path(args.depth), Path(args.intrinsics), Path(args.rgb).stem

    ex_dir = SUBMODULE / "examples" / str(args.example)
    if not ex_dir.is_dir():
        avail = ", ".join(sorted(p.name for p in (SUBMODULE / "examples").iterdir() if p.is_dir()))
        sys.exit(f"[depth] サンプルが見つかりません: {ex_dir}\n  利用可能: {avail}")
    rgb = next((ex_dir / f"rgb{e}" for e in (".png", ".jpg", ".jpeg") if (ex_dir / f"rgb{e}").exists()), None)
    if rgb is None:
        sys.exit(f"[depth] rgb.(png|jpg) が見つかりません: {ex_dir}")
    return rgb, ex_dir / "raw_depth.png", ex_dir / "intrinsics.txt", f"example{args.example}"


def build_parser():
    ap = argparse.ArgumentParser(prog="python depth_test.py")
    ap.add_argument("--example", default="0", help="付属サンプル番号（lingbot-depth/examples 配下, 既定 0）")
    ap.add_argument("--rgb", help="自前 RGB 画像パス（--depth/--intrinsics と併用）")
    ap.add_argument("--depth", help="自前 生深度 PNG(16bit) パス")
    ap.add_argument("--intrinsics", help="自前 内部パラメータ(.txt/.json, 3x3)")
    ap.add_argument("--depth-scale", type=float, default=1000.0, help="深度PNGをメートルに割る係数（mm=1000）")
    ap.add_argument("--model", default="robbyant/lingbot-depth-pretrain-vitl-14-v0.5", help="HF モデルID/ローカルckpt")
    return ap


def main():
    args = build_parser().parse_args()
    rgb_path, depth_path, intr_path, stem = resolve_inputs(args)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[depth] torch={torch.__version__} device={device}")
    for p in (rgb_path, depth_path, intr_path):
        if not Path(p).exists():
            sys.exit(f"[depth] 入力が見つかりません: {p}")

    # --- 入力読み込み ---
    image = cv2.cvtColor(cv2.imread(str(rgb_path)), cv2.COLOR_BGR2RGB)
    h, w = image.shape[:2]
    image_t = torch.tensor(image / 255.0, dtype=torch.float32, device=device).permute(2, 0, 1)[None]

    depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED).astype(np.float32) / args.depth_scale
    depth = np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)
    depth_t = torch.tensor(depth, dtype=torch.float32, device=device)[None]

    intr = np.loadtxt(str(intr_path)) if str(intr_path).endswith(".txt") else np.array(__import__("json").load(open(intr_path)), np.float32)
    intr = intr.astype(np.float32).copy()
    intr[0] /= w  # fx, cx を幅で正規化
    intr[1] /= h  # fy, cy を高さで正規化
    intr_t = torch.tensor(intr, dtype=torch.float32, device=device)[None]
    print(f"[depth] 入力 {rgb_path.name} {w}x{h}  生深度 {depth[depth > 0].min():.2f}-{depth.max():.2f} m")

    # --- モデル読み込み・推論 ---
    print(f"[depth] モデル読み込み中: {args.model}（初回は HF から DL）")
    model = MDMModel.from_pretrained(args.model).to(device)
    with torch.no_grad():
        # enable_depth_mask=False で xformers 不要の単一テンソル経路に通す
        out = model.infer(image_t, depth_in=depth_t, intrinsics=intr_t, enable_depth_mask=False)
    pred = out["depth"].squeeze().float().cpu().numpy()
    valid_pred = pred[np.isfinite(pred) & (pred > 0)]
    print(f"[depth] 精緻化深度 {valid_pred.min():.2f}-{valid_pred.max():.2f} m  "
          f"欠損 {(depth <= 0).mean() * 100:.1f}% -> {(~(np.isfinite(pred) & (pred > 0))).mean() * 100:.1f}%")

    # --- 可視化・保存（生と精緻化で同じレンジに合わせる） ---
    lo = float(min(depth[depth > 0].min(), valid_pred.min()))
    hi = float(np.percentile(valid_pred, 99))
    raw_c, _ = colorize_depth(depth, lo, hi)
    ref_c, _ = colorize_depth(pred, lo, hi)
    panel = np.concatenate([cv2.cvtColor(image, cv2.COLOR_RGB2BGR), raw_c, ref_c], axis=1)

    out_dir = REPO / "outputs" / "depth"
    out_dir.mkdir(parents=True, exist_ok=True)
    cmp_path = out_dir / f"{stem}_cmp.png"
    npy_path = out_dir / f"{stem}_refined.npy"
    cv2.imwrite(str(cmp_path), panel)
    np.save(str(npy_path), pred)
    print(f"[depth] 保存: {cmp_path}")
    print(f"[depth] 保存: {npy_path}")
    print("[depth] ✅ 完了（左=RGB / 中=生深度 / 右=精緻化深度）")


if __name__ == "__main__":
    main()
