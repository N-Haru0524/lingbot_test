"""Intel RealSense D405 で lingbot-vision をライブ試験する（ウィンドウ表示）.

D405 からカラー画像と深度を取得し、カラー画像を lingbot-vision バックボーンへ通して、
patch token の上位3 PCA 成分を RGB 可視化する。
ウィンドウに [入力 | PCA特徴 | 深度] の3面を並べて表示する。

CPU 推論なので重い。滑らかさが欲しければ --size を小さく（例 256）する。

操作（ウィンドウにフォーカスした状態で）:
    q ... 終了
    s ... 現在のパネルを outputs/d405/ に保存

使い方:
    uv run python d405_test.py                 # small / size=384 / 640x480@30 / 深度あり
    uv run python d405_test.py --size 256      # 軽く・速く
    uv run python d405_test.py --no-depth      # 深度パネルを出さない
    uv run python d405_test.py --width 848 --height 480
"""

import argparse
import sys
from pathlib import Path

# --- submodule を相対パスで解決（絶対パス依存を避ける） ---
REPO = Path(__file__).resolve().parent
SUBMODULE = REPO / "lingbot-vision"
if not SUBMODULE.exists():
    sys.exit(
        f"[d405] submodule が見つかりません: {SUBMODULE}\n"
        "  git submodule update --init --recursive を実行してください。"
    )
sys.path.insert(0, str(SUBMODULE))

import cv2
import numpy as np
import pyrealsense2 as rs
import torch

from lingbot_vision import extract_patch_tokens, load_pretrained_backbone
from lingbot_vision.pca_demo import _label, _pca_rgb

# カラー前処理は webcam 版と共通（cv2 BGR フレーム -> 正規化テンソル）
from webcam_test import preprocess_frame


def build_parser():
    ap = argparse.ArgumentParser(prog="python d405_test.py")
    ap.add_argument("--variant", default="small", choices=["small", "base", "large", "giant"])
    ap.add_argument("--size", type=int, default=384, help="推論入力の一辺(px)。小さいほど速い")
    ap.add_argument("--width", type=int, default=640, help="D405 ストリーム幅")
    ap.add_argument("--height", type=int, default=480, help="D405 ストリーム高さ")
    ap.add_argument("--fps", type=int, default=30, help="D405 フレームレート")
    ap.add_argument("--no-depth", dest="depth", action="store_false", help="深度パネルを表示しない")
    return ap


def start_pipeline(args):
    """D405 のカラー(+深度)ストリームを開始し、(pipeline, align) を返す。"""
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, args.width, args.height, rs.format.bgr8, args.fps)
    if args.depth:
        config.enable_stream(rs.stream.depth, args.width, args.height, rs.format.z16, args.fps)
    try:
        pipeline.start(config)
    except RuntimeError as e:
        sys.exit(
            f"[d405] ストリーム開始に失敗しました: {e}\n"
            "  - D405 が USB で接続されているか確認してください（USB3 推奨）。\n"
            "  - 解像度/FPS が D405 対応値か確認してください（例: 640x480@30, 848x480@30, 1280x720@30）。\n"
            "  - 他アプリ(RealSense Viewer 等)がカメラを掴んでいないか確認してください。"
        )
    # 深度をカラー視点に位置合わせ
    align = rs.align(rs.stream.color) if args.depth else None
    return pipeline, align


def main():
    args = build_parser().parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    print(f"[d405] torch={torch.__version__} device={device} dtype={dtype}")
    print(f"[d405] {args.variant} バックボーンを読み込み中 ...")
    backbone, embed_dim = load_pretrained_backbone(variant=args.variant, device=device, dtype=dtype)
    print(f"[d405] 読み込み完了 embed_dim={embed_dim} patch_size={backbone.patch_size}")

    pipeline, align = start_pipeline(args)
    colorizer = rs.colorizer() if args.depth else None

    out_dir = REPO / "outputs" / "d405"
    win = "lingbot-vision D405 (q=quit, s=save)"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    print("[d405] 起動しました。 ウィンドウで q=終了 / s=保存")
    saved = 0
    try:
        while True:
            frames = pipeline.wait_for_frames()
            if align is not None:
                frames = align.process(frames)
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue
            color_bgr = np.asanyarray(color_frame.get_data())  # BGR8

            # --- lingbot-vision 推論 ---
            img_norm, img_rgb, (H, W) = preprocess_frame(color_bgr, args.size, backbone.patch_size)
            tokens, (h, w) = extract_patch_tokens(backbone, img_norm, device, dtype)
            pca = _pca_rgb(tokens, h, w)
            pca_up = cv2.resize(pca, (W, H), interpolation=cv2.INTER_NEAREST)

            panels = [_label(img_rgb, f"input {H}x{W}"), _label(pca_up, f"patch PCA {h}x{w}")]

            # --- 深度パネル（任意） ---
            if args.depth:
                depth_frame = frames.get_depth_frame()
                if depth_frame:
                    depth_bgr = np.asanyarray(colorizer.colorize(depth_frame).get_data())
                    depth_rgb = cv2.cvtColor(depth_bgr, cv2.COLOR_BGR2RGB)
                    depth_rgb = cv2.resize(depth_rgb, (W, H), interpolation=cv2.INTER_NEAREST)
                    panels.append(_label(depth_rgb, "depth"))

            panel = np.concatenate(panels, axis=1)
            cv2.imshow(win, cv2.cvtColor(panel, cv2.COLOR_RGB2BGR))  # 表示は BGR

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("s"):
                out_dir.mkdir(parents=True, exist_ok=True)
                p = out_dir / f"panel_{saved:03d}.png"
                cv2.imwrite(str(p), cv2.cvtColor(panel, cv2.COLOR_RGB2BGR))
                print(f"[d405] 保存: {p}")
                saved += 1
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()
        print("[d405] 終了しました。")


if __name__ == "__main__":
    main()
