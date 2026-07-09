"""Intel RealSense D405 で lingbot-vision をライブ試験する（ウィンドウ表示）.

D405 からカラー画像と深度を取得し、2 つのモデルを同時にロードして可視化する:
  - lingbot-vision  ... カラーの patch token を PCA(上位3成分)＋輪郭で可視化
  - lingbot-depth   ... 生深度を精緻化・補完（--refine, 既定 ON）
ウィンドウに [入力 | PCA特徴 | 輪郭 | 生深度 | 精緻化深度] の面を並べて表示する。
輪郭は patch token の特徴不連続から作る（teaser 3列目の boundary 風, --edges 既定 ON）。
精緻化は enable_depth_mask=False で xformers 無しで動かす（depth_test.py と同じ）。

CPU 推論なので重い。滑らかさが欲しければ --size を小さく（例 256）する。

操作（ウィンドウにフォーカスした状態で）:
    q ... 終了
    s ... 現在のパネルを outputs/d405/ に保存

使い方:
    uv run python d405_test.py                 # vision + depth を同時ロード
    uv run python d405_test.py --size 256      # vision を軽く・速く
    uv run python d405_test.py --no-refine     # 深度精緻化を無効（生深度のみ表示）
    uv run python d405_test.py --no-edges      # 輪郭パネルを出さない
    uv run python d405_test.py --no-depth      # 深度・精緻化を出さない
    uv run python d405_test.py --width 848 --height 480
"""

import argparse
import os
import sys
import time
from pathlib import Path

# 深度エンコーダの nested-tensor 経路を避けるため xformers を無効化（mdm import 前に設定）。
os.environ.setdefault("XFORMERS_DISABLED", "1")

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
from lingbot_vision.pca_demo import _label

# カラー前処理・PCA 可視化・輪郭抽出は webcam 版と共通
from webcam_test import StablePCA, edges_to_rgb, feature_edges, preprocess_frame


def two_row_grid(panels):
    """パネル群を上下二段に並べて 1 枚にする（上段が同数か 1 枚多い）。全パネル同サイズ前提。"""
    n = len(panels)
    if n <= 1:
        return panels[0] if panels else None
    ncols = (n + 1) // 2  # 2 行になる列数（上段 >= 下段）
    blank = np.zeros_like(panels[0])
    rows = []
    for r in range(0, n, ncols):
        row = list(panels[r:r + ncols])
        row += [blank] * (ncols - len(row))  # 端数は黒で詰める
        rows.append(np.concatenate(row, axis=1))
    return np.concatenate(rows, axis=0)


def build_parser():
    ap = argparse.ArgumentParser(prog="python d405_test.py")
    ap.add_argument("--variant", default="small", choices=["small", "base", "large", "giant"])
    ap.add_argument("--size", type=int, default=1024, help="推論入力の一辺(px)。小さいほど速い")
    ap.add_argument("--width", type=int, default=1280, help="D405 ストリーム幅（最大 1280）")
    ap.add_argument("--height", type=int, default=720, help="D405 ストリーム高さ（最大 720）")
    ap.add_argument("--fps", type=int, default=30, help="D405 フレームレート")
    ap.add_argument("--no-edges", dest="edges", action="store_false", help="輪郭パネルを表示しない")
    ap.add_argument("--no-depth", dest="depth", action="store_false", help="深度・精緻化を表示しない")
    ap.add_argument("--no-refine", dest="refine", action="store_false", help="lingbot-depth 精緻化を無効にする")
    ap.add_argument("--refine-size", type=int, default=480, help="深度モデルの推論高さ(px)")
    ap.add_argument(
        "--depth-model", default="robbyant/lingbot-depth-pretrain-vitl-14-v0.5", help="lingbot-depth の HF モデルID/ckpt"
    )
    return ap


def start_pipeline(args):
    """D405 のカラー(+深度)ストリームを開始し、(pipeline, align, K_norm, depth_scale) を返す。

    K_norm は color 内部パラメータを幅・高さで正規化した 3x3（リサイズ不変。深度精緻化用）。
    深度を出さない場合 (K_norm, depth_scale) は (None, None)。
    """
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, args.width, args.height, rs.format.bgr8, args.fps)
    if args.depth:
        config.enable_stream(rs.stream.depth, args.width, args.height, rs.format.z16, args.fps)
    try:
        profile = pipeline.start(config)
    except RuntimeError as e:
        sys.exit(
            f"[d405] ストリーム開始に失敗しました: {e}\n"
            "  - D405 が USB で接続されているか確認してください（USB3 推奨）。\n"
            "  - 解像度/FPS が D405 対応値か確認してください（例: 640x480@30, 848x480@30, 1280x720@30）。\n"
            "  - 他アプリ(RealSense Viewer 等)がカメラを掴んでいないか確認してください。"
        )
    K = depth_scale = None
    if args.depth:
        ci = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
        depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
        K = np.array([[ci.fx, 0.0, ci.ppx], [0.0, ci.fy, ci.ppy], [0.0, 0.0, 1.0]], dtype=np.float32)
        K[0] /= ci.width   # fx, cx を幅で正規化
        K[1] /= ci.height  # fy, cy を高さで正規化
    # 深度をカラー視点に位置合わせ
    align = rs.align(rs.stream.color) if args.depth else None
    return pipeline, align, K, depth_scale


def main():
    args = build_parser().parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    print(f"[d405] torch={torch.__version__} device={device} dtype={dtype}")
    print(f"[d405] {args.variant} バックボーンを読み込み中 ...")
    backbone, embed_dim = load_pretrained_backbone(variant=args.variant, device=device, dtype=dtype)
    print(f"[d405] 読み込み完了 embed_dim={embed_dim} patch_size={backbone.patch_size}")

    # --- lingbot-depth（任意の 2 つ目のモデル）を同時ロード ---
    refiner = colorize_depth = None
    if args.depth and args.refine:
        depth_sub = REPO / "lingbot-depth"
        if not depth_sub.exists():
            print(f"[d405] lingbot-depth submodule が無いため精緻化を無効化: {depth_sub}")
        else:
            sys.path.insert(0, str(depth_sub))
            from mdm.model.v2 import MDMModel
            from depth_test import colorize_depth
            print(f"[d405] 深度モデル読み込み中: {args.depth_model}（初回は HF から DL）")
            refiner = MDMModel.from_pretrained(args.depth_model).to(device).eval()
            print("[d405] 深度モデル読み込み完了")

    pipeline, align, K, depth_scale = start_pipeline(args)
    intr_t = torch.tensor(K, dtype=torch.float32, device=device)[None] if (refiner is not None and K is not None) else None
    pca_rgb = StablePCA()

    out_dir = REPO / "outputs" / "d405"
    win = "lingbot-vision D405 (q=quit, s=save)"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    print("[d405] 起動しました。 ウィンドウで q=終了 / s=保存")
    saved = 0
    try:
        while True:
            try:
                frames = pipeline.wait_for_frames()
            except RuntimeError:
                print("[d405] フレーム待ちがタイムアウトしました。パイプラインを再起動します ...")
                pipeline.stop()
                pipeline, align, K, depth_scale = start_pipeline(args)
                intr_t = torch.tensor(K, dtype=torch.float32, device=device)[None] if (refiner is not None and K is not None) else None
                continue
            if align is not None:
                frames = align.process(frames)
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue
            color_bgr = np.asanyarray(color_frame.get_data()).copy()  # BGR8

            depth_m = None
            if args.depth:
                depth_frame = frames.get_depth_frame()
                if depth_frame:
                    depth_m = np.asanyarray(depth_frame.get_data()).astype(np.float32) * depth_scale  # メートル
                del depth_frame
            # フレームを保持したまま重い推論をするとプールが枯渇して取得が止まるため、
            # 必要なデータをコピーしたらすぐプールへ返す
            del frames, color_frame

            # --- lingbot-vision 推論 ---
            img_norm, img_rgb, (H, W) = preprocess_frame(color_bgr, args.size, backbone.patch_size)
            tokens, (h, w) = extract_patch_tokens(backbone, img_norm, device, dtype)
            pca = pca_rgb(tokens, h, w)
            pca_up = cv2.resize(pca, (W, H), interpolation=cv2.INTER_NEAREST)

            panels = [_label(img_rgb, f"input {H}x{W}"), _label(pca_up, f"patch PCA {h}x{w}")]

            # --- 輪郭パネル（任意, teaser 3列目風） ---
            if args.edges:
                edges = edges_to_rgb(feature_edges(tokens, h, w), W, H)
                panels.append(_label(edges, f"feature edges {h}x{w}"))

            # --- 深度パネル（生 + 精緻化。任意） ---
            if depth_m is not None:
                pred = None
                raw_view = depth_m
                if refiner is not None:
                    # 深度モデルへはアスペクト維持でリサイズ（正規化 K はリサイズ不変）
                    H0, W0 = color_bgr.shape[:2]
                    Wr = int(round(W0 * args.refine_size / H0))
                    color_r = cv2.resize(color_bgr, (Wr, args.refine_size), interpolation=cv2.INTER_AREA)
                    raw_view = cv2.resize(depth_m, (Wr, args.refine_size), interpolation=cv2.INTER_NEAREST)
                    rgb_r = cv2.cvtColor(color_r, cv2.COLOR_BGR2RGB)
                    image_t = torch.from_numpy(rgb_r.astype(np.float32) / 255.0).permute(2, 0, 1)[None].to(device)
                    depth_t = torch.from_numpy(raw_view)[None].to(device)
                    t0 = time.time()
                    with torch.no_grad():
                        out = refiner.infer(image_t, depth_in=depth_t, intrinsics=intr_t, enable_depth_mask=False)
                    pred = out["depth"].squeeze().float().cpu().numpy()
                    refine_ms = (time.time() - t0) * 1000.0

                # 生と精緻化で同じカラーレンジ（TURBO）に合わせる
                vr = raw_view[(np.isfinite(raw_view)) & (raw_view > 0)]
                vp = pred[(np.isfinite(pred)) & (pred > 0)] if pred is not None else np.empty(0, np.float32)
                pool = np.concatenate([vr, vp]) if vp.size else vr
                lo = float(np.percentile(pool, 2)) if pool.size else 0.0
                hi = float(np.percentile(pool, 98)) if pool.size else 1.0

                raw_rgb = cv2.cvtColor(colorize_depth(raw_view, lo, hi)[0], cv2.COLOR_BGR2RGB)
                panels.append(_label(cv2.resize(raw_rgb, (W, H), interpolation=cv2.INTER_NEAREST), "raw depth"))
                if pred is not None:
                    ref_rgb = cv2.cvtColor(colorize_depth(pred, lo, hi)[0], cv2.COLOR_BGR2RGB)
                    panels.append(
                        _label(cv2.resize(ref_rgb, (W, H), interpolation=cv2.INTER_NEAREST), f"refined {refine_ms:.0f}ms")
                    )

            panel = two_row_grid(panels)  # 上下二段組
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
