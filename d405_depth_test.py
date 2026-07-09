"""Intel RealSense D405 の生深度を lingbot-depth でライブ精緻化する（ウィンドウ表示）.

D405 からカラー・深度・内部パラメータを取得し、生深度を LingBot-Depth 2.0 に通して
精緻化・補完する。ウィンドウに [カラー | 生深度 | 精緻化深度] を並べて表示する。

深度は color 視点に align 済みなので color の内部パラメータを使う。深度エンコーダは
本来 xformers（nested tensor）を要求するが、enable_depth_mask=False で単一テンソル
経路に通して xformers 無しで動かす（depth_test.py と同じ）。

モデルは ViT-L/14 で重い。GPU 前提（CPU では実用的な速度が出ない）。--size で推論解像度
を下げると速くなる（正規化内部パラメータはリサイズ不変なので K の再計算は不要）。

操作（ウィンドウにフォーカスした状態で）:
    q ... 終了
    s ... 現在のパネルを outputs/d405_depth/ に保存

使い方:
    uv run python d405_depth_test.py                  # 1280x720@30 / 推論 size=480
    uv run python d405_depth_test.py --size 384       # 軽く・速く
    uv run python d405_depth_test.py --dmin 0.1 --dmax 1.0   # 深度カラーレンジを固定(m)
    uv run python d405_depth_test.py --model robbyant/lingbot-depth-postrain-dc-vitl14-v0.5
"""

import argparse
import os
import sys
import time
from pathlib import Path

# 深度エンコーダの nested-tensor 経路を避けるため xformers を無効化（import 前に設定）。
os.environ.setdefault("XFORMERS_DISABLED", "1")

# --- submodule を相対パスで解決（絶対パス依存を避ける） ---
REPO = Path(__file__).resolve().parent
SUBMODULE = REPO / "lingbot-depth"
if not SUBMODULE.exists():
    sys.exit(
        f"[d405-depth] submodule が見つかりません: {SUBMODULE}\n"
        "  git submodule update --init --recursive を実行してください。"
    )
sys.path.insert(0, str(SUBMODULE))

import cv2
import numpy as np
import pyrealsense2 as rs
import torch

from mdm.model.v2 import MDMModel

# 深度の色付けは depth_test 版と共通
from depth_test import colorize_depth


def _label(img, text):
    out = img.copy()
    cv2.rectangle(out, (0, 0), (out.shape[1], 28), (0, 0, 0), -1)
    cv2.putText(out, text, (7, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def build_parser():
    ap = argparse.ArgumentParser(prog="python d405_depth_test.py")
    ap.add_argument("--size", type=int, default=480, help="推論入力の高さ(px)。小さいほど速い")
    ap.add_argument("--width", type=int, default=1280, help="D405 ストリーム幅（最大 1280）")
    ap.add_argument("--height", type=int, default=720, help="D405 ストリーム高さ（最大 720）")
    ap.add_argument("--fps", type=int, default=30, help="D405 フレームレート")
    ap.add_argument("--dmin", type=float, default=None, help="深度カラーの下限(m)。未指定は毎フレーム自動")
    ap.add_argument("--dmax", type=float, default=None, help="深度カラーの上限(m)。未指定は毎フレーム自動")
    ap.add_argument("--model", default="robbyant/lingbot-depth-pretrain-vitl-14-v0.5", help="HF モデルID/ローカルckpt")
    return ap


def start_pipeline(args):
    """D405 のカラー+深度ストリームを開始し、(pipeline, align, K_norm, depth_scale) を返す。

    K_norm は color 内部パラメータを幅・高さで正規化した 3x3（リサイズ不変）。
    """
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, args.width, args.height, rs.format.bgr8, args.fps)
    config.enable_stream(rs.stream.depth, args.width, args.height, rs.format.z16, args.fps)
    try:
        profile = pipeline.start(config)
    except RuntimeError as e:
        sys.exit(
            f"[d405-depth] ストリーム開始に失敗しました: {e}\n"
            "  - D405 が USB3 で接続されているか確認してください。\n"
            "  - 解像度/FPS が D405 対応値か確認してください（最大 1280x720@30）。\n"
            "  - 他アプリ(RealSense Viewer 等)がカメラを掴んでいないか確認してください。"
        )
    ci = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
    depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
    K = np.array([[ci.fx, 0.0, ci.ppx], [0.0, ci.fy, ci.ppy], [0.0, 0.0, 1.0]], dtype=np.float32)
    K[0] /= ci.width   # fx, cx を幅で正規化
    K[1] /= ci.height  # fy, cy を高さで正規化
    align = rs.align(rs.stream.color)  # 深度を color 視点に位置合わせ
    return pipeline, align, K, depth_scale


def main():
    args = build_parser().parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[d405-depth] torch={torch.__version__} device={device}")
    if device.type == "cpu":
        print("[d405-depth] 警告: GPU 無し。ViT-L/14 の CPU 推論は非常に重い。")
    print(f"[d405-depth] モデル読み込み中: {args.model}（初回は HF から DL）")
    model = MDMModel.from_pretrained(args.model).to(device).eval()

    pipeline, align, K, depth_scale = start_pipeline(args)
    intr_t = torch.tensor(K, dtype=torch.float32, device=device)[None]
    print(f"[d405-depth] depth_scale={depth_scale:.2e} m/unit  推論高さ={args.size}px")

    out_dir = REPO / "outputs" / "d405_depth"
    win = "lingbot-depth D405 (q=quit, s=save)"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    print("[d405-depth] 起動しました。 ウィンドウで q=終了 / s=保存")
    saved = 0
    try:
        while True:
            try:
                frames = align.process(pipeline.wait_for_frames())
            except RuntimeError:
                print("[d405-depth] フレーム待ちタイムアウト。パイプラインを再起動します ...")
                pipeline.stop()
                pipeline, align, K, depth_scale = start_pipeline(args)
                intr_t = torch.tensor(K, dtype=torch.float32, device=device)[None]
                continue
            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()
            if not color_frame or not depth_frame:
                continue
            color_bgr = np.asanyarray(color_frame.get_data()).copy()               # BGR8
            depth_m = np.asanyarray(depth_frame.get_data()).astype(np.float32) * depth_scale  # メートル
            # 重い推論の前にフレームをプールへ返す（保持したままだと取得が止まる）
            del frames, color_frame, depth_frame

            # --- 推論解像度へリサイズ（正規化 K はリサイズ不変なので K は不変） ---
            H0, W0 = color_bgr.shape[:2]
            Hi = args.size
            Wi = int(round(W0 * Hi / H0))
            color_small = cv2.resize(color_bgr, (Wi, Hi), interpolation=cv2.INTER_AREA)
            depth_small = cv2.resize(depth_m, (Wi, Hi), interpolation=cv2.INTER_NEAREST)  # 深度は最近傍

            rgb = cv2.cvtColor(color_small, cv2.COLOR_BGR2RGB)
            image_t = torch.from_numpy(rgb.astype(np.float32) / 255.0).permute(2, 0, 1)[None].to(device)
            depth_t = torch.from_numpy(depth_small)[None].to(device)

            # --- lingbot-depth 推論 ---
            t0 = time.time()
            with torch.no_grad():
                out = model.infer(image_t, depth_in=depth_t, intrinsics=intr_t, enable_depth_mask=False)
            pred = out["depth"].squeeze().float().cpu().numpy()
            ms = (time.time() - t0) * 1000.0

            # --- 可視化（生と精緻化で同じカラーレンジ） ---
            valid_raw = depth_small[(np.isfinite(depth_small)) & (depth_small > 0)]
            valid_pred = pred[(np.isfinite(pred)) & (pred > 0)]
            if args.dmin is not None and args.dmax is not None:
                lo, hi = args.dmin, args.dmax
            else:
                pool = np.concatenate([valid_raw, valid_pred]) if valid_pred.size else valid_raw
                lo = float(np.percentile(pool, 2)) if pool.size else 0.0
                hi = float(np.percentile(pool, 98)) if pool.size else 1.0
            raw_c, _ = colorize_depth(depth_small, lo, hi)
            ref_c, _ = colorize_depth(pred, lo, hi)

            panel = np.concatenate([
                _label(color_small, f"color {Wi}x{Hi}"),
                _label(raw_c, "raw depth"),
                _label(ref_c, f"refined {ms:.0f}ms {lo:.2f}-{hi:.2f}m"),
            ], axis=1)
            cv2.imshow(win, panel)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("s"):
                out_dir.mkdir(parents=True, exist_ok=True)
                p = out_dir / f"panel_{saved:03d}.png"
                cv2.imwrite(str(p), panel)
                np.save(str(out_dir / f"refined_{saved:03d}.npy"), pred)
                print(f"[d405-depth] 保存: {p}")
                saved += 1
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()
        print("[d405-depth] 終了しました。")


if __name__ == "__main__":
    main()
