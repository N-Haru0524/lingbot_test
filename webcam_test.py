"""lingbot-vision を laptop のウェブカメラでライブ試験する（ウィンドウ表示）.

カメラ映像を 1 フレームずつ lingbot-vision バックボーンに通し、
patch token の上位3 PCA 成分を RGB 可視化して、入力と並べてウィンドウ表示する。

CPU 推論なので重い。滑らかさが欲しければ --size を小さく（例 256）する。

操作（ウィンドウにフォーカスした状態で）:
    q ... 終了
    s ... 現在のパネルを outputs/webcam/ に保存

使い方:
    uv run python webcam_test.py                 # small / size=384 / camera 0
    uv run python webcam_test.py --size 256      # 軽く・速く
    uv run python webcam_test.py --camera 1      # 別のカメラ番号
"""

import argparse
import sys
from pathlib import Path

# --- submodule を相対パスで解決（絶対パス依存を避ける） ---
REPO = Path(__file__).resolve().parent
SUBMODULE = REPO / "lingbot-vision"
if not SUBMODULE.exists():
    sys.exit(
        f"[webcam] submodule が見つかりません: {SUBMODULE}\n"
        "  git submodule update --init --recursive を実行してください。"
    )
sys.path.insert(0, str(SUBMODULE))

import cv2
import numpy as np
import torch
from PIL import Image

from lingbot_vision import extract_patch_tokens, load_pretrained_backbone
from lingbot_vision.pca_demo import _label, _pca_rgb
from lingbot_vision.preprocess import IMAGENET_MEAN, IMAGENET_STD, _snap


def preprocess_frame(frame_bgr, size, patch_size):
    """cv2 の BGR フレームを load_image と同じ正規化テンソルに変換する。

    戻り値: (img_norm, img_rgb(uint8), (H, W))
    """
    size = _snap(size, patch_size)
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb).resize((size, size), resample=Image.BILINEAR)  # mode="square"
    img_rgb = np.asarray(pil, dtype=np.uint8)
    img_t = torch.from_numpy(img_rgb.astype(np.float32) / 255.0)
    img_t = img_t.permute(2, 0, 1).unsqueeze(0)
    img_norm = (img_t - IMAGENET_MEAN) / IMAGENET_STD
    return img_norm, img_rgb, (size, size)


def build_parser():
    ap = argparse.ArgumentParser(prog="python webcam_test.py")
    ap.add_argument("--variant", default="small", choices=["small", "base", "large", "giant"])
    ap.add_argument("--size", type=int, default=384, help="入力の一辺(px)。小さいほど速い")
    ap.add_argument("--camera", type=int, default=0, help="cv2 のカメラ番号")
    return ap


def main():
    args = build_parser().parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    print(f"[webcam] torch={torch.__version__} device={device} dtype={dtype}")
    print(f"[webcam] {args.variant} バックボーンを読み込み中 ...")
    backbone, embed_dim = load_pretrained_backbone(variant=args.variant, device=device, dtype=dtype)
    print(f"[webcam] 読み込み完了 embed_dim={embed_dim} patch_size={backbone.patch_size}")

    # Windows では CAP_DSHOW が安定しやすい
    cap = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW)
    if not cap.isOpened():
        sys.exit(f"[webcam] カメラ {args.camera} を開けませんでした。--camera の番号を変えて試してください。")

    out_dir = REPO / "outputs" / "webcam"
    win = "lingbot-vision webcam (q=quit, s=save)"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    print("[webcam] 起動しました。 ウィンドウで q=終了 / s=保存")
    saved = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("[webcam] フレーム取得に失敗しました。")
                break

            img_norm, img_rgb, (H, W) = preprocess_frame(frame, args.size, backbone.patch_size)
            tokens, (h, w) = extract_patch_tokens(backbone, img_norm, device, dtype)
            pca = _pca_rgb(tokens, h, w)
            pca_up = cv2.resize(pca, (W, H), interpolation=cv2.INTER_NEAREST)
            panel = np.concatenate(
                [_label(img_rgb, f"input {H}x{W}"), _label(pca_up, f"patch PCA {h}x{w}")], axis=1
            )
            cv2.imshow(win, cv2.cvtColor(panel, cv2.COLOR_RGB2BGR))  # 表示は BGR

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("s"):
                out_dir.mkdir(parents=True, exist_ok=True)
                p = out_dir / f"panel_{saved:03d}.png"
                cv2.imwrite(str(p), cv2.cvtColor(panel, cv2.COLOR_RGB2BGR))
                print(f"[webcam] 保存: {p}")
                saved += 1
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("[webcam] 終了しました。")


if __name__ == "__main__":
    main()
