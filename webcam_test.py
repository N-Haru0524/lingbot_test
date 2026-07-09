"""lingbot-vision を laptop のウェブカメラでライブ試験する（ウィンドウ表示）.

カメラ映像を 1 フレームずつ lingbot-vision バックボーンに通し、
patch token の上位3 PCA 成分を RGB 可視化して、入力と並べてウィンドウ表示する。
--edges（既定 ON）で、patch token の特徴不連続から作った輪郭マップ
（teaser 3列目の boundary 風）も並べる。

CPU 推論なので重い。滑らかさが欲しければ --size を小さく（例 256）する。

操作（ウィンドウにフォーカスした状態で）:
    q ... 終了
    s ... 現在のパネルを outputs/webcam/ に保存

使い方:
    uv run python webcam_test.py                 # small / size=384 / D405 を自動検出
    uv run python webcam_test.py --size 256      # 軽く・速く
    uv run python webcam_test.py --no-edges      # 輪郭パネルを出さない
    uv run python webcam_test.py --camera 1      # D405 以外を使う場合はカメラ番号を指定
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
from lingbot_vision.pca_demo import _label
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


class StablePCA:
    """patch token の上位3 PCA 成分を RGB 画像 (uint8) にする（フレーム間で安定）。

    pca_demo._pca_rgb は CPU のフル SVD なので大きいトークン数（--size 1024 等）で
    毎フレーム数秒かかる上、毎フレーム独立に計算するため主成分の符号・向きが
    入れ替わって色がチカチカする。ここでは tokens のデバイス上で低ランク PCA を
    計算し、基底と正規化範囲を EMA で前フレームに追従させて色を安定化する。
    """

    def __init__(self, momentum=0.9):
        self.momentum = momentum
        self.mean = None
        self.basis = None  # (embed_dim, 3)
        self.lo = None
        self.hi = None

    def _update_basis(self, mean, v):
        m = self.momentum
        if self.basis is None:
            self.mean, self.basis = mean, v
            return
        self.mean = m * self.mean + (1 - m) * mean
        # 前フレームの基底に符号を合わせてから混ぜ、直交性を QR で回復する
        v = v * torch.sign((v * self.basis).sum(dim=0, keepdim=True))
        q, _ = torch.linalg.qr(m * self.basis + (1 - m) * v)
        q = q * torch.sign((q * self.basis).sum(dim=0, keepdim=True))
        self.basis = q

    def __call__(self, tokens, h, w):
        x = tokens[0].detach().float()
        _, _, v = torch.pca_lowrank(x, q=3)
        self._update_basis(x.mean(dim=0, keepdim=True), v)
        rgb = ((x - self.mean) @ self.basis).reshape(h, w, 3).cpu().numpy()
        lo = np.percentile(rgb, 1, axis=(0, 1), keepdims=True)
        hi = np.percentile(rgb, 99, axis=(0, 1), keepdims=True)
        if self.lo is None:
            self.lo, self.hi = lo, hi
        else:
            self.lo = self.momentum * self.lo + (1 - self.momentum) * lo
            self.hi = self.momentum * self.hi + (1 - self.momentum) * hi
        rgb = np.clip((rgb - self.lo) / np.maximum(self.hi - self.lo, 1e-6), 0, 1)
        return (rgb * 255).astype(np.uint8)


def feature_edges(tokens, h, w):
    """patch token の隣接不連続から輪郭マップを作る（学習不要, グレースケール uint8）。

    LingBot-Vision は masked boundary modeling で学習されているため、隣接する
    patch token 間の cosine 距離が物体の輪郭で大きくなる（teaser 3列目の
    boundary token に相当する挙動）。各パッチと右・下の近傍との cosine 距離を
    足し合わせて境界強度とし、1–99 パーセンタイルで正規化して (h, w) の uint8
    で返す。境界ヘッドを使わないので公開バックボーンだけで動く。
    """
    x = tokens[0].detach().float()
    x = x / x.norm(dim=1, keepdim=True).clamp_min(1e-6)  # cosine 用に L2 正規化
    g = x.reshape(h, w, -1)
    dx = 1.0 - (g[:, 1:] * g[:, :-1]).sum(-1)  # 横方向の非類似度 (h, w-1)
    dy = 1.0 - (g[1:, :] * g[:-1, :]).sum(-1)  # 縦方向の非類似度 (h-1, w)
    edge = torch.zeros(h, w, dtype=x.dtype, device=x.device)
    edge[:, 1:] += dx
    edge[:, :-1] += dx
    edge[1:, :] += dy
    edge[:-1, :] += dy
    edge = edge.cpu().numpy()
    lo, hi = np.percentile(edge, 1), np.percentile(edge, 99)
    return (np.clip((edge - lo) / max(hi - lo, 1e-6), 0, 1) * 255).astype(np.uint8)


def edges_to_rgb(edge_u8, W, H, line=(222, 52, 140)):
    """輪郭グレースケールを teaser 3列目風（白地にマゼンタ線）の RGB にして拡大する。"""
    e = cv2.resize(edge_u8, (W, H), interpolation=cv2.INTER_LINEAR).astype(np.float32)[..., None] / 255.0
    out = 255.0 * (1.0 - e) + np.asarray(line, dtype=np.float32) * e
    return out.clip(0, 255).astype(np.uint8)


def find_d405_camera():
    """/dev/v4l/by-id から D405(RealSense) のカラーノードを探して開く。

    D405 は深度(Z16)・IR(GREY/UYVY)・カラー(YUYV) の複数ノードを露出するので、
    YUYV をネゴシエートできたノードをカラーと判定する。
    """
    byid = Path("/dev/v4l/by-id")
    if not byid.exists():
        return None
    yuyv = cv2.VideoWriter_fourcc(*"YUYV")
    for dev in sorted(byid.glob("*RealSense*video-index*")):
        # OpenCV はパス指定で開けないビルドがあるので /dev/videoN に解決して番号で開く
        index = int(dev.resolve().name.removeprefix("video"))
        cap = cv2.VideoCapture(index, cv2.CAP_V4L2)
        if not cap.isOpened():
            continue
        cap.set(cv2.CAP_PROP_FOURCC, yuyv)
        if int(cap.get(cv2.CAP_PROP_FOURCC)) == yuyv and cap.read()[0]:
            print(f"[webcam] D405 を検出: /dev/video{index} ({dev.name})")
            return cap
        cap.release()
    return None


def build_parser():
    ap = argparse.ArgumentParser(prog="python webcam_test.py")
    ap.add_argument("--variant", default="small", choices=["small", "base", "large", "giant"])
    ap.add_argument("--size", type=int, default=384, help="入力の一辺(px)。小さいほど速い")
    ap.add_argument(
        "--camera", type=int, default=None,
        help="cv2 のカメラ番号（未指定なら D405 を自動検出）",
    )
    ap.add_argument("--no-edges", dest="edges", action="store_false", help="特徴由来の輪郭パネルを表示しない")
    return ap


def main():
    args = build_parser().parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    print(f"[webcam] torch={torch.__version__} device={device} dtype={dtype}")
    print(f"[webcam] {args.variant} バックボーンを読み込み中 ...")
    backbone, embed_dim = load_pretrained_backbone(variant=args.variant, device=device, dtype=dtype)
    print(f"[webcam] 読み込み完了 embed_dim={embed_dim} patch_size={backbone.patch_size}")

    if args.camera is None:
        cap = find_d405_camera()
        if cap is None:
            sys.exit("[webcam] D405 が見つかりませんでした。--camera でカメラ番号を指定してください。")
    else:
        # Windows では CAP_DSHOW、Linux では CAP_V4L2 が安定しやすい
        backend = cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_V4L2
        cap = cv2.VideoCapture(args.camera, backend)
        if not cap.isOpened():
            sys.exit(f"[webcam] カメラ {args.camera} を開けませんでした。--camera の番号を変えて試してください。")

    pca_rgb = StablePCA()
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
            pca = pca_rgb(tokens, h, w)
            pca_up = cv2.resize(pca, (W, H), interpolation=cv2.INTER_NEAREST)
            panels = [_label(img_rgb, f"input {H}x{W}"), _label(pca_up, f"patch PCA {h}x{w}")]
            if args.edges:
                edges = edges_to_rgb(feature_edges(tokens, h, w), W, H)
                panels.append(_label(edges, f"feature edges {h}x{w}"))
            panel = np.concatenate(panels, axis=1)
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
