import numpy as np
import cv2 as cv
import argparse
from typing import Tuple
import matplotlib.pyplot as plt


def split_helper(img: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    h = img.shape[0] // 3
    B = img[:h]
    G = img[h : 2 * h]
    R = img[2 * h : 3 * h]
    return B, G, R


def crop_border(
    img: np.ndarray, frac: float = 0.08
) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
    h, w = img.shape[:2]
    t = int(h * frac)
    b = h - t
    l = int(w * frac)
    r = w - l
    return img[t:b, l:r], (t, b, l, r)


def orb_kp_desc(gray: np.ndarray, n_features: int = 3000):
    orb = cv.ORB_create(
        nfeatures=n_features, fastThreshold=10, scaleFactor=1.2, edgeThreshold=15
    )
    kps, desc = orb.detectAndCompute(gray, None)
    return kps, desc


def match_ratio_test(desc1, desc2, ratio=0.75):
    bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=False)
    knn = bf.knnMatch(desc1, desc2, k=2)
    good = []
    for m, n in knn:
        if m.distance < ratio * n.distance:
            good.append(m)
    return good


def align_channel(
    moving: np.ndarray, ref: np.ndarray, crop_frac: float = 0.08
) -> np.ndarray:
    mov_c, (mt, mb, ml, mr) = crop_border(moving, crop_frac)
    ref_c, (rt, rb, rl, rr) = crop_border(ref, crop_frac)

    k1, d1 = orb_kp_desc(mov_c)
    k2, d2 = orb_kp_desc(ref_c)

    if d1 is None or d2 is None or len(k1) < 10 or len(k2) < 10:
        return moving

    good = match_ratio_test(d1, d2, ratio=0.75)
    if len(good) < 10:
        return moving

    pts1 = np.float32([k1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    pts2 = np.float32([k2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    pts1[:, 0, 0] += ml
    pts1[:, 0, 1] += mt
    pts2[:, 0, 0] += rl
    pts2[:, 0, 1] += rt

    H, mask = cv.findHomography(
        pts1,
        pts2,
        method=cv.RANSAC,
        ransacReprojThreshold=3.0,
        maxIters=5000,
        confidence=0.995,
    )
    if H is None:
        return moving

    h, w = ref.shape
    aligned = cv.warpPerspective(
        moving, H, (w, h), flags=cv.INTER_LINEAR, borderMode=cv.BORDER_REPLICATE
    )
    return aligned


def align_images(r: np.ndarray, g: np.ndarray, b: np.ndarray) -> np.ndarray:
    r_a = align_channel(r, g)
    b_a = align_channel(b, g)
    rgb = np.dstack([r_a, g, b_a])
    return np.clip(rgb, 0, 255).astype(np.uint8)


def main():
    parser = argparse.ArgumentParser(
        description="Align RGB channels from a stacked glass-plate scan."
    )
    parser.add_argument("--input", required=True, help="Input 3-panel grayscale image")
    parser.add_argument(
        "--output",
        required=True,
        help="Output color image path (e.g., results/out.png)",
    )
    parser.add_argument(
        "--crop",
        type=float,
        default=0.08,
        help="Border crop fraction for keypoints (default 0.08)",
    )
    args = parser.parse_args()

    img = cv.imread(args.input, cv.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Could not read input: {args.input}")

    B, G, R = split_helper(img)  # original plates are typically B, G, R (top→bottom)
    aligned = align_images(R, G, B)  # stack as RGB

    # Save in BGR order for OpenCV
    out_bgr = cv.cvtColor(aligned, cv.COLOR_RGB2BGR)
    ok = cv.imwrite(args.output, out_bgr)
    if not ok:
        raise IOError(f"Failed to write output: {args.output}")

    plt.imshow(aligned)
    plt.title("Aligned RGB")
    plt.axis("off")
    plt.show()


if __name__ == "__main__":
    main()
