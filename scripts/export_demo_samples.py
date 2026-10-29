"""Export a few test patches from the SEN12MS-CR cache as small .npz files for the demo.

Run where the cache is (e.g. Kaggle), then download demo/samples/ to your machine:
    python scripts/export_demo_samples.py --cache /kaggle/input/diewolke-cache --split test --n 8 --out demo/samples

Picks patches spread from lightly to heavily clouded. Each .npz holds s1 (2,H,W normalised float16),
cloudy and clear (13,H,W uint16 DN) and the estimated cloud fraction. About 4 MB per patch.
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from diewolke.data.clouds import CLOUD_PROB  # noqa: E402
from diewolke.data.sen12mscr import Cache  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--out", default="demo/samples")
    ap.add_argument("--max_target_cloud", type=float, default=0.05, help="skip pairs whose clear image is cloudier than this")
    a = ap.parse_args()
    c = Cache.load(a.cache, a.split)
    n = len(c["clear"])
    frac = np.array([(np.asarray(c["prob"][i]) > CLOUD_PROB).mean() for i in range(n)])
    tgt = np.array([(np.asarray(c["prob_clear"][i]) > CLOUD_PROB).mean() for i in range(n)])
    ok = np.where((tgt <= a.max_target_cloud) & (frac > 0.02))[0]
    ok = ok[np.argsort(frac[ok])]
    picks = ok[np.linspace(0, len(ok) - 1, min(a.n, len(ok))).astype(int)]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for k, i in enumerate(picks):
        name = f"patch{k:02d}_cloud{int(round(frac[i] * 100)):02d}pct"
        np.savez_compressed(out / f"{name}.npz",
                            s1=np.asarray(c["s1"][i], dtype=np.float32).astype(np.float16),
                            cloudy=np.clip(np.asarray(c["cloudy"][i]), 0, 65535).astype(np.uint16),
                            clear=np.clip(np.asarray(c["clear"][i]), 0, 65535).astype(np.uint16),
                            cloud_frac=np.float32(frac[i]))
        print("wrote", name)


if __name__ == "__main__":
    main()
