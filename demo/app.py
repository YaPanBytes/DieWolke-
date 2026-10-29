"""dieWolke demo: remove clouds from a Sentinel-2 patch using Sentinel-1 radar.

    pip install gradio
    python demo/app.py                      # then open http://127.0.0.1:7860
    python demo/app.py --runs runs --share  # --share gives a temporary public link

Inputs: a sample patch from demo/samples/*.npz (see scripts/export_demo_samples.py),
or your own Sentinel-1 + cloudy Sentinel-2 GeoTIFFs. Models: every folder under --runs that has
config.yaml and best.pt.
"""
import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import gradio as gr  # noqa: E402
import matplotlib  # noqa: E402

matplotlib.use("Agg")
from matplotlib import colormaps  # noqa: E402

from diewolke.bands import S2_MAX, normalize_s1, to_reflectance  # noqa: E402
from diewolke.engine import Pipeline  # noqa: E402
from diewolke.predict import reconstruct  # noqa: E402
from diewolke.utils import get_device  # noqa: E402

RED, GREEN, BLUE, NIR = 3, 2, 1, 7  # indices in B01,B02,B03,B04,B05,B06,B07,B08,B8A,B09,B10,B11,B12
DEVICE = get_device()
_PIPES = {}


def list_runs(runs_dir):
    runs = sorted(p.name for p in Path(runs_dir).glob("*") if (p / "config.yaml").exists() and (p / "best.pt").exists())
    return sorted(runs, key=lambda n: (n != "full", n))


def list_samples(sample_dir):
    return sorted(p.stem for p in Path(sample_dir).glob("*.npz"))


def get_pipe(runs_dir, name):
    if name not in _PIPES:
        _PIPES[name] = Pipeline.from_run(str(Path(runs_dir) / name), DEVICE)
    return _PIPES[name]


def rgb(refl):
    img = np.stack([refl[RED], refl[GREEN], refl[BLUE]], -1) / 0.3
    return (np.clip(img, 0, 1) ** (1 / 1.2) * 255).astype(np.uint8)


def ndvi(refl):
    return (refl[NIR] - refl[RED]) / (refl[NIR] + refl[RED] + 1e-6)


def colorize(a, cmap, lo, hi):
    return (colormaps[cmap](np.clip((a - lo) / (hi - lo + 1e-9), 0, 1))[..., :3] * 255).astype(np.uint8)


def psnr(a, b, m=None):
    d = (a - b) ** 2
    d = d[:, m] if m is not None else d
    return float("nan") if d.size == 0 else float(10 * np.log10(1.0 / max(d.mean(), 1e-12)))


def rmse(a, b, m=None):
    d = (a - b) ** 2
    d = d[m] if m is not None else d
    return float("nan") if d.size == 0 else float(np.sqrt(d.mean()))


def load_inputs(sample, s1_file, s2_file, sample_dir):
    if s1_file and s2_file:
        import rasterio
        with rasterio.open(s1_file) as f:
            s1 = normalize_s1(f.read().astype(np.float32))
        with rasterio.open(s2_file) as f:
            cloudy = np.clip(np.nan_to_num(f.read()), 0, S2_MAX).astype(np.uint16)
        return s1, cloudy, None, "uploaded files"
    if not sample:
        raise gr.Error("Pick a sample patch, or upload both a Sentinel-1 and a Sentinel-2 GeoTIFF.")
    z = np.load(Path(sample_dir) / f"{sample}.npz")
    clear = z["clear"].astype(np.float32) / S2_MAX if "clear" in z else None
    return z["s1"].astype(np.float32), z["cloudy"].astype(np.uint16), clear, sample


def run(sample, run_name, keep_pct, s1_file, s2_file, runs_dir, sample_dir):
    if not run_name:
        raise gr.Error(f"No trained runs found in {runs_dir}/ (need config.yaml and best.pt).")
    s1, cloudy_dn, clear, label = load_inputs(sample, s1_file, s2_file, sample_dir)
    pipe = get_pipe(runs_dir, run_name)
    recon, scale, mask = reconstruct(pipe, s1, cloudy_dn, DEVICE)
    cloudy = to_reflectance(cloudy_dn)
    m = mask > 0.5

    # keep only the most certain pixels (the "drop uncertain pixels" result from the paper)
    trusted = np.ones(m.shape, bool)
    unc_img = None
    if scale is not None:
        u = scale.mean(0)
        trusted = u <= np.percentile(u, keep_pct)
        unc_img = colorize(u, "magma", 0, np.percentile(u, 99))
    shown = rgb(recon).copy()
    shown[~trusted] = (shown[~trusted] * 0.25 + 40).astype(np.uint8)

    nd_c, nd_r = ndvi(cloudy), ndvi(recon)
    out = {
        "cloudy": rgb(cloudy),
        "mask": colorize(m.astype(np.float32), "gray", 0, 1),
        "recon": rgb(recon),
        "trusted": shown,
        "unc": unc_img if unc_img is not None else np.zeros_like(rgb(recon)),
        "ndvi_c": colorize(nd_c, "RdYlGn", -0.2, 0.9),
        "ndvi_r": colorize(nd_r, "RdYlGn", -0.2, 0.9),
        "clear": rgb(clear) if clear is not None else None,
    }
    lines = [f"**Input:** {label} &nbsp;|&nbsp; **Model:** `{run_name}` on {DEVICE} &nbsp;|&nbsp; "
             f"**Cloud/shadow pixels:** {100 * m.mean():.1f} %"]
    if clear is not None:
        nd_t = ndvi(clear)
        rows = [("Cloudy input", psnr(cloudy, clear), psnr(cloudy, clear, m), rmse(nd_c, nd_t, m)),
                ("dieWolke", psnr(recon, clear), psnr(recon, clear, m), rmse(nd_r, nd_t, m))]
        lines.append("\n| | PSNR all pixels (dB) | PSNR cloudy pixels (dB) | NDVI RMSE cloudy pixels |\n|---|---|---|---|")
        lines += [f"| {n} | {a:.2f} | {b:.2f} | {c:.3f} |" for n, a, b, c in rows]
        if scale is not None:
            sel = m & trusted
            lines.append(f"\nKeeping the {keep_pct:.0f} % most certain pixels: NDVI RMSE on cloudy pixels "
                         f"{rmse(nd_r, nd_t, sel):.3f} (all cloudy pixels: {rmse(nd_r, nd_t, m):.3f}).")
        lines.append("\n*The reference is a clear image of the same place on a different date, so land change adds some error.*")
    else:
        lines.append("\n*No clear reference for this input, so no metrics.*")
    return (out["cloudy"], out["mask"], out["recon"], out["trusted"], out["unc"],
            out["ndvi_c"], out["ndvi_r"], out["clear"], "\n".join(lines))


def build(runs_dir, sample_dir):
    runs, samples = list_runs(runs_dir), list_samples(sample_dir)
    with gr.Blocks(title="dieWolke") as demo:
        gr.Markdown("# dieWolke: cloud removal for all 13 Sentinel-2 bands\n"
                    "Sentinel-1 radar sees through clouds. A SAR-conditioned Brownian-bridge model fills in what the "
                    "clouds hide and predicts how uncertain each pixel is.")
        with gr.Row():
            sample = gr.Dropdown(samples, value=samples[0] if samples else None, label="Sample patch")
            model = gr.Dropdown(runs, value=runs[0] if runs else None, label="Trained model")
            keep = gr.Slider(10, 100, value=100, step=5, label="Keep the N % most certain pixels (dims the rest)")
        with gr.Accordion("Or use your own GeoTIFFs", open=False):
            with gr.Row():
                s1f = gr.File(label="Sentinel-1 (VV, VH) GeoTIFF", type="filepath")
                s2f = gr.File(label="Cloudy Sentinel-2 (13 bands) GeoTIFF", type="filepath")
        btn = gr.Button("Remove clouds", variant="primary")
        with gr.Row():
            i_cloudy = gr.Image(label="Cloudy input (RGB)")
            i_mask = gr.Image(label="Estimated cloud + shadow mask")
            i_recon = gr.Image(label="Reconstruction (RGB)")
            i_clear = gr.Image(label="Clear reference, other date (RGB)")
        with gr.Row():
            i_trust = gr.Image(label="Reconstruction, uncertain pixels dimmed")
            i_unc = gr.Image(label="Uncertainty (brighter = less certain)")
            i_nc = gr.Image(label="NDVI, cloudy input")
            i_nr = gr.Image(label="NDVI, reconstruction")
        info = gr.Markdown()
        btn.click(lambda *a: run(*a, runs_dir, sample_dir), [sample, model, keep, s1f, s2f],
                  [i_cloudy, i_mask, i_recon, i_trust, i_unc, i_nc, i_nr, i_clear, info])
    return demo


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default=str(ROOT / "runs"))
    ap.add_argument("--samples", default=str(ROOT / "demo" / "samples"))
    ap.add_argument("--share", action="store_true")
    ap.add_argument("--port", type=int, default=7860)
    a = ap.parse_args()
    build(a.runs, a.samples).launch(server_port=a.port, share=a.share)
