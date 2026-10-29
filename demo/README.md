# dieWolke demo

A small web app: pick a Sentinel-2 patch with clouds, press **Remove clouds**, and see the cloudy input,
the estimated cloud/shadow mask, the reconstruction, the uncertainty map and NDVI before and after.

## 1. Get sample patches (once)

The app needs a few test patches in `demo/samples/*.npz`. They come from the SEN12MS-CR cache you built
on Kaggle. In a Kaggle notebook (code and cache attached, like `03_evaluate.ipynb`):

```
!python scripts/export_demo_samples.py --cache {CACHE} --split test --n 8 --out /kaggle/working/demo_samples
!cd /kaggle/working && zip -r demo_samples.zip demo_samples
```

Download `demo_samples.zip`, unzip its `.npz` files into `demo/samples/`. About 4 MB per patch.
You can also upload your own Sentinel-1 + cloudy Sentinel-2 GeoTIFFs in the app (13-band S2 in DN, 2-band S1 in dB).

## 2. Run

```
.venv\Scripts\activate
pip install -e ".[clouds]" gradio
python demo/app.py
```

Open http://127.0.0.1:7860. It lists every folder in `runs/` that has `config.yaml` and `best.pt`
(`full` first). Use `--share` for a temporary public link, `--runs <dir>` for another folder.

## Notes

- Runs on CPU if there is no GPU. A 256 x 256 patch is reconstructed in one network pass; the first click
  also loads the model and the cloud detector, so it is slower.
- The mask is estimated with s2cloudless plus a shadow heuristic (the `full` model uses `mask_source: data`).
- The "clear reference" is another date than the cloudy image, so the metrics include real land change.
  The 28.12 dB in the paper is the mean over 1,500 patches; single patches will vary a lot.
- Uncertainty slider: dims all but the N % most certain pixels. The paper reports that keeping the
  50 % most certain pixels lowers NDVI RMSE on cloudy pixels by 26 %.
