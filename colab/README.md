# Seq2Point challenger on Google Colab

The CNN challenger needs a GPU, so it runs on Google Colab. Everything else runs locally.

**Data sent to Colab: development houses only** (16 files, ~89 MB), created by
`python -m src.export_colab` into `colab/data/` (not in git — rebuilt from `DATA/`).
The locked test houses (2, 15, 16) are never exported for model selection.

## Steps

1. In Google Drive, create the folder `MyDrive/refit_seq2point/` and upload the 16 files
   `colab/data/house*.parquet` into it.
2. Open `colab/seq2point_refit.ipynb` in Colab (File → Upload notebook).
3. Runtime → Change runtime type → **T4 GPU** → Save.
4. Runtime → **Run all**. Allow Drive access when asked.
   Finished folds are saved to `MyDrive/refit_seq2point/oof/`; if Colab disconnects, Run all
   again and finished folds are skipped.
5. Download `MyDrive/refit_seq2point/oof/seq2point.parquet` (and `seq2point_log.json`) and put
   them in `cache/oof/` in the project folder.
6. Locally: `python -m src.select_model` scores it with the same metric code as LightGBM and
   applies the promotion rule.

`seq2point_refit.py` is the same code as the notebook (`python colab/make_notebook.py`
regenerates the notebook from it). A CPU smoke test:
`S2P_DATA_DIR=colab/data S2P_SMOKE=1 python colab/seq2point_refit.py`.
