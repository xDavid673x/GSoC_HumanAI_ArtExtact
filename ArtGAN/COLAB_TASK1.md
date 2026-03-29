# Colab Setup for Task 1

## Recommended Upload Layout

Do **not** upload the current local `wikiart.zip` from this machine. It contains at least one corrupted image entry and will fail in Colab too.

Use this layout in Google Drive:

```text
MyDrive/
  ArtGAN/
    task1_crnn/
      __init__.py
      audit.py
      dataset.py
      evaluate.py
      metrics.py
      model.py
      train.py
    WikiArt Dataset/
      Artist/
        artist_class
        artist_train
        artist_val
      Genre/
        genre_class
        genre_train.csv
        genre_val.csv
      Style/
        style_class.txt
        style_train.csv
        style_val.csv
    TASK1_CONV_RECURRENT.md
    COLAB_TASK1.md
  wikiart.zip
```

## What You Can Skip

You do not need these files for Task 1 training in Colab:

- `.git/`
- `outputs/`
- the legacy `ArtGAN/` subfolder from the original repository
- `ICIP-16/`
- `data/`
- `models/`
- the current local `wikiart.zip`

If it is easier, you can still upload the whole repo folder. The important part is that `wikiart.zip` should be a **fresh** download.

## Fastest Colab Workflow

1. Put the minimal `ArtGAN/` code folder in Google Drive.
2. Put a clean `wikiart.zip` in Google Drive.
3. Open [`colab/Task1_CRNN_Colab.ipynb`](colab/Task1_CRNN_Colab.ipynb) in Colab.
4. Edit the two paths in the config cell:
   - `DRIVE_REPO_DIR`
   - `DRIVE_DATA_ZIP`
5. Run the notebook cells top to bottom.

## Result Files

After training and evaluation, the notebook copies results back into:

```text
MyDrive/ArtGAN_outputs/task1_crnn/
```

That folder will contain:

- `best.pt`
- `training_summary.json`
- `eval/metrics.json`
- `eval/outliers.json`
- `eval/style_outliers.csv`
- `eval/genre_outliers.csv`
- `eval/artist_outliers.csv`
