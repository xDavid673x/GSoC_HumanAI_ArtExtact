# Task 1: Convolutional-Recurrent Architectures for WikiArt

## Deliverables

- Trained model outputs and checkpoints: [Google Drive folder](https://drive.google.com/drive/folders/1m3cXlrUNEA3CDECekMavemhZnUDLqOFU?usp=sharing)

## Chosen Approach

The most appropriate design for this task is a **multi-task convolutional-recurrent network (CRNN)** rather than three independent classifiers.

Why:

- **Style** and **genre** depend heavily on global composition, color palette, repeated brushstroke patterns, and subject layout.
- **Artist** is the most specific target and benefits from the same shared visual cues plus the higher-level signals already useful for style and genre.
- The WikiArt labels are only partially aligned across tasks, so a shared encoder with **masked multi-task losses** uses more of the dataset than separate models while still respecting missing labels.

The implementation in [`task1_crnn/model.py`](task1_crnn/model.py) uses:

- A convolutional encoder (`ResNet-18` trunk without the classification head) for local texture, color, edge, and brushstroke evidence.
- Adaptive pooling to a fixed spatial grid, then flattening that grid into a token sequence.
- A bidirectional GRU over the token sequence to model long-range spatial dependencies across the canvas.
- Attention pooling over recurrent outputs to build a single painting embedding.
- Separate heads for `style`, `genre`, and `artist`.
- A **general-to-specific** artist head that consumes the shared embedding plus the model's own style/genre probabilities, which is appropriate because artist identity is usually narrower than style or subject matter.

## Data Strategy

The dataset release referenced in the assignment provides separate split files for style, genre, and artist classification. Those splits are not globally aligned across tasks:

- Style: 57,025 train / 24,421 val
- Genre: 45,503 train / 19,492 val
- Artist: 13,348 train / 5,708 val

When the three tasks are merged, there are **566 paintings that are train for one task and val for another**, so a naive multi-task loader would leak validation images into training. The dataset loader in [`task1_crnn/dataset.py`](task1_crnn/dataset.py) fixes this by creating a **global leak-free split**:

- Global train: 56,774 paintings
- Global val: 24,672 paintings

Label coverage under that global split:

- Train style labels: 56,774
- Train genre labels: 45,194
- Train artist labels: 13,201
- Val style labels: 24,672
- Val genre labels: 19,800
- Val artist labels: 5,851

The loader can read directly from [`wikiart.zip`](../wikiart.zip), so the 25 GB archive does not need to be extracted first.

## Training Strategy

The training script is [`task1_crnn/train.py`](task1_crnn/train.py).

Recommended strategy:

1. Train one shared CRNN with masked cross-entropy losses for all available labels in each batch.
2. Keep style and genre loss weights at `1.0`, and slightly up-weight artist to `1.2` because it is the most specific task and has the smallest labeled subset.
3. Use inverse-square-root class weighting to reduce dominance from large classes without making the rare classes unstable.
4. Validate with macro-sensitive metrics, not just raw accuracy, because WikiArt is imbalanced.
5. Select the best checkpoint using the mean macro-F1 across style, genre, and artist.

## Evaluation Metrics

The evaluation code is in [`task1_crnn/evaluate.py`](task1_crnn/evaluate.py) and [`task1_crnn/metrics.py`](task1_crnn/metrics.py).

I would report these metrics for each task:

- **Accuracy**: useful, but not enough on its own because of class imbalance.
- **Macro-F1**: the main metric for fair class-wise performance.
- **Weighted-F1**: shows overall performance while accounting for support.
- **Balanced accuracy**: helpful when some classes are much rarer than others.
- **Top-k accuracy**: especially useful for artist classification, where stylistically similar painters can be confused.
- **Expected calibration error (ECE)**: indicates whether model confidence is trustworthy enough for outlier detection.
- **Confusion matrix**: essential for seeing which styles, genres, or artists collapse into each other.

For model selection, **macro-F1** should be primary. For artist classification specifically, I would also monitor **top-3 accuracy**, because the nearest artistic alternatives are often semantically meaningful.

## Outlier Detection

The evaluation pipeline finds outliers in two ways:

1. **Low true-label confidence**: paintings where the assigned class receives low probability.
2. **High embedding distance from the class centroid**: paintings whose embedding is far from the visual center of their assigned class.

The outlier score combines both:

- `outlier_score = -log(p(true_class)) + standardized_centroid_distance`

This is implemented in [`task1_crnn/evaluate.py`](task1_crnn/evaluate.py), which writes:

- `style_outliers.csv`
- `genre_outliers.csv`
- `artist_outliers.csv`

These are the right paintings to inspect manually, because they include:

- likely label noise,
- transitional works,
- atypical subjects for a known artist,
- paintings with restoration, cropping, or digitization artifacts,
- pieces that genuinely sit between movements or genres.

For a faster first pass before training finishes, [`task1_crnn/metadata_outliers.py`](task1_crnn/metadata_outliers.py) also mines **metadata-only outlier candidates** from rare artist/style, artist/genre, and genre/style combinations in the official split files.

## Metadata-Based Outlier Candidates Already Visible

Even before training, the label metadata already exposes some plausible artist-side outliers because their assigned style is very rare for that artist in the training split:

- `Pointillism/pablo-picasso_woman-with-spanish-dress-1917.jpg`
- `Realism/childe-hassam_at-the-florist.jpg`
- `Realism/claude-monet_a-corner-of-the-studio(1).jpg`
- `Naive_Art_Primitivism/salvador-dali_exquisite-cadaver.jpg`
- `Art_Nouveau_Modern/martiros-saryan_costume-design-for-the-opera-by-rimsky-korsakov-s-golden-cockerel-1931.jpg`

These are not proof of mislabeling, but they are exactly the kind of paintings the outlier detector should rank highly after training.

## Files Added

- [`task1_crnn/dataset.py`](task1_crnn/dataset.py): leak-free multi-task dataset with zip-backed loading
- [`task1_crnn/model.py`](task1_crnn/model.py): CRNN architecture
- [`task1_crnn/train.py`](task1_crnn/train.py): training entry point
- [`task1_crnn/evaluate.py`](task1_crnn/evaluate.py): evaluation and outlier mining
- [`task1_crnn/metrics.py`](task1_crnn/metrics.py): metrics and confusion export
- [`task1_crnn/audit.py`](task1_crnn/audit.py): split audit helper
- [`task1_crnn/metadata_outliers.py`](task1_crnn/metadata_outliers.py): metadata-only rare-combination outlier mining

## Example Commands

Audit the merged split:

```bash
python3 -m task1_crnn.audit --dataset-dir "WikiArt Dataset" --output-json outputs/task1_crnn/audit.json
```

Train:

```bash
python3 -m task1_crnn.train \
  --dataset-dir "WikiArt Dataset" \
  --archive-path "../wikiart.zip" \
  --output-dir outputs/task1_crnn \
  --epochs 12 \
  --batch-size 32 \
  --num-workers 4
```

Evaluate and extract outliers:

```bash
python3 -m task1_crnn.evaluate \
  --checkpoint outputs/task1_crnn/best.pt \
  --dataset-dir "WikiArt Dataset" \
  --archive-path "../wikiart.zip" \
  --output-dir outputs/task1_crnn/eval \
  --batch-size 32 \
  --num-workers 4 \
  --top-outliers 25
```

Mine metadata-only outlier candidates:

```bash
python3 -m task1_crnn.metadata_outliers \
  --dataset-dir "WikiArt Dataset" \
  --output-dir outputs/task1_crnn/metadata_outliers \
  --top-n 25
```
