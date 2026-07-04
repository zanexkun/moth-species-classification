# Moth Species Classification

Transfer Learning with ResNet18 to classify 50 moth species. Built independently as a second Transfer Learning project to solidify the pattern on a different domain.

**Test Accuracy: 98.4%** across 50 classes.

---

## Dataset

[Moths Image Classification](https://www.kaggle.com/datasets/gpiosenka/moths-image-datasetclassification) — 50 moth species, pre-split into train/valid/test folders.

---

## Results

| Stage | Accuracy |
|-------|----------|
| Feature extraction (fc only) | 95.6% |
| Fine-tuning (layer4 + fc) | **98.4%** |

46 out of 50 classes achieved perfect F1 score of 1.00. The 4 classes with errors had at most one misclassification each out of 5 test images.

---

## Approach

**Stage 1 — Feature extraction:** froze all ResNet18 weights pretrained on ImageNet, replaced the final layer with `Linear(512, 50)` for the 50 moth classes. Trained for 89 epochs with Adam (lr=0.0001). Got to 95.6%.

**Stage 2 — Fine-tuning:** loaded the best feature extraction checkpoint, unfroze `layer4` (the last conv block), and continued training both `layer4` and `fc` at the same learning rate. The model overfitted after epoch 6 so the best checkpoint was at that point. Got to 98.4%.

Moths can appear at any orientation in real photos (wall, ceiling, flying) so both horizontal and vertical flips were used for augmentation, unlike scene classification where vertical flip would look unnatural.

---

## Why ResNet transfers well to moths

ResNet was pretrained on ImageNet which includes many insects and animals. Moth classification is closer to ImageNet's domain than scene classification is, which explains why feature extraction alone already reached 95.6% — the pretrained features already knew what wing textures and insect anatomy look like.

---

## Requirements

```
torch
torchvision
scikit-learn
```

## Usage

Download the dataset from Kaggle, update the paths at the top of the script, then run:

```bash
python moth_classification.py
```
