# ClothesMinded

A CNN-based image classifier trained on the FashionMNIST dataset with 10 clothing categories. This project explores model robustness under image distortions, and benchmarks three strategies for maintaining accuracy under rotated inputs:

1. **Rotation-Invariant Classifier** — a model trained with aggressive rotation augmentation 
   to learn features that are inherently robust to orientation changes.

2. **Rotation Correction** — a preprocessing approach that detects and corrects image rotation 
   before classification, normalizing inputs to a canonical orientation.

3. **Test Time Augmentation (TTA)** — an inference strategy that generates multiple rotated 
   versions of each test image, aggregates the model's predictions, and returns the consensus 
   classification.

## Categories
T-shirt/top · Trouser · Pullover · Dress · Coat · Sandal · Shirt · Sneaker · Bag · Ankle boot