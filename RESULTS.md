# Architecture and Results Report

---

## Section 1: CNN Cats vs Dogs — Overfitting Demonstration and Fix

### Dataset

- **CIFAR-10** (Cat and Dog classes only)
- Train: 10,000 images | Validation: 2,000 images
- Image size: 32x32 RGB
- Balanced binary classes: Cat (0) and Dog (1)

---

### Baseline Model Architecture (No Regularization)

```
Input: 3 x 32 x 32

Conv Block 1:
  Conv2d(3, 64, 3, padding=1) -> ReLU
  Conv2d(64, 64, 3, padding=1) -> ReLU
  MaxPool2d(2, 2)                            -> 64 x 16 x 16

Conv Block 2:
  Conv2d(64, 128, 3, padding=1) -> ReLU
  Conv2d(128, 128, 3, padding=1) -> ReLU
  MaxPool2d(2, 2)                            -> 128 x 8 x 8

Conv Block 3:
  Conv2d(128, 256, 3, padding=1) -> ReLU
  Conv2d(256, 256, 3, padding=1) -> ReLU
  MaxPool2d(2, 2)                            -> 256 x 4 x 4

Classifier:
  Flatten                                    -> 4,096
  Linear(4096, 4096) -> ReLU
  Linear(4096, 1024) -> ReLU
  Linear(1024, 2)

Total Parameters: 22,124,098 (~22M)
Optimizer: Adam (lr=0.001, no weight decay)
Loss: CrossEntropyLoss
Epochs: 50
```

**Design intent:** Deliberately large model with no regularization to demonstrate overfitting. The 22M parameters vs 10K training images creates a massive capacity-to-data imbalance.

---

### Improved Model Architecture (With Regularization)

```
Input: 3 x 32 x 32

Conv Block 1:
  Conv2d(3, 32, 3, padding=1) -> BatchNorm2d(32) -> ReLU
  Conv2d(32, 32, 3, padding=1) -> BatchNorm2d(32) -> ReLU
  MaxPool2d(2, 2) -> Dropout2d(0.25)         -> 32 x 16 x 16

Conv Block 2:
  Conv2d(32, 64, 3, padding=1) -> BatchNorm2d(64) -> ReLU
  Conv2d(64, 64, 3, padding=1) -> BatchNorm2d(64) -> ReLU
  MaxPool2d(2, 2) -> Dropout2d(0.25)         -> 64 x 8 x 8

Conv Block 3:
  Conv2d(64, 128, 3, padding=1) -> BatchNorm2d(128) -> ReLU
  Conv2d(128, 128, 3, padding=1) -> BatchNorm2d(128) -> ReLU
  MaxPool2d(2, 2) -> Dropout2d(0.25)         -> 128 x 4 x 4

Global Average Pooling:
  AdaptiveAvgPool2d(1)                       -> 128 x 1 x 1

Classifier:
  Flatten                                    -> 128
  Linear(128, 256) -> BatchNorm1d(256) -> ReLU -> Dropout(0.5)
  Linear(256, 2)

Total Parameters: 321,954 (~322K) — 69x fewer than baseline
Optimizer: Adam (lr=0.001, weight_decay=1e-4)
Scheduler: ReduceLROnPlateau (patience=3, factor=0.5)
Loss: CrossEntropyLoss
Epochs: 50 (with early stopping, patience=7)
```

### Regularization Techniques Applied

| # | Technique | Where | Purpose |
|---|---|---|---|
| 1 | Data Augmentation | Input transforms | RandomHorizontalFlip, RandomRotation(15), ColorJitter, RandomCrop with padding — artificially increases dataset diversity so the model sees different versions of each image every epoch |
| 2 | Batch Normalization | After every Conv2d and first Linear | Normalizes activations within each mini-batch, stabilizes training, smooths loss landscape, and acts as a mild regularizer |
| 3 | Dropout2d (0.25) | After each conv block | Randomly zeroes entire feature maps during training, forcing the network to learn redundant representations instead of relying on specific neurons |
| 4 | Dropout (0.5) | Before final FC layer | Prevents the classifier head from memorizing specific activation patterns |
| 5 | Weight Decay (L2) | Adam optimizer (1e-4) | Penalizes large weight values, keeping parameters small and preventing the model from fitting noise |
| 6 | Early Stopping | Training loop (patience=7) | Monitors validation loss and stops training when it stops improving, restoring the best model weights |
| 7 | LR Scheduling | ReduceLROnPlateau | Reduces learning rate by half when validation loss plateaus, allowing fine-grained convergence without oscillation |

### Architectural Changes That Reduce Overfitting

| Change | Baseline | Improved | Effect |
|---|---|---|---|
| Filter counts | 64/128/256 | 32/64/128 | Reduces model capacity, fewer parameters to memorize |
| FC head | Flatten -> 4096 -> 1024 -> 2 | GAP -> 128 -> 256 -> 2 | Global Average Pooling eliminates the massive FC layer, removing ~21M parameters |
| Total parameters | 22,124,098 | 321,954 | 69x reduction in capacity |

---

### Section 1 Results

| Metric | Baseline | Improved |
|---|---|---|
| Total Parameters | 22,124,098 | 321,954 |
| Epochs Trained | 50 | 50 |
| Final Train Accuracy | 50.0% | 82.8% |
| Final Val Accuracy | 50.0% | 84.5% |
| Best Val Accuracy | 50.0% (epoch 1) | 85.8% (epoch 48) |
| Final Train Loss | 0.6932 | 0.3799 |
| Final Val Loss | 0.6931 | 0.3580 |
| Train-Val Accuracy Gap | 0.0% | -1.7% |

### Why the Baseline Failed

The baseline model with 22M parameters completely failed to learn — it never moved beyond 50% accuracy (random guessing for a binary task). This happened because:

1. **Exploding/vanishing gradients:** Without Batch Normalization, the deep network with large FC layers suffered from unstable gradients. The loss stayed flat at 0.6931 (log(2)), meaning the model output near-uniform probabilities for both classes.
2. **Too many parameters:** 22M parameters for 10K tiny 32x32 images created an extremely ill-conditioned optimization landscape. The model had so many degrees of freedom that gradient descent couldn't find a meaningful direction.
3. **No augmentation on small images:** CIFAR-10 images are only 32x32 pixels. Without augmentation, the model had very little variation to learn from, making the optimization even harder.

### Why the Improved Model Succeeded

1. **Batch Normalization** stabilized gradients throughout the network, enabling the model to actually learn meaningful features from the start.
2. **69x fewer parameters** (322K vs 22M) created a much better conditioned optimization problem — the model was forced to learn generalizable features rather than having the capacity to memorize.
3. **Global Average Pooling** replaced the massive 4096-neuron FC layer, dramatically reducing the parameter space and acting as a structural regularizer.
4. **Data Augmentation** effectively multiplied the training set — each image appeared differently every epoch through random flips, rotations, and color shifts.
5. **Dropout** prevented co-adaptation of neurons, ensuring the learned features were robust and redundant.
6. **Weight Decay** kept weights small, preventing any single neuron from dominating.
7. **The negative train-val gap (-1.7%)** shows the model generalizes slightly better than it fits training data — a sign of healthy regularization without underfitting.

---

## Section 2: Diabetic Retinopathy Prediction

### Dataset

- **UCI Diabetic Retinopathy Debrecen Dataset**
- 1,151 samples, 19 features extracted from retinal images
- Binary target: 0 = No diabetic retinopathy, 1 = Diabetic retinopathy
- Class distribution: 540 (No DR) / 611 (DR) — slightly imbalanced
- No missing values
- Train: 920 samples | Test: 231 samples (80/20 stratified split)

### Features

The features are extracted from retinal images using image processing techniques:
- **ma1–ma6:** Microaneurysm detection results at different confidence levels
- **exudate1–exudate8:** Exudate detection results at different confidence levels
- **quality:** Image quality assessment (binary)
- **pre_screening:** Pre-screening result (binary)
- **macula_opticdisc_distance:** Distance between macula and optic disc
- **opticdisc_diameter:** Diameter of the optic disc
- **am_fm_classification:** AM/FM-based classification result

### Feature Engineering

| Feature | Formula | Purpose |
|---|---|---|
| ma_total | Sum of ma1–ma6 | Aggregate microaneurysm score across confidence levels |
| exudate_total | Sum of exudate1–exudate8 | Aggregate exudate score across confidence levels |
| macula_disc_ratio | macula_opticdisc_distance / opticdisc_diameter | Normalized distance metric independent of image scale |

### Models Used

**Individual Models:**
1. Logistic Regression (max_iter=1000)
2. Random Forest (100 trees)
3. XGBoost (100 trees, lr=0.1, max_depth=5)
4. SVM (RBF kernel, probability=True)
5. KNN (k=5)
6. MLP Neural Network (hidden layers: 64, 32)
7. Gradient Boosting (100 trees, lr=0.1, max_depth=5)
8. AdaBoost (100 estimators, lr=0.5)
9. Bagging (50 estimators)

**Ensemble Models:**
10. Voting (Soft) — Averaged probabilities of Logistic Regression + Random Forest + XGBoost
11. Stacking — RF + XGBoost as base learners, Logistic Regression as meta-learner (5-fold CV)

### Section 2 Results

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| **Logistic Regression** | **0.7273** | **0.8261** | **0.6179** | **0.7070** | **0.8294** |
| MLP | 0.7446 | 0.7581 | 0.7642 | 0.7611 | 0.8118 |
| Voting (Soft) | 0.7359 | 0.7981 | 0.6748 | 0.7313 | 0.8187 |
| Stacking | 0.7013 | 0.7500 | 0.6585 | 0.7013 | 0.7803 |
| Random Forest | 0.7100 | 0.7593 | 0.6667 | 0.7100 | 0.7757 |
| SVM | 0.6883 | 0.7576 | 0.6098 | 0.6757 | 0.7735 |
| XGBoost | 0.7056 | 0.7570 | 0.6585 | 0.7043 | 0.7723 |
| Bagging | 0.6753 | 0.7308 | 0.6179 | 0.6696 | 0.7695 |
| Gradient Boosting | 0.6970 | 0.7345 | 0.6748 | 0.7034 | 0.7533 |
| AdaBoost | 0.6840 | 0.8378 | 0.5041 | 0.6294 | 0.7345 |
| KNN | 0.6147 | 0.6667 | 0.5528 | 0.6044 | 0.6605 |

**Best model by ROC-AUC: Logistic Regression (0.8294)**
**Best model by Accuracy: MLP (0.7446)**
**Best model by F1-Score: MLP (0.7611)**

### Why Logistic Regression Performed Best (by ROC-AUC)

1. **Dataset size and feature count:** With only 1,151 samples and 19 features (plus 3 engineered), the dataset is relatively small. Logistic Regression is a low-variance model that performs well on smaller datasets because it has fewer parameters to estimate and is less prone to overfitting.

2. **Linearly separable features:** The retinal image features (microaneurysm counts, exudate levels) have a roughly linear relationship with diabetic retinopathy severity. Logistic Regression captures these linear decision boundaries efficiently without the noise that more complex models introduce.

3. **High precision (0.826):** Logistic Regression achieved the second-highest precision, meaning when it predicted retinopathy, it was correct 82.6% of the time. This indicates clean, well-calibrated probability estimates — which directly translates to a high ROC-AUC.

4. **Regularization by simplicity:** Complex models like Random Forest, XGBoost, and Gradient Boosting have many hyperparameters (tree depth, number of trees, learning rate). With only ~920 training samples, these models are more likely to overfit to noise in the training data, even with cross-validation.

### Why MLP Performed Best (by Accuracy and F1)

1. **Balanced precision-recall:** MLP achieved the best recall (0.764) among all models while maintaining good precision (0.758). This balance gave it the highest F1-score (0.761), meaning it was the best at finding true positives without excessive false positives.

2. **Non-linear feature interactions:** The hidden layers (64 and 32 neurons) can capture non-linear interactions between features that Logistic Regression misses. For example, the combination of microaneurysm count and exudate level may have a multiplicative effect on retinopathy risk.

3. **Feature scaling advantage:** MLP benefits more from StandardScaler preprocessing than tree-based models, and this dataset was properly scaled before training.

### Why Ensemble Models Didn't Win

1. **Small dataset:** Ensembles shine when there's enough data for diverse base learners to capture different patterns. With 920 training samples, the base learners (LR, RF, XGBoost) learn very similar patterns, so combining them adds limited value.

2. **Voting (Soft) came close:** It achieved the third-best ROC-AUC (0.8187) by averaging the probabilities of LR + RF + XGBoost. This shows ensembling does help, but the gain over the best individual model (LR: 0.8294) was negative because the weaker models (RF, XGBoost) pulled down the average.

3. **Stacking overhead:** Stacking uses 5-fold CV to train meta-features, which further reduces the effective training data per fold. On a 920-sample dataset, this means each fold trains on only ~736 samples — too little for the stacking approach to outperform simpler models.

---

## Key Takeaways

### Section 1
- An over-parameterized CNN without regularization can completely fail to learn on small datasets (stuck at random guessing).
- Applying regularization techniques (BatchNorm, Dropout, data augmentation, weight decay, early stopping, LR scheduling) along with a right-sized architecture achieved 85.8% validation accuracy.
- The 69x parameter reduction (22M to 322K) was as important as the regularization techniques themselves.

### Section 2
- Simpler models (Logistic Regression, MLP) often outperform complex ensembles on small-to-medium datasets.
- Model complexity should match dataset size — the best model is not always the most sophisticated one.
- Feature engineering (aggregating microaneurysm and exudate scores, computing the macula-disc ratio) provided meaningful signal for all models.
- For medical diagnostic tasks, the choice between optimizing for ROC-AUC vs F1-score depends on whether you prioritize ranking (screening) or classification accuracy (diagnosis).
