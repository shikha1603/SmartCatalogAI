# SmartCatalog AI — AI-Powered Product Catalog Automation System

SmartCatalog AI is an end-to-end, production-quality product catalog automation system designed for e-commerce marketplaces (similar to Meesho). It implements automated category routing, post-training logits calibration, class-specific threshold tuning, and margin-based ambiguity detection.

---

## 🚀 Key Features

1. **Two-Stage Deep Learning Inference**: Uses MobileNetV2 with transfer-learning heads to classify images into canonical categories: **Fashion**, **Electronics**, **Home**, **Beauty**, **Grocery**.
2. **Logits Temperature Calibration (Phase 1)**: Corrects model overconfidence/underconfidence using post-hoc temperature scaling ($T = 0.9411$), reducing Expected Calibration Error (ECE) from **`3.78%` to `2.99%`**.
3. **Precision-Optimized Per-Class Thresholds (Phase 2)**: Targets $\ge 90\%$ validation precision class-by-class.
   * Boosts catalog auto-approval rates from **`44.60%`** (global 85% threshold) to **`64.60%`** (optimized thresholds)—a **`20%` increase in automation throughput**!
4. **Margin-Based Ambiguity Routing (Phase 3)**: Implements class margin checks ($Top_1 - Top_2 \ge Margin$). If the model is split between two classes (e.g. 51% Fashion vs 49% Beauty), it flags it as **Ambiguous** and routes it to manual review.
5. **Human-in-the-Loop Moderation Interface**: A Streamlit dashboard supporting pending review queue resolution, audit logs, model health charts, and historical exports.
6. **Robust SQLite Audit Trail**: Decoupled schema tracking `predicted_category`, `corrected_category`, and `routing_status` with thread-safe database connection closures.
7. **Reset Confirmation Gate**: Checkbox-locked reset actions preventing accidental deletion of catalog logs.

---

## 📐 System Architecture

```
 Seller Image Upload
        │
        ▼
   [predict.py]
 ┌───────────────────────────────────────────────┐
 │ Preprocessing: Resize 224x224, Normalize      │
 │ Temperature Scaling: logits / T (T=0.9411)    │
 │ Inference: MobileNetV2 (PyTorch)               │
 └───────────────────────────────────────────────┘
        │
        ├─► Class Probabilities (Top-3)
        ▼
   [Confidence & Margin Routing Layer]
        │
        ├── (Confidence >= Class Threshold) AND (Margin >= Min Margin) ─► [✅ Auto-Approved] ──► Log to database.db
        │
        └── (Confidence < Class Threshold) OR (Margin < Min Margin) ────► [⚠️ Manual Review] ──► Log to database.db
                                                                                   │
                                                                                   ▼
                                                                       [Human Moderator Review]
                                                                       (Correction -> "Reviewed")
                                                                                   │
                                                                                   ▼
                                                                        [Update database.db]
```

---

## 📂 Directory Structure

```
SmartCatalogAI/
├── .streamlit/
│   └── config.toml             # Streamlit Light theme configuration
├── dataset/
│   ├── train/                  # Processed training images by folder
│   ├── val/                    # Processed validation images by folder
│   └── download_dataset.py     # Script to download Kagglehub or generate synthetic data
├── saved_model/
│   ├── model.pt                # PyTorch saved weights
│   ├── class_indices.json      # Class index mappings
│   ├── class_thresholds.json   # Calibrated per-class thresholds
│   └── calibration.json        # Calibrated temperature parameter T
├── outputs/
│   ├── thumbnails/             # Saved physical product thumbnails
│   ├── accuracy_curve.png      # Epoch training accuracy curve
│   ├── loss_curve.png          # Epoch training loss curve
│   ├── confusion_matrix.png    # Confusion matrix plots
│   └── classification_report.csv # Sklearn classification report
├── tests/
│   ├── test_predict.py         # Unit tests for prediction logic
│   └── test_database.py        # Unit tests for SQLite database transactions
├── train_model.py              # Two-stage MobileNetV2 training script
├── evaluate_model.py           # Evaluation script generating plots and reports
├── calibrate_model.py          # Logits temperature scaling script
├── compute_thresholds.py       # Per-class precision threshold optimizer
├── predict.py                  # Standalone inference wrapper
├── database.py                 # SQLite database transactions module
├── streamlit_app.py            # Streamlit dashboard app
├── Dockerfile                  # Production container configuration
├── requirements.txt            # Pinned requirements
├── config.py                   # Central configurations & parameters
└── README.md                   # Central documentation
```

---

## ⚙️ Setup & Running Instructions

### 1. Installation
Clone the repository and install the dependencies:
```bash
pip install -r requirements.txt
```

### 2. Dataset Setup
You can either download a real e-commerce dataset (recommended) or generate a mock synthetic dataset for CI/CD checks.

* **Option A: Real E-Commerce Dataset (Amazon 18k)**
  Automatically downloads and formats the public `fatihkgg/ecommerce-product-images-18k` dataset from Kaggle using public endpoints (no API credentials needed!):
  ```bash
  python -m dataset.download_dataset --kagglehub
  ```
* **Option B: Synthetic Mock Dataset**
  Generates randomized 224x224 PIL canvas drawings of geometric shapes (useful for offline syntax checks):
  ```bash
  python -m dataset.download_dataset --synthetic
  ```

### 3. Model Training
Train the two-stage transfer learning model:
```bash
python train_model.py
```
*Note: In Stage 1, the base MobileNetV2 layers are frozen to train the custom dense classification head. In Stage 2, the last blocks of the base model are unfrozen and fine-tuned at a low learning rate ($10^{-5}$).*

### 4. Calibration & Threshold Optimization
Run evaluation, calibrate logits, and calculate optimized per-class thresholds:
```bash
# 1. Run validation evaluation
python evaluate_model.py

# 2. Compute logits temperature scaling T
python calibrate_model.py

# 3. Optimize class thresholds targeting 90% precision
python compute_thresholds.py
```

### 5. Running Tests
Run the unit tests:
```bash
python -m unittest discover -s tests
```

### 6. Streamlit Dashboard
Launch the web interface for catalog moderators:
```bash
streamlit run streamlit_app.py
```

---

## 📈 Model Evaluation Results (Real Amazon Dataset)

### Dataset Count Summary
* **Total Training Images**: 2,000 (400 per class)
* **Total Validation Images**: 500 (100 per class)
* **Classes**: Fashion, Electronics, Home, Beauty, Grocery

### Training Convergence
* **Stage 1 (Head Warm-up)**: Best Validation Accuracy of **`71.80%`**
* **Stage 2 (Fine-tuning)**: Best Validation Accuracy of **`75.20%`**
* **Fashion Precision**: Reached **`92.00% precision` and `90.00% recall`** (Highly robust dress/shoe classifications).

### Class-Specific Thresholds ($\ge 90\%$ Precision target)
* **Beauty**: `67.9%`
* **Electronics**: `82.0%`
* **Fashion**: `50.0%` (clamped)
* **Grocery**: `67.5%`
* **Home**: `71.0%`

---

## 💼 Interview Talking Points

1. **Automation vs. Error Rate Trade-off**:
   * The confidence threshold serves as a lever for the business. Raising the threshold reduces catalog errors but increases the human review queue size. Lowering it reduces human moderation costs but risks misclassified products going live. This trade-off is optimized dynamically based on category risk.
2. **Logits Temperature Calibration**:
   * Neural networks are often overconfident. Post-hoc temperature scaling ($logits / T$) adjusts output confidence to match actual validation accuracy without affecting classifier accuracy, ensuring score thresholds correspond to true probabilities.
3. **Probability Margin Routing**:
   * Ambiguity checks prevent errors on images containing elements of multiple classes (e.g. beauty products in a fashion setting). Requiring a margin between Top-1 and Top-2 probabilities prevents routing highly contested edge cases into auto-approval.
4. **Human-in-the-Loop Active Learning**:
   * Separate fields for `predicted_category` and `corrected_category` preserve the audit history of model errors. This structured data creates a high-quality feedback dataset of "hard examples" that can be oversampled in subsequent fine-tuning runs.
