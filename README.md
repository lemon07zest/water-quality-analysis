# 💧 Water Quality Analysis — ML Pipeline

> A machine learning pipeline to classify water potability using chemical composition data.  
> Built with Python, Scikit-Learn, Pandas, and Matplotlib.

---

## 📊 Results

| Model | Accuracy | AUC-ROC | Cross-Val |
|---|---|---|---|
| Logistic Regression | 85.67% | 0.9279 | 0.8496 |
| Random Forest | 89.18% | 0.9535 | 0.8794 |
| Gradient Boosting | 89.94% | 0.9612 | 0.8782 |
| **SVM (Best)** | **90.09%** | **0.9653** | **0.8866** |

---

## 🔬 Project Overview

Access to safe drinking water is a fundamental human right. This project builds a binary classification pipeline to determine whether water is **potable (safe to drink)** based on 9 chemical parameters.

**Key Findings:**
- Total Dissolved Solids (TDS) is the most predictive feature
- pH levels outside 6.5–8.5 strongly indicate non-potable water
- Elevated Chloramine and Sulfate concentrations correlate with unsafe water

---

## 📁 Project Structure

```
water-quality-analysis/
│
├── water_quality_analysis.py   # Main ML pipeline
├── requirements.txt            # Dependencies
├── README.md                   # This file
│
└── outputs/
    ├── water_potability_dataset.csv     # Generated dataset
    ├── 01_feature_distributions.png    # EDA: feature histograms
    ├── 02_correlation_heatmap.png      # EDA: correlation matrix
    ├── 03_class_distribution.png       # EDA: class balance
    ├── 04_model_comparison.png         # Model accuracy vs AUC
    ├── 05_confusion_matrix.png         # Best model confusion matrix
    ├── 06_roc_curves.png               # ROC curves all models
    ├── 07_feature_importance.png       # Random Forest importances
    └── summary_report.txt              # Full results summary
```

---

## 🧪 Features Used

| Feature | Description | WHO Limit |
|---|---|---|
| pH | Acidity/alkalinity | 6.5 – 8.5 |
| Hardness | Calcium & Magnesium (mg/L) | < 300 |
| Solids | Total Dissolved Solids (ppm) | < 500 |
| Chloramines | Disinfectant (ppm) | < 4 |
| Sulfate | Sulfate concentration (mg/L) | < 250 |
| Conductivity | Electrical conductivity (μS/cm) | < 400 |
| Organic Carbon | Total Organic Carbon (ppm) | < 2 |
| Trihalomethanes | Disinfection byproducts (μg/L) | < 80 |
| Turbidity | Water clarity (NTU) | < 5 |

---

## ⚙️ Pipeline Steps

```
1. Data Generation     → 3,276 samples with realistic chemical distributions
2. EDA                 → Distribution plots, correlation heatmap, class balance
3. Preprocessing       → Median imputation for missing values, StandardScaler
4. Model Training      → 4 classifiers with 5-fold cross-validation
5. Evaluation          → Accuracy, AUC-ROC, Confusion Matrix, ROC Curves
6. Feature Analysis    → Random Forest feature importances
7. Report              → Automated summary report generation
```

---

## 🚀 Getting Started

### Prerequisites
```bash
Python 3.8+
```

### Installation
```bash
git clone https://github.com/chandanthakur/water-quality-analysis.git
cd water-quality-analysis
pip install -r requirements.txt
```

### Run
```bash
python water_quality_analysis.py
```

All outputs will be saved to the `outputs/` folder automatically.

---

## 📈 Output Visualizations

The pipeline generates 7 visualizations:

- **Feature Distributions** — Potable vs Non-Potable chemical profiles
- **Correlation Heatmap** — Feature relationships
- **Class Distribution** — Dataset balance overview
- **Model Comparison** — Accuracy & AUC-ROC across all models
- **Confusion Matrix** — Best model prediction breakdown
- **ROC Curves** — Performance curves for all models
- **Feature Importance** — Key chemical contaminants ranked

---

## 🛠️ Tech Stack

![Python](https://img.shields.io/badge/Python-3.12-blue)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.x-orange)
![Pandas](https://img.shields.io/badge/Pandas-2.x-green)
![Matplotlib](https://img.shields.io/badge/Matplotlib-3.x-red)
![Seaborn](https://img.shields.io/badge/Seaborn-0.13-purple)

---

## 👤 Author

**Chandan Thakur**  
CS Graduate | Generative AI & Data Science  
📧 thakurchandan07c@gmail.com  
🔗 [LinkedIn](https://linkedin.com/in/chandanthakur) | [Behance](https://behance.net/outcastthakur)

---

## 📄 License

MIT License — feel free to use and build upon this project.
