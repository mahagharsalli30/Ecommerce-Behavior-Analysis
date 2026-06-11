\# 🛒 E-Commerce User Behavior Analysis — Data Mining Project



> \*\*Tunis Business School | BA Major · IT Minor | April 2026\*\*  

> \*Maha Gharsalli · Rihem Abdelmoumen · Rayen Ghourabi\*



\---



\## 📌 Project Overview



This project applies data mining and machine learning techniques to analyze user behavior on an e-commerce platform and predict whether a user will complete a purchase.



The dataset contains \*\*8,000 simulated user sessions\*\* with 14 variables covering demographics, browsing behavior, marketing interactions, and device usage — with `purchase` as the binary target variable.



\---



\## 🎯 Objectives



\- Predict whether a user will make a purchase based on behavioral and demographic features

\- Identify the most important factors influencing purchase decisions

\- Segment customers based on behavioral characteristics

\- Generate actionable insights to improve conversion rates and marketing strategies



\---



\## 📂 Repository Structure



```

Data-Mining-ML-Project/

│

├── CleaningandMining2.py                   # Full data cleaning + mining pipeline

├── ecommerce\_user\_behavior\_cleaned.csv     # Preprocessed dataset (output of script)

├── Data\_Mining\_Project.pdf                 # Full project report

└── README.md

```



\---



\## 📊 Dataset



| Property | Details |

|---|---|

| Source | \[Kaggle — E-Commerce Behavior Dataset (8000 Users)](https://www.kaggle.com/datasets/asifxzaman/e-commerce-behavior-dataset8000-users) |

| Observations | 8,000 user sessions |

| Features | 14 (demographic + behavioral + marketing) |

| Target | `purchase` (0 = No Buy, 1 = Buy) |

| Class Imbalance | \~99.8% buyers vs \~0.2% non-buyers |



\*\*Key features:\*\* `age`, `gender`, `device\_type`, `time\_on\_site`, `pages\_viewed`, `previous\_purchases`, `cart\_items`, `discount\_seen`, `ad\_clicked`, `returning\_user`, `avg\_session\_time`, `bounce\_rate`



\---



\## ⚙️ Pipeline Summary



\### Data Preparation

\- Removed duplicates · dropped `user\_id`

\- Imputed missing values (median for numeric, mode for categorical)

\- IQR-based outlier detection

\- Label Encoding (`gender`) + One-Hot Encoding (`device\_type`)

\- Stratified 80/20 train-test split



\### Class Imbalance Handling

\- \*\*SMOTE\*\* (`sampling\_strategy=0.4`) applied on training set only

\- Training set expanded from 6,400 → 8,946 rows (minority: 10 → 2,556)



\### Feature Scaling

\- `StandardScaler` applied to 7 continuous features (fit on training set only)



\---



\## 🤖 Models \& Methods



| Method | Purpose |

|---|---|

| \*\*Logistic Regression\*\* | Linear baseline · coefficient-based feature interpretation |

| \*\*Decision Tree\*\* | Rule-based non-linear classifier · interpretable splits |

| \*\*Random Forest\*\* (200 trees) | Ensemble model · stable feature importance |

| \*\*PCA\*\* (3 components) | Dimensionality reduction · 3D behavioral visualization |



\---



\## 📈 Results



| Metric | Logistic Regression | Decision Tree | Random Forest |

|---|---|---|---|

| Accuracy | 99.69% | 99.81% | 99.81% |

| F1-Score (weighted) | 99.74% | 99.79% | 99.72% |

| ROC-AUC | \*\*99.85%\*\* | 83.23% | 99.79% |

| Macro F1 | \*\*0.72\*\* | 0.70 | 0.50 |

| No Buy Recall | \*\*67%\*\* | 33% | 0% |



\### Top Purchase Drivers (consensus across all models)

1\. `cart\_items` — strongest predictor of purchase intent

2\. `avg\_session\_time` — deeper engagement = higher conversion

3\. `previous\_purchases` — customer loyalty signal

4\. `bounce\_rate` — strongest negative predictor

5\. `time\_on\_site` — overall browsing engagement



\---



\## 🔍 Key Insights



\- \*\*Behavioral features dominate\*\* — engagement metrics outperform demographics in predicting purchases

\- \*\*Bounce rate\*\* is the only strong negative predictor; high-bounce users rarely convert

\- \*\*Ad clicks have minimal impact\*\* — clicking an ad does not reliably translate to a purchase

\- \*\*Random Forest\*\* is best for propensity scoring (ROC-AUC = 0.9979) but fails on minority class detection

\- \*\*Logistic Regression\*\* is best for identifying non-buyers / churn risk (67% minority recall)

\- \*\*Decision Tree\*\* is best for extracting simple, interpretable business rules



\---



\## 🚀 How to Run



\### Requirements

```bash

pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn

```



\### Steps

1\. Place `ecommerce\_user\_behavior\_8000.csv` in the same folder as the script

2\. Run the script:

```bash

python CleaningandMining2.py

```

3\. Outputs generated:

&#x20;  - `ecommerce\_user\_behavior\_cleaned.csv`

&#x20;  - `lr\_coefficients.png`

&#x20;  - `decision\_tree.png`

&#x20;  - `dt\_feature\_importance.png`

&#x20;  - `rf\_feature\_importance.png`

&#x20;  - `pca\_3d.png`

&#x20;  - `confusion\_matrices.png`



\---



\## 🛠️ Libraries Used



`pandas` · `numpy` · `matplotlib` · `seaborn` · `scikit-learn` · `imbalanced-learn`



\---



\## 📄 Full Report



See \[`Data\_Mining\_Project.pdf`](./Data\_Mining\_Project.pdf) for the complete methodology, visualizations, and interpretations.



