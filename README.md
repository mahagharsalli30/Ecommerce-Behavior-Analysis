# 🛒 E-Commerce User Behavior Analysis

Machine learning project focused on predicting customer purchase behavior using browsing, engagement, and demographic data from 8,000 e-commerce sessions.

### Technologies

* Python
* Pandas & NumPy
* Scikit-Learn
* Matplotlib & Seaborn
* SMOTE
* PCA

### Models

* Logistic Regression
* Decision Tree
* Random Forest

### Results

| Model               | Accuracy | ROC-AUC    |
| ------------------- | -------- | ---------- |
| Logistic Regression | 99.69%   | **99.85%** |
| Decision Tree       | 99.81%   | 83.23%     |
| Random Forest       | 99.81%   | 99.79%     |

### Key Findings

* Cart items added was the strongest predictor of purchase.
* User engagement metrics outperformed demographic variables.
* Bounce rate was the strongest negative predictor.
* Logistic Regression performed best for identifying non-buyers.
* Random Forest provided the most stable feature importance rankings.

### Project Files

* `CleaningandMining2.py` – complete analysis pipeline
* `ecommerce_user_behavior_cleaned.csv` – cleaned dataset
* `Data_Mining_Project.pdf` – full project report

### Run the Project

```bash
pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn
python CleaningandMining2.py
```

### Author

**Maha Gharsalli**
Business Analytics Major | IT Minor
Tunis Business School
