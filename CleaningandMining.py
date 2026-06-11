# _______________ Step 0: Import Libraries ──────────────────────────
import pandas as pd
import numpy as np

# Visualization (for outliers)
import matplotlib.pyplot as plt
import seaborn as sns

# Preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Handling class imbalance
from imblearn.over_sampling import SMOTE

# _______________ Models ────────────────────────────────────────────
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

# _______________ Evaluation ───────────────────────────────────────
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    f1_score
)

# _______________ Clustering ───────────────────────────────────────
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

import warnings
warnings.filterwarnings('ignore')


# ____________________ Load dataset ____________________
df = pd.read_csv(r"C:\Users\ramez\Downloads\ecommerce_user_behavior_8000.csv")

# Display dataset
print(f"Shape: {df.shape}")
print(df.head())
# Check Missing Values
print(df.isnull().sum())


# ____________________ Remove duplicates ____________________
df = df.drop_duplicates().reset_index(drop=True)


# ____________________ Drop identifier ____________________
df = df.drop(columns=['user_id'])


# ____________________ Handle Missing Values ____________________

# Fill numeric values with median
num_cols = df.select_dtypes(include=np.number).columns
df[num_cols] = df[num_cols].fillna(df[num_cols].median())

# Fill categorical values with mode
cat_cols = df.select_dtypes(include='object').columns
df[cat_cols] = df[cat_cols].fillna(df[cat_cols].mode().iloc[0])

# _________Continuous features to check for outliers__________
num_cols = ['age', 'time_on_site', 'pages_viewed',
             'previous_purchases', 'cart_items',
             'avg_session_time', 'bounce_rate']
 
for col in num_cols:
    Q1  = df[col].quantile(0.25)
    Q3  = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    n_out = ((df[col] < lower) | (df[col] > upper)).sum()
    print(f'{col}: {n_out} outliers')


# ____________________ Encoding ____________________
le = LabelEncoder()
df['gender'] = le.fit_transform(df['gender'])

df = pd.get_dummies(df, columns=['device_type'], prefix='device')

#_______________ Convert bool to int_________________________
bool_cols = df.select_dtypes(include='bool').columns
df[bool_cols] = df[bool_cols].astype(int)

# ____________Save cleaned dataset___________
df.to_csv("ecommerce_user_behavior_cleaned.csv", index=False)
print("Cleaned dataset saved successfully!")

# ____________________ Separate features and target ____________________
X = df.drop(columns=['purchase'])
y = df['purchase'].astype(int)

print("Before balancing:")
print(y.value_counts())


# ____________________ Train Test Split ____________________
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# ____________________ SMOTE (Partial Balancing) ____________________
smote = SMOTE(
    sampling_strategy=0.4,   # 40% balance (not equal)
    random_state=42
)

X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)

print("After SMOTE:")
print(y_train_bal.value_counts())


# ____________________ Feature Scaling ____________________
scale_cols = [
    'age', 'time_on_site', 'pages_viewed',
    'previous_purchases', 'cart_items',
    'avg_session_time', 'bounce_rate'
]

scaler = StandardScaler()

X_train_bal[scale_cols] = scaler.fit_transform(X_train_bal[scale_cols])
X_test[scale_cols] = scaler.transform(X_test[scale_cols])

# ================== MINING STARTS HERE ==================
X = df.drop(columns=['purchase'])
y = df['purchase'].astype(int)

from sklearn.linear_model import LogisticRegression
import warnings
warnings.filterwarnings('ignore')
 
lr = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42)
lr.fit(X_train_bal, y_train_bal)
y_pred_lr = lr.predict(X_test)
 
# Feature coefficients — direction and strength of each variable
coef_df = pd.DataFrame({'Feature': X_train_bal.columns,'Coefficient': lr.coef_[0]})
coef_df = coef_df.sort_values('Coefficient', ascending=False)
print(coef_df)
 
# Visualise coefficients
plt.figure(figsize=(9, 5))
sns.barplot(data=coef_df, x='Coefficient', y='Feature', palette='coolwarm')
plt.title('Logistic Regression — Feature Coefficients')
plt.axvline(x=0, color='black', linewidth=0.8, linestyle='--')
plt.tight_layout()
plt.savefig('lr_coefficients.png', dpi=150)
plt.show()
from sklearn.tree import DecisionTreeClassifier, plot_tree
 
dt = DecisionTreeClassifier(max_depth=5, class_weight='balanced', random_state=42)
dt.fit(X_train_bal, y_train_bal)
y_pred_dt = dt.predict(X_test)
 
# Visualise the tree structure (top 3 levels for readability)
plt.figure(figsize=(20, 8))
plot_tree(dt, max_depth=3, feature_names=X_train_bal.columns,
             class_names=['No Buy', 'Buy'], filled=True, rounded=True)
plt.title('Decision Tree — Purchase Decision Rules (Top 3 Levels)', fontsize=13)
plt.tight_layout()
plt.savefig('decision_tree.png', dpi=150, bbox_inches='tight')
plt.show()
 
# Feature importance bar chart
fi_dt = pd.DataFrame({'Feature': X_train_bal.columns,'Importance': dt.feature_importances_})
fi_dt = fi_dt.sort_values('Importance', ascending=False)
plt.figure(figsize=(9, 5))
sns.barplot(data=fi_dt, x='Importance', y='Feature', palette='Blues_r')
plt.title('Decision Tree — Feature Importance')
plt.tight_layout()
plt.savefig('dt_feature_importance.png', dpi=150)
plt.show()
from sklearn.ensemble import RandomForestClassifier
 
rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)
rf.fit(X_train_bal, y_train_bal)
y_pred_rf = rf.predict(X_test)
 
# Feature importance — averaged across all 200 trees
fi_rf = pd.DataFrame({'Feature': X_train_bal.columns,'Importance': rf.feature_importances_})
fi_rf = fi_rf.sort_values('Importance', ascending=False)
plt.figure(figsize=(9, 5))
sns.barplot(data=fi_rf, x='Importance', y='Feature', palette='Blues_r')
plt.title('Random Forest — Feature Importance (200 Trees)')
plt.tight_layout()
plt.savefig('rf_feature_importance.png', dpi=150)
plt.show()
from sklearn.cluster    import KMeans
from sklearn.decomposition import PCA
 
# Step 1 — Elbow Method: find the best K
inertia = []
for k in range(1, 9):
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X_test)
    inertia.append(km.inertia_)
 
plt.figure(figsize=(7, 4))
plt.plot(range(1, 9), inertia, marker='o', color='steelblue')
plt.title('Elbow Method — Optimal Number of Clusters')
plt.xlabel('K') 
plt.ylabel('Inertia')
plt.tight_layout() 
plt.savefig('elbow.png', dpi=150)
plt.show()
 
# Step 2 — Apply K-Means with K=3
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
clusters = kmeans.fit_predict(X_test)
 
# Step 3 — PCA for 2D visualisation
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_test)
 
plt.figure(figsize=(9, 6))
sc = plt.scatter(X_pca[:, 0], X_pca[:, 1],
               c=clusters, cmap='Set2', alpha=0.6, s=30)
plt.colorbar(sc, label='Cluster')
plt.title('User Segments — K-Means (K=3) + PCA')
plt.xlabel('PC 1')  
plt.ylabel('PC 2')
plt.tight_layout()  
plt.savefig('clusters_pca.png', dpi=150)
plt.show()
 
# Step 4 — Profile each cluster by average feature values
profile = X_test.copy()
profile['Cluster']  = clusters
profile['Purchase'] = y_test.values
print(profile.groupby('Cluster')[['time_on_site','cart_items','bounce_rate','Purchase']].mean().round(2))
from sklearn.metrics import (accuracy_score, f1_score,
                                 confusion_matrix, classification_report)
 
# ── Per-model evaluation ─────────────────────────────────────
models = {'Logistic Regression': y_pred_lr,
           'Decision Tree': y_pred_dt,
           'Random Forest': y_pred_rf}
 
for name, pred in models.items():
    print(f'\n=== {name} ===')
    print(f'Accuracy : {accuracy_score(y_test, pred):.4f}')
    print(f'F1 Score : {f1_score(y_test, pred, average="weighted"):.4f}')
    print(classification_report(y_test, pred,
                        target_names=['No Buy', 'Buy']))
 
# ── Confusion matrices — all 3 models side by side ───────────
fig, axes = plt.subplots(1, 3, figsize=(14, 4))
 
for ax, (name, pred) in zip(axes, models.items()):
    cm = confusion_matrix(y_test, pred)
    sns.heatmap(cm, annot=True, fmt='d', ax=ax, cmap='Blues', cbar=False,
               xticklabels=['No Buy','Buy'],
               yticklabels=['No Buy','Buy'])
    ax.set_title(name, fontsize=11)
    ax.set_xlabel('Predicted')  
    ax.set_ylabel('Actual')
 
plt.suptitle('Confusion Matrices — All Models', fontsize=13, y=1.02)
plt.tight_layout()
plt.savefig('confusion_matrices.png', dpi=150, bbox_inches='tight')
plt.show()
 
# ── Side-by-side feature importance: DT vs RF ───────────────
fi_compare = pd.DataFrame({
    'Feature'     : X_train_bal.columns,
    'Decision Tree': dt.feature_importances_,
    'Random Forest': rf.feature_importances_
})
fi_melt = fi_compare.melt(id_vars='Feature', var_name='Model', value_name='Importance')
 
plt.figure(figsize=(11, 6))
sns.barplot(data=fi_melt, x='Importance', y='Feature',
             hue='Model', palette=['#2E75B6','#1F3864'])
plt.title('Feature Importance: Decision Tree vs Random Forest')
plt.legend(loc='lower right')
plt.tight_layout()
plt.savefig('feature_importance_compare.png', dpi=150)
plt.show()

 
