"""
Diabetes Prediction Machine Learning Project
Dataset: Pima Indians Diabetes Dataset
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.metrics import roc_curve, roc_auc_score
import warnings
warnings.filterwarnings('ignore')

# Set style for better visualizations
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 8)

print("=" * 80)
print("DIABETES PREDICTION MACHINE LEARNING PROJECT")
print("=" * 80)

# 1. LOAD THE DATA
print("\n1. Loading Dataset...")
df = pd.read_csv('diabetes.csv')
print(f"Dataset loaded successfully! Shape: {df.shape}")
print(f"Number of samples: {df.shape[0]}")
print(f"Number of features: {df.shape[1]}")

# 2. DATA EXPLORATION
print("\n2. Data Exploration")
print("\nFirst 5 rows:")
print(df.head())

print("\nDataset Info:")
print(df.info())

print("\nStatistical Summary:")
print(df.describe())

print("\nMissing Values:")
print(df.isnull().sum())

print("\nClass Distribution:")
print(df['Outcome'].value_counts())
print(f"\nNo Diabetes: {df['Outcome'].value_counts()[0]} ({df['Outcome'].value_counts()[0]/len(df)*100:.2f}%)")
print(f"Diabetes: {df['Outcome'].value_counts()[1]} ({df['Outcome'].value_counts()[1]/len(df)*100:.2f}%)")

# 3. DATA VISUALIZATION
print("\n3. Creating Visualizations...")

# Outcome Distribution
plt.figure(figsize=(8, 6))
df['Outcome'].value_counts().plot(kind='pie', autopct='%1.1f%%', 
                                   labels=['No Diabetes', 'Diabetes'],
                                   colors=['lightgreen', 'salmon'])
plt.title('Diabetes Outcome Distribution', fontsize=16, fontweight='bold')
plt.ylabel('')
plt.savefig('outcome_distribution.png', dpi=300, bbox_inches='tight')
print("✓ Saved: outcome_distribution.png")
plt.close()

# Correlation Heatmap
plt.figure(figsize=(12, 10))
correlation = df.corr()
sns.heatmap(correlation, annot=True, cmap='coolwarm', center=0, 
            square=True, linewidths=1, cbar_kws={"shrink": 0.8})
plt.title('Feature Correlation Heatmap', fontsize=16, fontweight='bold')
plt.savefig('correlation_heatmap.png', dpi=300, bbox_inches='tight')
print("✓ Saved: correlation_heatmap.png")
plt.close()

# Feature Distributions
fig, axes = plt.subplots(3, 3, figsize=(15, 12))
features = df.columns[:-1]
axes = axes.flatten()

for i, col in enumerate(features):
    axes[i].hist(df[df['Outcome']==0][col], alpha=0.5, label='No Diabetes', bins=30, color='blue')
    axes[i].hist(df[df['Outcome']==1][col], alpha=0.5, label='Diabetes', bins=30, color='red')
    axes[i].set_title(col, fontweight='bold')
    axes[i].legend()
    axes[i].set_xlabel(col)
    axes[i].set_ylabel('Frequency')

plt.tight_layout()
plt.savefig('feature_distributions.png', dpi=300, bbox_inches='tight')
print("✓ Saved: feature_distributions.png")
plt.close()

# Box plots for key features
fig, axes = plt.subplots(2, 4, figsize=(16, 8))
axes = axes.flatten()

for i, col in enumerate(features):
    df.boxplot(column=col, by='Outcome', ax=axes[i])
    axes[i].set_title(f'{col} by Outcome')
    axes[i].set_xlabel('Outcome')

plt.tight_layout()
plt.savefig('boxplots.png', dpi=300, bbox_inches='tight')
print("✓ Saved: boxplots.png")
plt.close()

# 4. DATA PREPROCESSING
print("\n4. Data Preprocessing...")

# Separate features and target
X = df.drop('Outcome', axis=1)
y = df['Outcome']

print(f"Features shape: {X.shape}")
print(f"Target shape: {y.shape}")

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print(f"\nTraining set size: {X_train.shape[0]}")
print(f"Test set size: {X_test.shape[0]}")

# Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print("✓ Feature scaling completed")

# 5. MODEL TRAINING
print("\n5. Training Machine Learning Models...")

models = {
    'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'Support Vector Machine': SVC(probability=True, random_state=42)
}

results = {}

for name, model in models.items():
    print(f"\nTraining {name}...")
    model.fit(X_train_scaled, y_train)
    
    # Predictions
    y_pred = model.predict(X_test_scaled)
    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
    
    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    
    results[name] = {
        'model': model,
        'accuracy': accuracy,
        'roc_auc': roc_auc,
        'y_pred': y_pred,
        'y_pred_proba': y_pred_proba
    }
    
    print(f"✓ {name} - Accuracy: {accuracy:.4f}, ROC-AUC: {roc_auc:.4f}")

# 6. MODEL EVALUATION
print("\n6. Model Evaluation Results")
print("=" * 80)

for name, result in results.items():
    print(f"\n{name.upper()}")
    print("-" * 40)
    print(f"Accuracy: {result['accuracy']:.4f}")
    print(f"ROC-AUC Score: {result['roc_auc']:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, result['y_pred'], 
                                target_names=['No Diabetes', 'Diabetes']))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, result['y_pred']))

# 7. VISUALIZE RESULTS
print("\n7. Creating Model Comparison Visualizations...")

# Accuracy Comparison
fig, ax = plt.subplots(figsize=(10, 6))
model_names = list(results.keys())
accuracies = [results[name]['accuracy'] for name in model_names]
roc_aucs = [results[name]['roc_auc'] for name in model_names]

x = np.arange(len(model_names))
width = 0.35

bars1 = ax.bar(x - width/2, accuracies, width, label='Accuracy', color='steelblue')
bars2 = ax.bar(x + width/2, roc_aucs, width, label='ROC-AUC', color='coral')

ax.set_xlabel('Models', fontweight='bold')
ax.set_ylabel('Score', fontweight='bold')
ax.set_title('Model Performance Comparison', fontweight='bold', fontsize=14)
ax.set_xticks(x)
ax.set_xticklabels(model_names, rotation=15, ha='right')
ax.legend()
ax.grid(axis='y', alpha=0.3)

# Add value labels on bars
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}', ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
print("✓ Saved: model_comparison.png")
plt.close()

# ROC Curves
plt.figure(figsize=(10, 8))
for name, result in results.items():
    fpr, tpr, _ = roc_curve(y_test, result['y_pred_proba'])
    plt.plot(fpr, tpr, label=f"{name} (AUC = {result['roc_auc']:.3f})", linewidth=2)

plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier', linewidth=2)
plt.xlabel('False Positive Rate', fontweight='bold')
plt.ylabel('True Positive Rate', fontweight='bold')
plt.title('ROC Curves - Model Comparison', fontweight='bold', fontsize=14)
plt.legend(loc='lower right')
plt.grid(alpha=0.3)
plt.savefig('roc_curves.png', dpi=300, bbox_inches='tight')
print("✓ Saved: roc_curves.png")
plt.close()

# Confusion Matrices
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for idx, (name, result) in enumerate(results.items()):
    cm = confusion_matrix(y_test, result['y_pred'])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx],
                xticklabels=['No Diabetes', 'Diabetes'],
                yticklabels=['No Diabetes', 'Diabetes'])
    axes[idx].set_title(f'{name}\nAccuracy: {result["accuracy"]:.3f}', fontweight='bold')
    axes[idx].set_ylabel('True Label')
    axes[idx].set_xlabel('Predicted Label')

plt.tight_layout()
plt.savefig('confusion_matrices.png', dpi=300, bbox_inches='tight')
print("✓ Saved: confusion_matrices.png")
plt.close()

# 8. FEATURE IMPORTANCE (Random Forest)
print("\n8. Feature Importance Analysis...")
rf_model = results['Random Forest']['model']
feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': rf_model.feature_importances_
}).sort_values('Importance', ascending=False)

print("\nTop Features (Random Forest):")
print(feature_importance)

plt.figure(figsize=(10, 6))
sns.barplot(data=feature_importance, x='Importance', y='Feature', palette='viridis')
plt.title('Feature Importance - Random Forest', fontweight='bold', fontsize=14)
plt.xlabel('Importance Score', fontweight='bold')
plt.ylabel('Features', fontweight='bold')
plt.tight_layout()
plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
print("✓ Saved: feature_importance.png")
plt.close()

# 9. FINAL SUMMARY
print("\n" + "=" * 80)
print("PROJECT SUMMARY")
print("=" * 80)

best_model_name = max(results.keys(), key=lambda x: results[x]['accuracy'])
best_accuracy = results[best_model_name]['accuracy']
best_roc_auc = results[best_model_name]['roc_auc']

print(f"\nBest Performing Model: {best_model_name}")
print(f"Best Accuracy: {best_accuracy:.4f} ({best_accuracy*100:.2f}%)")
print(f"Best ROC-AUC Score: {best_roc_auc:.4f}")

print("\nAll Models Performance:")
for name, result in results.items():
    print(f"  • {name}: {result['accuracy']:.4f} accuracy, {result['roc_auc']:.4f} ROC-AUC")

print("\nTop 3 Most Important Features:")
for i, row in feature_importance.head(3).iterrows():
    print(f"  {i+1}. {row['Feature']}: {row['Importance']:.4f}")

print("\nFiles Generated:")
files = [
    'outcome_distribution.png',
    'correlation_heatmap.png',
    'feature_distributions.png',
    'boxplots.png',
    'model_comparison.png',
    'roc_curves.png',
    'confusion_matrices.png',
    'feature_importance.png'
]
for f in files:
    print(f"  ✓ {f}")

print("\n" + "=" * 80)
print("PROJECT COMPLETED SUCCESSFULLY!")
print("=" * 80)
