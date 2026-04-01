import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import shap

# -------------------------------
# 1. Load Data
# -------------------------------
df = pd.read_csv("german_credit_data.csv")

# -------------------------------
# 2. Basic Understanding
# -------------------------------
print("Shape:", df.shape)
print("Columns:", df.columns)
print(df.info())
print(df.describe())

# -------------------------------
# 3. Missing Values
# -------------------------------
print("Missing Values:\n", df.isnull().sum())

# -------------------------------
# 4. Target Variable
# -------------------------------
print("Target Distribution:\n", df.iloc[:, -1].value_counts())

# -------------------------------
# 5. Split Data
# -------------------------------
X = df.iloc[:, :-1]
y = df.iloc[:, -1]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# 6. Train Model (Balanced)

model = RandomForestClassifier(class_weight='balanced')
model.fit(X_train, y_train)


# 7. Prediction

y_pred = model.predict(X_test)


# 8. Evaluation

print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))


# 9. Feature Importance

importance = model.feature_importances_

feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': importance
}).sort_values(by='Importance', ascending=False)

print(feature_importance.head(10))

# Plot
feature_importance.head(10).plot(
    x='Feature', y='Importance', kind='barh', title="Top Features"
)
plt.show()


# 10. SHAP Explainability

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

shap.summary_plot(shap_values, X_test)