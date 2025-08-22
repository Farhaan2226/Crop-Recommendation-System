import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, classification_report, confusion_matrix

# Load dataset
df = pd.read_csv("Crop_recommendation.csv")

# Encode crop labels to numeric
le = LabelEncoder()
df["label"] = le.fit_transform(df["label"])  # 'label' column contains crop names

# Features and target
X = df.drop("label", axis=1)
y = df["label"]

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Initialize models
rf = RandomForestClassifier()
knn = KNeighborsClassifier()
xgb = XGBClassifier(use_label_encoder=False, eval_metric="mlogloss")

# Train and evaluate models
for model, name in zip([rf, knn, xgb], ["Random Forest", "KNN", "XGBoost"]):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='macro')

    print(f"\n🧠 {name} Evaluation")
    print(f"Accuracy: {acc:.2%}")
    print(f"Precision: {prec:.2%}")
    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    # Confusion Matrix Plot
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=False, cmap="Blues", fmt='g')
    plt.title(f"{name} - Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(f"{name.lower().replace(' ', '_')}_confusion_matrix.png")
    plt.close()

# Save models
pickle.dump(rf, open("random_forest_model.pkl", "wb"))
pickle.dump(knn, open("knn_model.pkl", "wb"))
pickle.dump(xgb, open("xgboost_model.pkl", "wb"))
pickle.dump(scaler, open("scaler.pkl", "wb"))
pickle.dump(le, open("label_encoder.pkl", "wb"))

print("\n✅ All models and encoders saved successfully!")

