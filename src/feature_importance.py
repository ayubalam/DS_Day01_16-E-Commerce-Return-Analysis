import pandas as pd
import joblib
import matplotlib.pyplot as plt

df = pd.read_csv("data/processed/cleaned_returns.csv")

pipeline = joblib.load("models/random_forest_model.joblib")

preprocessor = pipeline.named_steps["preprocessor"]
model = pipeline.named_steps["model"]

feature_names = preprocessor.get_feature_names_out()
importance = model.feature_importances_

feature_importance = pd.DataFrame({
    "feature": feature_names,
    "importance": importance
})

feature_importance = feature_importance.sort_values(
    "importance",
    ascending=False
)

print("Top 15 Feature Importances:")
print(feature_importance.head(15).to_string(index=False))

top_features = feature_importance.head(15)

plt.figure(figsize=(10, 6))
plt.barh(
    top_features["feature"][::-1],
    top_features["importance"][::-1]
)
plt.title("Top 15 Feature Importances")
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.tight_layout()
plt.savefig("visualizations/feature_importance.png")
plt.close()

print("\nFeature importance visualization saved to:")
print("visualizations/feature_importance.png")