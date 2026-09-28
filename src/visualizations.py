import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("data/processed/cleaned_returns.csv")

df["delivery_group"] = pd.cut(
    df["delivery_delay_days"],
    bins=[-float("inf"), 0, 2, 5, float("inf")],
    labels=["Early / On Time", "0–2 Days", "2–5 Days", "5+ Days"]
)

plt.figure(figsize=(7, 5))
sns.countplot(x="returned", data=df)
plt.title("Overall Return Distribution")
plt.xlabel("Returned")
plt.ylabel("Number of Orders")
plt.tight_layout()
plt.savefig("visualizations/overall_return_distribution.png")
plt.close()

category_returns = df.groupby("product_category", as_index=False)["returned"].mean()
category_returns["return_rate"] = category_returns["returned"] * 100
category_returns = category_returns.sort_values("return_rate", ascending=False)

plt.figure(figsize=(8, 5))
sns.barplot(data=category_returns, x="product_category", y="return_rate")
plt.title("Return Rate by Product Category")
plt.xlabel("Product Category")
plt.ylabel("Return Rate (%)")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("visualizations/category_return_rate.png")
plt.close()

delivery_returns = df.groupby("delivery_group", observed=True, as_index=False)["returned"].mean()
delivery_returns["return_rate"] = delivery_returns["returned"] * 100

plt.figure(figsize=(8, 5))
sns.barplot(data=delivery_returns, x="delivery_group", y="return_rate")
plt.title("Return Rate by Delivery Group")
plt.xlabel("Delivery Group")
plt.ylabel("Return Rate (%)")
plt.tight_layout()
plt.savefig("visualizations/delivery_return_rate.png")
plt.close()

customer_history = df.groupby("returned")[["past_purchase_count", "past_return_rate"]].mean().reset_index()
customer_history["return_status"] = customer_history["returned"].map({0: "Not Returned", 1: "Returned"})

history_plot = customer_history.melt(
    id_vars="return_status",
    value_vars=["past_purchase_count", "past_return_rate"],
    var_name="Metric",
    value_name="Average"
)

plt.figure(figsize=(8, 5))
sns.barplot(data=history_plot, x="Metric", y="Average", hue="return_status")
plt.title("Customer History by Return Status")
plt.xlabel("Customer History Metric")
plt.ylabel("Average Value")
plt.tight_layout()
plt.savefig("visualizations/customer_history_return.png")
plt.close()

print("All 4 visualizations created successfully.")