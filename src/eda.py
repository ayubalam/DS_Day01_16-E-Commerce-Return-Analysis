import pandas as pd

df = pd.read_csv("data/processed/cleaned_returns.csv")

print("Dataset Shape:", df.shape)

return_rate = df["returned"].mean() * 100
print(f"\nOverall Return Rate: {return_rate:.2f}%")

print("\nReturn Status:")
print(df["returned"].value_counts())

print("\nReturn Rate by Product Category:")
category_returns = df.groupby("product_category")["returned"].agg(["count", "sum", "mean"])
category_returns["return_rate"] = category_returns["mean"] * 100
print(category_returns.sort_values("return_rate", ascending=False))

df["delivery_group"] = pd.cut(
    df["delivery_delay_days"],
    bins=[-float("inf"), 0, 2, 5, float("inf")],
    labels=["Early / On Time", "0–2 Days", "2–5 Days", "5+ Days"]
)

delivery_returns = df.groupby("delivery_group", observed=True)["returned"].agg(["count", "sum", "mean"])
delivery_returns["return_rate"] = delivery_returns["mean"] * 100

print("\nReturn Rate by Delivery Group:")
print(delivery_returns)

print("\nReturn Rate by Coupon Usage:")
coupon_returns = df.groupby("used_coupon")["returned"].agg(["count", "sum", "mean"])
coupon_returns["return_rate"] = coupon_returns["mean"] * 100
print(coupon_returns)

print("\nReturn Rate by Device Type:")
device_returns = df.groupby("device_type")["returned"].agg(["count", "sum", "mean"])
device_returns["return_rate"] = device_returns["mean"] * 100
print(device_returns)

print("\nReturn Rate by Shipping Method:")
shipping_returns = df.groupby("shipping_method")["returned"].agg(["count", "sum", "mean"])
shipping_returns["return_rate"] = shipping_returns["mean"] * 100
print(shipping_returns)

print("\nReturn Rate by Payment Method:")
payment_returns = df.groupby("payment_method")["returned"].agg(["count", "sum", "mean"])
payment_returns["return_rate"] = payment_returns["mean"] * 100
print(payment_returns)

print("\nCustomer History Statistics:")
customer_history = df.groupby("returned")[["past_purchase_count", "past_return_rate"]].mean()
print(customer_history)

print("\nProduct Price Statistics by Return Status:")
price_analysis = df.groupby("returned")["product_price"].agg(["count", "mean", "median", "min", "max"])
print(price_analysis)

print("\nProduct Rating Statistics by Return Status:")
rating_analysis = df.groupby("returned")["product_rating"].agg(["mean", "median"])
print(rating_analysis)

print("\nDiscount Statistics by Return Status:")
discount_analysis = df.groupby("returned")["discount_percent"].agg(["mean", "median"])
print(discount_analysis)

print("\nSession Length Statistics by Return Status:")
session_analysis = df.groupby("returned")["session_length_minutes"].agg(["mean", "median"])
print(session_analysis)

print("\nProduct Views Statistics by Return Status:")
views_analysis = df.groupby("returned")["num_product_views"].agg(["mean", "median"])
print(views_analysis)