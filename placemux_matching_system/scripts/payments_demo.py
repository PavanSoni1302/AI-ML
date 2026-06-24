import pandas as pd
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

payments = [
    {
        "student": "Pavan",
        "job": "AI Engineer",
        "amount": 100,
        "status": "SUCCESS"
    },
    {
        "student": "Rahul",
        "job": "ML Engineer",
        "amount": 100,
        "status": "FAILED"
    }
]

df = pd.DataFrame(payments)

print("=" * 60)
print("PAYMENT TRANSACTION REPORT")
print("=" * 60)

print(df)

df.to_csv(
    os.path.join(BASE_DIR, "outputs", "payment_history.csv"),
    index=False
)

print("\nPayment history saved.")