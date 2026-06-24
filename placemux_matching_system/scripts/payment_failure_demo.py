import pandas as pd

transaction = {
    "student": "Pavan",
    "job": "AI Engineer",
    "payment_status": "FAILED",
    "application_status": "PENDING"
}

if transaction["payment_status"] == "SUCCESS":
    transaction["application_status"] = "SUBMITTED"
else:
    transaction["application_status"] = "CANCELLED"

df = pd.DataFrame([transaction])

print("=" * 60)
print("PAYMENT FAILURE HANDLING")
print("=" * 60)

print(df)

df.to_csv(
    "outputs/transaction_log.csv",
    index=False
)

print("\nTransaction log saved.")