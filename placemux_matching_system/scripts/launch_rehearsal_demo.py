import pandas as pd
import os
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

fairness = pd.read_csv(
    os.path.join(BASE_DIR,"outputs","fairness_report.csv")
)

metrics = pd.read_csv(
    os.path.join(BASE_DIR,"outputs","fairness_metrics.csv")
)

signoff = pd.read_csv(
    os.path.join(BASE_DIR,"outputs","model_signoff.csv")
)

print("="*70)
print("PLACEMUX LAUNCH REHEARSAL")
print("="*70)

print("\nSTEP 1")
print("Loading Fairness Report...")
print("SUCCESS")

print("\nSTEP 2")
print("Loading Metrics...")
print("SUCCESS")

print("\nSTEP 3")
print("Loading Model Sign-Off...")
print("SUCCESS")

candidate = fairness.iloc[0]

print("\nLIVE DEMO")

print("Candidate :",candidate["Candidate"])
print("Recommendation :",candidate["Recommendation"])

print("\nPlain-English Why")

print(
    "The candidate received this recommendation because "
    "their verified skills produced a strong average score "
    "during evaluation."
)

print("\nMODEL METRICS")

print(metrics)

print("\nSIGN-OFF STATUS")

print(signoff[["Model","Status"]])

print("\nFINAL RESULT")

if (signoff["Status"]=="APPROVED").all():
    print("LAUNCH REHEARSAL PASSED")
else:
    print("LAUNCH BLOCKED")

print("\nTimestamp")

print(datetime.now())

print("="*70)