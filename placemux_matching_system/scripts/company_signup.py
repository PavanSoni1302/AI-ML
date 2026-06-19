import pandas as pd
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

companies_path = os.path.join(BASE_DIR, "data", "companies.csv")

companies = pd.read_csv(companies_path)

print("=" * 50)
print("REGISTERED COMPANIES")
print("=" * 50)

print(companies)