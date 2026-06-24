import pandas as pd
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

companies_file = os.path.join(
    BASE_DIR,
    "data",
    "companies.csv"
)

companies = pd.read_csv(companies_file)

new_company = {
    "company_name": "Infosys",
    "industry": "IT Services"
}

companies = pd.concat(
    [companies, pd.DataFrame([new_company])],
    ignore_index=True
)

companies.to_csv(
    companies_file,
    index=False
)

print("=" * 60)
print("COMPANY SIGNUP")
print("=" * 60)

print("\nCompany Registered Successfully")

print(new_company)

print("\nTotal Companies")

print(len(companies))