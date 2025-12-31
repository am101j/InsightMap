import pandas as pd

# Load only the columns we want to check to save memory/time
cols_to_check = [
    'review_scores_rating', 
    'review_scores_cleanliness', 
    'review_scores_location', 
    'review_scores_value',
    'host_is_superhost', 
    'host_identity_verified',
    'property_type',
    'instant_bookable'
]

print(f"Checking {len(cols_to_check)} columns in listings.csv.gz...")
try:
    df = pd.read_csv('listings.csv.gz', usecols=cols_to_check)
    print("\n--- NON-NULL COUNTS ---")
    print(df.count())
    print("\n--- SAMPLE VALUES ---")
    print(df.iloc[0])
    print("\n--- DATA TYPES ---")
    print(df.dtypes)
except Exception as e:
    print(f"Error: {e}")
