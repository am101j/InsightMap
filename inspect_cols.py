import pandas as pd
try:
    df = pd.read_csv('listings.csv.gz', nrows=2)
    print("COLUMNS_START")
    print(list(df.columns))
    print("COLUMNS_END")
except Exception as e:
    print(e)
