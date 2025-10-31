import pandas as pd

# Read the CSV file
df = pd.read_csv("your_file.csv")

# Filter rows where any column contains the word 'MASH' (case-insensitive)
mash_rows = df[df.apply(lambda row: row.astype(str).str.contains('MASH', case=False, na=False).any(), axis=1)]

# Display the matching rows
print(mash_rows)

# (Optional) Save the filtered rows to a new CSV file
mash_rows.to_csv("mash_rows.csv", index=False)
