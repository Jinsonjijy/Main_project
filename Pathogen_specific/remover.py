# import pandas as pd

# filename = 'data/all_bacteria.csv' # Put your file name here
# column_to_remove = 'Entry Name' # Put the header name here

# # 1. Load the data
# df = pd.read_csv(filename)

# # 2. Remove the column
# if column_to_remove in df.columns:
#     df.drop(columns=[column_to_remove], inplace=True)
    
#     # 3. Save back to the SAME file
#     df.to_csv(filename, index=False)
#     print(f"Successfully removed '{column_to_remove}' and updated {filename}")
# else:
#     print(f"Error: Column '{column_to_remove}' not found in the file.")


import pandas as pd

filename = 'data/drug_lookup_enriched.csv'
target_column = 'drug_name'  # Change this to your actual column header
old_name = '3-(4-Benzenesulfonyl-Thiophene-2-Sulfonylamino)-Phenylboronic Acid'
new_name = 'Thiophenesulfonylboronic Acid'

# 1. Load the data
df = pd.read_csv(filename)

# 2. Replace the specific name
# .replace() is safe and won't affect other rows
df[target_column] = df[target_column].replace(old_name, new_name)

# 3. Save the changes back to the same file
df.to_csv(filename, index=False)

print(f"Update complete! Swapped '{old_name}' for '{new_name}' in {filename}")