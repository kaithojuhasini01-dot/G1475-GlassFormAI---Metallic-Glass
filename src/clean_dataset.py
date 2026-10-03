import pandas as pd

# Read the copied dataset from the cleaned folder
input_file = "C:\\Users\\Kaithoju Hasini\\OneDrive\\Documents\\metallic_glass_project\\data\\cleaned\\metallic_glass_cleaned.csv"

df = pd.read_csv(input_file)

# Check the original value
print("Before correction:")
print(df.loc[370, ["Composition", "Tg_[K]", "Tx_[K]", "Tl_[K]", "Dmax_[mm]"]])

# Correct the verified liquidus temperature
if df.loc[370, "Tl_[K]"] == 97:
    df.loc[370, "Tl_[K]"] = 971
    print("\nRow 370 corrected: Tl changed from 97 K to 971 K")
else:
    print("\nRow 370 was not changed because its value is not 97 K.")

# Save the cleaned dataset
output_file = "C:\\Users\\Kaithoju Hasini\\OneDrive\\Documents\\metallic_glass_project\\data\\cleaned\\metallic_glass_cleaned.csv"
df.to_csv(output_file, index=False)

# Check the corrected value
print("\nAfter correction:")
print(df.loc[370, ["Composition", "Tg_[K]", "Tx_[K]", "Tl_[K]", "Dmax_[mm]"]])

df["Tx_Flag"] = ""
if df.loc[780,"Tx_[K]"] == 75:
    df.loc[780,"Tx_Flag"] = "Unverified - Tx value reported as 75 K"
    print("Row 780 flagged successfully.")
    print("Tx = 75 K was NOT changed.")
else:
    print("Row 780 does not contain Tx = 75 K.")


output_file = "C:\\Users\\Kaithoju Hasini\\OneDrive\\Documents\\metallic_glass_project\\data\\cleaned\\metallic_glass_cleaned.csv"
df.to_csv(output_file, index=False)

print(df.loc[780, ["Composition", "Tg_[K]", "Tx_[K]", "Tl_[K]", "Dmax_[mm]", "Tx_Flag"]])

print("\nCleaned dataset saved successfully!")