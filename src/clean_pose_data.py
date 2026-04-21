import pandas as pd

# Load raw pose CSV
df = pd.read_csv("data/fitness_poses_csvs_out_full_list.csv")

print("Original shape:", df.shape)

# ------------------------------------------------------------
# STEP 1 — Identify columns
# ------------------------------------------------------------
# First column = image filename
image_col = df.columns[0]

# Second column = label column
label_col = df.columns[1]

print("Image column:", image_col)
print("Label column:", label_col)

# ------------------------------------------------------------
# STEP 2 — Drop the image filename column (we don't need it)
# ------------------------------------------------------------
df = df.drop(columns=[image_col])

# Rename label column to something cleaner
df = df.rename(columns={label_col: "label"})

# ------------------------------------------------------------
# STEP 3 — Convert all remaining columns to float
# ------------------------------------------------------------
for col in df.columns:
    if col != "label":
        df[col] = pd.to_numeric(df[col], errors="coerce")

# ------------------------------------------------------------
# STEP 4 — Handle missing values
# ------------------------------------------------------------
df = df.dropna()    # remove bad rows
print("After dropna:", df.shape)

# ------------------------------------------------------------
# STEP 5 — Show details
# ------------------------------------------------------------
print(df.head())
print(df.info())

# ------------------------------------------------------------
# STEP 6 — Save cleaned CSV
# ------------------------------------------------------------
df.to_csv("data/clean_pose.csv", index=False)
print("Saved cleaned file to data/clean_pose.csv")