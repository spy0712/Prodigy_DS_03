import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt


def load_bank_dataset():
    local_csv = "bank.csv"
    alternate_csv = "bank-additional-full.csv"
    if os.path.exists(local_csv):
        return pd.read_csv(local_csv)
    if os.path.exists(alternate_csv):
        return pd.read_csv(alternate_csv, sep=';')

    zip_url = (
        "https://archive.ics.uci.edu/ml/machine-learning-databases/00222/"
        "bank-additional.zip"
    )
    try:
        import urllib.request
        import zipfile
        from io import BytesIO

        with urllib.request.urlopen(zip_url, timeout=20) as response:
            content = response.read()

        with zipfile.ZipFile(BytesIO(content)) as archive:
            with archive.open("bank-additional/bank-additional-full.csv") as csv_file:
                return pd.read_csv(csv_file, sep=';')
    except Exception as exc:
        raise RuntimeError(
            "Could not load the bank marketing dataset. "
            "Ensure 'bank.csv' or 'bank-additional-full.csv' exists locally, "
            "or verify network access to download the dataset." 
            f"Original error: {exc}"
        ) from exc


# Load Dataset
df = load_bank_dataset()

# Encode categorical and string columns
for col in df.select_dtypes(include=['object', 'string']).columns:
    if col == 'y':
        continue
    df[col] = LabelEncoder().fit_transform(df[col].astype(str))

if 'y' not in df.columns:
    raise ValueError("Target column 'y' is missing from the dataset.")

# Normalize and encode target column
raw_y = df['y'].astype(str).str.strip().str.lower()
if set(raw_y.dropna().unique()) <= {'yes', 'no'}:
    df['y'] = raw_y.map({'yes': 1, 'no': 0})
else:
    df['y'] = LabelEncoder().fit_transform(raw_y)

# Features and Target
X = df.drop('y', axis=1)
y = df['y']

# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train Model
model = DecisionTreeClassifier(max_depth=4)
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Evaluation
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Visualize Tree
plt.figure(figsize=(15,10))
plot_tree(model, feature_names=X.columns,
          class_names=['No','Yes'],
          filled=True)
plt.show()
