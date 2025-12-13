import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from datetime import timedelta

class CustomLinearModel:
    def __init__(self, include_binary=False, special_dates=None):
        self.include_binary = include_binary
        self.special_dates = special_dates
        self.coef_ = None
        self.intercept_ = None

    def _add_binary_feature(self, X, dates):
        # Convert both to date only (strip time) to avoid mismatches
        dates_only = pd.to_datetime(dates).normalize()
        special_dates_only = pd.to_datetime(self.special_dates).normalize()
        
        # Create binary feature
        binary_feature = np.isin(dates_only, special_dates_only).astype(float)

        # Debug: print first few values
        print(binary_feature)

        # Append to X
        X_mod = np.column_stack((X, binary_feature))
        return X_mod

    def fit(self, X, y, dates=None):
        X_mod = np.copy(X)
        if self.include_binary:
            if dates is None:
                raise ValueError("dates must be provided if include_binary=True")
            X_mod = self._add_binary_feature(X_mod, dates)

        X_mod = np.column_stack((np.ones(X_mod.shape[0]), X_mod))
        beta = np.linalg.pinv(X_mod.T @ X_mod) @ (X_mod.T @ y)

        self.intercept_ = beta[0]
        self.coef_ = beta[1:]
        return self

    def predict(self, X, dates=None):
        X_mod = np.copy(X)
        if self.include_binary:
            if dates is None:
                raise ValueError("dates must be provided if include_binary=True")
            X_mod = self._add_binary_feature(X_mod, dates)
        return self.intercept_ + X_mod @ self.coef_


# Load data
df = pd.read_csv("/Users/dervint/Desktop/UMICH_25/Fall_2025/DATASCI-406/final-project/data/sp500.csv")
list_of_dates = pd.read_csv("/Users/dervint/Desktop/UMICH_25/Fall_2025/DATASCI-406/final-project/data/political_events.csv")
election_dates = pd.to_datetime(list_of_dates["date"])

# Expand binary feature to ±3 days around each election
expanded_special_dates = []
for d in election_dates:
    for offset in range(-3, 4):
        expanded_special_dates.append(d + timedelta(days=offset))
expanded_special_dates = pd.to_datetime(expanded_special_dates)

df.drop(columns=['Unnamed: 0'], inplace=True)
df['Return'] = np.log(df['Close'] / df['Close'].shift(1))
df["Date"] = pd.to_datetime(df["Date"])
df = df.sort_values("Date")
df = df.replace([np.inf, -np.inf], np.nan).dropna()
df["Target"] = df["Return"].shift(-1)
df = df.dropna()

features = ["Close", "High", "Low", "Open", "Volume", "Return"]
X = df[features]
y = df["Target"]

train_size = int(len(df) * 0.8)
X_train, X_test = X.iloc[:train_size], X.iloc[train_size:]
y_train, y_test = y.iloc[:train_size], y.iloc[train_size:]
all_dates = df["Date"].tolist()
train_dates = all_dates[:train_size]
test_dates = all_dates[train_size:]

models = {
    "Standard Linear": CustomLinearModel(include_binary=False),
    "Binary Feature Linear": CustomLinearModel(include_binary=True, special_dates=expanded_special_dates)
}

window_size = 3
predictions_window = {name: [] for name in models.keys()}
dates_window = []

for i in range(len(X_test)):
    start_idx = max(0, train_size - window_size + i)
    end_idx = train_size + i

    X_window_train = X.iloc[start_idx:train_size + i]
    y_window_train = y.iloc[start_idx:train_size + i]
    dates_window_train = all_dates[start_idx:train_size + i]

    X_window_test = X_test.iloc[i:i+1]
    dates_window_test = all_dates[train_size + i:train_size + i + 1]

    dates_window.append(dates_window_test[0])

    model_objects = {}
    for name, model in models.items():
        model.fit(X_window_train, y_window_train, dates_window_train)
        y_pred = model.predict(X_window_test, dates_window_test)
        predictions_window[name].append(y_pred[0])
        model_objects[name] = model

    # Print coefficients per window
    for name, model in model_objects.items():
        if model.include_binary:
            print(f"{name} binary feature coefficient:", model.coef_[-1])
        else:
            print(f"{name} has no binary feature.")

# Evaluate predictions
results = []
for name, y_pred_window in predictions_window.items():
    y_pred_window = np.array(y_pred_window)
    mae = mean_absolute_error(y_test, y_pred_window)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred_window))
    r2 = r2_score(y_test, y_pred_window)
    accuracy = r2 * 100
    results.append([name, mae, rmse, r2, accuracy])
    # Correctly reference the last model object
    last_model = model_objects[name]
    if last_model.include_binary:
        print(f"{name} binary feature coefficient (last window):", last_model.coef_[-1])
    else:
        print(f"{name} has no binary feature (last window).")

results_df = pd.DataFrame(results, columns=["Model", "MAE", "RMSE", "R2 Score", "Accuracy (%)"])
print("\nModel Comparison Results (Rolling Window):\n")
print(results_df)

# Plot
plt.figure(figsize=(12, 6))
plt.plot(dates_window, y_test, label="Actual", color="black")
for name, y_pred_window in predictions_window.items():
    plt.plot(dates_window, y_pred_window, label=name, alpha=0.7)
plt.title(f"Actual vs Predicted Returns (Window size={window_size})")
plt.xlabel("Date")
plt.ylabel("Return")
plt.legend()
plt.show()
