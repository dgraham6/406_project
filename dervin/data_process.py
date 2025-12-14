# ...existing code...
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


class CustomLinearModel:
    def __init__(self, include_binary: bool = False, special_dates=None):
        """
        include_binary: whether to include a binary feature for special days
        special_dates: list/array of dates (e.g., ['2025-01-01', '2025-01-15'])
        """
        self.include_binary = include_binary
        self.special_dates = pd.to_datetime(special_dates) if special_dates is not None else None
        self.coef_ = None
        self.intercept_ = None

    def _add_binary_feature(self, X: np.ndarray, dates):
        """Add a binary column that flags rows whose date is in special_dates."""
        n = X.shape[0]
        binary_feature = np.zeros(n, dtype=float)
        if self.special_dates is not None:
            dates_arr = pd.to_datetime(dates)
            binary_feature = np.isin(dates_arr, self.special_dates).astype(float)
        X_mod = np.column_stack((X, binary_feature))
        return X_mod

    def fit(self, X, y, dates=None):
        """
        Fit using ordinary least squares.
        X: array-like (n_samples, n_features)
        y: array-like (n_samples,)
        dates: array-like of datetimes (required if include_binary=True)
        """
        X_mod = np.asarray(X)
        if self.include_binary:
            if dates is None:
                raise ValueError("dates must be provided if include_binary=True")
            X_mod = self._add_binary_feature(X_mod, dates)

        # add intercept column
        X_design = np.column_stack((np.ones(X_mod.shape[0]), X_mod))

        # OLS solution (using pseudo-inverse for stability)
        beta = np.linalg.pinv(X_design.T @ X_design) @ (X_design.T @ np.asarray(y))
        self.intercept_ = float(beta[0])
        self.coef_ = beta[1:]
        return self

    def predict(self, X, dates=None):
        X_mod = np.asarray(X)
        if self.include_binary:
            if dates is None:
                raise ValueError("dates must be provided if include_binary=True")
            X_mod = self._add_binary_feature(X_mod, dates)
        return self.intercept_ + X_mod @ self.coef_


if __name__ == "__main__":
    # file paths
    sp500_path = "/Users/dervint/Desktop/UMICH_25/Fall_2025/DATASCI-406/final-project/data/sp500.csv"
    events_path = "/Users/dervint/Desktop/UMICH_25/Fall_2025/DATASCI-406/final-project/data/political_events.csv"

    # load data
    df = pd.read_csv(sp500_path)
    events = pd.read_csv(events_path)

    # extract election dates
    election_dates = events.loc[events["event_type"] == "presidential_election", "date"].tolist()
    election_dates = pd.to_datetime(election_dates)

    # cleanup
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date")

    # compute returns and target (next day's close)
    df["Return"] = np.log(df["Close"] / df["Close"].shift(1))
    df["Target"] = df["Return"].shift(-1)

    # remove infinities and rows with NaNs
    df = df.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)

    features = ["Close", "High", "Low", "Open", "Volume"]
    X = df[features]
    y = df["Target"]
    dates = df["Date"]

    # train/test split
    train_size = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:train_size], X.iloc[train_size:]
    y_train, y_test = y.iloc[:train_size], y.iloc[train_size:]
    train_dates = dates.iloc[:train_size]
    test_dates = dates.iloc[train_size:]

    # quick check for any election dates in dataset
    for d in dates:
        if d in election_dates:
            print(f"Date in election date: {d}")

    # models to compare
    models = {
        "Standard Linear": CustomLinearModel(include_binary=False),
        "Binary Feature Linear": CustomLinearModel(include_binary=True, special_dates=election_dates),
    }

    results = []
    predictions = {}

    for name, model in models.items():
        model.fit(X_train, y_train, train_dates)
        y_pred = model.predict(X_test, test_dates)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        accuracy = r2 * 100

        results.append([name, mae, rmse, r2, accuracy])
        predictions[name] = y_pred

        if model.include_binary:
            # binary feature is the last coefficient
            print(f"{name} binary feature coefficient: {model.coef_[-1]}")

    results_df = pd.DataFrame(results, columns=["Model", "MAE", "RMSE", "R2 Score", "Accuracy (%)"])
    print("\nModel Comparison Results:\n")
    print(results_df)

    # plotting
    plt.figure(figsize=(12, 6))
    plt.plot(test_dates, y_test, label="Actual", color="black")
    for name, y_pred in predictions.items():
        plt.plot(test_dates, y_pred, label=name, alpha=0.7)
    plt.title("Actual vs Predicted Close Price")
    plt.xlabel("Date")
    plt.ylabel("Stock Price")
    plt.legend()
    plt.show()

    plt.figure(figsize=(12, 6))
    for name, y_pred in predictions.items():
        errors = y_test - y_pred
        plt.hist(errors, bins=50, alpha=0.5, label=name, density=True)
    plt.title("Error Distribution Across Models")
    plt.xlabel("Prediction Error")
    plt.ylabel("Density")
    plt.legend()
    plt.show()
# ...existing code...