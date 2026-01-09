import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# ==========================
# CONFIG
# ==========================
INITIAL_CAPITAL = 100000
BUY_HOLD_CAPITAL = 50000
REQUIRED_COLUMNS = {"Date", "Open", "Close"}


# ==========================
# DATA INPUT
# ==========================
def load_excel_data(file_path: str) :
    try:
        df = pd.read_excel(file_path)
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {file_path}")

    missing_cols = REQUIRED_COLUMNS - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing columns {missing_cols} in {file_path}")

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce", dayfirst=True)
    df = df.dropna(subset=["Date"])

    return (
        df.set_index("Date")
          .sort_index()
          .drop(columns=["High", "Low"])
    )


# ==========================
# RESAMPLING OF MONTHLY DATA
# ==========================
def monthly_transform(df: pd.DataFrame, prefix: str) :
    monthly = (
        df.resample("ME")
          .agg({"Open": "first", "Close": "last"})
    )

    monthly.rename(
        columns={
            "Open": f"{prefix}Open",
            "Close": f"{prefix}Close"
        },
        inplace=True
    )

    monthly[f"{prefix}_returns"] = (
        monthly[f"{prefix}Close"]
        .pct_change(fill_method=None)
        .mul(100)
        .round(2)
    )

    return monthly.dropna()


# ==========================
# JOINING THE DATAFRAMES
# ==========================
def merge_data(df1: pd.DataFrame, df2: pd.DataFrame) :
    df = df1.join(df2, how="inner")
    if df.empty:
        raise ValueError("No overlapping dates after merge.")
    return df


# ==========================
# CAPITAL ALLOCATION WITH MY STRATEGY
# ==========================
def capitalallocation(df: pd.DataFrame):
    df = df.copy()

    conditions_1 = [
        df["df1_returns"] < df["df2_returns"],
        df["df1_returns"] > df["df2_returns"]
    ]

    choices_1 = [INITIAL_CAPITAL, 0]

    df["capital_allocation_1"] = np.select(
        conditions_1, choices_1, default=0
    )

    conditions_2 = [
        df["df1_returns"] > df["df2_returns"],
        df["df1_returns"] < df["df2_returns"]
    ]

    choices_2 = [INITIAL_CAPITAL, 0]

    df["capital_allocation_2"] = np.select(
        conditions_2, choices_2, default=0
    )

    return df


# ==========================
# NAV UNITS 
# ==========================
def calculate_units(df: pd.DataFrame) :
    df = df.copy()

    df["navunits1"] = (df["capital_allocation_1"] / df["df1Close"]).fillna(0)
    df["navunits2"] = (df["capital_allocation_2"] / df["df2Close"]).fillna(0)

    df["bhunits1"] = BUY_HOLD_CAPITAL / df["df1Close"]
    df["bhunits2"] = BUY_HOLD_CAPITAL / df["df2Close"]

    return df.round(0)


# ==========================
# NAV CALCULATION
# ==========================
def calculate_nav(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["strategy_nav"] = (
        df["navunits1"].cumsum() * df["df1Close"]
        + df["navunits2"].cumsum() * df["df2Close"]
    )

    df["buy_hold_nav"] = (
        df["bhunits1"].cumsum() * df["df1Close"]
        + df["bhunits2"].cumsum() * df["df2Close"]
    )

    return df


# ==========================
# PLOT
# ==========================
def plot_capital_growth(df: pd.DataFrame) :
    plt.figure(figsize=(10, 5))
    plt.plot(df.index, df["strategy_nav"], label="Rotation Strategy")
    plt.plot(df.index, df["buy_hold_nav"], label="Buy & Hold")

    plt.title("Capital Growth Comparison")
    plt.xlabel("Date")
    plt.ylabel("Portfolio Value in lacs")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# ==========================
# MAIN PROGRAMM
# ==========================
def run_backtest(asset1_path: str, asset2_path: str) :
    df1 = load_excel_data(asset1_path)
    df2 = load_excel_data(asset2_path)

    df1_m = monthly_transform(df1, "df1")
    df2_m = monthly_transform(df2, "df2")

    df = merge_data(df1_m, df2_m)
    df = capitalallocation(df)
    df = calculate_units(df)
    df = calculate_nav(df)

    return df


# ==========================
# LAST PART OF PROGRAM  *****************************please add dataframes **************************************
# ==========================
if __name__ == "__main__":
    result = run_backtest(
        # "dataframe1...........",
        # "dataframe2..........."
    )

    print("Final Strategy Value:", round(result["strategy_nav"].iloc[-1], 0))
    print("Final Buy & Hold Value:", round(result["buy_hold_nav"].iloc[-1], 0))

    plot_capital_growth(result)