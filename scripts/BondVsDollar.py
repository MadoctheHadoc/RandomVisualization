import yfinance as yf
import matplotlib.pyplot as plt
import pandas as pd

def fetch_data(period="6mo"):
    # US 10-Year Treasury Note (^TNX) and US Dollar Index (DX-Y.NYB or DXY)
    bond = yf.Ticker("^TNX")
    dollar = yf.Ticker("DX-Y.NYB")

    bond_hist = bond.history(period=period)
    dollar_hist = dollar.history(period=period)

    return bond_hist['Close'], dollar_hist['Close']


def plot_data(bond_data, dollar_data):
    fig, ax1 = plt.subplots(figsize=(10, 5))

    ax1.set_xlabel("Date")
    ax1.set_ylabel("10Y Bond Yield (%)", color="blue")
    l1, = ax1.plot(bond_data.index, bond_data, color="blue", label="10Y Bond Yield")
    ax1.tick_params(axis='y', labelcolor="blue")

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    ax2.set_ylabel("Dollar Index (DXY)", color="green")
    l2, = ax2.plot(dollar_data.index, dollar_data, color="green", label="Dollar Index")
    ax2.tick_params(axis='y', labelcolor="green")

    plt.title("US 10-Year Bond Yield vs Dollar Index (Last 6 Months)")
    fig.tight_layout()
    fig.legend(handles=[l1, l2], loc="upper left")
    plt.grid(True)
    plt.show()

def plot_ratio(bond_data, dollar_data):
    bond_series = bond_data.copy()
    dollar_series = dollar_data.copy()
    
    bond_series.index = bond_series.index.tz_localize(None).date
    dollar_series.index = dollar_series.index.tz_localize(None).date

    bond_df = pd.DataFrame({"bond": bond_series})
    dollar_df = pd.DataFrame({"dollar": dollar_series})
    df = pd.merge(bond_df, dollar_df, left_index=True, right_index=True, how='inner')

    # df.replace([0.0, float("inf"), -float("inf")], pd.NA, inplace=True)
    # df.dropna(inplace=True)
    df["ratio"] = df["bond"] / df["dollar"]

    plt.figure(figsize=(10, 5))
    plt.plot(df.index, df["ratio"], color="purple")
    plt.title("Ratio: 10Y Bond Yield / Dollar Index (Cleaned, Last 6 Months)")
    plt.xlabel("Date")
    plt.ylabel("Ratio")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def main():
    period = input("Enter time period (e.g: 5d, 3mo, 2y, max): ")
    bond_data, dollar_data = fetch_data(period)
    print("Latest 10Y Bond Yield:", bond_data[-1])
    print("Latest Dollar Index:", dollar_data[-1])
    plot_data(bond_data, dollar_data)
    plot_ratio(bond_data, dollar_data)

if __name__ == "__main__":
    main()
