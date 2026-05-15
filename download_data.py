import yfinance as yf
import pandas as pd
import os
from datetime import datetime, timedelta

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
os.makedirs(DATA_DIR, exist_ok=True)

STOCKS = {
    "AI": {
        "NVDA": "NVIDIA",
        "MSFT": "Microsoft",
        "GOOGL": "Google",
        "META": "Meta",
        "AMZN": "Amazon",
        "CRM": "Salesforce",
        "PLTR": "Palantir",
        "AI": "C3.ai",
        "SNOW": "Snowflake",
        "IBM": "IBM",
        "SMCI": "Super Micro Computer",
        "ORCL": "Oracle",
    },
    "Chip": {
        "AMD": "AMD",
        "INTC": "Intel",
        "TSM": "TSMC",
        "AVGO": "Broadcom",
        "QCOM": "Qualcomm",
        "TXN": "Texas Instruments",
        "MU": "Micron",
        "MRVL": "Marvell",
        "ARM": "Arm Holdings",
        "ASML": "ASML",
    },
    "NewEnergy": {
        "TSLA": "Tesla",
        "ENPH": "Enphase Energy",
        "SEDG": "SolarEdge",
        "FSLR": "First Solar",
        "RUN": "Sunrun",
        "PLUG": "Plug Power",
        "BE": "Bloom Energy",
        "NEE": "NextEra Energy",
        "RIVN": "Rivian",
        "LCID": "Lucid",
        "NIO": "NIO",
        "LI": "Li Auto",
        "XPEV": "XPeng",
    },
    "Index": {
        "^GSPC": "SP500",
        "^IXIC": "NASDAQ",
        "^DJI": "DowJones",
        "^RUT": "Russell2000",
        "^VIX": "VIX",
        "^SOX": "PHLX_Semiconductor",
    },
}

end_date = datetime.now()
start_date = end_date - timedelta(days=365)

all_records = []
failed = []

for sector, tickers in STOCKS.items():
    print(f"\n{'='*50}")
    print(f"Downloading {sector} sector...")
    print(f"{'='*50}")

    for ticker, name in tickers.items():
        safe_name = ticker.replace("^", "")
        print(f"  {ticker:8s} ({name})... ", end="", flush=True)
        try:
            df = yf.download(ticker, start=start_date, end=end_date, progress=False)
            if df.empty:
                print("NO DATA")
                failed.append((ticker, name, "empty"))
                continue

            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.droplevel(1)

            df.index.name = "Date"
            df = df.round(2)

            csv_path = os.path.join(DATA_DIR, f"{safe_name}.csv")
            df.to_csv(csv_path)

            all_records.append({
                "Ticker": ticker,
                "Name": name,
                "Sector": sector,
                "Records": len(df),
                "Start": df.index.min().strftime("%Y-%m-%d"),
                "End": df.index.max().strftime("%Y-%m-%d"),
                "File": f"{safe_name}.csv",
            })

            print(f"{len(df)} rows  [{df.index.min().strftime('%Y-%m-%d')} ~ {df.index.max().strftime('%Y-%m-%d')}]")
        except Exception as e:
            print(f"FAILED: {e}")
            failed.append((ticker, name, str(e)))

summary = pd.DataFrame(all_records)
summary.to_csv(os.path.join(DATA_DIR, "_summary.csv"), index=False)

print(f"\n{'='*50}")
print(f"DONE: {len(all_records)} stocks downloaded to data/")
if failed:
    print(f"FAILED ({len(failed)}):")
    for t, n, reason in failed:
        print(f"  {t} ({n}): {reason}")
print(f"{'='*50}")
