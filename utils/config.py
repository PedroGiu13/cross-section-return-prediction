from typing import TypedDict

# Paths
RAW_DATA_PATH = "data/raw/"
PROCESSED_DATA_PATH = "data/processed/"

TICKERS = [
    # Technology
    "AAPL",
    "MSFT",
    "NVDA",
    "GOOGL",
    "META",
    "AVGO",
    "ORCL",
    "CRM",
    "AMD",
    "INTC",
    # Financials
    "JPM",
    "BAC",
    "WFC",
    "GS",
    "MS",
    "BLK",
    "C",
    "AXP",
    "USB",
    "PNC",
    # Healthcare
    "JNJ",
    "UNH",
    "PFE",
    "ABBV",
    "MRK",
    "TMO",
    "ABT",
    "DHR",
    "BMY",
    "AMGN",
    # Consumer
    "AMZN",
    "TSLA",
    "HD",
    "MCD",
    "NKE",
    "SBUX",
    "TGT",
    "COST",
    "LOW",
    "TJX",
    # Industrials
    "CAT",
    "HON",
    "UPS",
    "BA",
    "GE",
    "MMM",
    "RTX",
    "LMT",
    "DE",
    "EMR",
    # Energy
    "XOM",
    "CVX",
    "COP",
    "SLB",
    "EOG",
]


class Config(TypedDict):
    start: str
    end: str
    tickers: list[str]
    min_history_months: int
    max_nan_pct: float


CONFIG: Config = {
    "start": "2000-01-01",
    "end": "2025-12-31",
    "tickers": TICKERS,
    "min_history_months": 24,
    "max_nan_pct": 0.1,
}
