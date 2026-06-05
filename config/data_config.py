from typing import TypedDict

# Files and directories
RAW_DATA_PATH = "data/raw/"
PROCESSED_DATA_PATH = "data/processed/"

PROCESSED_TICKER_PRICE = "processed_ticker_prices"
RAW_TICKER_PRICE = "raw_ticker_prices"


# Yahoo Finance Input Data
TICKERS = [
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
    "XOM",
    "CVX",
    "COP",
    "SLB",
    "EOG",
]


# Input details
class Config(TypedDict):
    start: str
    end: str
    tickers: list[str]
    min_history: int
    max_nan_pct: float


CONFIG: Config = {
    "start": "2000-01-01",
    "end": "2025-12-31",
    "tickers": TICKERS,
    "min_history": 251,
    "max_nan_pct": 0.1,
}
