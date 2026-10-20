
import math

import pandas as pd
import yfinance as yf
from langchain_core.tools import tool


def _clean_value(value):
    """Convert common Pandas and NumPy values to JSON-friendly values."""
    if value is None:
        return None

    if hasattr(value, "item"):
        try:
            value = value.item()
        except (ValueError, AttributeError):
            pass

    if pd.isna(value):
        return None

    if isinstance(value, float) and not math.isfinite(value):
        return None

    if hasattr(value, "isoformat"):
        return value.isoformat()

    if isinstance(value, (str, int, float, bool)):
        return value

    return str(value)


def _dataframe_to_records(dataframe, limit=20):
    """Convert a DataFrame to a list of JSON-friendly dictionaries."""
    if dataframe is None or dataframe.empty:
        return []

    records = []

    for index, row in dataframe.head(limit).iterrows():
        record = {"date": _clean_value(index)}

        for column, value in row.items():
            record[str(column)] = _clean_value(value)

        records.append(record)

    return records


@tool
def company_financial_snapshot(ticker: str) -> dict:
    """Retrieve a company profile and basic financial snapshot from Yahoo Finance.

    Example ticker symbols: NVDA, MSFT, AAPL.
    """
    symbol = ticker.strip().upper()

    if not symbol:
        return {"error": "Please provide a valid stock ticker."}

    try:
        company = yf.Ticker(symbol)
        info = company.info or {}

        fields = [
            "longName",
            "shortName",
            "symbol",
            "sector",
            "industry",
            "country",
            "website",
            "currency",
            "marketCap",
            "currentPrice",
            "regularMarketPrice",
            "previousClose",
            "totalRevenue",
            "grossProfits",
            "ebitda",
            "netIncomeToCommon",
            "totalDebt",
            "totalCash",
            "trailingPE",
            "forwardPE",
            "priceToBook",
            "returnOnEquity",
            "profitMargins",
            "dividendYield",
            "fiftyTwoWeekHigh",
            "fiftyTwoWeekLow",
        ]

        snapshot = {
            field: _clean_value(info.get(field))
            for field in fields
        }

        return {
            "ticker": symbol,
            "company": snapshot,
            "source": "Yahoo Finance",
            "source_url": (
                f"https://finance.yahoo.com/quote/{symbol}/"
            ),
            "disclaimer": (
                "Values may be unavailable, delayed, or subject to "
                "revision. Verify important figures against official "
                "company filings."
            ),
        }

    except Exception as exc:
        return {
            "ticker": symbol,
            "error": f"Could not retrieve the financial snapshot: {exc}",
        }


@tool
def company_financial_statements(ticker: str) -> dict:
    """Retrieve annual income statement, balance sheet, and cash flow data."""
    symbol = ticker.strip().upper()

    if not symbol:
        return {"error": "Please provide a valid stock ticker."}

    try:
        company = yf.Ticker(symbol)

        return {
            "ticker": symbol,
            "income_statement": _dataframe_to_records(
                company.financials
            ),
            "balance_sheet": _dataframe_to_records(
                company.balance_sheet
            ),
            "cash_flow": _dataframe_to_records(
                company.cashflow
            ),
            "source": "Yahoo Finance",
            "source_url": (
                f"https://finance.yahoo.com/quote/{symbol}/financials/"
            ),
        }

    except Exception as exc:
        return {
            "ticker": symbol,
            "error": f"Could not retrieve financial statements: {exc}",
        }


@tool
def historical_stock_prices(
    ticker: str,
    period: str = "1y",
    interval: str = "1mo",
) -> dict:
    """Retrieve historical stock prices for a ticker.

    Common periods include 1mo, 3mo, 6mo, 1y, 2y, and 5y.
    Common intervals include 1d, 1wk, and 1mo.
    """
    symbol = ticker.strip().upper()

    allowed_periods = {
        "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "max"
    }
    allowed_intervals = {"1d", "1wk", "1mo"}

    if not symbol:
        return {"error": "Please provide a valid stock ticker."}

    if period not in allowed_periods:
        return {
            "error": (
                f"Unsupported period: {period}. "
                f"Choose from {sorted(allowed_periods)}."
            )
        }

    if interval not in allowed_intervals:
        return {
            "error": (
                f"Unsupported interval: {interval}. "
                f"Choose from {sorted(allowed_intervals)}."
            )
        }

    try:
        company = yf.Ticker(symbol)

        history = company.history(
            period=period,
            interval=interval,
            auto_adjust=False,
        )

        return {
            "ticker": symbol,
            "period": period,
            "interval": interval,
            "result_count": len(history),
            "prices": _dataframe_to_records(
                history.reset_index().set_index(
                    history.index.name or "Date"
                ) if not history.empty else history
            ),
            "source": "Yahoo Finance",
            "source_url": (
                f"https://finance.yahoo.com/quote/{symbol}/history/"
            ),
        }

    except Exception as exc:
        return {
            "ticker": symbol,
            "error": f"Could not retrieve historical prices: {exc}",
        }

