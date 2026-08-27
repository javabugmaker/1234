from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def synthetic_bars() -> pd.DataFrame:
    dates = pd.bdate_range("2023-01-02", periods=340)
    symbols = ["000300.SH", *[f"{index:06d}.SZ" for index in range(1, 26)]]
    rows = []
    for symbol_index, symbol in enumerate(symbols):
        previous = 10.0 + symbol_index * 0.2
        for date_index, date in enumerate(dates):
            overnight = 0.0002 * np.sin(date_index / 7 + symbol_index)
            intraday = 0.0006 + 0.00015 * np.cos(date_index / 5 + symbol_index)
            open_price = previous * (1 + overnight)
            close = open_price * (1 + intraday)
            high = max(open_price, close) * 1.004
            low = min(open_price, close) * 0.996
            volume = 1_000_000 + symbol_index * 10_000 + date_index * 100
            rows.append(
                {
                    "symbol": symbol,
                    "trade_date": date,
                    "timestamp": int(pd.Timestamp(date, tz="Asia/Shanghai").timestamp() * 1000),
                    "open": open_price,
                    "high": high,
                    "low": low,
                    "close": close,
                    "prev_close": previous,
                    "volume": volume,
                    "amount": volume * close,
                }
            )
            previous = close
    return pd.DataFrame(rows)


@pytest.fixture
def small_market() -> tuple[pd.DataFrame, pd.DataFrame]:
    dates = pd.bdate_range("2024-01-02", periods=18)
    rows = []
    for symbol, initial in [("000300.SH", 100.0), ("AAA.SH", 10.0), ("BBB.SZ", 20.0)]:
        previous = initial
        for date in dates:
            open_price = previous
            close = open_price * 1.001
            rows.append(
                {
                    "symbol": symbol,
                    "trade_date": date,
                    "open": open_price,
                    "high": max(open_price, close) * 1.002,
                    "low": min(open_price, close) * 0.998,
                    "close": close,
                    "prev_close": previous,
                    "volume": 2_000_000,
                    "amount": 30_000_000.0,
                }
            )
            previous = close
    bars = pd.DataFrame(rows)
    prediction_rows = []
    for index, date in enumerate(dates[:-1]):
        leader = "AAA.SH" if index < 2 else "BBB.SZ"
        for symbol in ("AAA.SH", "BBB.SZ"):
            prediction_rows.append(
                {
                    "trade_date": date,
                    "symbol": symbol,
                    "NeuralAlpha": 0.05 if symbol == leader else 0.01,
                    "NeuralRank": 1 if symbol == leader else 2,
                    "Alpha20": 0.04,
                    "Alpha40": 0.05,
                    "Alpha60": 0.06,
                }
            )
    return bars, pd.DataFrame(prediction_rows)
