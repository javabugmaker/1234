from __future__ import annotations

from types import SimpleNamespace

import pandas as pd

from neural_a_share.config import TickFlowConfig
from neural_a_share.data.quality import check_bars
from neural_a_share.data.tickflow import TickFlowFreeClient


class FakeExchanges:
    def get_instruments(self, exchange, instrument_type=None):
        if exchange != "SH" or instrument_type != "stock":
            return []
        return [
            {
                "symbol": "600000.SH",
                "code": "600000",
                "name": "浦发银行",
                "exchange": "SH",
                "region": "CN",
                "type": "stock",
                "ext": {"listing_date": "1999-11-10"},
            }
        ]


class FakeKlines:
    def batch(self, symbols, **kwargs):
        assert kwargs["adjust"] == "none"
        dates = pd.bdate_range("2024-01-02", periods=3, tz="Asia/Shanghai")
        payload = {
            "timestamp": [int(date.timestamp() * 1000) for date in dates],
            "open": [10.0, 10.1, 10.2],
            "high": [10.2, 10.3, 10.4],
            "low": [9.9, 10.0, 10.1],
            "close": [10.1, 10.2, 10.3],
            "prev_close": [10.0, 10.1, 10.2],
            "volume": [1000, 1100, 1200],
            "amount": [10000, 11220, 12360],
        }
        return {symbol: payload for symbol in symbols}


def test_tickflow_free_adapter_uses_sdk_and_normalizes_data() -> None:
    sdk = SimpleNamespace(exchanges=FakeExchanges(), klines=FakeKlines())
    config = TickFlowConfig(exchanges=("SH",), instrument_types=("stock",), benchmark="600000.SH")
    client = TickFlowFreeClient(config, sdk=sdk)
    result = client.update()
    assert result.symbols_requested == 1
    assert result.rows_received == 3
    assert result.bars["trade_date"].dt.tz is None
    assert result.catalog.iloc[0]["name"] == "浦发银行"


def test_tickflow_completeness_detects_latest_partial_cross_section(synthetic_bars) -> None:
    latest = synthetic_bars["trade_date"].max()
    partial = synthetic_bars[~((synthetic_bars["trade_date"] == latest) & (synthetic_bars["symbol"] != "000300.SH"))]
    report = check_bars(partial, minimum_latest_coverage=0.75)
    assert report.status == "FAIL"
    assert any(issue.code == "INCOMPLETE_LATEST" for issue in report.issues)
