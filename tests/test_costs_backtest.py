from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from neural_a_share.backtest import PortfolioBacktester, estimate_cost
from neural_a_share.config import PortfolioConfig


def _test_config(**kwargs) -> PortfolioConfig:
    base = PortfolioConfig(
        top_k=1,
        rebalance_every=1,
        initial_cash=100_000.0,
        max_weight=1.0,
        max_participation=1.0,
        fixed_slippage_bps=0.0,
        impact_coefficient=0.0,
        max_impact_bps=0.0,
    )
    return replace(base, **kwargs)


def test_configured_stock_and_etf_costs_are_exact() -> None:
    config = _test_config()
    stock_buy = estimate_cost(100_000, "stock", "BUY", config, 10_000_000)
    stock_sell = estimate_cost(100_000, "stock", "SELL", config, 10_000_000)
    etf_sell = estimate_cost(100_000, "etf", "SELL", config, 10_000_000)
    assert stock_buy.commission == pytest.approx(100_000 * 0.00008499999)
    assert stock_sell.stamp_duty == pytest.approx(100_000 * 0.0005)
    assert etf_sell.commission == pytest.approx(100_000 * 0.00005000001)
    assert etf_sell.stamp_duty == 0


def test_signal_at_close_executes_no_earlier_than_t_plus_one(small_market) -> None:
    bars, predictions = small_market
    result = PortfolioBacktester(_test_config()).run(bars, predictions)
    first_signal = pd.to_datetime(predictions["trade_date"]).min()
    first_trade = result.trades.iloc[0]
    assert first_trade["signal_date"] == first_signal
    assert first_trade["execution_date"] > first_signal


def test_position_ledger_and_nav_mark_to_market(small_market) -> None:
    bars, predictions = small_market
    result = PortfolioBacktester(_test_config()).run(bars, predictions)
    assert not result.position_ledger.empty
    assert np.allclose(result.nav["nav"], result.nav["cash"] + result.nav["market_value"])
    assert (result.nav["nav"] > 0).all()
    assert {"shares", "market_value", "unrealized_pnl", "status"} <= set(result.position_ledger)


def test_limit_down_exit_is_deferred_and_actual_date_recorded(small_market) -> None:
    bars, predictions = small_market
    dates = sorted(bars["trade_date"].unique())
    # AAA drops from Top-K at signal dates[2], so the first sell attempt is dates[3].
    for index in (3, 4, 5):
        date = dates[index]
        mask = (bars["symbol"] == "AAA.SH") & (bars["trade_date"] == date)
        previous = float(bars.loc[mask, "prev_close"].iloc[0])
        locked = previous * 0.90
        bars.loc[mask, ["open", "high", "low", "close"]] = locked
        # Keep the raw chain internally consistent for the next session.
        if index + 1 < len(dates):
            next_mask = (bars["symbol"] == "AAA.SH") & (bars["trade_date"] == dates[index + 1])
            bars.loc[next_mask, "prev_close"] = locked
    result = PortfolioBacktester(_test_config()).run(bars, predictions)
    deferred = result.exit_events[(result.exit_events["symbol"] == "AAA.SH") & (result.exit_events["status"] == "DEFERRED")]
    filled = result.exit_events[(result.exit_events["symbol"] == "AAA.SH") & (result.exit_events["status"] == "FILLED")]
    assert len(deferred) >= 3
    assert not filled.empty
    assert pd.Timestamp(filled.iloc[0]["actual_exit_date"]) == pd.Timestamp(dates[6])


def test_exit_stays_in_nav_after_ten_failed_sessions(small_market) -> None:
    bars, predictions = small_market
    dates = sorted(bars["trade_date"].unique())
    for index in range(3, 15):
        date = dates[index]
        mask = (bars["symbol"] == "AAA.SH") & (bars["trade_date"] == date)
        previous = float(bars.loc[mask, "prev_close"].iloc[0])
        locked = previous * 0.90
        bars.loc[mask, ["open", "high", "low", "close"]] = locked
        if index + 1 < len(dates):
            next_mask = (bars["symbol"] == "AAA.SH") & (bars["trade_date"] == dates[index + 1])
            bars.loc[next_mask, "prev_close"] = locked
    result = PortfolioBacktester(_test_config(max_exit_deferral_days=10)).run(bars, predictions)
    unresolved = result.exit_events[
        (result.exit_events["symbol"] == "AAA.SH") & (result.exit_events["status"] == "UNRESOLVED")
    ]
    assert len(unresolved) == 1
    assert result.position_ledger["status"].eq("LOCKED_UNRESOLVED").any()
