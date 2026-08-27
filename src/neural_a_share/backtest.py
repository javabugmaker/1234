from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Mapping

import numpy as np
import pandas as pd

from .config import PortfolioConfig


@dataclass
class Position:
    symbol: str
    instrument_type: str
    shares: float
    average_cost: float
    entry_date: pd.Timestamp
    last_raw_close: float
    exit_requested_date: pd.Timestamp | None = None
    exit_reason: str | None = None
    exit_deferrals: int = 0
    locked_unresolved: bool = False


@dataclass(frozen=True)
class TransactionCost:
    commission: float
    stamp_duty: float
    slippage: float
    impact: float

    @property
    def total(self) -> float:
        return self.commission + self.stamp_duty + self.slippage + self.impact


@dataclass
class BacktestResult:
    nav: pd.DataFrame
    trades: pd.DataFrame
    position_ledger: pd.DataFrame
    exit_events: pd.DataFrame
    metrics: dict[str, float]


def commission_rate(instrument_type: str, side: str, config: PortfolioConfig) -> float:
    is_fund = instrument_type.lower() in {"etf", "lof", "fund"}
    if is_fund:
        return config.fund_commission_buy if side == "BUY" else config.fund_commission_sell
    return config.stock_commission_buy if side == "BUY" else config.stock_commission_sell


def estimate_cost(
    notional: float,
    instrument_type: str,
    side: str,
    config: PortfolioConfig,
    market_amount: float,
) -> TransactionCost:
    gross = abs(float(notional))
    commission = max(float(config.minimum_commission), gross * commission_rate(instrument_type, side, config))
    stamp = (
        gross * config.stock_stamp_duty_sell
        if side == "SELL" and instrument_type.lower() not in {"etf", "lof", "fund"}
        else 0.0
    )
    participation = gross / market_amount if market_amount > 0 else 1.0
    impact_rate = min(
        config.max_impact_bps / 10_000.0,
        config.impact_coefficient * participation ** config.impact_exponent,
    )
    impact = gross * impact_rate
    slippage = gross * config.fixed_slippage_bps / 10_000.0
    return TransactionCost(commission, stamp, slippage, impact)


def _one_price(row: Mapping[str, Any]) -> bool:
    values = [float(row[field]) for field in ("open", "high", "low", "close")]
    return max(values) - min(values) <= max(1e-8, abs(values[0]) * 1e-8)


def _is_limit_down(row: Mapping[str, Any]) -> bool:
    if not _one_price(row):
        return False
    explicit = row.get("limit_down")
    if explicit is not None and pd.notna(explicit):
        return float(row["open"]) <= float(explicit) + 1e-8
    previous = row.get("prev_close")
    return bool(previous and pd.notna(previous) and float(row["open"]) / float(previous) <= 0.951)


def _is_limit_up(row: Mapping[str, Any]) -> bool:
    if not _one_price(row):
        return False
    explicit = row.get("limit_up")
    if explicit is not None and pd.notna(explicit):
        return float(row["open"]) >= float(explicit) - 1e-8
    previous = row.get("prev_close")
    return bool(previous and pd.notna(previous) and float(row["open"]) / float(previous) >= 1.049)


def _tradable_row(row: Mapping[str, Any] | None) -> bool:
    if row is None:
        return False
    return bool(
        pd.notna(row.get("open"))
        and float(row.get("open", 0)) > 0
        and float(row.get("volume", 0)) > 0
        and float(row.get("amount", 0)) > 0
    )


def _capacity_shares(row: Mapping[str, Any], config: PortfolioConfig) -> int:
    capacity_notional = float(row.get("amount", 0.0)) * float(config.max_participation)
    return max(0, int(capacity_notional / float(row["open"]) // config.lot_size) * config.lot_size)


def performance_metrics(nav: pd.DataFrame) -> dict[str, float]:
    if nav.empty:
        return {
            "total_return": float("nan"),
            "sharpe": float("nan"),
            "max_drawdown": float("nan"),
            "calmar": float("nan"),
        }
    series = nav["nav"].astype(float)
    returns = series.pct_change().dropna()
    total_return = float(series.iloc[-1] / series.iloc[0] - 1.0)
    elapsed = max(len(returns), 1)
    annual_return = float((1.0 + total_return) ** (252 / elapsed) - 1.0) if total_return > -1 else -1.0
    volatility = float(returns.std(ddof=1)) if len(returns) > 1 else float("nan")
    sharpe = float(returns.mean() / volatility * np.sqrt(252)) if volatility and np.isfinite(volatility) else float("nan")
    drawdown = series / series.cummax() - 1.0
    max_drawdown = float(drawdown.min())
    calmar = annual_return / abs(max_drawdown) if max_drawdown < 0 else float("nan")
    return {
        "total_return": total_return,
        "annual_return": annual_return,
        "sharpe": sharpe,
        "max_drawdown": max_drawdown,
        "calmar": float(calmar),
    }


class PortfolioBacktester:
    """Event-driven long-only backtester.

    Signals are observed after the close. Orders are first eligible at the next
    market session's open. Cash, real positions, executions, failed exits and
    daily mark-to-market snapshots are separate ledgers.
    """

    def __init__(self, config: PortfolioConfig) -> None:
        self.config = config

    def run(
        self,
        bars: pd.DataFrame,
        predictions: pd.DataFrame,
        metadata: pd.DataFrame | None = None,
        benchmark: str = "000300.SH",
    ) -> BacktestResult:
        required_predictions = {"symbol", "trade_date", "NeuralRank", "NeuralAlpha"}
        if missing := required_predictions - set(predictions):
            raise ValueError(f"predictions missing columns: {sorted(missing)}")
        market = bars.copy()
        market["trade_date"] = pd.to_datetime(market["trade_date"]).dt.normalize()
        market = market.sort_values(["trade_date", "symbol"]).drop_duplicates(
            ["trade_date", "symbol"], keep="last"
        )
        prediction_frame = predictions.copy()
        prediction_frame["trade_date"] = pd.to_datetime(prediction_frame["trade_date"]).dt.normalize()
        calendar = pd.DatetimeIndex(
            market.loc[market["symbol"].eq(benchmark), "trade_date"].drop_duplicates().sort_values()
        )
        if len(calendar) < 2:
            raise ValueError("benchmark calendar is missing or too short")
        price_lookup = {
            (pd.Timestamp(row.trade_date), row.symbol): row._asdict()
            for row in market.itertuples(index=False)
        }
        type_lookup: dict[str, str] = {}
        if metadata is not None and not metadata.empty:
            column = "instrument_type" if "instrument_type" in metadata else "type"
            if column in metadata:
                type_lookup = dict(zip(metadata["symbol"], metadata[column].fillna("stock")))
        signal_targets: dict[pd.Timestamp, list[str]] = {}
        signal_dates = set(prediction_frame["trade_date"])
        for index, date in enumerate(calendar[:-1]):
            if date not in signal_dates or index % int(self.config.rebalance_every) != 0:
                continue
            ranked = prediction_frame[prediction_frame["trade_date"].eq(date)].sort_values(
                ["NeuralRank", "symbol"]
            )
            target = ranked.head(int(self.config.top_k))["symbol"].tolist()
            signal_targets[calendar[index + 1]] = target

        cash = float(self.config.initial_cash)
        positions: dict[str, Position] = {}
        trades: list[dict[str, Any]] = []
        ledger: list[dict[str, Any]] = []
        exits: list[dict[str, Any]] = []
        nav_rows: list[dict[str, Any]] = []

        def record_exit_event(date: pd.Timestamp, position: Position, status: str, reason: str) -> None:
            exits.append(
                {
                    "requested_date": position.exit_requested_date,
                    "actual_exit_date": date if status == "FILLED" else pd.NaT,
                    "symbol": position.symbol,
                    "status": status,
                    "reason": reason,
                    "deferral_days": position.exit_deferrals,
                }
            )

        def execute(symbol: str, side: str, requested_shares: float, date: pd.Timestamp, reason: str) -> float:
            nonlocal cash
            row = price_lookup.get((date, symbol))
            if not _tradable_row(row):
                return 0.0
            if side == "SELL" and _is_limit_down(row):
                return 0.0
            if side == "BUY" and _is_limit_up(row):
                return 0.0
            available = _capacity_shares(row, self.config)
            shares = min(float(requested_shares), float(available))
            if side == "BUY":
                shares = int(shares // self.config.lot_size) * self.config.lot_size
            if shares <= 0:
                return 0.0
            instrument_type = type_lookup.get(symbol, "stock")
            open_price = float(row["open"])
            reference_notional = shares * open_price
            costs = estimate_cost(
                reference_notional,
                instrument_type,
                side,
                self.config,
                float(row.get("amount", 0.0)),
            )
            price_adjustment = (costs.slippage + costs.impact) / reference_notional
            execution_price = open_price * (1.0 + price_adjustment if side == "BUY" else 1.0 - price_adjustment)
            notional = shares * execution_price
            explicit_fees = costs.commission + costs.stamp_duty
            if side == "BUY":
                affordable = max(0.0, cash - explicit_fees)
                if notional > affordable:
                    shares = int(affordable / execution_price // self.config.lot_size) * self.config.lot_size
                    if shares <= 0:
                        return 0.0
                    reference_notional = shares * open_price
                    costs = estimate_cost(reference_notional, instrument_type, side, self.config, float(row["amount"]))
                    price_adjustment = (costs.slippage + costs.impact) / reference_notional
                    execution_price = open_price * (1 + price_adjustment)
                    notional = shares * execution_price
                    explicit_fees = costs.commission
                cash -= notional + explicit_fees
                current = positions.get(symbol)
                if current:
                    total_cost = current.shares * current.average_cost + notional + explicit_fees
                    current.shares += shares
                    current.average_cost = total_cost / current.shares
                else:
                    positions[symbol] = Position(
                        symbol=symbol,
                        instrument_type=instrument_type,
                        shares=shares,
                        average_cost=(notional + explicit_fees) / shares,
                        entry_date=date,
                        last_raw_close=float(row["close"]),
                    )
            else:
                current = positions[symbol]
                shares = min(shares, current.shares)
                notional = shares * execution_price
                cash += notional - explicit_fees
                current.shares -= shares
            trades.append(
                {
                    "signal_date": calendar[calendar.get_loc(date) - 1],
                    "execution_date": date,
                    "symbol": symbol,
                    "side": side,
                    "shares": shares,
                    "open_price": open_price,
                    "execution_price": execution_price,
                    "notional": notional,
                    "commission": costs.commission,
                    "stamp_duty": costs.stamp_duty,
                    "slippage": costs.slippage,
                    "impact": costs.impact,
                    "reason": reason,
                }
            )
            return float(shares)

        for date_index, date in enumerate(calendar):
            # Apply only information available at today's open. This preserves
            # portfolio value across ex-right events without current-day forward adjustment.
            for symbol, position in list(positions.items()):
                row = price_lookup.get((date, symbol))
                if row and pd.notna(row.get("prev_close")) and position.last_raw_close > 0:
                    reference = float(row["prev_close"])
                    factor = position.last_raw_close / reference if reference > 0 else 1.0
                    if np.isfinite(factor) and abs(factor - 1.0) > 1e-6:
                        position.shares *= factor
                        position.average_cost /= factor

            target_symbols = signal_targets.get(date)
            if target_symbols is not None:
                target_set = set(target_symbols)
                for symbol, position in positions.items():
                    held_sessions = date_index - calendar.get_indexer([position.entry_date])[0]
                    if symbol not in target_set:
                        position.exit_requested_date = position.exit_requested_date or date
                        position.exit_reason = position.exit_reason or "RANK_DROP"
                    elif held_sessions >= self.config.max_holding_days:
                        position.exit_requested_date = position.exit_requested_date or date
                        position.exit_reason = position.exit_reason or "MAX_HOLDING"

            # Sells always happen before buys. A position bought today can never be sold today.
            for symbol, position in list(positions.items()):
                if position.exit_requested_date is None or position.locked_unresolved:
                    continue
                if position.entry_date >= date:
                    continue
                row = price_lookup.get((date, symbol))
                if not _tradable_row(row):
                    failure_reason = "SUSPENDED_OR_MISSING"
                elif _is_limit_down(row):
                    failure_reason = "ONE_PRICE_LIMIT_DOWN"
                else:
                    sold = execute(symbol, "SELL", position.shares, date, position.exit_reason or "EXIT")
                    if sold > 0 and position.shares <= 1e-8:
                        record_exit_event(date, position, "FILLED", position.exit_reason or "EXIT")
                        del positions[symbol]
                        continue
                    failure_reason = "PARTIAL_LIQUIDITY" if sold > 0 else "NO_LIQUIDITY_CAPACITY"
                position.exit_deferrals += 1
                if position.exit_deferrals >= self.config.max_exit_deferral_days:
                    position.locked_unresolved = True
                    record_exit_event(date, position, "UNRESOLVED", f"{failure_reason}_AFTER_10")
                else:
                    record_exit_event(date, position, "DEFERRED", failure_reason)

            if target_symbols is not None:
                open_equity = cash
                for symbol, position in positions.items():
                    row = price_lookup.get((date, symbol))
                    mark = float(row["open"]) if _tradable_row(row) else position.last_raw_close
                    open_equity += position.shares * mark
                target_weight = min(1.0 / max(len(target_symbols), 1), self.config.max_weight)
                target_value = open_equity * target_weight
                for symbol in target_symbols:
                    position = positions.get(symbol)
                    row = price_lookup.get((date, symbol))
                    current_value = position.shares * float(row["open"]) if position and _tradable_row(row) else 0.0
                    gap = max(0.0, target_value - current_value)
                    if row and gap >= float(row["open"]) * self.config.lot_size:
                        execute(
                            symbol,
                            "BUY",
                            gap / float(row["open"]),
                            date,
                            "TOP_K_TARGET",
                        )

            market_value = 0.0
            gross_cost = 0.0
            for symbol, position in positions.items():
                row = price_lookup.get((date, symbol))
                close = float(row["close"]) if row and pd.notna(row.get("close")) else position.last_raw_close
                if row and pd.notna(row.get("close")):
                    position.last_raw_close = close
                value = position.shares * close
                cost_basis = position.shares * position.average_cost
                market_value += value
                gross_cost += cost_basis
                ledger.append(
                    {
                        "trade_date": date,
                        "symbol": symbol,
                        "shares": position.shares,
                        "mark_price": close,
                        "market_value": value,
                        "cost_basis": cost_basis,
                        "unrealized_pnl": value - cost_basis,
                        "entry_date": position.entry_date,
                        "exit_requested_date": position.exit_requested_date,
                        "exit_deferrals": position.exit_deferrals,
                        "status": "LOCKED_UNRESOLVED" if position.locked_unresolved else "OPEN",
                    }
                )
            nav_value = cash + market_value
            nav_rows.append(
                {
                    "trade_date": date,
                    "cash": cash,
                    "market_value": market_value,
                    "nav": nav_value,
                    "positions": len(positions),
                }
            )

        nav = pd.DataFrame(nav_rows)
        if not nav.empty:
            nav["daily_return"] = nav["nav"].pct_change().fillna(0.0)
            nav["drawdown"] = nav["nav"].div(nav["nav"].cummax()).sub(1.0)
        trade_frame = pd.DataFrame(trades)
        metrics = performance_metrics(nav)
        if not trade_frame.empty:
            metrics["turnover"] = float(trade_frame["notional"].abs().sum() / nav["nav"].mean())
            metrics["trading_costs"] = float(
                trade_frame[["commission", "stamp_duty", "slippage", "impact"]].sum().sum()
            )
        else:
            metrics.update({"turnover": 0.0, "trading_costs": 0.0})
        return BacktestResult(
            nav=nav,
            trades=trade_frame,
            position_ledger=pd.DataFrame(ledger),
            exit_events=pd.DataFrame(exits),
            metrics=metrics,
        )
