from __future__ import annotations

import pandas as pd

from neural_a_share.reports import ReportContext, StaticReportPublisher


def test_static_reports_publish_all_core_pages_without_cdn(tmp_path) -> None:
    dates = pd.bdate_range("2024-01-02", periods=5)
    predictions = pd.DataFrame(
        {
            "symbol": ["A.SH", "B.SZ"],
            "name": ["A", "B"],
            "NeuralRank": [1, 2],
            "Alpha20": [0.02, 0.01],
            "Alpha40": [0.03, 0.02],
            "Alpha60": [0.04, 0.03],
            "NeuralAlpha": [0.03, 0.02],
        }
    )
    nav = pd.DataFrame({"trade_date": dates, "nav": [1.0, 1.01, 1.02, 1.015, 1.03]})
    rolling = pd.DataFrame(
        {"trade_date": dates, "rolling_ic_20": 0.03, "rolling_ic_40": 0.02, "rolling_ic_60": 0.01}
    )
    quantiles = pd.DataFrame({"quantile": [1, 2, 3, 4, 5], "mean": [-0.01, 0.0, 0.01, 0.02, 0.03]})
    context = ReportContext(
        data_date="2024-01-08",
        tickflow_status="PASS",
        model_version="mlp-test",
        training_cutoff="2024-01-05",
        predictions=predictions,
        rolling_ic=rolling,
        nav=nav,
        benchmark_nav=nav,
        quantiles=quantiles,
        metrics={"total_return": 0.03, "sharpe": 1.2, "max_drawdown": -0.01, "calmar": 2.0},
    )
    StaticReportPublisher(tmp_path).publish(context)
    for filename in ("index.html", "daily.html", "weekly.html"):
        content = (tmp_path / filename).read_text(encoding="utf-8")
        assert "</html>" in content
        assert "https://cdn" not in content
    assert (tmp_path / "reports" / "2024-01-08" / "daily.html").exists()
    assert (tmp_path / "assets" / "generated" / "2024-01-08" / "nav.svg").exists()
