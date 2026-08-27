# TickFlow Neural Alpha

一个从零实现的 A 股/ETF 纯神经网络选股研究项目。数据只来自官方 `TickFlow.free()` Python SDK；项目不需要 API Key，不接入其他行情源，也不继承任何旧项目的 RuleScore、InstitutionalScore 或人工打分。

模型唯一的选股链路是：

```text
122 causal features
  → Multi-Task MLP (128 → 64 → 32)
  → Alpha20 / Alpha40 / Alpha60
  → NeuralAlpha = mean(three neural heads)
  → NeuralRank
  → Top-K
```

RSI、MACD、CMF、OBV、ATR、Momentum、Volume 和 Volatility 只作为网络输入。任何技术指标都不会绕过模型直接改变排名。

## 核心约束

- 数据：`TickFlow.free()` 的日 K、标的信息、交易所目录与标的池。
- 缓存：原始数据按年写入原子 Parquet 分区；日更自动覆盖重叠窗口并去重。
- 复权/PIT：只存 `adjust="none"`；利用每个交易日当时可见的 `prev_close` 构造收益链，不把今天的前复权序列倒灌到过去。
- 特征：122 个仅依赖 `t` 日及以前数据的量价特征，当日横截面缩尾标准化。
- 标签：信号在 `t` 日收盘生成，标签从 `t+1` 开盘到 `t+h` 收盘；缺失开盘、停牌或未成熟标签不会被评价。
- 划分：Expanding Purged Walk-Forward；没有 random split，Train/Validation 与 Validation/Test 之间都隔离最长 60 日标签区间，另加 embargo。
- 调参：Early Stopping 只看 Validation；Test 只产生 Historical OOS 预测，不参与模型选择。
- 回测：真实现金、持仓、交易、退出事件和逐日 Mark-to-Market NAV 四套账本。
- DAILY：只做增量、检查、特征更新、champion 推理、排名和报告；不会每天训练。
- 页面：静态 HTML + matplotlib SVG，不依赖外部 CDN；Pages 只部署通过测试和页面校验的完整 artifact。

## Survivorship Bias 的严格处理

TickFlow 免费服务当前可返回“查询当日看到的交易所目录”，但没有一个可把今天的成分安全回填到多年前的历史成分接口。因此项目不会假装当前成分表天然没有 Survivorship Bias：

1. 每次更新都会保存不可变的 TickFlow universe snapshot。
2. strict 模式只允许某个信号日使用该日以前已观察到的 snapshot。
3. 如果回测开始日期早于第一份 snapshot，训练和 Walk-Forward 默认直接失败，并在周报显示 `DEGRADED/FAIL`。
4. `--allow-degraded-survivorship` 是显式的研究调试开关，不能把其结果标成 strict Historical OOS。

这是有意的安全阀。免费服务无法提供的信息不能靠代码猜出来；随着每日 snapshot 积累，Forward Shadow OOS 会自然成为严格无幸存者偏差的样本。

## Windows 安装

建议使用 Python 3.11。RTX 3060 需要已安装兼容的 NVIDIA 驱动；PyTorch 会自动检测 CUDA，检测不到时回退 CPU。

```powershell
git clone https://github.com/javabugmaker/1234.git
cd 1234
powershell -ExecutionPolicy Bypass -File .\scripts\setup_windows.ps1
.\.venv\Scripts\python.exe run_gui.py
```

也可以手动安装：

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
```

如果使用 CUDA，请按 [PyTorch 官方安装选择器](https://pytorch.org/get-started/locally/) 安装与本机驱动匹配的 wheel。代码中的 `device: auto` 会自动选择 `cuda` 或 `cpu`，AMP 只在 CUDA 上开启。

## 运行顺序

首次研究构建：

```powershell
# 免费 TickFlow 历史全量 + 当前 universe snapshot
.\.venv\Scripts\neural-alpha.exe update --full

# 分年生成 PIT 特征和 20/40/60 日标签
.\.venv\Scripts\neural-alpha.exe features

# strict 模式训练；历史成分审计不通过会拒绝运行
.\.venv\Scripts\neural-alpha.exe train

# Expanding Purged Walk-Forward 与真实账本回测
.\.venv\Scripts\neural-alpha.exe walk-forward
.\.venv\Scripts\neural-alpha.exe backtest

# 报告
.\.venv\Scripts\neural-alpha.exe daily
.\.venv\Scripts\neural-alpha.exe weekly
```

开发阶段若只是验证模型和管线（结果会被标记为 degraded）：

```powershell
.\.venv\Scripts\neural-alpha.exe train --allow-degraded-survivorship
.\.venv\Scripts\neural-alpha.exe walk-forward --max-folds 2 --allow-degraded-survivorship
```

常用入口也可直接运行：

```powershell
.\.venv\Scripts\python.exe run_daily.py
.\.venv\Scripts\python.exe run_train.py
.\.venv\Scripts\python.exe run_backtest.py
.\.venv\Scripts\python.exe run_weekly.py
```

## DAILY

```text
TickFlow.free() 增量日 K
→ schema / OHLC / 重复 / 日历 / 完整性 / 新鲜度检查
→ 当前年份 Feature/Label 分区更新
→ 加载 champion checkpoint
→ 全市场 MLP 推理
→ Alpha20 / Alpha40 / Alpha60
→ NeuralAlpha / NeuralRank / Top-K
→ predictions/YYYY-MM-DD.{parquet,csv}
→ docs/index.html + daily.html + 历史归档
```

免费服务盘中不会实时更新。GUI 和页面显示的是 TickFlow 已提供的最近一根完整日 K，绝不会把实时快照伪装成收盘数据。

## WEEKLY

`weekly.html` 明确区分：

- `IN-SAMPLE`
- `VALIDATION`
- `HISTORICAL OOS`
- `FORWARD SHADOW OOS`

并展示 Strategy NAV vs CSI300、Total Return、Sharpe、Max Drawdown、Calmar、Rolling Rank IC20/40/60、ICIR、Newey-West t、IC Decay、Yearly/Regime IC、Top-K、Quantile Monotonicity、Turnover、Trading Costs、Champion/Challenger、Train/Validation/Test、Purge/Embargo、Mature Labels、Data Quality 和 PIT/Survivorship 状态。没有成熟样本时显示 `N/A/PENDING`，不会拿 IS 指标填充 OOS 区块。

## 回测口径

- 佣金：股票单边 `0.00008499999`；ETF/LOF/基金单边 `0.00005000001`；最低佣金默认 `0`。
- 股票卖出印花税：独立配置，默认 `0.0005`。
- 成交价：下一交易日开盘价 + 固定不利滑点 + 基于成交额参与率的非线性冲击；冲击有上限。
- 容量：单笔成交额不超过当日成交额的 `max_participation`。
- 卖出失败：停牌、缺失行情或一字跌停逐日顺延；最多尝试 10 个交易日。仍无法成交时持仓继续计入 NAV，并记录 `UNRESOLVED`，不虚构退出。
- 账本：`nav.parquet`、`trades.parquet`、`position_ledger.parquet`、`exit_events.parquet`。

## GUI

`run_gui.py` 使用标准库 Tkinter。更新、训练、Walk Forward、回测和报告都在后台线程执行；主线程只更新控件和日志，因此长任务不会卡死窗口。GUI 显示：

- TickFlow 最新日期
- CUDA/GPU 或 CPU fallback
- Champion 版本与 TrainingCutoff
- Rolling IC20/40/60
- Top 股票、Alpha20/40/60、NeuralAlpha、NeuralRank
- 进度、日志和 Champion/Challenger 管理

## 测试

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

测试覆盖 PIT、标签对齐与成熟度、Purge、Expanding Walk-Forward、checkpoint、CUDA/CPU fallback、Rank IC、T+1、股票/ETF 成本、退出顺延、Position Ledger、NAV 和 TickFlow 完整性。

## 目录

```text
config/default.yaml          参数与交易成本
src/neural_a_share/          数据、特征、模型、回测、报告、GUI
tests/                       严格性与账本测试
docs/                        GitHub Pages 健康页面
data/cache/                  TickFlow Parquet（不提交）
data/models/                 checkpoint 与 registry（不提交）
data/predictions/            每日预测（不提交）
data/backtests/              OOS 与账本（不提交）
.github/workflows/           CI 与原子 Pages 部署
```

## GitHub Pages 首次启用

仓库管理员首次需要在 `Settings → Pages → Build and deployment → Source` 选择
`GitHub Actions`。这是 GitHub 的一次性仓库设置；标准 `GITHUB_TOKEN` 无权替仓库
开启 Pages。启用后，`CI` 成功会触发 `Deploy Pages`，完整 artifact 校验、上传和部署
都成功后才切换线上版本，因此发布失败不会覆盖上一版健康页面。
