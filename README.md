# TickFlow Neural Alpha

一个从零实现的 A 股股票纯神经网络选股研究项目。数据只来自官方 `TickFlow.free()` Python SDK；项目不需要 API Key，不接入其他行情源，也不继承任何旧项目的 RuleScore、InstitutionalScore 或人工打分。TickFlow 缓存可以保留 ETF/基金行情供市场特征与交易成本研究使用，但默认候选池只允许 A 股股票。

模型唯一的选股链路是：

```text
122 causal features
  → Multi-Task MLP (128 → 64 → 32)
  → Alpha20 / Alpha40 / Alpha60
  → NeuralAlpha = mean(three neural heads)
  → NeuralRank
  → Neural Stock Top-K
```

RSI、MACD、CMF、OBV、ATR、Momentum、Volume 和 Volatility 只作为网络输入。任何技术指标都不会绕过模型直接改变排名。

## 核心约束

- 数据：`TickFlow.free()` 的日 K、标的信息、交易所目录与标的池。
- 缓存：原始数据按年写入原子 Parquet 分区；日更自动覆盖重叠窗口并去重。
- 复权/PIT：只存 `adjust="none"`；利用每个交易日当时可见的 `prev_close` 构造收益链，不把今天的前复权序列倒灌到过去。
- 特征：122 个仅依赖 `t` 日及以前数据的量价特征，当日横截面缩尾标准化。
- 候选资格：使用信号日已经观察到的 TickFlow `instrument_type` 收口为股票，并要求默认至少 80% Feature 可用；这两个条件只决定网络可推理的研究范围，不产生分数，也不改变范围内的 NeuralRank。
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
4. `--allow-degraded-survivorship` 是显式的研究调试开关：它保留已有历史 K 线样本、不把首份快照错误套用到过去；模型版本、checkpoint、训练清单和 Walk-Forward 输出都会标记为 `DEGRADED`，不能把其结果标成 strict Historical OOS。

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

Walk Forward 默认启用逐折断点续跑。输入数据、折边界、Feature、模型配置或
Survivorship 模式的签名没有变化时，已完成折会直接复用；需要强制重算所选折时添加
`--no-resume`。`--max-folds 2` 只发布最近两折并明确标为 `PARTIAL`，不能当成完整
Historical OOS；不传 `--max-folds` 才发布全部可用折。Walk Forward 折只用于历史
OOS 评价与回测，不会替代 Champion，也不会直接生成 DAILY 榜单。

该开关不会改变 Feature、Label、日期切分、Purge/Embargo 或训练抽样，只跳过无法由免费服务追溯证明的历史 membership 过滤。它不能消除现有历史缓存可能包含的 Survivorship Bias；严格模式仍会 fail-closed，且不会自动降级。

常用入口也可直接运行：

```powershell
.\.venv\Scripts\python.exe run_daily.py
.\.venv\Scripts\python.exe run_train.py
.\.venv\Scripts\python.exe run_backtest.py
.\.venv\Scripts\python.exe run_weekly.py
.\.venv\Scripts\python.exe run_publish_pages.py
```

## 16 GB 内存训练

训练和 Walk-Forward 不会再把全部年份的 features 与 labels 一次性读入后 merge。
程序先逐年扫描键、成熟日期和必要标签，再逐年读取所需列、校验
`symbol + trade_date` one-to-one、merge，并立即执行全局有界抽样。特征和数值标签
在 Arrow 转为 pandas 之前收窄为 `float32`；Validation 保留全部验证日期及可计算
Rank IC 的横截面结构。现有 `data/cache/derived/**/year=YYYY/*.parquet` 可直接使用，
不需要重新下载 TickFlow，也不需要重跑 `update --full` 或 `features`。

首次训练或 Walk Forward 会从这些现有分区逐年生成
`data/cache/derived/research_cache/year=YYYY/research.parquet`。后续 expanding folds 直接读取
已校验的合并分区，避免反复 merge 和排序。训练 DataLoader 使用张量批量索引，避免
逐行 Python collation；GPU loss 只在每个 epoch 结束时同步。每折预测会立即原子写入
`data/backtests/walk_forward_cache/fold=NNN/`，中断后无需从第 1 折重来，最终结果也按折
流式合并，不会一次性 `concat` 全部 OOS 预测。

## DAILY

```text
TickFlow.free() 增量日 K
→ schema / OHLC / 重复 / 日历 / 完整性 / 新鲜度检查
→ 当前年份 Feature/Label 分区更新
→ 加载 champion checkpoint
→ 全市场因果 Feature
→ PIT 股票类别 + Feature 完整度资格过滤
→ Champion MLP 推理
→ Alpha20 / Alpha40 / Alpha60
→ NeuralAlpha / NeuralRank / Top-K
→ predictions/YYYY-MM-DD.{parquet,csv}
→ docs/index.html + daily.html + publish.html + 历史归档
→ 完整站点校验后安全自动推送 docs/（失败不影响 DAILY，也不覆盖健康站点）
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
- 显式 `DEGRADED 研究模式` 开关；当前 Champion 已标记 degraded 时首次启动会自动勾选
- Walk Forward 范围选择：最近 2 折（默认快速）、5 折、10 折或全部折；自动断点续跑
- Rolling IC20/40/60
- Top 股票、Alpha20/40/60、NeuralAlpha、NeuralRank
- 进度、日志和 Champion/Challenger 管理

GUI 中“训练模型”和“Walk Forward”使用同一个 Survivorship 模式开关。关闭时严格
fail-closed；开启时保留历史缓存样本，并把模型和 Walk-Forward 产物明确标记为
`DEGRADED`，不会改变日期切分、Purge/Embargo 或统计口径。
快速范围的产物同时标记 `PARTIAL`；要生成完整历史 OOS 报告，请选择“全部折（完整范围）”。

GUI 的 Walk Forward 范围旁会明确显示“仅历史 OOS 评估”，并提供独立的
“发布 Pages”后台按钮。榜单只显示股票；`FeatureCoverage` 会同时写入预测文件、
日报和首页，方便审计候选资格。

## Pages 自动发布

`docs/publish.html` 是独立的发布状态页。默认配置 `reports.auto_push_pages: true`，
因此 DAILY 和 WEEKLY 在本地四个核心页面、历史归档和图表全部生成并校验后，会自动
尝试推送。也可单独执行：

```powershell
.\.venv\Scripts\neural-alpha.exe publish-pages
# 或
.\.venv\Scripts\python.exe run_publish_pages.py
```

发布器使用临时 Git index，把本地 `docs/` 叠加到最新远端 `main` 后做非强制推送；
不会切换当前分支，不会修改源代码工作区，也不会碰已有 staged changes。远端并发更新、
身份认证、分支保护、CI 或部署失败时安全停止，线上继续保留上一版健康页面，DAILY/WEEKLY
本身仍算成功。Windows 需要当前仓库的 Git 凭据能够正常执行 `git push origin main`；
若只需要本地报告，可把 `auto_push_pages` 设为 `false`，以后再用 GUI/CLI 手动重试。

从旧版本升级不需要重新下载 TickFlow，也不需要 `update --full`。当前 Champion 会立即只在
合格股票中推理；要让训练样本也完全收口为股票，请重新执行训练，并在模型管理中审阅后
Promote 新 Challenger。Walk Forward 的缓存签名包含股票范围和完整度阈值，旧 ETF 范围的
折不会被误复用。收益好坏必须用足够长的 Historical OOS 或后续 Forward Shadow OOS 判断；
最近 2 折只适合快速验证管线，不能据此调整统计口径。

## 测试

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

测试覆盖 PIT、标签对齐与成熟度、Purge、Expanding Walk-Forward、逐折缓存/断点续跑、
年度研究缓存、向量批处理、checkpoint、CUDA/CPU fallback、Rank IC、T+1、股票/ETF
成本、退出顺延、Position Ledger、NAV、股票候选资格、Feature 覆盖率、Pages 临时 index
安全推送和 TickFlow 完整性。

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
