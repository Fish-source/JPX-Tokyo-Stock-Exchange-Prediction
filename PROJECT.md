# JPX Tokyo Stock Exchange Prediction

## 项目概述

本项目参与 Kaggle [JPX Tokyo Stock Exchange Prediction](https://www.kaggle.com/competitions/jpx-tokyo-stock-exchange-prediction) 竞赛，目标是预测东京证券交易所约 2000 只股票的未来收益率排名。

竞赛特点：
- **预测目标**：每只股票的 `Target` = 从 t+1 收盘价到 t+2 收盘价的调整后收益率
- **评估指标**：Spread Return Sharpe Ratio — 每日做多预测排名最高的 N 只股票、做空排名最低的 N 只股票，计算线性加权利差收益的年化 Sharpe Ratio
- **代码竞赛**：通过 Kaggle Notebook 提交，使用时间序列 API 逐日预测
- **实时评估**：提交截止后，模型将运行在真实市场数据上

---

## 项目架构

```
JPX/
├── run_pipeline.py                  # 主流水线入口（训练+评估）
├── generate_submission.py           # 独立提交脚本
├── src/
│   ├── config.py                    # 全局配置（路径、超参数、CV折）
│   ├── data/
│   │   ├── loader.py                # 数据加载（stock_prices + stock_list 合并）
│   │   └── preprocessor.py          # 价格调整、缺失值填充、Target清洗
│   ├── features/
│   │   ├── build_features.py        # 特征编排、缓存、清洗、列选择
│   │   ├── price_features.py        # 时序技术指标（~30个特征）
│   │   ├── cross_section.py         # 截面因子+市场情绪（~25个特征）
│   │   └── financial_features.py    # 财务基本面因子（~10个特征）
│   ├── models/
│   │   ├── train_lgb.py             # LightGBM 训练（regression + lambdarank + CV + 特征筛选）
│   │   ├── train_xgb.py             # XGBoost 训练
│   │   └── ensemble.py              # 多模型加权集成
│   ├── evaluation/
│   │   └── metrics.py               # Spread Return Sharpe + Spearman 相关性
│   └── inference/
│       └── submit.py                # Kaggle API 逐日预测提交
├── notebooks/
│   ├── 01_eda.ipynb                 # 探索性数据分析
│   ├── 02_feature_analysis.ipynb    # 特征重要性与相关性分析
│   └── 03_kaggle_submission.ipynb   # Kaggle 提交专用 Notebook
├── output/
│   ├── features/                    # 特征缓存（Parquet）
│   ├── models/                      # 训练好的模型文件
│   └── submissions/                 # 提交文件
└── jpx-tokyo-stock-exchange-prediction/  # 竞赛数据
    ├── train_files/
    ├── supplemental_files/
    ├── example_test_files/
    └── jpx_tokyo_market_prediction/ # Kaggle API
```

---

## 数据说明

### 核心数据：stock_prices.csv

| 列名 | 类型 | 说明 |
|------|------|------|
| RowId | string | `{Date}_{SecuritiesCode}` |
| Date | date | 交易日期 |
| SecuritiesCode | int | 股票代码 |
| Open/High/Low/Close | float | 开盘/最高/最低/收盘价 |
| Volume | int | 成交量 |
| AdjustmentFactor | float | 累积调整因子（分红/拆股） |
| ExpectedDividend | float | 预期分红 |
| SupervisionFlag | bool | 监管/退市标志 |
| **Target** | float | **预测目标**：t+1 到 t+2 的调整后收益率 |

- 训练数据范围：~2017-01-04 至 ~2021-12-03
- 补充数据延续至更晚日期
- 每日约 2000 只股票

### 辅助数据

| 文件 | 说明 | 是否使用 |
|------|------|---------|
| financials.csv | 季度财报（营收、利润、EPS、ROE等） | ✅ 用于基本面特征 |
| secondary_stock_prices.csv | 非主板股票行情 | ✅ 用于市场宽度 |
| options.csv | 期权数据 | ❌ 暂未使用 |
| trades.csv | 周度交易汇总 | ❌ 暂未使用 |
| stock_list.csv | 股票元数据（行业、市值等） | ✅ 用于截面特征 |

---

## 目标变量详解

Target 的精确定义：

```
Target(t, i) = [AdjustedClose(t+1, i) / AdjustedClose(t, i)] - 1
```

其中 `AdjustedClose` 是经过 AdjustmentFactor 调整后的收盘价。这代表"在 t 日收盘买入，在 t+1 日收盘卖出"的收益率。

### 评估指标：Spread Return Sharpe Ratio

每日的 spread return 计算如下：

1. 将所有股票按预测排名排列
2. 选出排名最高（做多）和最低（做空）的 N = min(200, stocks//10×2) 只
3. 对做多组赋予线性递减权重：`w_i = 2 × (N - rank_i) / (N × (N+1))`
4. 对做空组赋予线性递减的负权重
5. Spread Return = Σ(weight_i × Target_i)

**Sharpe Ratio = mean(daily_spread) / std(daily_spread)**

这意味着：
- 预测的**相对排名**比绝对值更重要
- 排名中等的股票不影响得分
- 正确识别极端（头部/尾部）股票是关键

---

## 数据预处理

### 1. 价格调整 (`adjust_prices`)
- 计算 `CumAdjustmentFactor` = AdjustmentFactor 的反向累积乘积
- 生成 `AdjustedClose = Close / CumAdjustmentFactor`（及 Open/High/Low）
- 消除分红和拆股对价格序列的影响

### 2. 缺失值填充 (`fill_missing_prices`)
- 调整后价格按股票向前填充（ffill）
- Volume 缺失填 0

### 3. Target 清洗 (`clean_target`)
- 将 ±inf 替换为 NaN
- ExpectedDividend 缺失填 0

---

## 特征工程

### 时序技术指标 (~30 个特征)

| 特征组 | 特征名 | 计算 |
|--------|--------|------|
| 收益率 | Return_{1,5,10,20,40,60}d | AdjustedClose / AdjustedClose.shift(w) - 1 |
| 波动率 | Volatility_{5,10,20,60}d | 日收益率的滚动标准差 |
| 成交量 | VolumeRatio_{5,10,20}d | Volume / 滚动均值 |
| 成交量 | VolumeChange | Volume 的日变化率 |
| 日内 | IntradayRange | (High - Low) / Close |
| 日内 | UpperShadow | (High - max(Open,Close)) / Close |
| 日内 | LowerShadow | (min(Open,Close) - Low) / Close |
| 日内 | BodyRatio | abs(Close - Open) / (High - Low) |
| 日内 | OpenCloseReturn | (Close - Open) / Open |
| 回撤 | Drawdown_{20,40,60}d | Close / 滚动最高价 - 1，clip 至 [-1, 0] |
| RSI | RSI_14 | 14日相对强弱指标（EMA平滑） |
| MACD | MACD / MACD_Signal / MACD_Hist | 12/26/9 参数标准 MACD |
| 布林带 | BollingerPos | (Close - MA20) / (2 × Std20) |
| 分红 | DividendYield | ExpectedDividend / AdjustedClose |
| 滞后目标 | TargetLag_{1,2,3,5,10,20}d | Target 的各期滞后值 |
| 滞后目标 | TargetRollingMean_5d/20d, TargetRollingStd_20d | Target 的滚动均值和标准差 |
| 反转 | MomentumReversal_{5,20,60}d | 短期收益与中期收益之差（捕捉反转效应） |

### 截面因子 (~25 个特征)

| 特征组 | 特征名 | 计算 |
|--------|--------|------|
| 收益排名 | ReturnRank / ReturnRankPct | 日内 1d 收益的跨股票排名 |
| 板块收益 | SectorReturn_{1,5,20}d | 33 行业平均收益率 |
| 板块超额 | SectorExcess_{1,5,20}d | 个股收益 - 板块平均收益 |
| 板块排名 | SectorRank / SectorRankPct | 板块内收益排名 |
| 市场收益 | MarketReturn_1d / ExcessReturn_1d | 全市场平均收益及个股超额 |
| 波动排名 | VolatilityRank / VolatilityRankPct | 20d 波动率的跨股票排名 |
| 成交量排名 | VolumeRank / VolumeRankPct | 成交量的跨股票排名 |
| 目标排名 | TargetLag1dRank / TargetLag1dRankPct | 滞后1日 Target 的跨股票排名 |
| 日历 | DayOfWeek, Month, DayOfMonth, IsMonthStart, IsMonthEnd | 日期特征 |
| 动量排名 | MomentumRank_{5,20,60}d | 多周期收益率的百分位排名 |
| 短期反转 | ShortTermReversal / Reversal_5d | 短期收益百分位排名（反转因子） |
| 板块轮动 | SectorDailyRankPct | 板块当日收益在全市场板块中的排名 |
| 规模因子 | SizeRank | 市值百分位排名 |
| 市场宽度 | MarketBreadth | 次级市场当日上涨股票占比 |
| 成交量集中 | VolumeConcentration | 前10%大市值股票成交量占比 |

### 基本面因子 (~10 个特征)

| 特征名 | 计算 |
|--------|------|
| OperatingMargin | OperatingProfit / NetSales（营业利润率） |
| ROE | Profit / Equity（净资产收益率） |
| ProfitMargin | Profit / NetSales（净利率） |
| EPS_surprise | (实际EPS - 预测EPS) / |预测EPS|（盈余惊喜） |
| Profit_surprise | (实际利润 - 预测利润) / |预测利润|（利润惊喜） |
| EquityToAssetRatio | 股东权益比率（直接取自财报） |
| NetSales_YoY | 营收同比增长率（4个季度前对比） |
| Profit_YoY | 利润同比增长率 |
| EarningsPerShare_YoY | EPS 同比增长率 |

**处理方式**：财务数据按 DisclosedDate 前向填充到每日行情数据中，确保不使用未来信息。

### 特征后处理

1. 所有特征替换 ±inf 为 NaN
2. 连续特征 clip 到 1% 和 99% 分位数（去除极端值）
3. 特征缓存为 Parquet 文件，避免重复计算

---

## 模型

### 1. LightGBM (Regression)

- **目标函数**：regression（RMSE）
- **核心超参数**：
  - learning_rate = 0.01
  - num_leaves = 31, max_depth = 6
  - feature_fraction = 0.6, bagging_fraction = 0.6
  - lambda_l1 = 1.0, lambda_l2 = 1.0
  - min_child_samples = 500
- **训练策略**：1500 轮 + 200 轮早停
- **输出**：连续值预测，后续通过排名转换为 Rank

### 2. LightGBM (LambdaRank)

- **目标函数**：lambdarank（NDCG 优化）
- **组信息**：按交易日分组，每日约 2000 只股票为一个 query
- **自定义评估函数**：直接计算 Sharpe Ratio 作为评估指标
- **优势**：直接优化排名质量，与竞赛指标对齐

### 3. XGBoost (Regression)

- **目标函数**：reg:squarederror
- **核心超参数**：
  - learning_rate = 0.01, max_depth = 5
  - min_child_weight = 500
  - subsample = 0.6, colsample_bytree = 0.6
  - tree_method = "hist"（高效直方图方法）
- **训练策略**：1500 轮 + 200 轮早停

### 集成策略

三模型加权平均：

```
pred_ensemble = w1 × pred_lgb_reg + w2 × pred_lgb_rank + w3 × pred_xgb
```

- 权重在验证集上通过网格搜索优化，目标为最大化 Sharpe Ratio
- 默认等权重 (1:1:1) 作为初始方案
- 2 模型时搜索 9 个权重点，3 模型时搜索 8×8 网格

---

## 验证策略

### 时间序列分割

| 数据集 | 时间范围 | 用途 |
|--------|---------|------|
| 训练集 | < 2021-01-01 | 模型训练 |
| 验证集 | 2021-01-01 ~ 2021-07-01 | 早停、超参选择、集成权重 |
| 测试集 | ≥ 2021-07-01 | 最终评估 |

### 滚动时间序列 CV（3 折）

| 折 | 训练截止 | 验证截止 |
|----|---------|---------|
| 1 | 2020-01-01 | 2020-07-01 |
| 2 | 2020-07-01 | 2021-01-01 |
| 3 | 2021-01-01 | 2021-07-01 |

- 使用扩展窗口：每一折的训练集包含之前所有数据
- 报告 3 折平均 Sharpe Ratio 和 Spearman 相关性
- 用于评估模型稳定性和泛化能力

### 特征重要性筛选

- 基于 LightGBM 的 gain importance
- 保留累积重要性 ≤ 95% 的特征
- 剔除低贡献特征，减少噪声和过拟合

---

## 评估指标

### Spread Return Sharpe Ratio（主指标）

竞赛官方指标，完整实现：

1. 每日选取 Top-N 和 Bottom-N 股票（N = min(200, 股票数//10×2)）
2. 线性加权：Top 股票赋予正权重，Bottom 股票赋予负权重
3. 每日 Spread Return = Σ(weight_i × Target_i)
4. Sharpe = mean(daily_spread) / std(daily_spread)

### Spearman Rank Correlation（辅助指标）

- 每日计算预测值与 Target 的 Spearman 秩相关
- 取所有交易日的平均值
- 直接衡量排名预测的准确性

---

## 提交流程

### 方式一：本地流水线

```bash
python run_pipeline.py
```

完整流程：数据加载 → 预处理 → 特征工程 → CV → 训练3模型 → 集成评估

### 方式二：Kaggle Notebook 提交

将 `notebooks/03_kaggle_submission.ipynb` 上传到 Kaggle，该 Notebook 包含：
- 完整的特征工程代码（内联，无外部依赖）
- 3 个模型的训练逻辑
- 时间序列 API 逐日预测循环

### 方式三：独立提交脚本

```bash
python generate_submission.py
```

加载已训练的模型文件，通过 Kaggle API 逐日预测。

### 提交逻辑

Kaggle API 以时间序列方式逐日提供数据：

1. 初始化 65 日历史价格缓冲区
2. 每个交易日：
   - 将新数据拼入缓冲区
   - 运行完整预处理 + 特征工程
   - 只提取当日特征进行预测
   - 集成三模型预测，排名后写入 submission
   - 更新缓冲区
3. 预测排名 (Rank) 为 0-indexed 降序整数

---

## 计算负载

| 任务 | 预估时间 | 内存 |
|------|---------|------|
| 特征工程（含金融+截面+情绪） | ~3-5 分钟 | <4GB |
| LightGBM 训练 ×2 | ~5-10 分钟 | <1GB |
| XGBoost 训练 ×1 | ~3-5 分钟 | <1GB |
| 3 折 CV | ~15-25 分钟 | <1GB |
| 集成权重优化 | ~1 分钟 | <1GB |
| **端到端流水线** | **~30 分钟** | **<4GB** |

---

## 实验记录与改进路线

### 当前方案 (v4)

- 3 模型集成：LightGBM-reg + LightGBM-rank + XGBoost
- ~65 个特征（时序 + 截面 + 基本面 + 情绪）
- 3 折滚动 CV + 特征重要性筛选

### 未来改进方向

1. **Options 数据**：隐含波动率期限结构、看跌/看涨比率
2. **Trades 数据**：外资/散户净买入流、机构资金流向
3. **CatBoost**：有序提升策略，对类别特征（行业）更友好
4. **更细粒度的超参搜索**：Optuna 贝叶斯优化
5. **滚动集成**：根据近期表现动态调整模型权重
6. **堆叠集成**：用线性模型作为第二层学习器
7. **自适应特征**：根据市场状态（高波/低波）切换特征组合

---

## 环境依赖

```
python >= 3.7
pandas
numpy
lightgbm >= 4.0
xgboost >= 2.0
scipy
scikit-learn
matplotlib (notebooks)
statsmodels (notebooks)
```

安装：
```bash
pip install pandas numpy lightgbm xgboost scipy scikit-learn matplotlib statsmodels
```
