# Agent Memory — 量化开发知识沉淀

---

## [2026-05-10] 任务：DataLoader 核心模块 — Parquet 惰性加载 + 反未来函数日线对齐

### 1. 踩坑记录

#### 坑 1：虚拟环境 `.venv` 与系统 Python 分离，pytest/polars 需在 venv 中单独安装
- **现象**：运行 `python -m pytest` 提示 `No module named pytest`；安装 pytest 后又报 `No module named polars`
- **根因**：项目使用 `.venv` 虚拟环境（`c:/Users/rui.yan/project/b1/.venv`），系统 Python 与 venv 完全隔离
- **解法**：通过 `configure_python_environment` 工具识别 venv，再用 `install_python_packages` 依次安装 `pytest`、`polars`
- **经验**：新环境首次执行前，务必先 `configure_python_environment` 确认激活的 Python 解释器路径

#### 坑 2：Polars `join_asof` + `by` 参数触发 UserWarning
- **现象**：`UserWarning: Sortedness of columns cannot be checked when 'by' groups provided`
- **根因**：Polars 在有 `by=` 分组时无法对每个分组独立进行 O(1) 排序断言，降级为运行时警告
- **影响**：纯警告，不影响结果正确性；已在代码中显式调用 `.sort(["symbol", "bob"])` / `.sort(["symbol", "eob"])` 保证排序
- **经验**：如需消除警告，可在 sort 后追加 `.set_sorted("bob")` 手动声明已排序（但需确保确实有序）

#### 坑 3：`join_asof` 要求 `left_on` / `right_on` 类型字节级一致
- **规避措施**：测试 fixture 中所有 `bob`/`eob` 列均显式 `.cast(pl.Datetime("us", "Asia/Shanghai"))`，与 `build_database.py` 的生产 schema 完全对齐
- **经验**：在伪造测试数据时，`datetime` 对象不带 tzinfo，cast 前先构造为 naive datetime，再通过 Polars Series `.cast(DT)` 统一加时区，避免 Python 层的 tzinfo 与 Polars 层冲突

---

### 2. 量化逻辑备忘：反未来函数设计

#### 核心等式
$$\text{prev\_1d} = \arg\max\{ \text{eob}_{1d} \mid \text{eob}_{1d} \leq \text{bob}_{15m},\ \text{symbol 相同} \}$$

#### 为什么用 `join_asof(strategy="backward")` 而非 `shift(1) + date join`

| 对比维度 | `shift(1) + date join` | `join_asof(strategy="backward")` |
|----------|------------------------|----------------------------------|
| 停牌/节假日断层 | `_date` 在 1d 中缺失 → 产生 null，策略中断 | 自动回溯到最近已确认日线，不产生意外 null |
| 跨年/跨月边界 | 需额外处理边界 | 时间轴连续，天然无边界问题 |
| 代码复杂度 | 需提取 `_date` 辅助列、shift、再 join | 一行 `join_asof` 搞定 |
| 语义精确性 | 基于日历日期匹配 | 基于时间戳，精确到微秒 |

#### 首个交易日 null 是正确行为
- 当 15m `bob` 早于所有已知日线 `eob` 时，`join_asof` 返回 null
- 这表示"无前一日数据可用"，策略层应用 `.fill_null()` 或 `.drop_nulls()` 按需处理
- **绝不应**用 `fill_null(strategy="forward")` 填充，否则引入前视偏差

---

### 3. 下一步架构建议

1. **`load_kbars` 分区剪枝增强**：当前过滤仅靠 `bob` 列的谓词下推；对于大时间跨度查询，可额外加入 `pl.col("year").cast(pl.Int32) >= start_year` 等分区键过滤，触发 Hive 目录级剪枝，减少 `scan_parquet` 扫描的文件数量

2. **`align_daily_to_15m` 扩展**：目前只导出原始 OHLCV 6 列；后续可按需加入涨跌幅（`pct_change`）、振幅等派生字段——但严格遵守 SOP 铁律，**所有派生字段必须在内存中动态计算，严禁写入 Parquet**

3. **多频率泛化**：当前 `align_daily_to_15m` 硬编码了 1d→15m 的对齐；可抽象为 `align_lower_to_higher(df_high, df_low, left_on, right_on)` 泛型方法，支持 60m→15m、周线→日线等任意周期组合

4. **`requirements.txt` 补全**：当前项目缺少依赖声明，建议添加：
   ```
   polars>=1.0.0
   duckdb>=0.10.0
   pytest>=8.0.0
   ```

5. **`DataLoader` 集成测试**：在 CI 中对真实 Parquet 文件执行 `load_kbars` 端到端测试（需将测试数据小样本纳入版本控制，或在 CI 中生成）

---

## [2026-05-10] 任务：IndicatorFactory 动态技术指标工厂 (SMA/EMA/MACD/RSI/ATR)

### 1. 踩坑记录

#### 坑 1：Polars 1.21+ 将 `ewm_mean` 的 `min_periods` 参数重命名为 `min_samples`
- **现象**：`DeprecationWarning: the argument 'min_periods' for 'Expr.ewm_mean' is deprecated. It was renamed to 'min_samples' in version 1.21.0`
- **影响**：代码功能正常（向后兼容），但产生大量警告，会污染 pytest 输出
- **解法**：将所有 `ewm_mean(..., min_periods=N)` 改为 `ewm_mean(..., min_samples=N)`
- **经验**：使用 Polars 时优先查阅当前版本 API 文档（本项目 Polars 1.40.1），`ewm_mean` 签名为 `(span, com, half_life, alpha, adjust, min_samples, ignore_nulls)`

#### 坑 2：`rolling_mean` vs `ewm_mean` 的 null 行为差异
- **`rolling_mean(window_size=N)`**：前 N-1 行**自然产生 null**（需要恰好 N 个非 null 值才计算），无需额外参数
- **`ewm_mean(span=N)`（不加 min_samples）**：从**第 1 行**开始就有计算值（无前置 null），EWM 数学上可从任意点递推
- **解决方案**：统一追加 `min_samples=window`，让 EWM 与 SMA 的冷启动行为一致：前 N-1 行强制 null
- **量化意义**：EWM 初期权重分布极度失衡（第 1 行 = 第 1 个值，第 2 行 = 0.5*v1 + 0.5*v2），这些"假收敛"值进入回测会产生虚假信号。`min_samples` 是防早期假信号的关键保护

#### 坑 3：Wilder 平滑 — `com=N-1` vs `span=N` 的精确区分
- **Wilder 原版定义**：alpha = 1/N（每日 "修正平均数"，RSI 和 ATR 原书均使用此公式）
- **Polars `com` 参数**：alpha = 1/(1+com)，故 `com=N-1` → alpha=1/N ✓
- **Polars `span` 参数**：alpha = 2/(1+span)，故 `span=N` → alpha=2/(N+1) ≠ 1/N ✗
- **结论**：计算 RSI 和 ATR 必须用 `com=N-1`，不可用 `span=N`，两者在小 N 时差异显著
  - N=14: span 给出 alpha≈0.133，com=13 给出 alpha≈0.0714（几乎差 2 倍）

#### 坑 4：`.over("symbol")` 是多品种 DataFrame 的必要条件
- **现象**：若不加 `.over("symbol")`，`diff(1)` / `shift(1)` / `rolling_mean` / `ewm_mean` 会在品种 B 的第一行错误地接续品种 A 的最后一行，产生幻象 delta 和错误指标
- **具体危害**：
  - `diff(1)` 在品种边界处：B 的 row 0 delta = B.close[0] - A.close[-1]（跨品种差值）
  - `shift(1)` 在品种边界处：B 的 row 0 prev_close = A 的最后一根收盘价
- **解法**：所有依赖上下文的算子**全部**追加 `.over("symbol")`
  - `pl.col("close").diff(1).over("symbol")`
  - `pl.col("close").shift(1).over("symbol")`
  - `pl.col("close").rolling_mean(window_size=N).over("symbol")`
  - `pl.col("col").ewm_mean(..., min_samples=N).over("symbol")`
- **验证**：测试中构造 TEST（递增）+ OTHER（递减）混合 DataFrame，RSI 结果：TEST > 70，OTHER < 30，跨品种隔离完美

#### 坑 5：ATR row 0 的 TR 并非 null（max_horizontal 自动忽略 null 分量）
- **现象**：预期 ATR(3) 前 3 行（含 TR 首行 null）为 null，实际只有 2 行为 null
- **根因**：`prev_close = close.shift(1).over("symbol")` 在 row 0 为 null，但：
  ```
  TR = max_horizontal(high-low, |high-null|, |low-null|)
     = max_horizontal(1.0, null, null) = 1.0  ← null 被忽略！
  ```
  `pl.max_horizontal` 在有非 null 分量时自动返回最大非 null 值，TR row 0 = 1.0（非 null）
- **影响**：ATR(3) 的 null 数量 = min_samples - 1 = 2（rows 0, 1），不是 3
- **量化意义**：TR 退化为 `high-low` 在 row 0 是合理的（没有前一日数据时，只计算当根 K 线的价格区间），不引入前视偏差

#### 坑 6：MACD 冷启动期必须对齐 slow 周期
- **问题**：若 fast EMA 用 `min_samples=fast`、slow EMA 用 `min_samples=slow`，DIF = fast_ema - slow_ema 的冷启动期会不对称：fast EMA 从第 fast 行开始非 null，slow EMA 从第 slow 行开始非 null，导致 DIF 在 [fast, slow) 行段出现"半幻象值"（fast 有值但 slow 为 null，DIF 也是 null，但容易混淆）
- **解法**：fast EMA 也使用 `min_samples=slow`，强制两者冷启动期一致，DIF 从第 slow 行开始统一非 null

---

### 2. 量化逻辑备忘：指标计算中的防未来函数措施

#### RSI 的 null 传播链（全部为防未来函数设计）
```
close[t]
  → diff(1).over("symbol")         → delta[t] = close[t] - close[t-1]  (row 0 = null)
  → clip(lower=0)                  → gain[t]  (null 传播，row 0 = null)
  → (-delta).clip(lower=0)         → loss[t]  (null 传播，row 0 = null)
  → ewm_mean(com=N-1, min_samples=N).over("symbol")
                                   → avg_gain[t]  (rows 0..N-1 = null)
                                   → avg_loss[t]  (rows 0..N-1 = null)
  → when(is_null).then(null)
    .when(avg_loss==0).then(100.0)
    .otherwise(100 - 100/(1 + avg_gain/avg_loss))
                                   → RSI[t]  (rows 0..N-1 = null, 严格保留)
```

#### ATR 的 prev_close 设计（防跨品种污染 + 防未来函数）
```python
pl.col("close").shift(1).over("symbol")
```
- `shift(1)` 防未来函数：T 时刻只使用 T-1 的收盘价
- `.over("symbol")` 防跨品种污染：品种 B 的 row 0 prev_close = null（不接续品种 A）

#### MACD bar 乘以 2 的行业背景
- A 股主流软件（通达信、同花顺）的 MACD 柱 = (DIF - DEA) × 2
- 国际标准（Bloomberg、TradingView）的 MACD 柱 = DIF - DEA（不乘 2）
- 本项目按 A 股惯例实现，使用时注意与信号阈值的口径一致性

---

### 3. 关键 API 速查（Polars 1.40.1）

| 算子 | 正确写法 | 注意事项 |
|------|---------|---------|
| SMA | `rolling_mean(window_size=N).over("symbol")` | 前 N-1 行自然 null |
| EMA | `ewm_mean(span=N, adjust=False, min_samples=N).over("symbol")` | `min_samples`（非`min_periods`） |
| Wilder 平滑 | `ewm_mean(com=N-1, adjust=False, min_samples=N).over("symbol")` | `com=N-1`（非`span=N`） |
| diff | `diff(1).over("symbol")` | row 0 严格 null |
| shift | `shift(1).over("symbol")` | row 0 严格 null |
| max_horizontal | `pl.max_horizontal(expr1, expr2, expr3)` | 自动忽略 null 分量 |

---

### 4. 下一步架构建议

1. **指标工厂与信号生成器解耦**：IndicatorFactory 目前只负责计算指标，下一步应创建 `SignalGenerator` 类，将"指标计算"与"信号逻辑（买入/卖出）"分离，符合单一职责原则

2. **日内平仓强制约束**：SOP 要求"绝不过夜"。下一步策略模块应实现交易时段末（15:00 前最后一根 K 线）的强制平仓逻辑，并在单元测试中用跨日边界数据验证

3. **向量化回测引擎**：以 IndicatorFactory 为基础，下一步可搭建纯 Polars 的向量化回测引擎，用 `with_columns` 链式计算持仓、盈亏、资金曲线，避免 Python 层 for 循环

4. **滑点模型**：当前信号以收盘价为基准，实际成交时应引入滑点（如 `fill_price = signal_price * (1 + slippage)`），防止回测过于理想化
