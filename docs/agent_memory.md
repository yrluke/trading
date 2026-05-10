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
