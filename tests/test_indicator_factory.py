"""
tests/test_indicator_factory.py
================================
IndicatorFactory 单元测试：全部在内存中构造微型数据集，无任何文件 I/O。

重点验证：
    - 各指标初期 null 严格保留（防未来函数最高优先级）
    - 精确数学计算验证（SMA、ATR）
    - 合理范围验证（RSI）
    - 多品种混合 DataFrame 下 .over("symbol") 的跨品种隔离

ATR 数学推导（已人工验证）：
    close = [10, 11, ..., 24]，high = close + 0.5，low = close - 0.5
    prev_close（row 1+）= close - 1
    TR = max(high-low, |high-prev_close|, |low-prev_close|)
       = max(1.0, |close+0.5 - (close-1)|, |close-0.5 - (close-1)|)
       = max(1.0, 1.5, 0.5) = 1.5
    故稳定后 ATR ≈ 1.5（Wilder 平滑会从初始值 1.0 收敛至 1.5）
"""

from datetime import datetime, timedelta

import polars as pl
import pytest

from src.indicator_factory import IndicatorFactory

# ──────────────────────────────────────────────────────────────────────────────
# 辅助常量与工具函数
# ──────────────────────────────────────────────────────────────────────────────

TZ = "Asia/Shanghai"
DT = pl.Datetime("us", TZ)

# 15 行单调递增序列 [10, 11, ..., 24]
CLOSES_INC = list(range(10, 25))
# 15 行单调递减序列 [24, 23, ..., 10]
CLOSES_DEC = list(range(24, 9, -1))


def _make_df(closes: list, symbol: str = "TEST") -> pl.DataFrame:
    """
    在内存中构造微型 K 线 DataFrame。

    Parameters
    ----------
    closes : list[float]
        收盘价序列。
    symbol : str
        品种代码，默认 "TEST"。

    Returns
    -------
    pl.DataFrame
        含 symbol, bob, eob, open, close, high, low, volume, amount 列。
        high = close + 0.5，low = close - 0.5，使 TR 结果可精确推算。
    """
    n = len(closes)
    base = datetime(2024, 1, 1, 9, 30, 0)
    bobs = [base + timedelta(minutes=15 * i) for i in range(n)]
    eobs = [b + timedelta(minutes=15) for b in bobs]

    return pl.DataFrame(
        {
            "symbol": [symbol] * n,
            "bob":    pl.Series(bobs).cast(DT),
            "eob":    pl.Series(eobs).cast(DT),
            "open":   [float(c) for c in closes],
            "close":  [float(c) for c in closes],
            "high":   [float(c) + 0.5 for c in closes],
            "low":    [float(c) - 0.5 for c in closes],
            "volume": [1000.0] * n,
            "amount": [float(c) * 1000.0 for c in closes],
        }
    )


# ──────────────────────────────────────────────────────────────────────────────
# SMA 测试
# ──────────────────────────────────────────────────────────────────────────────


class TestSMA:
    """简单移动平均线：null 保留 + 精确数学"""

    @pytest.fixture
    def df(self) -> pl.DataFrame:
        return _make_df(CLOSES_INC)

    def test_sma_null_row0_row1(self, df: pl.DataFrame) -> None:
        """
        断言 1（最高优先）：SMA(3) 前 2 行严格为 null。
        rolling_mean 初期产生的 null 必须原样保留，绝不前向填充。
        """
        out = IndicatorFactory.add_sma(df, window=3)
        col = out["close_sma_3"]
        assert col[0] is None, "row 0 应为 null（窗口数据不足）"
        assert col[1] is None, "row 1 应为 null（窗口数据不足）"
        assert col[2] is not None, "row 2 应为非 null（恰好满足 window=3）"

    def test_sma_exact_value_row2(self, df: pl.DataFrame) -> None:
        """断言 2：SMA(3) row 2 精确等于 (10+11+12)/3 = 11.0"""
        out = IndicatorFactory.add_sma(df, window=3)
        assert out["close_sma_3"][2] == pytest.approx(11.0), (
            f"期望 11.0，实际 {out['close_sma_3'][2]}"
        )

    def test_sma_null_count(self, df: pl.DataFrame) -> None:
        """断言 3：SMA(3) 恰好有 window-1=2 个 null，不多不少"""
        out = IndicatorFactory.add_sma(df, window=3)
        null_count = out["close_sma_3"].null_count()
        assert null_count == 2, f"SMA(3) 应有 2 个 null，实际 {null_count} 个"


# ──────────────────────────────────────────────────────────────────────────────
# EMA 测试
# ──────────────────────────────────────────────────────────────────────────────


class TestEMA:
    """指数移动平均线：冷启动保护验证"""

    @pytest.fixture
    def df(self) -> pl.DataFrame:
        return _make_df(CLOSES_INC)

    def test_ema_coldstart_window3(self, df: pl.DataFrame) -> None:
        """
        断言 4：EMA(3) 前 2 行为 null，row 2 非 null。
        min_periods=window 冷启动保护生效，与 SMA null 行为一致。
        """
        out = IndicatorFactory.add_ema(df, window=3)
        col = out["close_ema_3"]
        assert col[0] is None, "EMA row 0 应为 null（min_periods 冷启动保护）"
        assert col[1] is None, "EMA row 1 应为 null（min_periods 冷启动保护）"
        assert col[2] is not None, "EMA row 2 应为非 null（已满足 min_periods=3）"

    def test_ema_null_count_window5(self, df: pl.DataFrame) -> None:
        """断言 5：EMA(5) 恰好有 window-1=4 个 null（min_periods=5 生效）"""
        out = IndicatorFactory.add_ema(df, window=5)
        null_count = out["close_ema_5"].null_count()
        assert null_count == 4, f"EMA(5) 应有 4 个 null，实际 {null_count} 个"


# ──────────────────────────────────────────────────────────────────────────────
# MACD 测试
# ──────────────────────────────────────────────────────────────────────────────


class TestMACD:
    """MACD：三列完整性 + 冷启动 null + 趋势方向"""

    @pytest.fixture
    def df(self) -> pl.DataFrame:
        return _make_df(CLOSES_INC)

    def test_macd_columns_exist_no_tmp(self, df: pl.DataFrame) -> None:
        """
        断言 6：结果含 diff, dea, bar 三列；无 '_' 开头临时列残留。
        临时列 _macd_fast_tmp / _macd_slow_tmp 必须在计算后 drop。
        """
        out = IndicatorFactory.add_macd(df, fast=3, slow=5, signal=3)
        assert "close_macd_diff" in out.columns
        assert "close_macd_dea" in out.columns
        assert "close_macd_bar" in out.columns
        tmp_cols = [c for c in out.columns if c.startswith("_")]
        assert tmp_cols == [], f"存在未清理的临时列: {tmp_cols}"

    def test_macd_coldstart_null(self, df: pl.DataFrame) -> None:
        """
        断言 7：fast=3, slow=5 时，diff 前 slow-1=4 行为 null。
        fast/slow EMA 均使用 min_periods=slow，确保冷启动期对齐。
        """
        out = IndicatorFactory.add_macd(df, fast=3, slow=5, signal=3)
        diff_col = out["close_macd_diff"]
        for i in range(4):
            assert diff_col[i] is None, f"diff row {i} 应为 null（冷启动期）"
        assert diff_col[4] is not None, "diff row 4 应为非 null（冷启动期结束）"

    def test_macd_diff_positive_for_uptrend(self, df: pl.DataFrame) -> None:
        """断言 8：单调递增序列末尾 diff > 0（fast EMA 收敛快，始终高于 slow EMA）"""
        out = IndicatorFactory.add_macd(df, fast=3, slow=5, signal=3)
        last_diff = out["close_macd_diff"][-1]
        assert last_diff is not None, "末尾 diff 不应为 null"
        assert last_diff > 0, f"单调递增时末尾 diff 应 > 0，实际: {last_diff}"


# ──────────────────────────────────────────────────────────────────────────────
# RSI 测试
# ──────────────────────────────────────────────────────────────────────────────


class TestRSI:
    """RSI：高值验证 + null 数量 + 合法范围"""

    @pytest.fixture
    def df(self) -> pl.DataFrame:
        return _make_df(CLOSES_INC)

    def test_rsi_high_for_uptrend(self, df: pl.DataFrame) -> None:
        """
        断言 9：全正收益序列（单调递增）末尾 RSI 应 > 90。
        avg_loss → 0 时 RSI → 100，断言 > 90 留有宽松容差。
        """
        out = IndicatorFactory.add_rsi(df, window=5)
        last_rsi = out["close_rsi_5"][-1]
        assert last_rsi is not None, "末尾 RSI 不应为 null"
        assert last_rsi > 90, f"全正收益末尾 RSI 应 > 90，实际: {last_rsi:.4f}"

    def test_rsi_null_count(self, df: pl.DataFrame) -> None:
        """
        断言 10：RSI(5) 恰好有 window=5 个 null。
        diff(1) 使 row 0 为 null（1 个 null），
        ewm_mean(min_periods=5) 需 5 个非 null 值才计算，
        故 rows 0-4（共 5 行）均为 null。
        """
        out = IndicatorFactory.add_rsi(df, window=5)
        null_count = out["close_rsi_5"].null_count()
        assert null_count == 5, f"RSI(5) 应有 5 个 null，实际 {null_count} 个"

    def test_rsi_range(self, df: pl.DataFrame) -> None:
        """断言 11：所有非 null RSI 值应在 [0, 100] 范围内"""
        out = IndicatorFactory.add_rsi(df, window=5)
        rsi_valid = out["close_rsi_5"].drop_nulls()
        assert len(rsi_valid) > 0, "应存在非 null RSI 值"
        assert float(rsi_valid.min()) >= 0.0, f"RSI 最小值 {rsi_valid.min()} < 0"
        assert float(rsi_valid.max()) <= 100.0, f"RSI 最大值 {rsi_valid.max()} > 100"


# ──────────────────────────────────────────────────────────────────────────────
# ATR 测试
# ──────────────────────────────────────────────────────────────────────────────


class TestATR:
    """ATR：null 保留 + 数学精度（Wilder 平滑）+ 临时列清理"""

    @pytest.fixture
    def df(self) -> pl.DataFrame:
        return _make_df(CLOSES_INC)

    def test_atr_null_preservation(self, df: pl.DataFrame) -> None:
        """
        断言 12：ATR(3) 前 2 行为 null（min_periods=3 冷启动保护）。
        row 0 的 TR 虽非 null（prev_close null 时退化为 high-low），
        但 ewm_mean(min_periods=3) 需 3 个非 null TR 才计算，故 rows 0,1 为 null。
        """
        out = IndicatorFactory.add_atr(df, window=3)
        col = out["atr_3"]
        assert col[0] is None, "ATR row 0 应为 null（min_periods 冷启动保护）"
        assert col[1] is None, "ATR row 1 应为 null（min_periods 冷启动保护）"
        assert col[2] is not None, "ATR row 2 应为非 null（3 个 TR 满足 min_periods）"

    def test_atr_converges_to_expected(self, df: pl.DataFrame) -> None:
        """
        断言 13：末尾 ATR(3) 应收敛至约 1.5（相对误差 ≤ 5%）。

        数学推导：
            high = close + 0.5, low = close - 0.5
            row 0: prev_close = null → TR = max(1.0, null, null) = 1.0
            row 1+: prev_close = close - 1
                    TR = max(1.0, |close+0.5 - (close-1)|, |close-0.5 - (close-1)|)
                       = max(1.0, 1.5, 0.5) = 1.5
            Wilder 平滑从 1.0 开始，随时间指数收敛至 1.5。
        """
        out = IndicatorFactory.add_atr(df, window=3)
        last_atr = float(out["atr_3"][-1])
        assert abs(last_atr - 1.5) / 1.5 < 0.05, (
            f"ATR(3) 末尾值应约为 1.5，实际 {last_atr:.6f}（相对误差超过 5%）"
        )

    def test_atr_no_tmp_columns(self, df: pl.DataFrame) -> None:
        """断言 14：结果 DataFrame 无 '_' 开头临时列（_atr_prev_tmp / _atr_tr_tmp 已清理）"""
        out = IndicatorFactory.add_atr(df, window=3)
        tmp_cols = [c for c in out.columns if c.startswith("_")]
        assert tmp_cols == [], f"存在未清理的临时列: {tmp_cols}"


# ──────────────────────────────────────────────────────────────────────────────
# 多品种隔离测试（.over("symbol") 核心验证）
# ──────────────────────────────────────────────────────────────────────────────


class TestMultiSymbolIsolation:
    """
    验证多品种混合 DataFrame 下 .over("symbol") 的跨品种隔离效果。

    测试方法：
        - 构造 TEST（递增 [10..24]）+ OTHER（递减 [24..10]）拼接的混合 DataFrame
        - 直接传入工厂方法，通过 .over("symbol") 分组计算
        - filter 各品种结果，验证数学正确性与品种独立性
    """

    @pytest.fixture
    def mixed_df(self) -> pl.DataFrame:
        """TEST（递增）+ OTHER（递减）拼接的混合 DataFrame，模拟真实多品种场景"""
        df_test  = _make_df(CLOSES_INC, symbol="TEST")
        df_other = _make_df(CLOSES_DEC, symbol="OTHER")
        return pl.concat([df_test, df_other])

    def test_sma_isolation(self, mixed_df: pl.DataFrame) -> None:
        """
        断言 15：多品种 SMA 隔离验证。
        TEST row 2  = (10+11+12)/3 = 11.0
        OTHER row 2 = (24+23+22)/3 = 23.0
        两者前 2 行均为 null；若无 .over("symbol")，OTHER row 0 会接续 TEST 数据。
        """
        out = IndicatorFactory.add_sma(mixed_df, window=3)
        test_rows  = out.filter(pl.col("symbol") == "TEST")
        other_rows = out.filter(pl.col("symbol") == "OTHER")

        # 前两行 null（各品种独立冷启动）
        assert test_rows["close_sma_3"][0]  is None, "TEST row 0 应为 null"
        assert test_rows["close_sma_3"][1]  is None, "TEST row 1 应为 null"
        assert other_rows["close_sma_3"][0] is None, "OTHER row 0 应为 null"
        assert other_rows["close_sma_3"][1] is None, "OTHER row 1 应为 null"

        # 精确数学验证
        assert test_rows["close_sma_3"][2]  == pytest.approx(11.0), "TEST SMA(3) row2 应为 11.0"
        assert other_rows["close_sma_3"][2] == pytest.approx(23.0), "OTHER SMA(3) row2 应为 23.0"

    def test_rsi_isolation(self, mixed_df: pl.DataFrame) -> None:
        """
        断言 16：多品种 RSI 隔离验证。
        TEST（单调递增，全涨）末尾 RSI 应 > 70。
        OTHER（单调递减，全跌）末尾 RSI 应 < 30。
        若无 .over("symbol")，品种边界处会产生错误的 delta，污染 RSI 计算。
        """
        out = IndicatorFactory.add_rsi(mixed_df, window=5)
        test_rows  = out.filter(pl.col("symbol") == "TEST")
        other_rows = out.filter(pl.col("symbol") == "OTHER")

        last_rsi_test  = test_rows["close_rsi_5"][-1]
        last_rsi_other = other_rows["close_rsi_5"][-1]

        assert last_rsi_test  is not None, "TEST 末尾 RSI 不应为 null"
        assert last_rsi_other is not None, "OTHER 末尾 RSI 不应为 null"
        assert last_rsi_test  > 70, f"TEST（递增）末尾 RSI 应 > 70，实际: {last_rsi_test:.4f}"
        assert last_rsi_other < 30, f"OTHER（递减）末尾 RSI 应 < 30，实际: {last_rsi_other:.4f}"
