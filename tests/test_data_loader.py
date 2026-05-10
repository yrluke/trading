"""
tests/test_data_loader.py
=========================
DataLoader 单元测试：全部在内存中构造微型数据集，无任何文件 I/O。

重点验证：
    - align_daily_to_15m 的反未来函数正确性
    - 停牌/节假日（日线缺失）场景下的自动回溯
    - 多品种隔离（不同 symbol 数据互不污染）
    - load_kbars 非法 freq 参数的异常抛出
"""

from datetime import datetime

import polars as pl
import pytest

from src.data_loader import DataLoader

# ──────────────────────────────────────────────────────────────────────────────
# 辅助常量：与 build_database.py / data_loader.py 保持完全一致的时间类型
# ──────────────────────────────────────────────────────────────────────────────
TZ = "Asia/Shanghai"
DT = pl.Datetime("us", TZ)


def _dt(s: str) -> datetime:
    """将 'YYYY-MM-DD HH:MM:SS' 字符串解析为带时区的 datetime（仅用于测试）。"""
    return datetime.fromisoformat(s).replace(tzinfo=None)


# ──────────────────────────────────────────────────────────────────────────────
# Fixtures：内存中构造微型数据集
# ──────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def loader() -> DataLoader:
    return DataLoader(data_dir="data/kbars")


@pytest.fixture
def df_1d_normal() -> pl.DataFrame:
    """
    正常日线：TEST 品种，含 2024-01-01 和 2024-01-02 两根日线。
    eob = 当天 15:00（收盘时刻，数据正式确定时间）。
    """
    return pl.DataFrame(
        {
            "symbol": ["TEST", "TEST"],
            "bob": pl.Series(
                [
                    datetime(2024, 1, 1, 9, 30, 0),
                    datetime(2024, 1, 2, 9, 30, 0),
                ],
            ).cast(DT),
            "eob": pl.Series(
                [
                    datetime(2024, 1, 1, 15, 0, 0),
                    datetime(2024, 1, 2, 15, 0, 0),
                ],
            ).cast(DT),
            "open":   [100.0, 110.0],
            "close":  [105.0, 115.0],
            "high":   [108.0, 118.0],
            "low":    [98.0,  108.0],
            "volume": [1000.0, 2000.0],
            "amount": [1050000.0, 2200000.0],
        }
    )


@pytest.fixture
def df_1d_with_gap(df_1d_normal: pl.DataFrame) -> pl.DataFrame:
    """
    停牌场景：删除 2024-01-02 的日线，模拟该日停牌/数据缺失。
    2024-01-02 的 15m 行应回溯到 2024-01-01 的日线。
    """
    return df_1d_normal.filter(pl.col("bob").dt.date() != pl.lit("2024-01-02").str.to_date())


@pytest.fixture
def df_15m_normal() -> pl.DataFrame:
    """
    正常 15m 数据：TEST 品种，跨两个交易日，含边界时刻。
    """
    return pl.DataFrame(
        {
            "symbol": ["TEST", "TEST", "TEST"],
            "bob": pl.Series(
                [
                    datetime(2024, 1, 1, 9, 30, 0),   # 首日第一根，无前一日
                    datetime(2024, 1, 2, 10, 0, 0),   # 第二日日内
                    datetime(2024, 1, 2, 14, 45, 0),  # 第二日尾盘
                ],
            ).cast(DT),
            "open":   [101.0, 111.0, 112.0],
            "close":  [102.0, 112.0, 113.0],
            "high":   [103.0, 114.0, 115.0],
            "low":    [99.0,  109.0, 110.0],
            "volume": [100.0, 200.0, 150.0],
            "amount": [10200.0, 22400.0, 16950.0],
        }
    )


@pytest.fixture
def df_15m_multi_symbol(df_15m_normal: pl.DataFrame) -> pl.DataFrame:
    """
    多品种场景：在 TEST 之外追加 OTHER 品种的一根 15m 行。
    """
    other = pl.DataFrame(
        {
            "symbol": ["OTHER"],
            "bob": pl.Series([datetime(2024, 1, 2, 10, 0, 0)]).cast(DT),
            "open":   [200.0],
            "close":  [201.0],
            "high":   [202.0],
            "low":    [199.0],
            "volume": [50.0],
            "amount": [10050.0],
        }
    )
    return pl.concat([df_15m_normal, other])


@pytest.fixture
def df_1d_multi_symbol(df_1d_normal: pl.DataFrame) -> pl.DataFrame:
    """
    多品种场景：在 TEST 之外追加 OTHER 品种的日线（close 与 TEST 明显不同）。
    """
    other = pl.DataFrame(
        {
            "symbol": ["OTHER", "OTHER"],
            "bob": pl.Series(
                [datetime(2024, 1, 1, 9, 30, 0), datetime(2024, 1, 2, 9, 30, 0)]
            ).cast(DT),
            "eob": pl.Series(
                [datetime(2024, 1, 1, 15, 0, 0), datetime(2024, 1, 2, 15, 0, 0)]
            ).cast(DT),
            "open":   [500.0, 510.0],
            "close":  [505.0, 515.0],
            "high":   [508.0, 518.0],
            "low":    [498.0, 508.0],
            "volume": [3000.0, 4000.0],
            "amount": [1515000.0, 2060000.0],
        }
    )
    return pl.concat([df_1d_normal, other])


# ──────────────────────────────────────────────────────────────────────────────
# 测试用例
# ──────────────────────────────────────────────────────────────────────────────

class TestAlignDailyTo15m:

    def test_assert_a_second_day_morning_gets_first_day_close(
        self,
        loader: DataLoader,
        df_15m_normal: pl.DataFrame,
        df_1d_normal: pl.DataFrame,
    ) -> None:
        """
        断言 A：2024-01-02 10:00 的 15m 行，prev_1d_close 应等于 2024-01-01 日线的 close(105.0)，
        而不是 2024-01-02 日线的 close(115.0)。
        这是反未来函数的核心验证。
        """
        result = loader.align_daily_to_15m(df_15m_normal, df_1d_normal)
        row = result.filter(
            pl.col("bob") == pl.lit(datetime(2024, 1, 2, 10, 0, 0)).cast(DT)
        )
        assert len(row) == 1
        prev_close = row["prev_1d_close"][0]
        assert prev_close == 105.0, (
            f"2024-01-02 10:00 应匹配 2024-01-01 close=105.0，实际得到 {prev_close}"
        )

    def test_assert_b_second_day_close_bar_same_prev(
        self,
        loader: DataLoader,
        df_15m_normal: pl.DataFrame,
        df_1d_normal: pl.DataFrame,
    ) -> None:
        """
        断言 B：2024-01-02 14:45 的 15m 行，prev_1d_close 与 10:00 相同（均为 105.0）。
        一天内所有 15m bar 共享同一前一日日线特征。
        """
        result = loader.align_daily_to_15m(df_15m_normal, df_1d_normal)
        row = result.filter(
            pl.col("bob") == pl.lit(datetime(2024, 1, 2, 14, 45, 0)).cast(DT)
        )
        assert len(row) == 1
        prev_close = row["prev_1d_close"][0]
        assert prev_close == 105.0, (
            f"2024-01-02 14:45 应匹配 2024-01-01 close=105.0，实际得到 {prev_close}"
        )

    def test_assert_c_first_bar_has_null_prev(
        self,
        loader: DataLoader,
        df_15m_normal: pl.DataFrame,
        df_1d_normal: pl.DataFrame,
    ) -> None:
        """
        断言 C：2024-01-01 09:30 的 15m 行，prev_1d_close 应为 null。
        当时 eob_1d <= bob_15m 不存在满足条件的日线（首个交易日）。
        """
        result = loader.align_daily_to_15m(df_15m_normal, df_1d_normal)
        row = result.filter(
            pl.col("bob") == pl.lit(datetime(2024, 1, 1, 9, 30, 0)).cast(DT)
        )
        assert len(row) == 1
        assert row["prev_1d_close"][0] is None, (
            "首个交易日第一根 15m bar 不应有前一日数据，应为 null"
        )

    def test_assert_d_suspension_fallback_not_null(
        self,
        loader: DataLoader,
        df_15m_normal: pl.DataFrame,
        df_1d_with_gap: pl.DataFrame,
    ) -> None:
        """
        断言 D（停牌测试）：2024-01-02 日线缺失时，该日的 15m 行应自动回溯到
        2024-01-01 的日线数据（prev_1d_close == 105.0），而不是 null。
        这是 join_asof 相比 shift+date_join 的核心优势。
        """
        result = loader.align_daily_to_15m(df_15m_normal, df_1d_with_gap)
        rows_day2 = result.filter(
            pl.col("bob").dt.date() == pl.lit("2024-01-02").str.to_date()
        )
        assert len(rows_day2) == 2  # 10:00 和 14:45 两行
        for prev_close in rows_day2["prev_1d_close"].to_list():
            assert prev_close == 105.0, (
                f"停牌日应回溯到 2024-01-01 close=105.0，实际得到 {prev_close}"
            )

    def test_assert_e_multi_symbol_isolation(
        self,
        loader: DataLoader,
        df_15m_multi_symbol: pl.DataFrame,
        df_1d_multi_symbol: pl.DataFrame,
    ) -> None:
        """
        断言 E（多品种隔离）：TEST 品种的 15m 行只能匹配 TEST 的日线，
        OTHER 品种的日线（close=505.0）不能污染 TEST 的 prev_1d_close。
        """
        result = loader.align_daily_to_15m(df_15m_multi_symbol, df_1d_multi_symbol)

        test_rows = result.filter(
            (pl.col("symbol") == "TEST")
            & (pl.col("bob").dt.date() == pl.lit("2024-01-02").str.to_date())
        )
        other_rows = result.filter(pl.col("symbol") == "OTHER")

        # TEST 的 prev_1d_close 应为 TEST 日线的值（105.0），而非 OTHER 的（505.0）
        for prev_close in test_rows["prev_1d_close"].to_list():
            assert prev_close == 105.0, (
                f"TEST 品种不应匹配 OTHER 的日线，期望 105.0，实际 {prev_close}"
            )

        # OTHER 的 prev_1d_close 应为 OTHER 日线的值（505.0）
        for prev_close in other_rows["prev_1d_close"].to_list():
            assert prev_close == 505.0, (
                f"OTHER 品种应匹配自己的日线 505.0，实际 {prev_close}"
            )


class TestLoadKbarsValidation:

    def test_invalid_freq_raises_value_error(self, loader: DataLoader) -> None:
        """非法 freq 参数应立即抛出 ValueError，不进行任何文件 I/O。"""
        with pytest.raises(ValueError, match="freq 必须为"):
            loader.load_kbars("000001", "2024-01-01", "2024-01-31", freq="5m")
