"""
src/data_loader.py
==================
DataLoader：从 Parquet 分区目录高效加载 K 线数据，并实现反未来函数的多周期对齐。

设计约束（SOP 量化铁律）：
    - 严禁引用未来数据（No Look-ahead Bias）
    - 所有指标/特征在内存中动态计算，严禁持久化
    - align_daily_to_15m 使用 join_asof(strategy="backward")，
      数学等价于：prev_1d = max{eob_1d | eob_1d <= bob_15m, same symbol}
      自然规避停牌/节假日日线断层导致的 null 异常
"""

import logging
import time
from pathlib import Path

import polars as pl

logger = logging.getLogger(__name__)

# 与 build_database.py 保持完全一致的时间戳类型
_DT_TYPE = pl.Datetime("us", "Asia/Shanghai")

# 日线特征列：重命名后带入 15m（原始 OHLCV，不含技术指标）
_1D_FEATURE_COLS = ["open", "close", "high", "low", "volume", "amount"]


class DataLoader:
    """
    从 Hive 分区 Parquet 目录加载 K 线数据，并提供多周期安全对齐。

    Parameters
    ----------
    data_dir : str | Path
        Parquet 根目录，默认 "data/kbars"。
        期望子结构：
            <data_dir>/15m/year=YYYY/month=YYYYMM/data_0.parquet
            <data_dir>/1d/year=YYYY/data_0.parquet
    """

    def __init__(self, data_dir: str | Path = "data/kbars") -> None:
        self.data_dir = Path(data_dir)

    # ──────────────────────────────────────────────────────────────────────
    # 公开方法 1：惰性加载 K 线
    # ──────────────────────────────────────────────────────────────────────

    def load_kbars(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        freq: str,
    ) -> pl.DataFrame:
        """
        从 Parquet 分区目录中惰性加载指定品种、时间范围的 K 线数据。

        利用 Polars scan_parquet 的谓词下推（predicate pushdown）和
        Hive 分区剪枝（partition pruning）特性，只读取必要的文件分片。

        Parameters
        ----------
        symbol : str
            品种代码，如 "000001"。
        start_date : str
            起始日期（含），格式 "YYYY-MM-DD"。
        end_date : str
            结束日期（含），格式 "YYYY-MM-DD"。
        freq : str
            频率，"15m" 或 "1d"。

        Returns
        -------
        pl.DataFrame
            按 bob 升序排列的 K 线 DataFrame。

        Raises
        ------
        ValueError
            freq 不在 {"15m", "1d"} 时抛出。
        """
        if freq not in {"15m", "1d"}:
            raise ValueError(f"freq 必须为 '15m' 或 '1d'，收到: {freq!r}")

        pattern = str(self.data_dir / freq / "**" / "*.parquet")

        # 将日期字符串转为带时区的 datetime，用于 Polars 谓词过滤
        start_dt = pl.lit(start_date).str.to_datetime(
            "%Y-%m-%d", time_unit="us", time_zone="Asia/Shanghai"
        )
        end_dt = (
            pl.lit(end_date)
            .str.to_datetime("%Y-%m-%d", time_unit="us", time_zone="Asia/Shanghai")
            .dt.offset_by("1d")  # end_date 当天全天均包含
        )

        t0 = time.perf_counter()
        df = (
            pl.scan_parquet(pattern, hive_partitioning=True)
            .filter(
                (pl.col("symbol") == symbol)
                & (pl.col("bob") >= start_dt)
                & (pl.col("bob") < end_dt)
            )
            .sort("bob")
            .collect()
        )
        elapsed = time.perf_counter() - t0

        logger.info(
            "load_kbars | symbol=%s freq=%s [%s, %s] → %d 行，耗时 %.3fs",
            symbol, freq, start_date, end_date, len(df), elapsed,
        )
        return df

    # ──────────────────────────────────────────────────────────────────────
    # 公开方法 2：反未来函数多周期对齐
    # ──────────────────────────────────────────────────────────────────────

    def align_daily_to_15m(
        self,
        df_15m: pl.DataFrame,
        df_1d: pl.DataFrame,
    ) -> pl.DataFrame:
        """
        将日线特征安全地合并到 15 分钟线，严格防止未来函数。

        核心逻辑（join_asof，strategy="backward"）：
            对每根 15m K 线（bob_15m），在同一品种的日线中寻找
            满足 eob_1d <= bob_15m 的最新一根日线，取其 OHLCV 特征。

        数学等价：
            prev_1d = argmax{ eob_1d | eob_1d <= bob_15m, symbol 相同 }

        停牌/节假日处理：
            若某交易日日线缺失，自动回溯到更早的已确认日线，不产生 null。
            只有在 15m 数据早于所有已知日线 eob 时才出现 null（正确行为）。

        Parameters
        ----------
        df_15m : pl.DataFrame
            15 分钟 K 线，必须含 "bob"(Datetime us Asia/Shanghai) 和 "symbol" 列。
        df_1d : pl.DataFrame
            日线 K 线，必须含 "eob"(Datetime us Asia/Shanghai)、"symbol" 及
            open/close/high/low/volume/amount 列。

        Returns
        -------
        pl.DataFrame
            df_15m 附加 prev_1d_open / prev_1d_close / prev_1d_high /
            prev_1d_low / prev_1d_volume / prev_1d_amount 六列。
        """
        # 1. 日线：只保留必要列，重命名 OHLCV 为 prev_1d_* 避免与 15m 冲突
        rename_map = {col: f"prev_1d_{col}" for col in _1D_FEATURE_COLS}
        df_1d_slim = (
            df_1d
            .select(["symbol", "eob"] + _1D_FEATURE_COLS)
            .rename(rename_map)
            .sort(["symbol", "eob"])          # join_asof 强制要求右表按 key 有序
        )

        # 2. 15m：按 (symbol, bob) 排序（join_asof 要求左表也有序）
        df_15m_sorted = df_15m.sort(["symbol", "bob"])

        # 3. join_asof：天然反未来函数
        #    left_on="bob"  → 15m K 线开始时间
        #    right_on="eob" → 日线收盘时间（数据正式确定时刻）
        #    strategy="backward" → 寻找 eob_1d <= bob_15m 的最近日线
        #    by="symbol"    → 严格按品种隔离
        df_aligned = df_15m_sorted.join_asof(
            df_1d_slim,
            left_on="bob",
            right_on="eob",
            by="symbol",
            strategy="backward",
        )

        logger.info(
            "align_daily_to_15m | 15m 行数=%d，对齐后行数=%d，"
            "null prev_1d_close 行数=%d",
            len(df_15m), len(df_aligned),
            df_aligned["prev_1d_close"].null_count(),
        )
        return df_aligned
