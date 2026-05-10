"""
src/indicator_factory.py
========================
IndicatorFactory：在内存中动态计算技术指标。

设计原则（SOP 量化铁律）：
    - 输入一个含原始 K 线数据的 Polars DataFrame（支持多品种混合输入）
    - 输出一个追加了各项技术指标列的新 DataFrame
    - 绝对不将指标持久化到磁盘（Compute on-the-fly）
    - 所有依赖上下文的算子（shift/diff/rolling_mean/ewm_mean）均追加
      .over("symbol")，Polars 在底层按品种分组后再滑动，彻底杜绝
      跨品种边界污染（否则品种 B 的第一行会错误地接续品种 A 的最后一行）
    - 滑动窗口初期产生的 null 值严格保留，不做前向填充或插值（防未来函数）
    - EMA 统一使用 min_periods=window 冷启动保护，与 SMA 的 null 行为对齐，
      防止权重未收敛的早期假信号进入回测

依赖列要求：
    - 公共列：symbol (Utf8), close (Float64)
    - add_atr 额外需要：high (Float64), low (Float64)
"""

import logging

import polars as pl

logger = logging.getLogger(__name__)


class IndicatorFactory:
    """
    动态技术指标工厂。

    所有方法均为静态方法，接受 Polars DataFrame 作为输入，
    返回追加了指标列的新 DataFrame。

    多品种支持：
        DataFrame 中可包含多个 symbol，所有滑动/累积算子均追加
        .over("symbol")，确保按品种分组计算，互不污染。

    量化铁律：
        - 所有依赖上下文算子追加 .over("symbol")，防止跨品种数据污染
        - 初期 null 值严格保留，绝不前向填充
        - 所有计算均在内存中完成，不写入磁盘
    """

    # ──────────────────────────────────────────────────────────────────────
    # SMA：简单移动平均线
    # ──────────────────────────────────────────────────────────────────────

    @staticmethod
    def add_sma(
        df: pl.DataFrame,
        column: str = "close",
        window: int = 20,
    ) -> pl.DataFrame:
        """
        添加简单移动平均线 (Simple Moving Average, SMA)。

        Parameters
        ----------
        df : pl.DataFrame
            含 symbol、{column} 列的原始 K 线 DataFrame（支持多品种混合）。
        column : str
            目标价格列，默认 "close"。
        window : int
            滑动窗口大小，默认 20。

        Returns
        -------
        pl.DataFrame
            追加了 "{column}_sma_{window}" 列的 DataFrame。
            前 window-1 行严格为 null（防未来函数，rolling_mean 自然保证）。

        Notes
        -----
        使用 .over("symbol") 按品种分组滑动，彻底防止跨品种边界污染。
        """
        col_name = f"{column}_sma_{window}"
        logger.debug("计算 SMA: column=%s, window=%d, output=%s", column, window, col_name)
        return df.with_columns(
            pl.col(column)
            .rolling_mean(window_size=window)
            .over("symbol")
            .alias(col_name)
        )

    # ──────────────────────────────────────────────────────────────────────
    # EMA：指数移动平均线
    # ──────────────────────────────────────────────────────────────────────

    @staticmethod
    def add_ema(
        df: pl.DataFrame,
        column: str = "close",
        window: int = 20,
    ) -> pl.DataFrame:
        """
        添加指数移动平均线 (Exponential Moving Average, EMA)，含冷启动保护。

        Parameters
        ----------
        df : pl.DataFrame
            含 symbol、{column} 列的原始 K 线 DataFrame（支持多品种混合）。
        column : str
            目标价格列，默认 "close"。
        window : int
            EMA 周期（span），默认 20。权重衰减系数 alpha = 2 / (window + 1)。

        Returns
        -------
        pl.DataFrame
            追加了 "{column}_ema_{window}" 列的 DataFrame。
            前 window-1 行严格为 null（冷启动保护）。

        Notes
        -----
        - min_periods=window 是关键冷启动保护参数：确保前 window-1 行为 null，
          与 SMA 的 null 行为完全一致，防止权重未收敛的早期假信号。
        - adjust=False 使用递推公式（非加权求和），与 Pandas 的 ewm 行为一致。
        - 使用 .over("symbol") 按品种分组计算。
        """
        col_name = f"{column}_ema_{window}"
        logger.debug("计算 EMA: column=%s, window=%d, output=%s", column, window, col_name)
        return df.with_columns(
            pl.col(column)
            .ewm_mean(span=window, adjust=False, min_samples=window)
            .over("symbol")
            .alias(col_name)
        )

    # ──────────────────────────────────────────────────────────────────────
    # MACD：移动平均线收敛/发散指标
    # ──────────────────────────────────────────────────────────────────────

    @staticmethod
    def add_macd(
        df: pl.DataFrame,
        column: str = "close",
        fast: int = 12,
        slow: int = 26,
        signal: int = 9,
    ) -> pl.DataFrame:
        """
        添加 MACD 三线（DIF / DEA / MACD 柱）。

        Parameters
        ----------
        df : pl.DataFrame
            含 symbol、{column} 列的原始 K 线 DataFrame（支持多品种混合）。
        column : str
            目标价格列，默认 "close"。
        fast : int
            快线 EMA 周期，默认 12。
        slow : int
            慢线 EMA 周期，默认 26。
        signal : int
            信号线（DEA）EMA 周期，默认 9。

        Returns
        -------
        pl.DataFrame
            追加了以下三列的 DataFrame：
            - "{column}_macd_diff"：快慢线差值（DIF）
            - "{column}_macd_dea" ：信号线（DEA）
            - "{column}_macd_bar" ：柱状图 = (DIF - DEA) × 2（中国市场惯例）

        Notes
        -----
        - fast/slow EMA 均使用 min_periods=slow：冷启动期对齐 slow 周期，
          确保前 slow-1 行的 DIF 为 null，防止早期假信号。
        - MACD 柱乘以 2 是 A 股软件（通达信/同花顺）的行业惯例。
        - 中间临时列以 "_macd_*_tmp" 命名，计算完成后自动 drop，不污染结果。
        - 使用 .over("symbol") 按品种分组计算。
        """
        fast_tmp = "_macd_fast_tmp"
        slow_tmp = "_macd_slow_tmp"
        diff_col = f"{column}_macd_diff"
        dea_col  = f"{column}_macd_dea"
        bar_col  = f"{column}_macd_bar"

        logger.debug(
            "计算 MACD: column=%s, fast=%d, slow=%d, signal=%d",
            column, fast, slow, signal,
        )

        return (
            df
            # Step 1: 快线与慢线 EMA（冷启动期统一对齐 slow 周期）
            .with_columns(
                pl.col(column)
                .ewm_mean(span=fast, adjust=False, min_samples=slow)
                .over("symbol")
                .alias(fast_tmp),

                pl.col(column)
                .ewm_mean(span=slow, adjust=False, min_samples=slow)
                .over("symbol")
                .alias(slow_tmp),
            )
            # Step 2: DIF = 快线 EMA - 慢线 EMA
            .with_columns(
                (pl.col(fast_tmp) - pl.col(slow_tmp)).alias(diff_col)
            )
            # Step 3: DEA = DIF 的 EMA（信号线）
            .with_columns(
                pl.col(diff_col)
                .ewm_mean(span=signal, adjust=False, min_samples=signal)
                .over("symbol")
                .alias(dea_col),
            )
            # Step 4: MACD 柱 = (DIF - DEA) × 2
            .with_columns(
                ((pl.col(diff_col) - pl.col(dea_col)) * 2.0).alias(bar_col)
            )
            # Step 5: 清理临时列
            .drop([fast_tmp, slow_tmp])
        )

    # ──────────────────────────────────────────────────────────────────────
    # RSI：相对强弱指标
    # ──────────────────────────────────────────────────────────────────────

    @staticmethod
    def add_rsi(
        df: pl.DataFrame,
        column: str = "close",
        window: int = 14,
    ) -> pl.DataFrame:
        """
        添加相对强弱指标 (Relative Strength Index, RSI)，使用 Wilder 平滑法。

        Parameters
        ----------
        df : pl.DataFrame
            含 symbol、{column} 列的原始 K 线 DataFrame（支持多品种混合）。
        column : str
            目标价格列，默认 "close"。
        window : int
            RSI 周期，默认 14。

        Returns
        -------
        pl.DataFrame
            追加了 "{column}_rsi_{window}" 列的 DataFrame。
            前 window 行为 null（diff 首行 null + min_periods=window 共同作用）。

        Notes
        -----
        - Wilder 平滑：ewm_mean(com=window-1, adjust=False)，
          等价 alpha=1/window。注意此处使用 com（Center of Mass）而非 span，
          因为 com=N-1 → alpha=1/N，与 span=2N-1 → alpha=1/N 等价，
          是 Wilder 原版定义（RSI 书中的"修正平均数"）。
        - avg_loss == 0（全涨期）时 RSI 强制返回 100.0，防止除零产生 inf。
        - avg_gain 为 null（冷启动期）时显式返回 null，严格保留。
        - diff(1).over("symbol") 保证首行为 null 且不跨品种。
        - 中间临时列用完即 drop，不污染结果。
        """
        delta_tmp    = "_rsi_delta_tmp"
        gain_tmp     = "_rsi_gain_tmp"
        loss_tmp     = "_rsi_loss_tmp"
        avg_gain_tmp = "_rsi_avg_gain_tmp"
        avg_loss_tmp = "_rsi_avg_loss_tmp"
        col_name     = f"{column}_rsi_{window}"

        logger.debug("计算 RSI: column=%s, window=%d, output=%s", column, window, col_name)

        return (
            df
            # Step 1: 价格变化量（diff(1).over("symbol") 保证首行为 null，防未来函数）
            .with_columns(
                pl.col(column).diff(1).over("symbol").alias(delta_tmp)
            )
            # Step 2: 分离涨跌幅（clip 为纯列操作，null 自动传播）
            .with_columns(
                pl.col(delta_tmp).clip(lower_bound=0).alias(gain_tmp),
                (-pl.col(delta_tmp)).clip(lower_bound=0).alias(loss_tmp),
            )
            # Step 3: Wilder 平滑均值（com=window-1 等价 alpha=1/window）
            .with_columns(
                pl.col(gain_tmp)
                .ewm_mean(com=window - 1, adjust=False, min_samples=window)
                .over("symbol")
                .alias(avg_gain_tmp),

                pl.col(loss_tmp)
                .ewm_mean(com=window - 1, adjust=False, min_samples=window)
                .over("symbol")
                .alias(avg_loss_tmp),
            )
            # Step 4: 计算 RSI，含除零保护与显式 null 保留
            .with_columns(
                pl.when(pl.col(avg_gain_tmp).is_null())
                .then(pl.lit(None).cast(pl.Float64))       # 冷启动期严格保 null
                .when(pl.col(avg_loss_tmp) == 0.0)
                .then(pl.lit(100.0))                       # 全涨期：RSI = 100
                .otherwise(
                    100.0
                    - 100.0 / (1.0 + pl.col(avg_gain_tmp) / pl.col(avg_loss_tmp))
                )
                .alias(col_name)
            )
            # Step 5: 清理所有临时列
            .drop([delta_tmp, gain_tmp, loss_tmp, avg_gain_tmp, avg_loss_tmp])
        )

    # ──────────────────────────────────────────────────────────────────────
    # ATR：真实波动幅度（Wilder 原版定义）
    # ──────────────────────────────────────────────────────────────────────

    @staticmethod
    def add_atr(
        df: pl.DataFrame,
        window: int = 14,
    ) -> pl.DataFrame:
        """
        添加真实波动幅度 (Average True Range, ATR)，使用 Wilder 平滑法。

        Parameters
        ----------
        df : pl.DataFrame
            含 symbol、high、low、close 列的原始 K 线 DataFrame（支持多品种混合）。
        window : int
            ATR 周期，默认 14。

        Returns
        -------
        pl.DataFrame
            追加了 "atr_{window}" 列的 DataFrame。
            前 window-1 行为 null（min_periods=window Wilder 冷启动保护）。

        Notes
        -----
        True Range 定义（取三项最大值）：
            TR = max(high - low, |high - prev_close|, |low - prev_close|)

        当 prev_close 为 null（品种首行）时，max_horizontal 自动忽略 null 分量，
        TR 退化为 high-low（纯当根 K 线区间），不引入前视数据。

        ATR 平滑使用与 RSI 完全一致的 Wilder 公式：
            ewm_mean(com=window-1, adjust=False, min_periods=window)
        这是 Wilder 原版定义，与简单滚动均值 rolling_mean 有所不同。

        使用 shift(1).over("symbol") 防止跨品种边界污染：品种 B 的首行
        prev_close 严格为 null，而非品种 A 最后一行的收盘价。
        """
        prev_tmp = "_atr_prev_tmp"
        tr_tmp   = "_atr_tr_tmp"
        col_name = f"atr_{window}"

        logger.debug("计算 ATR: window=%d, output=%s", window, col_name)

        return (
            df
            # Step 1: 前一根 K 线收盘价（.over("symbol") 保证不跨品种，首行 null 防未来函数）
            .with_columns(
                pl.col("close").shift(1).over("symbol").alias(prev_tmp)
            )
            # Step 2: True Range（max_horizontal 自动忽略 null 分量）
            .with_columns(
                pl.max_horizontal(
                    pl.col("high") - pl.col("low"),
                    (pl.col("high") - pl.col(prev_tmp)).abs(),
                    (pl.col("low")  - pl.col(prev_tmp)).abs(),
                ).alias(tr_tmp)
            )
            # Step 3: Wilder 平滑 ATR（与 RSI 平滑方式完全一致）
            .with_columns(
                pl.col(tr_tmp)
                .ewm_mean(com=window - 1, adjust=False, min_samples=window)
                .over("symbol")
                .alias(col_name)
            )
            # Step 4: 清理临时列
            .drop([prev_tmp, tr_tmp])
        )
