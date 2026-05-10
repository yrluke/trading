"""
build_database.py
=================
将 ./data 目录下的 15 分钟和日线 CSV 数据转换为分区 Parquet 格式。

输出结构：
    data/kbars/
    ├── 15m/  year=2024/symbol=000001/data_0.parquet
    └── 1d/   year=2024/data_0.parquet

用法：
    python build_database.py                 # 全部频率，增量模式
    python build_database.py --freq 15m      # 仅 15 分钟
    python build_database.py --freq 1d       # 仅日线
    python build_database.py --overwrite     # 强制覆盖
"""

import argparse
import logging
import shutil
import time
from collections import defaultdict
from pathlib import Path

import duckdb
import polars as pl

# ──────────────────────────────────────────────────────────────────────────────
# 常量
# ──────────────────────────────────────────────────────────────────────────────

# 带 UTC 偏移的时间字符串格式（%:z 匹配 +08:00）
DATETIME_FMT = "%Y-%m-%d %H:%M:%S%:z"

# 15m 列类型覆写（CSV 有表头，仅覆写需要精确控制的列）
SCHEMA_15M: dict[str, type] = {
    "exchange": pl.Utf8,
    "symbol":   pl.Utf8,
    "open":     pl.Float64,
    "close":    pl.Float64,
    "high":     pl.Float64,
    "low":      pl.Float64,
    "amount":   pl.Float64,
    "volume":   pl.Float64,
    "bob":      pl.Utf8,
    "eob":      pl.Utf8,
    "type":     pl.Int32,
    "sequence": pl.Int32,
}

# 日线列类型覆写（无 sequence 列）
SCHEMA_1D: dict[str, type] = {
    "exchange": pl.Utf8,
    "symbol":   pl.Utf8,
    "open":     pl.Float64,
    "close":    pl.Float64,
    "high":     pl.Float64,
    "low":      pl.Float64,
    "amount":   pl.Float64,
    "volume":   pl.Float64,
    "bob":      pl.Utf8,
    "eob":      pl.Utf8,
    "type":     pl.Int32,
}

PARQUET_COMPRESSION = "zstd"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# 目录发现（仅枚举目录，不遍历文件，速度极快）
# ──────────────────────────────────────────────────────────────────────────────

def discover_day_dirs(freq_dir: Path) -> dict[str, list[Path]]:
    """
    枚举所有 YYYYMMDD 格式的交易日目录，按年月（YYYYMM）分组。

    只枚举目录，不迭代文件，对百万级文件场景比 rglob("*.csv") 快 ~2000 倍。
    兼容两种路径结构：
        data/15m/2024/202401/20240102/   →  month=202401
        data/15m/202601/20260105/         →  month=202601
    """
    month_to_dirs: dict[str, list[Path]] = defaultdict(list)
    for subdir in freq_dir.rglob("*/"):
        name = subdir.name
        if len(name) == 8 and name.isdigit():
            month_to_dirs[name[:6]].append(subdir)
    return dict(month_to_dirs)


# ──────────────────────────────────────────────────────────────────────────────
# Polars 向量化处理（作用于 LazyFrame，collect() 前不触发 I/O）
# ──────────────────────────────────────────────────────────────────────────────

def _parse_time_columns(lf: pl.LazyFrame) -> pl.LazyFrame:
    """
    向量化将 bob/eob 从字符串转换为带时区的 Datetime。

    单次 with_columns() 同时处理两列，Polars 将其合并为一个 SIMD 执行节点，
    充分利用向量化指令，比逐行解析快 10–100 倍。
    time_unit="us" 与 Parquet timestamp 标准对齐。
    """
    return lf.with_columns([
        pl.col("bob")
          .str.to_datetime(format=DATETIME_FMT, time_unit="us")
          .dt.convert_time_zone("Asia/Shanghai")
          .alias("bob"),
        pl.col("eob")
          .str.to_datetime(format=DATETIME_FMT, time_unit="us")
          .dt.convert_time_zone("Asia/Shanghai")
          .alias("eob"),
    ])


def _add_year_col(lf: pl.LazyFrame) -> pl.LazyFrame:
    """
    从已解析的 bob 列提取年份，新增 year 字符串列。
    DuckDB PARTITION_BY 将此值作为 Hive 分区目录名（year=2024）。
    """
    return lf.with_columns(
        pl.col("bob").dt.year().cast(pl.Utf8).alias("year")
    )


# ──────────────────────────────────────────────────────────────────────────────
# Polars 批量读取（按月 glob，Rust 层文件发现）
# ──────────────────────────────────────────────────────────────────────────────

def load_month_glob(
    month_dir: Path,
    day_dirs: list[Path],
    schema: dict[str, type],
) -> pl.DataFrame:
    """
    用 Polars glob 字符串一次性扫描整月所有 CSV 文件。

    【性能关键】传入 glob 字符串而非 list[Path]：
    - Polars（Rust 层）调用 OS 原生目录枚举，无 Python 逐路径对象开销
    - 单个 LazyFrame → 单次 collect()，消除 22 次 scan+collect 的重复开销
    - infer_schema_length=0：schema 已由调用方指定，跳过类型推断 I/O

    glob 模式 "month_dir/*/*.csv" 匹配：
        202401/20240102/000001.csv
        202401/20260105/000001.csv
    等所有交易日目录下的 CSV 文件。

    Parameters
    ----------
    month_dir : Path
        月份目录（YYYYMM），如 data/15m/2024/202401
    day_dirs : list[Path]
        该月所有交易日目录（用于采样 schema）
    schema : dict
        列类型覆写字典

    Returns
    -------
    pl.DataFrame
        含 year 列的处理后 DataFrame
    """
    # 采样第一个交易日的首个文件，确定实际列名（部分列可能不存在，需过滤）
    sample_files = list(day_dirs[0].glob("*.csv"))
    if not sample_files:
        return pl.DataFrame()

    actual_cols = set(pl.read_csv(sample_files[0], n_rows=0).columns)
    filtered_schema = {k: v for k, v in schema.items() if k in actual_cols}

    # glob 字符串：Rust 层展开，匹配 month_dir/YYYYMMDD/*.csv
    # Windows 上 Polars scan_csv(glob) 只支持相对路径，转换为相对 cwd 的路径
    try:
        month_glob = month_dir.relative_to(Path.cwd()).as_posix() + "/*/*.csv"
    except ValueError:
        # 若 month_dir 不在 cwd 下，则退回绝对路径
        month_glob = month_dir.as_posix() + "/*/*.csv"

    # 惰性扫描 → 向量化时间解析 → 添加年份列 → 一次触发执行
    lf = pl.scan_csv(
        month_glob,
        has_header=True,
        schema_overrides=filtered_schema,
        infer_schema_length=0,   # schema 已知，跳过推断节省 I/O
    )
    lf = _parse_time_columns(lf)
    lf = _add_year_col(lf)
    return lf.collect()


# ──────────────────────────────────────────────────────────────────────────────
# DuckDB 分区 Parquet 写出
# ──────────────────────────────────────────────────────────────────────────────

def write_partitioned_parquet(
    df: pl.DataFrame,
    output_dir: Path,
    partition_by: list[str],
) -> None:
    """
    通过 DuckDB 写出 Hive 分区 Parquet。

    Polars → Arrow（零拷贝）→ DuckDB → 并行写 Parquet。
    OVERWRITE_OR_IGNORE 允许同分区多次追加（15m 月内逐月累积写入）。
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()
    try:
        con.register("df_view", df.to_arrow())
        out_path = str(output_dir).replace("\\", "/")
        con.execute(f"""
            COPY (SELECT * FROM df_view)
            TO '{out_path}'
            (
                FORMAT PARQUET,
                COMPRESSION {PARQUET_COMPRESSION},
                PARTITION_BY ({", ".join(partition_by)}),
                OVERWRITE_OR_IGNORE TRUE
            )
        """)
    finally:
        con.close()


# ──────────────────────────────────────────────────────────────────────────────
# 15m 处理器
# ──────────────────────────────────────────────────────────────────────────────

def process_15m(
    freq_input_dir: Path,
    freq_output_dir: Path,
    overwrite: bool,
) -> None:
    """
    处理 15 分钟数据（Polars 全链路：向量化 CSV 读取 → Parquet 分区写出）。

    【架构】逐月处理，单月 ~160 万行完全放入内存，无 OOM 风险：
        for month:
            load_month_glob(month_dir, day_dirs, SCHEMA_15M)  ← Polars 向量化读取
            → write_partitioned_parquet(df, ..., ["year", "month"])

    【分区结构】year=XXXX/month=YYYYMM/data_0.parquet
    - month 分区确保每次写入目标目录唯一，无冲突
    - 读取时 DuckDB/Polars 可透明识别 Hive 分区，month 列自动推断
    - 增量检测粒度：年份（year=XXXX 目录存在则跳过该年）
    """
    logger.info("══════ 开始处理 [15m] 频率数据 ══════")

    if not freq_input_dir.exists():
        logger.warning("输入目录不存在，跳过：%s", freq_input_dir)
        return

    month_to_dirs = discover_day_dirs(freq_input_dir)
    if not month_to_dirs:
        logger.warning("[15m] 未找到任何交易日目录")
        return

    years = sorted({m[:4] for m in month_to_dirs})
    logger.info("[15m] 共发现 %d 个年份：%s", len(years), years)

    for year in years:
        year_output_dir = freq_output_dir / f"year={year}"

        if year_output_dir.exists():
            if overwrite:
                logger.info("[15m/%s] 覆盖模式：删除 %s", year, year_output_dir)
                shutil.rmtree(year_output_dir)
            else:
                logger.info("[15m/%s] 增量模式：已存在，跳过", year)
                continue

        for month in sorted(m for m in month_to_dirs if m[:4] == year):
            day_dirs = sorted(month_to_dirs[month])
            month_dir = day_dirs[0].parent

            logger.info("[15m/%s/%s] 处理 %d 个交易日……", year, month, len(day_dirs))
            t0 = time.perf_counter()

            df = load_month_glob(month_dir, day_dirs, SCHEMA_15M)
            if df.is_empty():
                logger.warning("[15m/%s/%s] 无有效数据，跳过", year, month)
                continue

            df = df.with_columns(pl.lit(month).alias("month"))
            write_partitioned_parquet(df, freq_output_dir, ["year", "month"])

            logger.info("[15m/%s/%s] 完成，耗时 %.1f 秒",
                        year, month, time.perf_counter() - t0)

    logger.info("══════ [15m] 处理完毕 ══════")


# ──────────────────────────────────────────────────────────────────────────────
# 1d 处理器
# ──────────────────────────────────────────────────────────────────────────────

def process_1d(
    freq_input_dir: Path,
    freq_output_dir: Path,
    overwrite: bool,
) -> None:
    """
    处理日线数据。

    1d 每年仅 ~243 个文件（每文件含全市场数据），整年批量读取即可。
    部分年份文件列数不同（有无 sequence 列），用目录采样 + schema 分组处理。
    增量检测粒度：年份。
    """
    logger.info("══════ 开始处理 [1d] 频率数据 ══════")

    if not freq_input_dir.exists():
        logger.warning("输入目录不存在，跳过：%s", freq_input_dir)
        return

    month_to_dirs = discover_day_dirs(freq_input_dir)
    if not month_to_dirs:
        logger.warning("[1d] 未找到任何交易日目录")
        return

    # 按年收集所有 CSV 文件
    year_to_files: dict[str, list[Path]] = defaultdict(list)
    for month, day_dirs in month_to_dirs.items():
        for day_dir in day_dirs:
            year_to_files[month[:4]].extend(day_dir.glob("*.csv"))

    years = sorted(year_to_files.keys())
    logger.info("[1d] 共发现 %d 个年份：%s", len(years), years)

    for year in years:
        csv_files = year_to_files[year]
        year_output_dir = freq_output_dir / f"year={year}"

        if year_output_dir.exists():
            if overwrite:
                logger.info("[1d/%s] 覆盖模式：删除 %s", year, year_output_dir)
                shutil.rmtree(year_output_dir)
            else:
                logger.info("[1d/%s] 增量模式：已存在，跳过", year)
                continue

        logger.info("[1d/%s] 读取 %d 个 CSV 文件……", year, len(csv_files))
        t0 = time.perf_counter()

        # 按交易日目录采样 schema（目录内文件结构相同，每目录读一次表头）
        dir_to_files: dict[Path, list[Path]] = defaultdict(list)
        for f in csv_files:
            dir_to_files[f.parent].append(f)

        schema_groups: dict[tuple, list[Path]] = defaultdict(list)
        for day_dir, files in dir_to_files.items():
            sample_cols = tuple(pl.read_csv(files[0], n_rows=0).columns)
            schema_groups[sample_cols].extend(files)

        partial_dfs: list[pl.DataFrame] = []
        for col_names, group_files in schema_groups.items():
            group_schema = {k: v for k, v in SCHEMA_1D.items() if k in col_names}
            lf = pl.scan_csv(
                [str(f) for f in group_files],
                has_header=True,
                schema_overrides=group_schema,
                infer_schema_length=0,
            )
            lf = _parse_time_columns(lf)
            lf = _add_year_col(lf)
            partial_dfs.append(lf.collect())

        if not partial_dfs:
            logger.warning("[1d/%s] 无有效数据，跳过", year)
            continue

        # diagonal_relaxed：缺失列自动填 null，兼容不同 schema 的分组合并
        df = (pl.concat(partial_dfs, how="diagonal_relaxed")
              if len(partial_dfs) > 1 else partial_dfs[0])

        logger.info("[1d/%s] 读取完成：%d 行，耗时 %.1f 秒",
                    year, len(df), time.perf_counter() - t0)

        t1 = time.perf_counter()
        write_partitioned_parquet(df, freq_output_dir, ["year"])
        logger.info("[1d/%s] 写出完成，耗时 %.1f 秒", year, time.perf_counter() - t1)

    logger.info("══════ [1d] 处理完毕 ══════")


# ──────────────────────────────────────────────────────────────────────────────
# 命令行入口
# ──────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="CSV K 线 → 分区 Parquet（混合存储架构）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--freq", choices=["15m", "1d", "all"], default="all",
        help="处理频率（默认 all）",
    )
    parser.add_argument(
        "--overwrite", action="store_true", default=False,
        help="强制覆盖已有 Parquet 分区（默认增量跳过）",
    )
    parser.add_argument(
        "--data-dir", type=Path, default=Path("data"),
        help="原始数据根目录（默认 ./data）",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_root  = args.data_dir.resolve()
    kbars_root = data_root / "kbars"

    logger.info("数据根目录：%s", data_root)
    logger.info("Parquet 输出：%s", kbars_root)
    logger.info("处理频率：%s | 覆盖模式：%s", args.freq, args.overwrite)

    t = time.perf_counter()
    if args.freq in ("15m", "all"):
        process_15m(data_root / "15m", kbars_root / "15m", args.overwrite)
    if args.freq in ("1d", "all"):
        process_1d(data_root / "1d", kbars_root / "1d", args.overwrite)
    logger.info("全部处理完毕，总耗时 %.1f 秒", time.perf_counter() - t)


if __name__ == "__main__":
    main()