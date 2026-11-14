
#!/usr/bin/env python
# coding: utf-8

# In[5]:


#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import os
from pathlib import Path
import re
import time
from typing import Optional, Literal

import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup

# =========================
# 設定
# =========================

def _find_project_root(start: Path) -> Path:
    """
    start から上に向かって pyproject.toml を探す。
    見つかればそのディレクトリを返す。
    """
    for p in [start, *start.parents]:
        if (p / "pyproject.toml").exists():
            return p
    return start

def _lib_root() -> Path:
    # meteo_lib リポジトリ（pyproject.tomlがある場所）
    return Path(__file__).resolve().parents[1]

def _project_root() -> Path:
    """
    プロジェクトルート決定ロジック

    優先順位:
      1) METEO_PROJECT_ROOT（明示指定）
      2) pyproject.toml を探索
      3) cwd（最終フォールバック）
    """
    env = os.getenv("METEO_PROJECT_ROOT")
    if env:
        return Path(env).resolve()

    cwd = Path.cwd().resolve()
    return _find_project_root(cwd)

LIB_ROOT = _lib_root()
PROJECT_ROOT = _project_root()

# 共通マスタ（amd_list等）
LIB_DATA_DIR = LIB_ROOT / "data"

# プロジェクト固有の入出力
PROJECT_DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

Freq = Literal["daily", "hourly", "10min"]

# v1と同じ都道府県→prec_no（そのまま流用）
prec_list = {
    "宗谷総合振興局":11,"上川総合振興局":12,"留萌振興局":13,"石狩振興局":14,
    "空知総合振興局":15,"後志総合振興局":16,"オホーツク総合振興局":17,
    "根室振興局":18,"釧路振興局":19,"十勝総合振興局":20,"胆振総合振興局":21,
    "日高振興局":22,"渡島総合振興局":23,"檜山振興局":24,
    "青森県":31,"秋田県":32,"岩手県":33,"宮城県":34,"山形県":35,"福島県":36,
    "茨城県":40,"栃木県":41,"群馬県":42,"埼玉県":43,"東京都":44,"千葉県":45,
    "神奈川県":46,"長野県":48,"山梨県":49,"静岡県":50,"愛知県":51,"岐阜県":52,
    "三重県":53,"新潟県":54,"富山県":55,"石川県":56,"福井県":57,
    "滋賀県":60,"京都府":61,"大阪府":62,"兵庫県":63,"奈良県":64,"和歌山県":65,
    "岡山県":66,"広島県":67,"島根県":68,"鳥取県":69,"徳島県":71,"香川県":72,
    "愛媛県":73,"高知県":74,"山口県":81,"福岡県":82,"大分県":83,"長崎県":84,
    "佐賀県":85,"熊本県":86,"宮崎県":87,"鹿児島県":88,"沖縄県":91,
    "南極":99
#    "南大東村":92,"北大東村":92,"宮古島市":93,"宮古郡":93,"石垣市":94,"八重山郡":94
}

# =========================
# 列レイアウト（v1を踏襲） :contentReference[oaicite:5]{index=5}
# =========================
cols_k_day = (
    "日","現地気圧","海面気圧","降水量_合計","降水量_1時間","降水量_10分間",
    "平均気温","最高気温","最低気温","平均湿度","最小湿度",
    "平均風速","最大風速","最大時風向","最大瞬間風速","最大瞬間時風向",
    "日照時間","降雪","最深積雪","天気概況_昼","天気概況_夜"
)
cols_k_hour = (
    "時","現地気圧","海面気圧","降水量","気温","露点温度","蒸気圧",
    "湿度","風速","風向","日照","全天日射量","降雪","積雪","天気","雲量","視程"
)
cols_k_10m = (
    "時分","気圧","海面気圧","降水量","気温","相対湿度","平均風速","平均風向",
    "最大瞬間風速","最大瞬間時風向","日照時間"
)
cols_a_day = (
    "日","降水量_合計","降水量_1時間","降水量_10分間","平均気温","最高気温","最低気温",
    "平均湿度","最小湿度","平均風速","最大風速","最大時風向","最大瞬間風速",
    "最大瞬間時風向","最多風向","日照時間","降雪","最深積雪"
)
cols_a_hour = (
    "時","降水量","気温","露点温度","蒸気圧","湿度","風速","風向","日照","降雪","積雪"
)
cols_a_10m = (
    "時分","降水量","気温","相対湿度","平均風速","平均風向","最大瞬間風速","最大瞬間時風向","日照時間"
)

# =========================
# HTMLテーブル読取（v1踏襲：img altを拾う） :contentReference[oaicite:6]{index=6}
# =========================
def _read_table_preserve_flags(html_text: str, cols: tuple[str, ...], table_css: str = "table.data2_s") -> pd.DataFrame:
    soup = BeautifulSoup(html_text, "lxml")
    table = soup.select_one(table_css) or soup.find("table")
    if not table:
        return pd.DataFrame(columns=cols)

    rows: list[list[str]] = []
    for tr in table.find_all("tr"):
        tds = tr.find_all(["td", "th"])
        if not tds:
            continue
        row: list[str] = []
        for td in tds:
            img = td.find("img")
            if img and (img.get("alt") or img.get("src")):
                cell = img.get("alt") or img.get("src")
            else:
                cell = td.get_text(separator="", strip=True).replace("\xa0", "")
            row.append(cell)
        rows.append(row)

    n_expected = len(cols)
    rows = [r[:n_expected] + [""] * (n_expected - len(r)) for r in rows]
    df = pd.DataFrame(rows, columns=cols)
    df = df.stack().map(lambda x: "" if pd.isna(x) else str(x).strip()).unstack()
    return df

# =========================
# amd_list と amd_list_ext のロード（案B）
# =========================
@lru_cache(maxsize=1)
def _load_amd_list(path: Optional[str] = None) -> pd.DataFrame:
#    p = Path(path) if path else (LIB_DATA_DIR / "amd_list.csv")    # prec_no問題（石垣島が91なのか94なのかなど）のため、当面、地点コード.csvを使う
    p = Path(path) if path else (LIB_DATA_DIR / "地点コード.csv")
    return pd.read_csv(p, encoding="utf-8")

@lru_cache(maxsize=1)
def _load_amd_ext(path: Optional[str] = None) -> pd.DataFrame:
    p = Path(path) if path else (LIB_DATA_DIR / "amd_list_ext.csv")
    return pd.read_csv(p, encoding="utf-8")

@dataclass(frozen=True)
class Station:
    pref: str
    place: str
    prec: int
    block: int
    station_class: str = "unknown"  # amedas / kansho_wmo / etc.

def get_station(pref: str, place: str, amd_list_path: Optional[str] = None, amd_ext_path: Optional[str] = None) -> Optional[Station]:
    if pref not in prec_list:
        return None

    prec = int(prec_list[pref])
    amd = _load_amd_list(amd_list_path)

    m = (amd["prec_no"] == prec) & (amd["Name"] == place)
    if not m.any():
        return None

    row = amd.loc[m].iloc[0]
    block = int(row["block_no"])

#    station_class = "unknown"
#    ext = _load_amd_ext(amd_ext_path)
#    if ext is not None and "block" in ext.columns:
#        hit = ext.loc[ext["block"] == block]
#        if not hit.empty and "station_class" in hit.columns:
#            station_class = str(hit.iloc[0]["station_class"])

#    return Station(pref=pref, place=place, prec=prec, block=block, station_class=station_class)
    return Station(pref=pref, place=place, prec=prec, block=block)

# =========================
# QC処理（v2仕様）
# =========================
IGNORE_COLUMNS = (
    "風向","平均風向","最大時風向","最大瞬間時風向","最多風向","天気","雲量",
    "天気概況_昼","天気概況_夜"
)

NUMERIC_LIKE_COLS = {
    "現地気圧":"float32","海面気圧":"float32","降水量_合計":"float32",
    "降水量_1時間":"float32","降水量_10分間":"float32",
    "平均気温":"float32","最高気温":"float32","最低気温":"float32",
    "平均湿度":"float32","最小湿度":"float32",
    "平均風速":"float32","最大風速":"float32","最大瞬間風速":"float32",
    "日照時間":"float32","降雪":"float32","最深積雪":"float32",
    "降水量":"float32","気温":"float32","露点温度":"float32",
    "蒸気圧":"float32","湿度":"float32","風速":"float32",
    "日照":"float32","全天日射量":"float32","積雪":"float32","視程":"float32",
    "気圧":"float32","相対湿度":"float32",
}

# 数値+記号（例 "12.3)" / "0.0" / "--" / "///" / "×" / "12.3]"）
_NUM_FLAG_RE = re.compile(r"^\s*([+-]?\d+(?:\.\d+)?)?\s*([^\d+\-\.].*)?$", re.DOTALL)

QC_CODE_MAP = {
    ")": 5,
    "]": 4,
    "#": 2,
    "///": 1,
    "×": 1,
    "--": 1,
}
# 値を不採用に落とすフラグ（あなたの方針＋安全運用）
INVALID_FLAGS = {"]", "#", "///", "×", "--"}

def split_num_and_qc(series: pd.Series) -> tuple[pd.Series, pd.Series, pd.Series]:
    """
    返り値: (num, qc, qc_code)
    - num: 数値（不採用は NaN）
    - qc: 元記号（正常は NA）
    - qc_code: 正常=8、記号ありは help3 相当（未知は1）
    """
    s = series.astype("string").str.strip()
    s = s.replace({"na": None, "NaN": None})

    m = s.str.extract(_NUM_FLAG_RE)
    num_str = m[0]
    qc = m[1].fillna("").str.strip()

    # 値を落とす（準正常 ')' は落とさない）
    invalid = qc.isin(INVALID_FLAGS) | qc.str.contains(r"\]", na=False)
    num = pd.to_numeric(num_str, errors="coerce").mask(invalid, np.nan)

    # QC列（空はNA）
    qc = qc.replace("", pd.NA)

    # QCコード（正常=8）
    qc_code = qc.map(QC_CODE_MAP).fillna(8).astype("Int8")

    return num, qc, qc_code

def handle_data_v2(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = df.columns.str.strip()
    df = df.astype("string")

    for col, dtype in NUMERIC_LIKE_COLS.items():
        if col in df.columns and col not in IGNORE_COLUMNS:
            values, qc, qc_code = split_num_and_qc(df[col])

            df[col] = pd.to_numeric(values, errors="coerce").astype(dtype)

            # QC列は元記号のまま（視認性）
            if qc.notna().any():
                df[f"{col}_qc"] = qc.astype("string")

            # codeは常に持たせる（正常は8）
            df[f"{col}_qc_code"] = qc_code

    return df

# ---- Canonical column names (English) ----
COLUMN_MAP = {
    # pressure
    "現地気圧": "pres_local",
    "海面気圧": "pres_msl",
    "気圧": "pres_local",          # 10分値の「気圧」
    "海面気圧": "pres_msl",

    # precipitation
    "降水量": "precip",
    "降水量_合計": "precip_day",
    "降水量_1時間": "precip_1h",
    "降水量_10分間": "precip_10m",

    # temperature / humidity
    "気温": "temp",
    "平均気温": "temp_mean",
    "最高気温": "temp_max",
    "最低気温": "temp_min",
    "相対湿度": "rh",
    "湿度": "rh",
    "平均湿度": "rh_mean",
    "最小湿度": "rh_min",
    "露点温度": "dewpoint",
    "蒸気圧": "vapor_pressure",

    # wind
    "風速": "wind_speed",
    "平均風速": "wind_speed_mean",
    "最大風速": "wind_speed_max",
    "最大瞬間風速": "wind_gust",
    "最大時風向": "wind_dir_max",
    "最大瞬間時風向": "wind_dir_gust",
    "最多風向": "wind_dir_mode",
    "風向": "wind_dir",
    "平均風向": "wind_dir_mean",

    # sunshine / radiation
    "日照": "sunshine",
    "日照時間": "sunshine",
    "全天日射量": "solar_rad",

    # snow
    "降雪": "snowfall",
    "積雪": "snow_depth",
    "最深積雪": "snow_depth_max",

    # visibility (only kansho hourly)
    "視程": "visibility",

}


def rename_with_qc(df: pd.DataFrame, col_map: dict[str, str]) -> pd.DataFrame:
    """
    日本語列を英語に統一。
    併せて *_qc, *_qc_code も追随してリネーム。
    """
    rename = {}
    for src, dst in col_map.items():
        if src in df.columns:
            rename[src] = dst
        qc = f"{src}_qc"
        if qc in df.columns:
            rename[qc] = f"{dst}_qc"
        qcc = f"{src}_qc_code"
        if qcc in df.columns:
            rename[qcc] = f"{dst}_qc_code"

    # メタ列（日本語ではないので原則そのまま）
    return df.rename(columns=rename)

    # メタ情報（先に付与）
    out.insert(0, "prec_no", station.prec)
    out.insert(1, "block_no", station.block)
    out.insert(2, "pref", station.pref)
    out.insert(3, "station", station.place)
    out.insert(4, "station_class", station.station_class)

    # source_url は後でメタに移すので、いったん残していてもOK（次の①で消す）
    out.insert(5, "source_url", url)

    # ★追加：英語列名へ統一（QC列も追随）
    out = rename_with_qc(out, COLUMN_MAP)

# =========================
# HTTP（Session＋礼儀スリープ＋簡易リトライ）
# =========================
def _get_text(url: str, session: requests.Session, timeout: int = 30, sleep_sec: float = 1.0, max_retries: int = 2) -> str:
    for i in range(max_retries + 1):
        r = session.get(url, timeout=timeout)
        r.encoding = r.apparent_encoding
        if r.status_code == 200:
            if sleep_sec > 0:
                time.sleep(sleep_sec)
            return r.text
        # 429/5xx は少し待ってリトライ
        if r.status_code in (429, 500, 502, 503, 504) and i < max_retries:
            time.sleep(max(1.0, sleep_sec) * (i + 1))
            continue
        r.raise_for_status()
    raise RuntimeError("unreachable")

# =========================
# URLビルド（station_class があればそれ優先）
# =========================
def _is_kansho(block: int, station_class: str) -> bool:
    if station_class in ("kansho_wmo", "kansho"):
        return True
    if station_class == "amedas":
        return False
    # フォールバック：v1互換の近道（block>40000は官署側ページに寄せる） :contentReference[oaicite:7]{index=7}
    return block > 40000

def _url(freq: Freq, prec: int, block: int, station_class: str, year: int, month: int, day: int) -> str:
    kansho = _is_kansho(block, station_class)
    if freq == "daily":
        if kansho:
            return f"https://www.data.jma.go.jp/stats/etrn/view/daily_s1.php?prec_no={prec}&block_no={block}&year={year}&month={month}&day={day}&view="
        return f"https://www.data.jma.go.jp/stats/etrn/view/daily_a1.php?prec_no={prec}&block_no={block:04}&year={year}&month={month}&day={day}&view="
    if freq == "hourly":
        if kansho:
            return f"https://www.data.jma.go.jp/stats/etrn/view/hourly_s1.php?prec_no={prec}&block_no={block}&year={year}&month={month}&day={day}&view="
        return f"https://www.data.jma.go.jp/stats/etrn/view/hourly_a1.php?prec_no={prec}&block_no={block:04}&year={year}&month={month}&day={day}&view="
    if freq == "10min":
        if kansho:
            return f"https://www.data.jma.go.jp/stats/etrn/view/10min_s1.php?prec_no={prec}&block_no={block}&year={year}&month={month}&day={day}&view="
        return f"https://www.data.jma.go.jp/stats/etrn/view/10min_a1.php?prec_no={prec}&block_no={block:04}&year={year}&month={month}&day={day}&view="
    raise ValueError(freq)

# =========================
# 取得（1日分）
# =========================
def fetch_one_day(
    station: Station,
    freq: Freq,
    year: int,
    month: int,
    day: int,
    session: Optional[requests.Session] = None,
    sleep_sec: float = 1.0,
) -> pd.DataFrame:
    """
    1日分のデータを取得し、index を日時にして返す（JST想定のnaive datetime）
    """
    sess = session or requests.Session()
    url = _url(freq, station.prec, station.block, station.station_class, year, month, day)

    html = _get_text(url, sess, sleep_sec=sleep_sec)

    # どの列レイアウトで読むか（官署/アメダス × freq）
    kansho = _is_kansho(station.block, station.station_class)
    if freq == "daily":
        cols = cols_k_day if kansho else cols_a_day
        df = _read_table_preserve_flags(html, cols)
        df = (df.iloc[4:] if kansho else df.iloc[3:]).reset_index(drop=True)

        df["日時"] = pd.to_datetime(dict(year=year, month=month, day=pd.to_numeric(df["日"], errors="coerce")), errors="coerce")
        df = df.drop(columns=["日"]).set_index("日時")
        out = handle_data_v2(df)

    elif freq == "hourly":
        cols = cols_k_hour if kansho else cols_a_hour
        df = _read_table_preserve_flags(html, cols)
        df = df.iloc[2:].reset_index(drop=True)

        hh = pd.to_numeric(df["時"], errors="coerce")
        mask24 = (hh == 24)
        df.loc[mask24, "時分"] = "00:00"
        df.loc[~mask24, "時分"] = hh.astype("Int16").astype(str) + ":00"

        df["日時"] = pd.to_datetime(f"{year}-{month}-{day} " + df["時分"], format="%Y-%m-%d %H:%M", errors="coerce")
        df.loc[mask24, "日時"] += pd.Timedelta(days=1)
        df = df.drop(columns=["時", "時分"]).set_index("日時")
        out = handle_data_v2(df)

    else:  # "10min"
        cols = cols_k_10m if kansho else cols_a_10m
        df = _read_table_preserve_flags(html, cols)
        df = (df.iloc[2:] if kansho else df.iloc[3:]).reset_index(drop=True)

        df["日付"] = pd.to_datetime(f"{year}-{month}-{day}")
        mask2400 = (df["時分"] == "24:00")
        df.loc[mask2400, "時分"] = "00:00"
        df.loc[mask2400, "日付"] += pd.Timedelta(days=1)
        df["日時"] = pd.to_datetime(
            df["日付"].dt.strftime("%Y-%m-%d") + " " + df["時分"],
            format="%Y-%m-%d %H:%M",
            errors="coerce",
        )
        df = df.drop(columns=["時分", "日付"]).set_index("日時")
        out = handle_data_v2(df)

    out = rename_with_qc(out, COLUMN_MAP)

    meta = {
        "prec_no": str(station.prec),
        "block_no": str(station.block),
        "pref": station.pref,
        "station": station.place,
        "station_class": station.station_class,
        "freq": freq,
        "date": f"{year:04d}-{month:02d}-{day:02d}",
        "source_url": url,
    }
    return out, meta

# =========================
# 保存（Parquet正本＋任意CSV）
# =========================

#      DatetimeIndex を datetime列に変換して、保存用に正規化する。
#    - DatetimeIndex なら reset_index() で列化
#    - 列名は time_col に統一
#    - datetime列は pd.to_datetime で型を揃える

def ensure_datetime_column(df: pd.DataFrame, time_col: str = "datetime") -> pd.DataFrame:
    """
    保存前の正規化：
    1) DatetimeIndex を必要に応じて列化（ただし datetime 列があるなら index は捨てる）
    2) time_col が重複している場合は 1本に潰す（左優先、欠損は右で埋める）
    3) time_col を datetime64 に寄せる
    """
    df2 = df.copy()

    # --- (A) index -> column（ただし衝突するなら index は捨てる）---
    if isinstance(df2.index, pd.DatetimeIndex):
        if time_col in df2.columns:
            df2 = df2.reset_index(drop=True)
        else:
            idx_name = df2.index.name or time_col
            df2 = df2.reset_index()
            if idx_name != time_col:
                df2 = df2.rename(columns={idx_name: time_col})

    # --- (B) time_col が重複しているなら 1本に潰す ---
    # pandasは重複列名OKなので、位置で処理する
    pos = [i for i, c in enumerate(df2.columns) if c == time_col]
    if len(pos) >= 2:
        # まず datetime 型に寄せてから統合した方が安全
        tmp = df2.iloc[:, pos].apply(pd.to_datetime, errors="coerce")
        merged = tmp.bfill(axis=1).iloc[:, 0]  # 左を優先し、NaTは右で埋める

        # 元の重複 datetime 列を「位置」で削除
        keep_mask = [i not in pos for i in range(df2.shape[1])]
        df2 = df2.iloc[:, keep_mask].copy()

        # 1本だけ戻す（先頭に置きたいなら insert）
        df2.insert(0, time_col, merged)

    # --- (C) time_col の型を最終調整 ---
    if time_col in df2.columns:
        df2[time_col] = pd.to_datetime(df2[time_col], errors="coerce")

    return df2


# データをParquetで保存する

def save_dataset(df, out_dir, basename, export_csv=False, parquet_metadata=None):
    import pyarrow as pa
    import pyarrow.parquet as pq
    from pathlib import Path

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pq_path = out_dir / f"{basename}.parquet"

    # ★ 追加：index/列の正規化（datetime重複対策込み）
    df = ensure_datetime_column(df, time_col="datetime")
    
# 一旦ここで止めてもOK（原因切り分け用）
# raise SystemExit

#    print("BEFORE from_pandas")
#    table = pa.Table.from_pandas(df, preserve_index=False)
#    print("AFTER  from_pandas")

    # ★ 重要：indexは保存しない
    table = pa.Table.from_pandas(df, preserve_index=False)

#    print("BEFORE write_table")
#    pq.write_table(table, pq_path)
#    print("AFTER  write_table")

    meta = table.schema.metadata or {}
    meta2 = dict(meta)

    if parquet_metadata:
        for k, v in parquet_metadata.items():
            meta2[str(k).encode("utf-8")] = str(v).encode("utf-8")

    table = table.replace_schema_metadata(meta2)
    pq.write_table(table, pq_path)

    csv_path = None
    if export_csv:
        csv_path = out_dir / f"{basename}.csv"
        df.to_csv(csv_path, index=False, encoding="utf-8")

    return pq_path, csv_path


def make_clean_view(
    df: pd.DataFrame,
    strict: bool = False,
) -> pd.DataFrame:
    """
    strict=False: 準正常 ')' も許容（qc_code 8 or 5）
    strict=True : 完全正常のみ（qc_code 8）
    """
    qc_code_cols = [c for c in df.columns if c.endswith("_qc_code")]
    if not qc_code_cols:
        return df.copy()

    allowed = {8} if strict else {8, 5}

    mask = np.ones(len(df), dtype=bool)
    for c in qc_code_cols:
        s = df[c]
        # 欠損があれば 8扱いにしない（安全側）
        mask &= s.isin(list(allowed))

    return df.loc[mask].copy()

import re

def slug(s: str) -> str:
    # 英数字・アンダースコア以外は _ に寄せる（日本語は残す方針でもOK）
    s = s.strip()
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"[^\w\-]+", "_", s)  # 日本語を残したいならここは緩める
    s = re.sub(r"_+", "_", s).strip("_")
    return s.lower()

def make_basename(meta: dict[str, str]) -> str:
    freq = meta["freq"]
    pref = slug(meta["pref"])
    station = slug(meta["station"])
    prec = meta["prec_no"]
    block = meta["block_no"]
    date = meta["date"]
    return f"{freq}_{pref}_{station}_p{prec}_b{block}_{date}"


from datetime import date, timedelta

def _daterange(d0: date, d1: date):
    d = d0
    while d <= d1:
        yield d
        d += timedelta(days=1)

def fetch_range(
    station: Station,
    freq: Freq,
    start: str,           # "YYYY-MM-DD"
    end: str,             # "YYYY-MM-DD"
    out_root: str | Path = "outputs/v2",
    export_csv: bool = False,
    sleep_sec: float = 1.0,
    skip_existing: bool = True,
    stop_on_error: bool = False,
) -> pd.DataFrame:
    """
    期間を回して日別Parquetを書き出す。
    戻り値：実行ログ（成功/失敗・パス等）
    """
    out_root = Path(out_root)
    out_dir = out_root / freq
    out_dir.mkdir(parents=True, exist_ok=True)

    d0 = pd.to_datetime(start).date()
    d1 = pd.to_datetime(end).date()

    results = []

    sess = requests.Session()
    for d in _daterange(d0, d1):
        y, m, dd = d.year, d.month, d.day
        try:
            df, meta = fetch_one_day(
                station,
                freq=freq,
                year=y,
                month=m,
                day=dd,
                session=sess,
                sleep_sec=sleep_sec,
            )
            basename = make_basename(meta)
            pq_path = out_dir / f"{basename}.parquet"

            if skip_existing and pq_path.exists():
                results.append({
                    "date": meta["date"],
                    "status": "skipped",
                    "parquet": str(pq_path),
                    "error": "",
                })
                continue

            pq_written, csv_written = save_dataset(
                df,
                out_dir=out_dir,
                basename=basename,
                export_csv=export_csv,
                parquet_metadata=meta,
            )

            results.append({
                "date": meta["date"],
                "status": "ok",
                "parquet": str(pq_written),
                "error": "",
            })

        except Exception as e:
            results.append({
                "date": f"{y:04d}-{m:02d}-{dd:02d}",
                "status": "error",
                "parquet": "",
                "error": repr(e),
            })
            if stop_on_error:
                break

    return pd.DataFrame(results)
