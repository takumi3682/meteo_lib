#!/usr/bin/env python
# coding: utf-8

# In[5]:


#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import os
import numpy as np
import pandas as pd
from pathlib import Path
import pyarrow.parquet as pq
import re
import time
from typing import Optional, Literal
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
    # core_v4.py と同じ階層（meteo_lib/）を返す
    return Path(__file__).resolve().parent

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
# Canonical (English) schemas by (freq, station_kind)
# - station_kind: "kansho" or "amedas"
# - The HTML is Japanese, but we immediately replace column names by position.
#   This makes downstream processing stable and avoids JP->EN rename steps.
# =========================
EN_COLS: dict[tuple[Freq, str], tuple[str, ...]] = {
    ("daily", "kansho"): (
        "day","pres_local","pres_msl","precip_day","precip_1h","precip_10m",
        "temp_mean","temp_max","temp_min","rh_mean","rh_min",
        "wind_speed_mean","wind_speed_max","wind_dir_max","wind_gust","wind_dir_gust",
        "sunshine","snowfall","snow_depth_max","weather_summary_day","weather_summary_night",
    ),
    ("daily", "amedas"): (
        "day","precip_day","precip_1h","precip_10m",
        "temp_mean","temp_max","temp_min",
        "rh_mean","rh_min",
        "wind_speed_mean","wind_speed_max","wind_dir_max","wind_gust","wind_dir_gust",
        "wind_dir_mode","sunshine","snowfall","snow_depth_max",
    ),
    ("hourly", "kansho"): (
        "hour","pres_local","pres_msl","precip","temp","dewpoint","vapor_pressure",
        "rh","wind_speed","wind_dir","sunshine","solar_rad","snowfall","snow_depth",
        "weather","cloud_amount","visibility",
    ),
    ("hourly", "amedas"): (
        "hour","precip","temp","dewpoint","vapor_pressure",
        "rh","wind_speed","wind_dir","sunshine","snowfall","snow_depth",
    ),
    ("10min", "kansho"): (
        "time","pres_local","pres_msl","precip","temp","rh",
        "wind_speed_mean","wind_dir_mean","wind_gust","wind_dir_gust","sunshine",
    ),
    ("10min", "amedas"): (
        "time","precip","temp","rh",
        "wind_speed_mean","wind_dir_mean","wind_gust","wind_dir_gust","sunshine",
    ),
}

def _apply_schema(df: pd.DataFrame, freq: Freq, station_kind: str) -> tuple[pd.DataFrame, dict]:
    expected = EN_COLS[(freq, station_kind)]
    if len(df.columns) != len(expected):
        raise ValueError(
            f"Column length mismatch for {(freq, station_kind)}: "
            f"expected {len(expected)} cols, got {len(df.columns)}"
        )
    df = df.copy()
    df.columns = list(expected)
    return df


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

    return Station(pref=pref, place=place, prec=prec, block=block)


def get_station_start_year(station_no: str | int, master_path: Optional[str | Path] = None) -> int:
    """
    amdmaster.index4 から指定された地点番号 (5桁のアメダス番号) の観測開始年を取得する。
    見つからない場合のデフォルトは 1910 年を返す。
    """
    if master_path is None:
        master_path = Path(LIB_DATA_DIR, "amdmaster.index4")
    else:
        master_path = Path(master_path)

    station_no_str = str(station_no).zfill(5)

    # 1行目がヘッダー、2行目は単位行 (Kanji, Kana 等) なので skiprows=[1] で読み込む
    df = pd.read_csv(
        master_path,
        skiprows=[1],
        usecols=["Station Number", "Start Date"],
        dtype={"Station Number": str, "Start Date": str},
        encoding="utf-8"
    )

    # 地点番号で抽出
    matched = df[df["Station Number"] == station_no_str]
    if matched.empty:
        # 気象台・測候所など官署でアメダス番号と異なる場合は 1910 を採用
        return 1910

    # Start Date の最小値（最初期）から「年」を取り出す
    dates = pd.to_datetime(matched["Start Date"], format="mixed", errors="coerce")
    min_year = dates.min().year

    return int(min_year) if not pd.isna(min_year) else 1910

# =========================
# QC処理（v2仕様）
# =========================
IGNORE_COLUMNS = (
    # categorical / non-numeric
    "wind_dir","wind_dir_mean","wind_dir_max","wind_dir_gust","wind_dir_mode",
    "weather","cloud_amount","weather_summary_day","weather_summary_night",
)

# Columns that should be treated as numeric-like strings possibly suffixed with QC marks.
# Keys are canonical (English) column names.
NUMERIC_LIKE_COLS = {
    # pressure
    "pres_local":"float32","pres_msl":"float32",

    # precipitation
    "precip":"float32","precip_day":"float32","precip_1h":"float32","precip_10m":"float32",

    # temperature / humidity
    "temp":"float32","temp_mean":"float32","temp_max":"float32","temp_min":"float32",
    "rh":"float32","rh_mean":"float32","rh_min":"float32",
    "dewpoint":"float32","vapor_pressure":"float32",

    # wind / sunshine / radiation / snow / visibility
    "wind_speed":"float32","wind_speed_mean":"float32","wind_speed_max":"float32","wind_gust":"float32",
    "sunshine":"float32","solar_rad":"float32",
    "snowfall":"float32","snow_depth":"float32","snow_depth_max":"float32",
    "visibility":"float32",
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

def _url(freq: Freq, prec: int, block: int, station_class: str, year: int, month: int, day: int) -> tuple[str, str]:
    """Return (url, station_kind). station_kind is "kansho" or "amedas"."""
    station_kind = "kansho" if _is_kansho(block, station_class) else "amedas"

    if freq == "daily":
        if station_kind == "kansho":
            url = f"https://www.data.jma.go.jp/stats/etrn/view/daily_s1.php?prec_no={prec}&block_no={block}&year={year}&month={month}&day={day}&view="
        else:
            url = f"https://www.data.jma.go.jp/stats/etrn/view/daily_a1.php?prec_no={prec}&block_no={block:04d}&year={year}&month={month}&day={day}&view="
        return url, station_kind

    if freq == "hourly":
        if station_kind == "kansho":
            url = f"https://www.data.jma.go.jp/stats/etrn/view/hourly_s1.php?prec_no={prec}&block_no={block}&year={year}&month={month}&day={day}&view="
        else:
            url = f"https://www.data.jma.go.jp/stats/etrn/view/hourly_a1.php?prec_no={prec}&block_no={block:04d}&year={year}&month={month}&day={day}&view="
        return url, station_kind

    if freq == "10min":
        if station_kind == "kansho":
            url = f"https://www.data.jma.go.jp/stats/etrn/view/10min_s1.php?prec_no={prec}&block_no={block}&year={year}&month={month}&day={day}&view="
        else:
            url = f"https://www.data.jma.go.jp/stats/etrn/view/10min_a1.php?prec_no={prec}&block_no={block:04d}&year={year}&month={month}&day={day}&view="
        return url, station_kind

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
    url, station_kind = _url(freq, station.prec, station.block, station.station_class, year, month, day)

    html = _get_text(url, sess, sleep_sec=sleep_sec)

    # Read with v1-style Japanese layouts (to preserve flags), then immediately
    # replace columns by canonical English schema (position-based).
    if freq == "daily":
        cols = cols_k_day if station_kind == "kansho" else cols_a_day
        df = _read_table_preserve_flags(html, cols)
        df = (df.iloc[4:] if station_kind == "kansho" else df.iloc[3:]).reset_index(drop=True)
        df = _apply_schema(df, freq="daily", station_kind=station_kind)

        # The page is monthly; for backward compatibility, keep only the requested day.
        df["datetime"] = pd.to_datetime(
            dict(year=year, month=month, day=pd.to_numeric(df["day"], errors="coerce")),
            errors="coerce",
        )
        df = df.drop(columns=["day"]).set_index("datetime")
        out = handle_data_v2(df)

        target = pd.to_datetime(f"{year:04d}-{month:02d}-{day:02d}")
        out = out.loc[out.index == target]

    elif freq == "hourly":
        cols = cols_k_hour if station_kind == "kansho" else cols_a_hour
        df = _read_table_preserve_flags(html, cols)
        df = df.iloc[2:].reset_index(drop=True)
        df = _apply_schema(df, freq="hourly", station_kind=station_kind)

        hh = pd.to_numeric(df["hour"], errors="coerce")
        mask24 = (hh == 24)

        hh2 = hh.astype("Int16").astype("string")
        time_str = hh2 + ":00"
        time_str = time_str.where(~mask24, "00:00")

        dt = pd.to_datetime(
            f"{year:04d}-{month:02d}-{day:02d} " + time_str,
            format="%Y-%m-%d %H:%M",
            errors="coerce",
        )
        dt = dt + pd.to_timedelta(mask24.astype("int16"), unit="D")

        df["datetime"] = dt
        df = df.drop(columns=["hour"]).set_index("datetime")
        out = handle_data_v2(df)

    else:  # "10min"
        cols = cols_k_10m if station_kind == "kansho" else cols_a_10m
        df = _read_table_preserve_flags(html, cols)
        df = (df.iloc[2:] if station_kind == "kansho" else df.iloc[3:]).reset_index(drop=True)
        df = _apply_schema(df, freq="10min", station_kind=station_kind)

        t = df["time"].astype("string").str.strip()
        mask2400 = (t == "24:00")
        t = t.where(~mask2400, "00:00")

        base = pd.to_datetime(f"{year:04d}-{month:02d}-{day:02d}")
        dt = pd.to_datetime(
            base.strftime("%Y-%m-%d") + " " + t,
            format="%Y-%m-%d %H:%M",
            errors="coerce",
        )
        dt = dt + pd.to_timedelta(mask2400.astype("int16"), unit="D")

        df["datetime"] = dt
        df = df.drop(columns=["time"]).set_index("datetime")
        out = handle_data_v2(df)

    meta = {
        "prec_no": str(station.prec),
        "block_no": str(station.block),
        "pref": station.pref,
        "station": station.place,
        "station_class": station.station_class,
        "station_kind": station_kind,
        "freq": freq,
        "date": f"{year:04d}-{month:02d}-{day:02d}",
        "source_url": url,
    }
    return out, meta

# =========================
# 取得（daily: 月単位）
# =========================
def fetch_daily_month(
    station: Station,
    year: int,
    month: int,
    session: Optional[requests.Session] = None,
    sleep_sec: float = 1.0,
) -> tuple[pd.DataFrame, dict]:
    """Fetch daily (日別値) as a monthly table, returned with DatetimeIndex."""
    sess = session or requests.Session()
    # JMA daily page is monthly; day is a dummy (use 1).
    url, station_kind = _url("daily", station.prec, station.block, station.station_class, year, month, 1)
    html = _get_text(url, sess, sleep_sec=sleep_sec)

    cols = cols_k_day if station_kind == "kansho" else cols_a_day
    df = _read_table_preserve_flags(html, cols)
    df = (df.iloc[4:] if station_kind == "kansho" else df.iloc[3:]).reset_index(drop=True)
    df = _apply_schema(df, freq="daily", station_kind=station_kind)

    df["datetime"] = pd.to_datetime(
        dict(year=year, month=month, day=pd.to_numeric(df["day"], errors="coerce")),
        errors="coerce",
    )
    df = df.drop(columns=["day"]).set_index("datetime")
    out = handle_data_v2(df)

    # Safety: keep only the target month
    out = out[(out.index.year == year) & (out.index.month == month)]

    meta = {
        "prec_no": str(station.prec),
        "block_no": str(station.block),
        "pref": station.pref,
        "station": station.place,
        "station_class": station.station_class,
        "station_kind": station_kind,
        "freq": "daily",
        "period": f"{year:04d}-{month:02d}",
        "source_url": url,
    }
    return out, meta

def _month_starts(start: str, end: str) -> list[pd.Timestamp]:
    """Return list of month-start timestamps between start/end (inclusive)."""
    d0 = pd.to_datetime(start).to_period("M").to_timestamp()
    d1 = pd.to_datetime(end).to_period("M").to_timestamp()
    months = pd.period_range(d0.to_period("M"), d1.to_period("M"), freq="M")
    return [m.to_timestamp() for m in months]



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

    # 1. データの正規化
    df2 = ensure_datetime_column(df, time_col="datetime")

    # datetimeが naive なら JST を付与する
    if df2["datetime"].dt.tz is None:
        df2["datetime"] = df2["datetime"].dt.tz_localize("Asia/Tokyo")
    # 2. 型を明示的に指定して変換（ここが重要）
    # datetime列を明確に timestamp[s] (秒単位のタイムスタンプ) として変換
    schema = pa.Schema.from_pandas(df2)
    
    # datetime 列の型を強制的に timestamp[s] に変更
    # もし datetime 列が型推論で date になっている場合をここで防ぐ
    new_types = []
    for field in schema:
        if field.name == "datetime":
            new_types.append(pa.field("datetime", pa.timestamp("s")))
        else:
            new_types.append(field)
    
    schema = pa.schema(new_types)
    table = pa.Table.from_pandas(df2, schema=schema, preserve_index=False)

    # 3. メタデータ付与
    meta = table.schema.metadata or {}
    meta2 = dict(meta)
    if parquet_metadata:
        for k, v in parquet_metadata.items():
            meta2[str(k).encode("utf-8")] = str(v).encode("utf-8")
    table = table.replace_schema_metadata(meta2)

    # 4. 書き込み
    pq.write_table(table, pq_path)

    # CSV出力
    csv_path = None
    if export_csv:
        csv_path = out_dir / f"{basename}.csv"
        df2.to_csv(csv_path, index=False, encoding="utf-8")

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
    out_root: str | Path = "outputs",
    export_csv: bool = False,
    sleep_sec: float = 1.0,
    skip_existing: bool = True,
    stop_on_error: bool = False,
    make_monthly: bool = False,          # hourly/10min: build derived monthly files
    export_csv_monthly: bool = False,    # derived monthly CSV
) -> pd.DataFrame:
    """
    Fetch and write datasets as Parquet.

    - daily: fetch by month (JMA page is monthly) and write monthly Parquet:
        <out_root>/daily/bXXXXX/YYYY-MM.parquet
    - hourly / 10min: fetch by day and write daily Parquet (as before):
        <out_root>/<freq>/... (basename includes date)
    Return: execution log DataFrame.
    """
    out_root = Path(out_root)
    results: list[dict[str, str]] = []

    sess = requests.Session()

    # Station folder (stable)
    station_id = f"b{int(station.block):05d}"

    if freq == "daily":
        out_dir = out_root / "daily" / station_id
        out_dir.mkdir(parents=True, exist_ok=True)

        for ts in _month_starts(start, end):
            y, m = ts.year, ts.month
            try:
                df, meta = fetch_daily_month(
                    station,
                    year=y,
                    month=m,
                    session=sess,
                    sleep_sec=sleep_sec,
                )
                basename = meta["period"]  # "YYYY-MM"
                pq_path = out_dir / f"{basename}.parquet"

                if skip_existing and pq_path.exists():
                    results.append({
                        "period": meta["period"],
                        "status": "skipped",
                        "parquet": str(pq_path),
                        "error": "",
                    })
                    continue

                pq_written, _ = save_dataset(
                    df,
                    out_dir=out_dir,
                    basename=basename,
                    export_csv=export_csv,  # daily monthly: follow export_csv flag
                    parquet_metadata=meta,
                )

                results.append({
                    "period": meta["period"],
                    "status": "ok",
                    "parquet": str(pq_written),
                    "error": "",
                })

            except Exception as e:
                results.append({
                    "period": f"{y:04d}-{m:02d}",
                    "status": "error",
                    "parquet": "",
                    "error": repr(e),
                })
                if stop_on_error:
                    break

        return pd.DataFrame(results)

    # hourly / 10min (day loop)
    out_dir = out_root / freq / station_id
    out_dir.mkdir(parents=True, exist_ok=True)

    d0 = pd.to_datetime(start).date()
    d1 = pd.to_datetime(end).date()

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

            pq_written, _ = save_dataset(
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

    # derived monthly (hourly/10min)
    if make_monthly and freq in ("hourly", "10min"):
        monthly_dir = out_root / f"{freq}_monthly" / station_id
        monthly_dir.mkdir(parents=True, exist_ok=True)

        for ts in _month_starts(start, end):
            y, m = ts.year, ts.month
            out_path = monthly_dir / f"{y:04d}-{m:02d}.parquet"
            existed = out_path.exists()

            try:
                built = build_monthly_from_daily(
                    station=station,
                    freq=freq,
                    year=y,
                    month=m,
                    out_root=out_root,
                    export_csv=export_csv_monthly,
                    skip_existing=skip_existing,
                )
                results.append({
                    "period": f"{y:04d}-{m:02d}",
                    "status": "skipped" if (skip_existing and existed) else "ok",
                    "parquet": str(built),
                    "error": "",
                    "kind": "derived_monthly",
                })
            except Exception as e:
                results.append({
                    "period": f"{y:04d}-{m:02d}",
                    "status": "error",
                    "parquet": "",
                    "error": repr(e),
                    "kind": "derived_monthly",
                })

    return pd.DataFrame(results)


# =========================
# Derived monthly builder (hourly / 10min)
# =========================

def _daily_parquet_glob_pattern(freq: Freq, station: Station, year: int, month: int) -> str:
    """Pattern to find daily parquet files written by fetch_range for a given station & month."""
    block = int(station.block)
    # 余計な "_*_" を 제거して、b{block}の直後に日付が来る形に合わせる
    return f"{freq}_*_b{block}_{year:04d}-{month:02d}-*.parquet"

from pathlib import Path
import pandas as pd

def build_temperature_indices_monthly(
    daily_monthly_root: str | Path,
    out_path: str | Path,
    export_csv: bool = False,
    station_ids: list[str] | None = None,
    block_nos: list[int] | None = None,
) -> pd.DataFrame:
    """
    daily_monthly（1地点×1ヶ月）Parquetから、
    気温由来の月次指標を自己完結型テーブルとして生成する。

    station_ids: ["b47588", ...] のようにフォルダ名で対象を限定
    block_nos  : [47588, ...] のようにblock番号で対象を限定（station_idsに変換して使用）
    両方指定された場合は union（和集合）で処理する。
    """

    daily_monthly_root = Path(daily_monthly_root)

    # --- 対象フォルダ集合を作る ---
    targets: set[str] | None = None
    if station_ids or block_nos:
        targets = set()
        if station_ids:
            targets |= {sid if sid.startswith("b") else f"b{sid}" for sid in station_ids}
        if block_nos:
            targets |= {f"b{int(b)}" for b in block_nos}

    records = []

    for station_dir in sorted(daily_monthly_root.glob("b*/")):
        # 対象限定があるならフィルタ
        if targets is not None and station_dir.name not in targets:
            continue

        for pq_path in sorted(station_dir.glob("????-??.parquet")):
            df = pd.read_parquet(pq_path)
            meta = read_parquet_metadata(pq_path)

            ym = pq_path.stem  # YYYY-MM
            year = int(ym[:4])
            month = int(ym[5:7])

            tmax = pd.to_numeric(df["temp_max"], errors="coerce")
            tmin = pd.to_numeric(df["temp_min"], errors="coerce")

            records.append({
                # ---- 識別情報（自己完結） ----
                "pref": meta.get("pref", ""),
                "station": meta.get("station", ""),
                "block_no": int(meta.get("block_no", 0)),
                "prec_no": int(meta.get("prec_no", 0)),
                "station_class": meta.get("station_class", ""),
                "station_kind": meta.get("station_kind", ""),

                # ---- 時間軸 ----
                "year": year,
                "month": month,

                # ---- 気温指標（JMA定義） ----
                "n_extreme_hot_days": int((tmax >= 35).sum()),
                "n_very_hot_days": int((tmax >= 30).sum()),
                "n_summer_days": int((tmax >= 25).sum()),
                "n_ice_days": int((tmax < 0).sum()),
                "n_frost_days": int((tmin < 0).sum()),

                # ---- 欠測 ----
                "n_missing_tmax": int(tmax.isna().sum()),
                "n_missing_tmin": int(tmin.isna().sum()),
            })

    out_df = (
        pd.DataFrame(records)
        .sort_values(["block_no", "year", "month"])
        .reset_index(drop=True)
    )

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(out_path, index=False)

    if export_csv:
        out_df.to_csv(out_path.with_suffix(".csv"), index=False, encoding="utf-8")

    return out_df
    
# ============================================================
# Monthly indices (temperature day counts)
# ============================================================

def read_parquet_metadata(path: Path) -> dict:
    pf = pq.ParquetFile(path)
    meta = pf.schema_arrow.metadata or {}
    return {k.decode(): v.decode() for k, v in meta.items()}

# ============================================================
# 年次集計関数
# ============================================================

def build_temperature_indices_yearly(
    monthly_indices_path: str | Path,
    out_path: str | Path,
    export_csv: bool = False,
) -> pd.DataFrame:
    """
    月次気温指標テーブルから、地点別年次集計を生成する。

    入力: 1地点×1ヶ月 テーブル
    出力: 1地点×1年 テーブル
    """

    monthly_indices_path = Path(monthly_indices_path)
    df = pd.read_parquet(monthly_indices_path)

    indicator_cols = [
        "n_extreme_hot_days",
        "n_very_hot_days",
        "n_summer_days",
        "n_ice_days",
        "n_frost_days",
        "n_missing_tmax",
        "n_missing_tmin",
    ]

    group_cols = [
        "block_no",
        "pref",
        "station",
        "prec_no",
        "station_class",
        "station_kind",
        "year",
    ]

    yearly_df = (
        df
        .groupby(group_cols, as_index=False)[indicator_cols]
        .sum()
        .sort_values(["block_no", "year"])
        .reset_index(drop=True)
    )

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    yearly_df.to_parquet(out_path, index=False)

    if export_csv:
        yearly_df.to_csv(out_path.with_suffix(".csv"), index=False, encoding="utf-8")

    return yearly_df

# ============================================================
# stations から get_station() を使って block_no のリストを作る
# ============================================================

def stations_to_block_nos(
    stations: list[dict],
    *,
    strict: bool = True,
) -> list[int]:
    """
    stations: [{"jblock": "...", "Name": "..."}, ...]
    から get_station() を使って block_no のリストを作る。

    strict=True なら見つからない地点があれば例外。
    strict=False ならスキップして続行。
    """
    block_nos: list[int] = []
    missing: list[tuple[str, str]] = []

    for s in stations:
        block = s["jblock"]
        name = s["Name"]

        st = get_station(block, name)  # core_v4.py 内なら cv4. は不要
        if st is None:
            missing.append((block, name))
            if strict:
                raise ValueError(f"Station not found: {block=} {name=}")
            continue

        block_nos.append(int(st.block))

    # 重複除去（順序維持）
    seen = set()
    out = []
    for b in block_nos:
        if b not in seen:
            seen.add(b)
            out.append(b)

    if missing and not strict:
        print("WARNING: missing stations:")
        for block, name in missing:
            print("  -", block, name)

    return out


# ======================================================================
# Parquet 統合・クレンジング処理
# ======================================================================
from pathlib import Path
import polars as pl


def consolidate_daily_station(block_no: str, base_dir: Path | str | None = None) -> Path | None:
    if base_dir is None:
        base_dir = Path("/Volumes/Transcend/projects/MetData/JMA/daily_monthly")
    else:
        base_dir = Path(base_dir)

    station_dir = base_dir / f"b{block_no}"
    files = sorted(list(station_dir.glob("[0-9][0-9][0-9][0-9]-[0-9][0-9].parquet")))

    if not files:
        print(f"[Warn] 日別統合対象ファイルが見つかりません: {station_dir}")
        return None

    output_file = station_dir / f"b{block_no}.parquet"

    # 各ファイルを個別にスキャンし、スキーマの差異を許容して斜め結合（diagonal_relaxed）する
    lfs = [pl.scan_parquet(f) for f in files]
    lf = pl.concat(lfs, how="diagonal_relaxed")

    # datetime のキャストとソート
    lf = lf.with_columns(pl.col("datetime").cast(pl.Date)).sort("datetime")

    schema = lf.collect_schema()
    val_cols = [c for c in schema.names() if c != "datetime" and not c.endswith("_code")]

    # datetime と _code 列以外の全要素が欠損・空文字の行を除外
    not_all_empty = pl.any_horizontal([
        pl.col(c).is_not_null() & (
            pl.col(c).cast(pl.String).str.strip_chars() != "" 
            if schema[c] == pl.String 
            else pl.col(c).is_not_null()
        )
        for c in val_cols
    ])

    df_cleaned = lf.filter(not_all_empty).collect()
    df_cleaned.write_parquet(output_file, compression="zstd", compression_level=7)

    print(f"[Done] 日別統合完了: {output_file.name} ({df_cleaned.height:,} 行)")
    return output_file


def consolidate_hourly_station(
    block_no: str, base_dir: Path | str | None = None
) -> Path | None:
    """指定した地点（block_no）の日別毎時データ(YYYY-MM-DD.parquet)を統合・クレンジングし、

    hourly/b{block_no}/b{block_no}.parquet を生成する。
    """
    if base_dir is None:
        base_dir = Path("/Volumes/Transcend/projects/MetData/JMA")
    else:
        base_dir = Path(base_dir)

    station_dir = base_dir / "hourly" / f"b{block_no}"
    files = sorted(
        list(
            station_dir.glob(
                "[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9].parquet"
            )
        )
    )

    if not files:
        print(f"[Warn] 毎時統合対象ファイルが見つかりません: {station_dir}")
        return None

    out_dir = base_dir / "hourly" / f"b{block_no}"
    out_dir.mkdir(parents=True, exist_ok=True)
    output_file = out_dir / f"b{block_no}.parquet"

    lf = pl.scan_parquet(files).sort("datetime")

    schema = lf.collect_schema()
    val_cols = [
        c
        for c in schema.names()
        if c != "datetime" and not c.endswith("_code")
    ]

    not_all_empty = pl.any_horizontal(
        [
            pl.col(c).is_not_null()
            & (
                pl.col(c).cast(pl.String).str.strip_chars() != ""
                if schema[c] == pl.String
                else pl.col(c).is_not_null()
            )
            for c in val_cols
        ]
    )

    df_cleaned = lf.filter(not_all_empty).collect()
    df_cleaned.write_parquet(
        output_file, compression="zstd", compression_level=7
    )

    print(
        f"[Done] 毎時統合完了: {output_file.name} ({df_cleaned.height:,} 行)"
    )
    return output_file

