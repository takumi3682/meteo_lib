#!/usr/bin/env python
# coding: utf-8

# In[35]:


# -*- coding: utf-8 -*-
"""
JMA AMeDAS/観測所データ取得ユーティリティ（最終安定版）
- BeautifulSoupで直接HTMLを解析し、']', ')', '#' などの記号を保持
- '--' は列ごとに0またはNaNへ
- *_flag列に品質フラグ（], ), #など）を分離
- 将来のpandas仕様変更(FutureWarning)に非依存
"""

from __future__ import annotations
from functools import lru_cache
from io import StringIO
import re
import requests
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
import lxml
import os
from pathlib import Path

PROJECT = Path(__file__).resolve().parent

# =========================
# 地点検索（prec, block）
# =========================
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
    "南大東村":92,"北大東村":92,"宮古島市":93,"宮古郡":93,"石垣市":94,"八重山郡":94
}

@lru_cache(maxsize=1)
def _load_amd_list() -> pd.DataFrame:
    ROOT = PROJECT
    DATA = ROOT / "data"
    OUT  = ROOT / "outputs"
    df = pd.read_csv(DATA / "amd_list.csv", encoding="utf-8")
    return df

from typing import Optional, TypedDict

class StationInfo(TypedDict, total=False):
    prec: int
    block: int
    Name_2: str
    Alt: float
    H_Anemo: float
    Lat: float
    Lon: float

def getBlock(pref: str, place: str) -> Optional[StationInfo]:
    df_amd = _load_amd_list()  # 列: ["prec","Name","Name_2","block","Alt","H_Anemo","Lat","Lon"] を想定
    if pref not in prec_list:
        return None

    prec = int(prec_list[pref])
    m = (df_amd["prec"] == prec) & (df_amd["Name"] == place)
    rows = df_amd.loc[m, ["Name_2", "block", "Alt", "H_Anemo", "Lat", "Lon"]]

    if rows.empty:
        return None

    # 複数ヒット時は先頭を採用（方針に応じてここで弾くことも可）
    row = rows.iloc[0]

    info: StationInfo = {
        "prec": prec,
        "block": int(row["block"]) if not pd.isna(row["block"]) else None,
        "Name_2": str(row["Name_2"]) if not pd.isna(row["Name_2"]) else None,
        "Alt": float(row["Alt"]) if not pd.isna(row["Alt"]) else None,
        "H_Anemo": float(row["H_Anemo"]) if not pd.isna(row["H_Anemo"]) else None,
        "Lat": float(row["Lat"]) if not pd.isna(row["Lat"]) else None,
        "Lon": float(row["Lon"]) if not pd.isna(row["Lon"]) else None,
    }
    return info


# =========================
# 列レイアウト
# =========================
cols_k_day = (
    '日','現地気圧', '海面気圧', '降水量_合計', '降水量_1時間', '降水量_10分間',
    '平均気温', '最高気温', '最低気温', '平均湿度', '最小湿度',
    '平均風速', '最大風速', '最大時風向', '最大瞬間風速', '最大瞬間時風向',
    '日照時間', '降雪', '最深積雪', '天気概況_昼', '天気概況_夜'
)
cols_k_hour = (
    '時', '現地気圧', '海面気圧', '降水量', '気温', '露点温度', '蒸気圧',
    '湿度', '風速', '風向', '日照', '全天日射量', '降雪', '積雪', '天気', '雲量', '視程'
)
cols_k_10m = (
    '時分','気圧', '海面気圧', '降水量', '気温', '相対湿度', '平均風速', '平均風向',
    '最大瞬間風速', '最大瞬間時風向', '日照時間'
)
cols_a_day = (
    '日','降水量_合計','降水量_1時間','降水量_10分間','平均気温','最高気温','最低気温',
    '平均湿度','最小湿度','平均風速','最大風速','最大時風向','最大瞬間風速',
    '最大瞬間時風向','最多風向','日照時間','降雪','最深積雪'
)
cols_a_hour = (
    '時','降水量','気温','露点温度','蒸気圧','湿度','風速','風向','日照','降雪','積雪'
)
cols_a_10m = (
    '時分','降水量','気温','相対湿度','平均風速','平均風向','最大瞬間風速','最大瞬間時風向','日照時間'
)

# =========================
# HTMLテーブルを完全に文字列で読む
# =========================
def _read_table_preserve_flags(html_text: str, cols: tuple[str, ...], table_css: str = "table.data2_s") -> pd.DataFrame:
    soup = BeautifulSoup(html_text, "lxml")
    table = soup.select_one(table_css) or soup.find("table")
    if not table:
        return pd.DataFrame(columns=cols)

    rows = []
    for tr in table.find_all("tr"):
        tds = tr.find_all(["td", "th"])
        if not tds: continue
        row = []
        for td in tds:
            img = td.find("img")
            if img and (img.get("alt") or img.get("src")):
                cell = img.get("alt") or img.get("src")
            else:
                cell = td.get_text(separator="", strip=True).replace("\xa0", "")
            row.append(cell)
        rows.append(row)

    n_expected = len(cols)
    rows = [r[:n_expected] + [""]*(n_expected - len(r)) for r in rows]
    df = pd.DataFrame(rows, columns=cols)
    # DataFrame.applymap は非推奨 → stack+map+unstackで代替
    df = df.stack().map(lambda x: '' if pd.isna(x) else str(x).strip()).unstack()
    return df


# =========================
# データ整形ユーティリティ
# =========================
IGNORE_COLUMNS = (
    '風向','平均風向','最大時風向','最大瞬間時風向','最多風向','天気','雲量',
    '天気概況_昼', '天気概況_夜')

NUMERIC_LIKE_COLS = {
    '現地気圧':'float32', '海面気圧':'float32', '降水量_合計':'float32',
    '降水量_1時間':'float32', '降水量_10分間':'float32',
    '平均気温':'float32', '最高気温':'float32', '最低気温':'float32',
    '平均湿度':'float32', '最小湿度':'float32',
    '平均風速':'float32', '最大風速':'float32','最大瞬間風速':'float32', 
    '日照時間':'float32', '降雪':'float32', '最深積雪':'float32',
    '降水量':'float32', '気温':'float32', '露点温度':'float32',
    '蒸気圧':'float32', '湿度':'float32', '風速':'float32',
    '日照':'float32', '全天日射量':'float32','積雪':'float32','視程':'float32',
    '気圧':'float32','相対湿度':'float32'
}

ZERO_AS_ZERO = {'降水量','降水量_合計','降水量_1時間','降水量_10分間','降雪','積雪'}

_NUM_FLAG_RE = re.compile(r'^\s*([+-]?\d+(?:\.\d+)?)?\s*([^\d+\-\.].*)?$', re.DOTALL)

def split_num_and_flag(series: pd.Series, col_name: str = "") -> tuple[pd.Series, pd.Series]:
    s = series.astype('string').str.strip()
    s = s.replace({'///': None,'×': None,'na': None,'NaN': None})
    s = s.replace({'--': '0' if col_name in ZERO_AS_ZERO else None})
    m = s.str.extract(_NUM_FLAG_RE)
    num_str = m[0]; flag = m[1].fillna('').str.strip()
    invalid = flag.str.contains(r']', na=False)
    num = pd.to_numeric(num_str, errors='coerce').mask(invalid, np.nan)
    flag = flag.replace('', pd.NA)
    return num, flag

def handle_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = df.columns.str.strip()

    # ぜんぶ string（日時は後で to_datetime）
    df = df.astype('string')

    for col, dtype in NUMERIC_LIKE_COLS.items():
        if col in df.columns and col not in IGNORE_COLUMNS:
            values, flags = split_num_and_flag(df[col], col)
            df[col] = pd.to_numeric(values, errors='coerce').astype(dtype)
            if flags is not None and flags.notna().any():
                df[f'{col}_flag'] = flags.astype('string')
    return df

# =========================
# 各取得関数
# =========================
def get_DailyData(prec:int, block:int, year:int, month:int, day:int) -> pd.DataFrame:
    if block > 40000:
        url = f"https://www.data.jma.go.jp/stats/etrn/view/daily_s1.php?prec_no={prec}&block_no={block}&year={year}&month={month}&day={day}&view="
        res = requests.get(url, timeout=30); res.encoding = res.apparent_encoding
        df = _read_table_preserve_flags(res.text, cols_k_day)
        df = df.iloc[4:].reset_index(drop=True)
    else:
        url = f"https://www.data.jma.go.jp/stats/etrn/view/daily_a1.php?prec_no={prec}&block_no={block:04}&year={year}&month={month}&day={day}&view="
        res = requests.get(url, timeout=30); res.encoding = res.apparent_encoding
        df = _read_table_preserve_flags(res.text, cols_a_day)
        df = df.iloc[3:].reset_index(drop=True)

    df['日時'] = pd.to_datetime(dict(year=year, month=month, day=pd.to_numeric(df['日'], errors='coerce')), errors='coerce')
    df = df.drop(columns=['日']).set_index('日時')
    return handle_data(df)

def get_HourlyData(prec:int, block:int, year:int, month:int, day:int) -> pd.DataFrame:
    if block > 40000:
        url = f"https://www.data.jma.go.jp/stats/etrn/view/hourly_s1.php?prec_no={prec}&block_no={block}&year={year}&month={month}&day={day}&view="
        res = requests.get(url, timeout=30); res.encoding = res.apparent_encoding
        df = _read_table_preserve_flags(res.text, cols_k_hour)
    else:
        url = f"https://www.data.jma.go.jp/stats/etrn/view/hourly_a1.php?prec_no={prec}&block_no={block:04}&year={year}&month={month}&day={day}&view="
        res = requests.get(url, timeout=30); res.encoding = res.apparent_encoding
        df = _read_table_preserve_flags(res.text, cols_a_hour)

    df = df.iloc[2:].reset_index(drop=True)
    hh = pd.to_numeric(df['時'], errors='coerce')
    mask24 = (hh == 24)
    df.loc[mask24, '時分'] = '00:00'
    df.loc[~mask24, '時分'] = hh.astype('Int16').astype(str) + ':00'
    df['日時'] = pd.to_datetime(f"{year}-{month}-{day} " + df['時分'], format="%Y-%m-%d %H:%M", errors='coerce')
    df.loc[mask24, '日時'] += pd.Timedelta(days=1)
    df = df.drop(columns=['時','時分']).set_index('日時')
    return handle_data(df)

def get_10minData(prec:int, block:int, year:int, month:int, day:int) -> pd.DataFrame:
    if block > 40000:
        url = f"https://www.data.jma.go.jp/stats/etrn/view/10min_s1.php?prec_no={prec}&block_no={block}&year={year}&month={month}&day={day}&view="
        res = requests.get(url, timeout=30); res.encoding = res.apparent_encoding
        df = _read_table_preserve_flags(res.text, cols_k_10m)
        df = df.iloc[2:].reset_index(drop=True)
    else:
        url = f"https://www.data.jma.go.jp/stats/etrn/view/10min_a1.php?prec_no={prec}&block_no={block:04}&year={year}&month={month}&day={day}&view="
        res = requests.get(url, timeout=30); res.encoding = res.apparent_encoding
        df = _read_table_preserve_flags(res.text, cols_a_10m)
        df = df.iloc[3:].reset_index(drop=True)

    df['日付'] = pd.to_datetime(f"{year}-{month}-{day}")
    mask2400 = (df['時分'] == '24:00')
    df.loc[mask2400, '時分'] = '00:00'
    df.loc[mask2400, '日付'] += pd.Timedelta(days=1)
    df['日時'] = pd.to_datetime(df['日付'].dt.strftime('%Y-%m-%d') + ' ' + df['時分'], format="%Y-%m-%d %H:%M", errors='coerce')
    df = df.drop(columns=['時分','日付']).set_index('日時')
    return handle_data(df)
