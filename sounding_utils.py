"""
sounding_utils.py
=================
1次元鉛直大気プロファイル（ゾンデ実測、MSM客観解析、ERA5再解析）の取得・抽出、
大気安定度指数（CAPE, CIN, SSI, LI, PWAT等）の計算、および Skew-T Log-P 図描画モジュール。

提供関数一覧
------------
1. get_sounding(source="msm", dt=None, lat=None, lon=None, station=None, station_id=None, step_idx=0)
   - 'msm', 'era5', 'igra' の各データソースから鉛直プロファイルを統合取得するファサード関数。
   - 地点名（例: "仙台", "館野", "東京"）の指定により緯度経度・観測所IDを自動補完。
2. plot_sounding(sounding, title_info="")
   - 抽出したプロファイル辞書を受け取り、Skew-T Log-P 図、状態曲線、風向風速矢羽根、各種安定度指標を描画。
3. resolve_station_info(station_input)
   - 地点名・英名・IGRA IDから辞書マスターを照合し {lat, lon, id, name} を解決。
4. download_rish_msm_p(year, month, day, hour)
   - 京都大学RISHからMSM気圧面GPV(GRIB2)をダウンロードする下位関数。
5. extract_msm_sounding(msm_file_path, target_lat, target_lon, step_idx=0)
   - MSM気圧面GRIB2から特定地点のプロファイルを抽出・計算。
6. extract_igra_sounding(station_id, dt)
   - NOAA IGRA v2 実測データから特定日時のプロファイルを抽出・計算。
7. extract_era5_sounding(nc_file_path, target_lat, target_lon)
   - ERA5 NetCDFからプロファイルを抽出・計算。

使用例
------
from datetime import datetime
from sounding_utils import get_sounding, plot_sounding

dt = datetime(2026, 8, 13, 12)

# 1. 気象庁 MSM による仙台上空のサウンディング取得と描画
msm_sd = get_sounding(source="msm", dt=dt, station="仙台")
plot_sounding(msm_sd, title_info="MSM - Sendai (2026-08-13 12UTC)")

# 2. NOAA IGRA 実測ゾンデ（館野/つくば）の取得と描画
igra_sd = get_sounding(source="igra", dt=dt, station="館野")
plot_sounding(igra_sd, title_info="NOAA IGRA - Tateno (2026-08-13 12UTC)")

# 3. 任意の緯度経度での MSM サウンディング取得
custom_sd = get_sounding(source="msm", dt=dt, lat=35.60, lon=140.10)
"""

import io
import os
from pathlib import Path
from typing import Any, Dict, Optional, Union
import urllib.request
import zipfile
import cdsapi
import matplotlib.pyplot as plt
import metpy.calc as mpcalc
from metpy.plots import SkewT
from metpy.units import units
import numpy as np
import pandas as pd
import xarray as xr
from datetime import datetime

# ==============================================================================
# 基本ディレクトリ設定（プロジェクトルート基準）
# ==============================================================================
_cwd = Path.cwd()
_root = _cwd if (_cwd / "raw_grib").exists() or (_cwd / "data").exists() else _cwd.parent

CACHE_DIR = str(_root / "cache")
MSM_CACHE_DIR = str(_root / "cache" / "msm")
ERA5_CACHE_DIR = str(_root / "cache" / "era5")
IGRA_CACHE_DIR = str(_root / "cache" / "igra")

for d in [MSM_CACHE_DIR, ERA5_CACHE_DIR, IGRA_CACHE_DIR]:
    os.makedirs(d, exist_ok=True)

# ==============================================================================
# 観測地点マスター（緯度経度・IGRA ID統合）
# ==============================================================================
STATION_COORDS = {
    # 東北・北海道
    "札幌": {"id": "JAM00047412", "lat": 43.06, "lon": 141.33, "en": "SAPPORO"},
    "秋田": {"id": "JAM00047582", "lat": 39.72, "lon": 140.10, "en": "AKITA"},
    "仙台": {"id": "JAM00047590", "lat": 38.26, "lon": 140.90, "en": "SENDAI"},
    # 関東・中部・北陸
    "館野": {"id": "JAM00047646", "lat": 36.06, "lon": 140.13, "en": "TATENO"},
    "つくば": {"id": "JAM00047646", "lat": 36.06, "lon": 140.13, "en": "TATENO"},
    "東京": {"id": None, "lat": 35.69, "lon": 139.75, "en": "TOKYO"},
    "千葉": {"id": None, "lat": 35.60, "lon": 140.10, "en": "CHIBA"},
    "輪島": {"id": "JAM00047600", "lat": 37.39, "lon": 136.90, "en": "WAJIMA"},
    "名古屋": {"id": None, "lat": 35.18, "lon": 136.90, "en": "NAGOYA"},
    # 近畿・西日本
    "潮岬": {"id": "JAM00047778", "lat": 33.43, "lon": 135.77, "en": "SHIONOMISAKI"},
    "大阪": {"id": None, "lat": 34.68, "lon": 135.52, "en": "OSAKA"},
    "福岡": {"id": "JAM00047807", "lat": 33.58, "lon": 130.38, "en": "FUKUOKA"},
    "鹿児島": {"id": "JAM00047827", "lat": 31.55, "lon": 130.55, "en": "KAGOSHIMA"},
    # 南西諸島・海洋島
    "八丈島": {"id": "JAM00047678", "lat": 33.12, "lon": 139.78, "en": "HACHIJOJIMA"},
    "名瀬": {"id": "JAM00047909", "lat": 28.38, "lon": 129.49, "en": "NAZE"},
    "那覇": {"id": "JAM00047936", "lat": 26.21, "lon": 127.68, "en": "NAHA"},
    "石垣島": {"id": "JAM00047918", "lat": 24.34, "lon": 124.16, "en": "ISHIGAKIJIMA"},
    "南大東島": {"id": "JAM00047945", "lat": 25.83, "lon": 131.23, "en": "MINAMIDAITOJIMA"},
    "父島": {"id": "JAM00047971", "lat": 27.09, "lon": 142.19, "en": "CHICHIJIMA"},
    "南鳥島": {"id": "JAM00047991", "lat": 24.29, "lon": 153.98, "en": "MINAMITORISHIMA"},
}


def resolve_station_info(station_input: Any) -> Optional[Dict[str, Any]]:
    """地点名（日本語/英字/ID）から {lat, lon, id, name} を検索・解決"""
    if not station_input:
        return None

    s = str(station_input).strip()

    # 1. 日本語キーでの完全一致
    if s in STATION_COORDS:
        info = STATION_COORDS[s].copy()
        info["name"] = s
        return info

    # 2. 英名・IDでの検索
    for name, info in STATION_COORDS.items():
        if (
            s.upper() == info["en"].upper()
            or (info["id"] and s.upper() == info["id"].upper())
            or (s.upper() in info["en"].upper())
            or (name in s)
        ):
            res = info.copy()
            res["name"] = name
            return res

    # 3. 未登録の独自ID（JAM...形式）が直接渡された場合
    if s.startswith("JAM"):
        return {"id": s, "lat": None, "lon": None, "en": s, "name": s}

    return None


# IGRA v2 固定長フォーマット定義
COLSPECS = [
    (0, 1),
    (1, 2),
    (3, 8),
    (9, 15),
    (16, 21),
    (22, 27),
    (28, 33),
    (34, 39),
    (40, 45),
    (46, 51),
]
NAMES = [
    "lvl1",
    "lvl2",
    "etime",
    "press",
    "gph",
    "temp",
    "rh",
    "dpdp",
    "wdir",
    "wspd",
]


# ==============================================================================
# Layer 1 & 2: NOAA IGRA v2 (実測ゾンデ)
# ==============================================================================
class IGRAMemoryCache:
    def __init__(self):
        self.cached_station_id = None
        self.raw_text = ""

    def load_station(self, station_id: str):
        if self.cached_station_id == station_id:
            return
        local_zip = os.path.join(IGRA_CACHE_DIR, f"{station_id}-data.txt.zip")
        if not os.path.exists(local_zip):
            url = f"https://www.ncei.noaa.gov/pub/data/igra/data/data-por/{station_id}-data.txt.zip"
            print(f"NOAAからIGRAデータをダウンロード中... ({station_id})")
            urllib.request.urlretrieve(url, local_zip)

        with zipfile.ZipFile(local_zip) as z:
            fname = z.namelist()[0]
            self.raw_text = z.read(fname).decode("utf-8")
        self.cached_station_id = station_id


igra_cache = IGRAMemoryCache()


def extract_igra_sounding(station_id: str, dt: Any) -> Optional[Dict[str, Any]]:
    """NOAA IGRAテキストから指定日時を抽出し、プロファイルを計算"""
    igra_cache.load_station(station_id)
    target_year, target_month, target_day, target_hour = (
        dt.year,
        dt.month,
        dt.day,
        dt.hour,
    )

    lines = []
    recording = False
    for line in igra_cache.raw_text.splitlines():
        if line.startswith("#"):
            p = line.split()
            y, m, d, h = int(p[1]), int(p[2]), int(p[3]), int(p[4])
            if (y, m, d, h) == (
                target_year,
                target_month,
                target_day,
                target_hour,
            ):
                recording = True
                continue
            elif recording:
                break
        elif recording:
            lines.append(line)

    if not lines:
        print(f"IGRA観測データが見つかりません: {dt} (Station: {station_id})")
        return None

    df = pd.read_fwf(
        io.StringIO("\n".join(lines)),
        colspecs=COLSPECS,
        names=NAMES,
        na_values=[-9999, -8888],
    )
    df = df.dropna(subset=["press", "temp"]).sort_values(
        "press", ascending=False
    )
    df = df.drop_duplicates(subset=["press"], keep="first")

    p = (df["press"].values / 100.0) * units.hPa
    T = (df["temp"].values / 10.0) * units.degC
    Td = ((df["temp"].values - df["dpdp"].values) / 10.0) * units.degC

    wspd_val = df["wspd"].fillna(0).values / 10.0
    wdir_val = df["wdir"].fillna(0).values
    wind_spd = (wspd_val * units("m/s")).to(units.knots)
    wind_dir = wdir_val * units.degrees
    u, v = mpcalc.wind_components(wind_spd, wind_dir)

    return _calc_sounding_indices(p, T, Td, u, v)


# ==============================================================================
# Layer 1 & 2: 気象庁 MSM (京都大学RISH)
# ==============================================================================
def download_rish_msm_p(year: int, month: int, day: int, hour: int) -> Optional[str]:
    """京都大学RISHからMSM気圧面GPV (GRIB2) をダウンロード"""
    date_str = f"{year:04d}{month:02d}{day:02d}{hour:02d}0000"
    file_name = f"Z__C_RJTD_{date_str}_MSM_GPV_Rjp_L-pall_FH00-15_grib2.bin"
    local_path = os.path.join(MSM_CACHE_DIR, file_name)

    if os.path.exists(local_path):
        return local_path

    url = (
        f"http://database.rish.kyoto-u.ac.jp/arch/jmadata/data/gpv/original/"
        f"{year:04d}/{month:02d}/{day:02d}/{file_name}"
    )
    print(f"京都大学RISHからMSM気圧面データを取得中...\nURL: {url}")
    try:
        urllib.request.urlretrieve(url, local_path)
        return local_path
    except Exception as e:
        print(f"ダウンロード失敗: {e}")
        if os.path.exists(local_path):
            os.remove(local_path)
        return None


def extract_msm_sounding(
    msm_file_path: str, target_lat: float, target_lon: float, step_idx: int = 0
) -> Optional[Dict[str, Any]]:
    """MSM気圧面GRIB2から特定地点のプロファイルを抽出・計算"""
    try:
        ds_t = xr.open_dataset(
            msm_file_path,
            engine="cfgrib",
            backend_kwargs={
                "filter_by_keys": {
                    "typeOfLevel": "isobaricInhPa",
                    "shortName": "t",
                }
            },
        )
        ds_u = xr.open_dataset(
            msm_file_path,
            engine="cfgrib",
            backend_kwargs={
                "filter_by_keys": {
                    "typeOfLevel": "isobaricInhPa",
                    "shortName": "u",
                }
            },
        )
        ds_v = xr.open_dataset(
            msm_file_path,
            engine="cfgrib",
            backend_kwargs={
                "filter_by_keys": {
                    "typeOfLevel": "isobaricInhPa",
                    "shortName": "v",
                }
            },
        )
        ds_r = xr.open_dataset(
            msm_file_path,
            engine="cfgrib",
            backend_kwargs={
                "filter_by_keys": {
                    "typeOfLevel": "isobaricInhPa",
                    "shortName": "r",
                }
            },
        )
        ds = xr.merge([ds_t, ds_u, ds_v, ds_r], join="outer", compat="override")
    except Exception as e:
        print(f"MSM GRIB2展開エラー: {e}")
        return None

    if "step" in ds.dims:
        ds_point = ds.isel(step=step_idx).sel(
            latitude=target_lat, longitude=target_lon, method="nearest"
        )
    else:
        ds_point = ds.sel(
            latitude=target_lat, longitude=target_lon, method="nearest"
        )

    ds_point = ds_point.squeeze().sortby("isobaricInhPa", ascending=False)

    p = np.asarray(ds_point["isobaricInhPa"].values).flatten() * units.hPa
    T = (np.asarray(ds_point["t"].values).flatten() * units.kelvin).to("degC")
    u = (np.asarray(ds_point["u"].values).flatten() * units("m/s")).to(units.knots)
    v = (np.asarray(ds_point["v"].values).flatten() * units("m/s")).to(units.knots)

    if "r" in ds_point:
        rh_raw = ds_point["r"].values
        rh_arr = np.full(p.shape, np.nan)
        r_plevs = ds_point["r"]["isobaricInhPa"].values
        for i, p_val in enumerate(p.magnitude):
            if p_val in r_plevs:
                idx = np.where(r_plevs == p_val)[0][0]
                rh_arr[i] = rh_raw[idx]
        rh = rh_arr * units.percent
        Td = mpcalc.dewpoint_from_relative_humidity(T, rh)
    else:
        Td = np.full(p.shape, np.nan) * units.degC

    res = _calc_sounding_indices(p, T, Td, u, v)
    if res:
        res["lat"] = float(ds_point["latitude"].values)
        res["lon"] = float(ds_point["longitude"].values)
    return res


# ==============================================================================
# Layer 1 & 2: ECMWF ERA5 (再解析)
# ==============================================================================
def download_era5_sounding_nc(
    year: int, month: int, day: int, hour: int, lat: float, lon: float
) -> Optional[str]:
    """ECMWF CDSからERA5気圧面データをNetCDFでダウンロード"""
    file_name = (
        f"era5_sounding_{year:04d}{month:02d}{day:02d}_{hour:02d}UTC_{lat:.1f}N_{lon:.1f}E.nc"
    )
    local_path = os.path.join(ERA5_CACHE_DIR, file_name)

    if os.path.exists(local_path):
        return local_path

    pressure_levels = [
        "100", "125", "150", "175", "200", "225", "250", "300", "350", "400",
        "450", "500", "550", "600", "650", "700", "750", "775", "800", "825",
        "850", "875", "900", "925", "950", "975", "1000",
    ]

    print(f"ERA5 CDSからデータを取得中... ({year:04d}-{month:02d}-{day:02d} {hour:02d}:00 UTC)")
    c = cdsapi.Client()
    area = [lat + 0.5, lon - 0.5, lat - 0.5, lon + 0.5]

    try:
        c.retrieve(
            "reanalysis-era5-pressure-levels",
            {
                "product_type": "reanalysis",
                "variable": [
                    "temperature",
                    "relative_humidity",
                    "u_component_of_wind",
                    "v_component_of_wind",
                ],
                "pressure_level": pressure_levels,
                "year": f"{year:04d}",
                "month": f"{month:02d}",
                "day": f"{day:02d}",
                "time": f"{hour:02d}:00",
                "area": area,
                "format": "netcdf",
            },
            local_path,
        )
        return local_path
    except Exception as e:
        print(f"ERA5取得失敗: {e}")
        if os.path.exists(local_path):
            os.remove(local_path)
        return None


def extract_era5_sounding(
    nc_file_path: str, target_lat: float, target_lon: float
) -> Optional[Dict[str, Any]]:
    """ERA5 NetCDFからプロファイルを抽出・計算"""
    try:
        ds = xr.open_dataset(nc_file_path)
    except Exception as e:
        print(f"ERA5 NetCDF読み込みエラー: {e}")
        return None

    lat_key = "latitude" if "latitude" in ds else "lat"
    lon_key = "longitude" if "longitude" in ds else "lon"
    plev_key = "pressure_level" if "pressure_level" in ds else "level"

    ds_point = ds.sel({lat_key: target_lat, lon_key: target_lon}, method="nearest")
    if "time" in ds_point.dims:
        ds_point = ds_point.isel(time=0)

    ds_point = ds_point.squeeze().sortby(plev_key, ascending=False)

    p = np.asarray(ds_point[plev_key].values).flatten() * units.hPa
    t_val = np.asarray(ds_point["t"].values).flatten()
    r_val = np.asarray(ds_point["r"].values).flatten()
    u_val = np.asarray(ds_point["u"].values).flatten()
    v_val = np.asarray(ds_point["v"].values).flatten()

    T = (t_val * units.kelvin).to("degC")
    rh = r_val * units.percent
    Td = mpcalc.dewpoint_from_relative_humidity(T, rh)

    u = (u_val * units("m/s")).to(units.knots)
    v = (v_val * units("m/s")).to(units.knots)

    res = _calc_sounding_indices(p, T, Td, u, v)
    if res:
        res["lat"] = float(ds_point[lat_key].values)
        res["lon"] = float(ds_point[lon_key].values)
    return res


# ==============================================================================
# 内部共通関数: 各種大気安定度指標の計算
# ==============================================================================
def _calc_sounding_indices(p, T, Td, u, v) -> Optional[Dict[str, Any]]:
    valid_mask = ~np.isnan(T) & ~np.isnan(Td)
    if not np.any(valid_mask):
        return None

    p_valid = p[valid_mask]
    T_valid = T[valid_mask]
    Td_valid = Td[valid_mask]

    indices = {}
    try:
        lcl_p, lcl_t = mpcalc.lcl(p_valid[0], T_valid[0], Td_valid[0])
        indices["lcl_p"] = lcl_p

        parcel_prof = mpcalc.parcel_profile(p_valid, T_valid[0], Td_valid[0]).to("degC")
        indices["parcel_prof"] = parcel_prof

        cape, cin = mpcalc.cape_cin(p_valid, T_valid, Td_valid, parcel_prof)
        indices["cape"] = cape
        indices["cin"] = cin

        lfc_p, lfc_t = mpcalc.lfc(p_valid, T_valid, Td_valid, parcel_prof)
        el_p, el_t = mpcalc.el(p_valid, T_valid, Td_valid, parcel_prof)
        indices["lfc_p"] = lfc_p
        indices["el_p"] = el_p
    except Exception:
        indices["parcel_prof"] = None
        indices["cape"] = np.nan * (units.joule / units.kilogram)
        indices["cin"] = np.nan * (units.joule / units.kilogram)
        indices["lcl_p"] = np.nan * units.hPa
        indices["lfc_p"] = np.nan * units.hPa
        indices["el_p"] = np.nan * units.hPa

    try:
        ssi_val = mpcalc.showalter_index(p_valid, T_valid, Td_valid)
        indices["ssi"] = np.atleast_1d(ssi_val)[0]
    except Exception:
        indices["ssi"] = np.nan * units.delta_degC

    try:
        if indices["parcel_prof"] is not None:
            li_val = mpcalc.lifted_index(p_valid, T_valid, indices["parcel_prof"])
            indices["li"] = np.atleast_1d(li_val)[0]
        else:
            indices["li"] = np.nan * units.delta_degC
    except Exception:
        indices["li"] = np.nan * units.delta_degC

    try:
        k_val = mpcalc.k_index(p_valid, T_valid, Td_valid)
        indices["k_idx"] = np.atleast_1d(k_val)[0]
    except Exception:
        indices["k_idx"] = np.nan * units.degC

    try:
        tt_val = mpcalc.total_totals_index(p_valid, T_valid, Td_valid)
        indices["tt"] = np.atleast_1d(tt_val)[0]
    except Exception:
        indices["tt"] = np.nan * units.delta_degC

    try:
        pwat_val = mpcalc.precipitable_water(p_valid, Td_valid)
        indices["pwat"] = pwat_val.to("mm")
    except Exception:
        indices["pwat"] = np.nan * units.mm

    return {
        "p": p,
        "T": T,
        "Td": Td,
        "p_valid": p_valid,
        "T_valid": T_valid,
        "Td_valid": Td_valid,
        "u": u,
        "v": v,
        "indices": indices,
    }


# ==============================================================================
# 統一ファサード関数
# ==============================================================================
def get_sounding(
    source: str = "msm",
    dt: Any = None,
    lat: Optional[float] = None,
    lon: Optional[float] = None,
    station: Optional[str] = None,
    station_id: Optional[str] = None,
    step_idx: int = 0,
    **kwargs,
) -> Optional[Dict[str, Any]]:
    """'msm', 'era5', 'igra' を統合取得する窓口関数"""
    st_target = station if station is not None else station_id
    st_info = resolve_station_info(st_target) if st_target else None

    target_lat = (
        lat if lat is not None else (st_info["lat"] if st_info and "lat" in st_info else None)
    )
    target_lon = (
        lon if lon is not None else (st_info["lon"] if st_info and "lon" in st_info else None)
    )
    resolved_id = st_info["id"] if st_info and st_info["id"] else st_target

    src = source.lower()

    # 1. 気象庁 MSM
    if src == "msm":
        if target_lat is None or target_lon is None:
            raise ValueError("MSMの抽出には lat/lon または有効な station が必要です。")
        file_path = download_rish_msm_p(dt.year, dt.month, dt.day, dt.hour)
        if file_path:
            sounding = extract_msm_sounding(
                file_path, target_lat, target_lon, step_idx=step_idx
            )
            if sounding and st_info:
                sounding["station_name"] = st_info["name"]
            return sounding

    # 2. ECMWF ERA5
    elif src == "era5":
        if target_lat is None or target_lon is None:
            raise ValueError("ERA5の抽出には lat/lon または有効な station が必要です。")
        file_path = download_era5_sounding_nc(
            dt.year, dt.month, dt.day, dt.hour, target_lat, target_lon
        )
        if file_path:
            sounding = extract_era5_sounding(file_path, target_lat, target_lon)
            if sounding and st_info:
                sounding["station_name"] = st_info["name"]
            return sounding

    # 3. NOAA IGRA (実測)
    elif src == "igra":
        if not resolved_id:
            raise ValueError(
                f"指定された地点 '{st_target}' は高層ゾンデ実測（IGRA）に対応していません。"
            )
        sounding = extract_igra_sounding(resolved_id, dt)
        if sounding:
            sounding["station_name"] = st_info["name"] if st_info else resolved_id
            sounding["station_id"] = resolved_id
        return sounding

    else:
        raise ValueError(f"未対応のデータソース: {source} ('msm', 'era5', 'igra')")


# ==============================================================================
# Skew-T Log-P 描画関数
# ==============================================================================
def plot_sounding(sounding: Optional[Dict[str, Any]], title_info: str = "", rotation=45):
    """抽出したプロファイル辞書を受け取り Skew-T Log-P 図を描画"""
    if sounding is None:
        print("描画データが存在しません。")
        return

    p, T, Td = sounding["p"], sounding["T"], sounding["Td"]
    u, v = sounding["u"], sounding["v"]
    idx = sounding["indices"]

    fig = plt.figure(figsize=(9.0, 9.5))
    skew = SkewT(fig, rotation=rotation)

    skew.plot(p, T, "r", linewidth=2.0, label="Temperature (T)")
    skew.plot(p, Td, "g", linewidth=2.0, label="Dewpoint (Td)")
    skew.plot_barbs(p[::1], u[::1], v[::1])

    if idx.get("parcel_prof") is not None:
        skew.plot(
            sounding["p_valid"],
            idx["parcel_prof"],
            "k--",
            linewidth=1.5,
            label="Parcel Profile",
        )
        skew.shade_cape(
            sounding["p_valid"],
            sounding["T_valid"],
            idx["parcel_prof"],
            color="red",
            alpha=0.25,
        )
        skew.shade_cin(
            sounding["p_valid"],
            sounding["T_valid"],
            idx["parcel_prof"],
            sounding["Td_valid"],
            color="lightblue",
            alpha=0.3,
        )

    if not np.isnan(idx["lcl_p"].magnitude):
        skew.ax.axhline(
            idx["lcl_p"],
            color="navy",
            linestyle=":",
            alpha=0.7,
            linewidth=1.2,
            label=f"LCL ({idx['lcl_p'].magnitude:.0f} hPa)",
        )
    if not np.isnan(idx["lfc_p"].magnitude):
        skew.ax.axhline(
            idx["lfc_p"],
            color="darkorange",
            linestyle="--",
            alpha=0.8,
            linewidth=1.2,
            label=f"LFC ({idx['lfc_p'].magnitude:.0f} hPa)",
        )
    if not np.isnan(idx["el_p"].magnitude):
        skew.ax.axhline(
            idx["el_p"],
            color="crimson",
            linestyle="-.",
            alpha=0.8,
            linewidth=1.2,
            label=f"EL  ({idx['el_p'].magnitude:.0f} hPa)",
        )

    skew.ax.set_ylim(1050, 100)
    skew.ax.set_xlim(-40, 40)
    skew.plot_dry_adiabats(color="darkred", alpha=0.15, linestyle="--")
    skew.plot_moist_adiabats(color="darkgreen", alpha=0.15, linestyle=":")
    skew.plot_mixing_lines(color="blue", alpha=0.12, linestyle="--")
    skew.ax.axvline(0, color="cyan", linestyle="--", alpha=0.5)

    skew.ax.set_xlabel("Temperature [degC]", fontsize=11)
    skew.ax.set_ylabel("Pressure [hPa]", fontsize=11)

    def fmt(val, unit_str, precision=0):
        if np.isnan(val.magnitude):
            return "--"
        return f"{val.magnitude:.{precision}f} {unit_str}"

    info_lines = [
        f"CAPE:  {fmt(idx['cape'], 'J/kg', 0):>11}",
        f"CIN:   {fmt(idx['cin'], 'J/kg', 0):>11}",
        f"PWAT:  {fmt(idx['pwat'], 'mm', 1):>11}",
        f"--------------------",
        f"SSI:   {fmt(idx['ssi'], 'degC', 1):>11}",
        f"LI:    {fmt(idx['li'], 'degC', 1):>11}",
        f"K-Idx: {fmt(idx['k_idx'], 'degC', 1):>11}",
        f"TT:    {fmt(idx['tt'], 'degC', 1):>11}",
    ]
    info_text = "\n".join(info_lines)

    skew.ax.text(
        0.03,
        0.04,
        info_text,
        transform=skew.ax.transAxes,
        fontsize=8.5,
        verticalalignment="bottom",
        family="monospace",
        bbox=dict(
            boxstyle="round,pad=0.5",
            facecolor="white",
            alpha=0.92,
            edgecolor="gray",
        ),
    )

    plt.title(f"Atmospheric Sounding Profile: {title_info}", fontsize=11, pad=10)
    skew.ax.legend(loc="upper left", fontsize=8)
    plt.tight_layout()
    plt.show()


def get_msm_level_plane(dt: datetime, level: int = 850, step_idx: int = 0):
  """MSM気圧面GPVから指定気圧面（hPa）の2次元水平データを一括取得する関数"""
  file_path = download_rish_msm_p(dt.year, dt.month, dt.day, dt.hour)
  if not file_path:
    return None

  # cfgrib によるデータセット結合（shortName を filter_by_keys の中に含める）
  ds_list = []
  for var in ["t", "u", "v", "r", "gh"]:
    try:
      ds_var = xr.open_dataset(
          file_path,
          engine="cfgrib",
          backend_kwargs={
              "filter_by_keys": {
                  "typeOfLevel": "isobaricInhPa",
                  "shortName": var,
              }
          },
      )
      ds_list.append(ds_var)
    except Exception as e:
      print(f"変数 {var} の読み込みスキップまたはエラー: {e}")

  if not ds_list:
    return None

  ds = xr.merge(ds_list, join="outer", compat="override")

  if "step" in ds.dims:
    ds_plane = ds.isel(step=step_idx).sel(isobaricInhPa=level)
  else:
    ds_plane = ds.sel(isobaricInhPa=level)

  lons_1d = ds_plane["longitude"].values
  lats_1d = ds_plane["latitude"].values
  lons, lats = np.meshgrid(lons_1d, lats_1d)

  return {
      "lons": lons,
      "lats": lats,
      "z": ds_plane["gh"].values if "gh" in ds_plane else None,
      "t": (ds_plane["t"].values * units.kelvin).to("degC").magnitude,
      "rh": ds_plane["r"].values if "r" in ds_plane else None,
      "u": ds_plane["u"].values,
      "v": ds_plane["v"].values,
      "level": level,
  }
