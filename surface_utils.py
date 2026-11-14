"""surface_utils.py.

================
気象庁GPV（MSM地上客観解析、解析雨量RA、全国合成レーダー）およびアメダス実況データの
自動ダウンロード・抽出・NetCDF4キャッシュ管理・メソ実況作図モジュール。

提供関数一覧
------------
1. get_msm_surface(dt_str, extent=None, cache_dir=None, raw_dir=None)
   - MSM地上客観解析 (FT=0) データを取得。海面更正気圧(hPa)の単位換算および指定領域のNetCDF4キャッシュを返却。
2. get_ra_dataset(dt_str, extent=None, cache_dir=None, raw_dir=None)
   - 解析雨量(RA: 1kmメッシュ・1時間積算降水量[mm/h])を取得し、指定領域をNetCDF4キャッシュして返却。
3. get_radar_dataset(dt_str, extent=None, cache_dir=None, raw_dir=None)
   - 全国合成レーダーGPV(1kmメッシュ降水強度[mm/h])をダウンロードし、wgrib2経由でNetCDF4化して返却。
4. load_amedas_parquet(base_dir, target_utc, extent)
   - 指定日時のアメダス実況Parquetデータを読み込み、指定領域でフィルタリングして返却。
5. plot_surface_meso_analysis(dt_str, amedas_dir, extent=None, title_suffix="", save_path=None)
   - MSM海面更正気圧等圧線 ＋ アメダス実況（気温カラー散布図・風矢羽）の地上メソ解析図を作図。
6. plot_nowcast_surface_quicklook(dt_str, amedas_dir, extent="関東", save_path=None)
   - 全国合成レーダー降水強度(面) ＋ アメダス実況（気温数値・風矢羽）の速報メソ解析図を作図。
7. download_msm_grib / download_ra_grib / download_radar_grib
   - 各種GRIB2生データをRISH（京都大学生存圏研究所）アーカイブからダウンロードする下位関数。

使用例
------
import surface_utils as surf

# 1. 地上メソ解析図（MSM等圧線 + アメダス気温・風）
surf.plot_surface_meso_analysis(
    dt_str="2026081312",
    amedas_dir="./data/amedas_parquet/",
    extent=surf.REGION_EXTENTS["南関東"]
)

# 2. 速報メソ実況図（合成レーダー降水強度面 + アメダス実況）
surf.plot_nowcast_surface_quicklook(
    dt_str="202608131200",
    amedas_dir="./data/amedas_parquet/",
    extent="南関東"  # 地域プリセット名または (lon_min, lon_max, lat_min, lat_max)
)
"""

from datetime import datetime, timedelta, timezone
import os
from pathlib import Path
import subprocess
from typing import Optional, Tuple, Union
import urllib.error
import urllib.request
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.colors import BoundaryNorm, ListedColormap
import matplotlib.patheffects as PathEffects
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import xarray as xr
import io
import math
from PIL import Image
import requests
from zoneinfo import ZoneInfo

# ==============================================================================
# 基本パス・地域プリセット定義
# ==============================================================================
_cwd = Path.cwd()
_root = (
    _cwd
    if (_cwd / "raw_grib").exists() or (_cwd / "data").exists()
    else _cwd.parent
)

RAW_DIR = str(_root / "raw_grib")
CACHE_DIR = str(_root / "cache")
SURF_CACHE_DIR = str(_root / "cache" / "surface")
ERA5_SURF_DIR = str(_root / "cache" / "era5_surf")
ERA5_UPPER_DIR = str(_root / "cache" / "era5_upper")

os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(SURF_CACHE_DIR, exist_ok=True)
os.makedirs(ERA5_SURF_DIR, exist_ok=True)
os.makedirs(ERA5_UPPER_DIR, exist_ok=True)

REGION_EXTENTS = {
    "全国": [128.0, 146.0, 30.0, 46.0],
    "関東": [138.5, 141.0, 34.5, 37.5],
    "南関東": [138.8, 140.9, 34.8, 36.4],
    "東北": [138.5, 142.5, 36.5, 41.8],
    "北陸": [135.5, 139.5, 35.5, 38.5],
    "中部": [135.5, 139.5, 34.0, 37.5],
    "近畿": [134.0, 137.0, 33.0, 36.0],
    "中国": [130.5, 135.0, 33.5, 36.5],
    "四国": [132.0, 135.0, 32.5, 35.0],
    "九州": [129.0, 132.5, 31.0, 34.5],
    "北海道": [139.0, 146.0, 41.0, 45.8],
}

REGION_EN = {
    "全国": "Japan",
    "関東": "Kanto",
    "南関東": "S-Kanto",
    "東北": "Tohoku",
    "北陸": "Hokuriku",
    "中部": "Chubu",
    "近畿": "Kinki",
    "中国": "Chugoku",
    "四国": "Shikoku",
    "九州": "Kyushu",
    "北海道": "Hokkaido",
}

METAR_STATIONS_JAPAN = [
    {"icao": "RJCC", "name": "新千歳", "lon": 141.69, "lat": 42.78},
    {"icao": "RJCO", "name": "札幌(丘珠)", "lon": 141.38, "lat": 43.12},
    {"icao": "RJEC", "name": "旭川", "lon": 142.45, "lat": 43.67},
    {"icao": "RJCH", "name": "函館", "lon": 140.82, "lat": 41.77},
    {"icao": "RJSS", "name": "仙台", "lon": 140.92, "lat": 38.14},
    {"icao": "RJSA", "name": "青森", "lon": 140.69, "lat": 40.73},
    {"icao": "RJSM", "name": "三沢", "lon": 141.37, "lat": 40.70},
    {"icao": "RJSI", "name": "花巻", "lon": 141.14, "lat": 39.43},
    {"icao": "RJSK", "name": "秋田", "lon": 140.22, "lat": 39.62},
    {"icao": "RJSC", "name": "山形", "lon": 140.37, "lat": 38.41},
    {"icao": "RJSF", "name": "福島", "lon": 140.43, "lat": 37.23},
    {"icao": "RJTT", "name": "東京(羽田)", "lon": 139.78, "lat": 35.55},
    {"icao": "RJAA", "name": "成田", "lon": 140.39, "lat": 35.76},
    {"icao": "RJAH", "name": "百里", "lon": 140.41, "lat": 36.18},
    {"icao": "RJTY", "name": "横田", "lon": 139.35, "lat": 35.75},
    {"icao": "RJTA", "name": "厚木", "lon": 139.45, "lat": 35.45},
    {"icao": "RJTL", "name": "下総", "lon": 140.01, "lat": 35.80},
    {"icao": "RJSN", "name": "新潟", "lon": 139.11, "lat": 37.96},
    {"icao": "RJNT", "name": "富山", "lon": 137.19, "lat": 36.65},
    {"icao": "RJNK", "name": "小松", "lon": 136.41, "lat": 36.40},
    {"icao": "RJGG", "name": "中部国際", "lon": 136.81, "lat": 34.86},
    {"icao": "RJNA", "name": "名古屋(小牧)", "lon": 136.92, "lat": 35.26},
    {"icao": "RJNS", "name": "静岡", "lon": 138.19, "lat": 34.80},
    {"icao": "RJBB", "name": "関西", "lon": 135.23, "lat": 34.43},
    {"icao": "RJOO", "name": "大阪(伊丹)", "lon": 135.44, "lat": 34.78},
    {"icao": "RJBE", "name": "神戸", "lon": 135.22, "lat": 34.63},
    {"icao": "RJOB", "name": "岡山", "lon": 133.86, "lat": 34.76},
    {"icao": "RJOA", "name": "広島", "lon": 132.92, "lat": 34.44},
    {"icao": "RJOT", "name": "高松", "lon": 134.02, "lat": 34.21},
    {"icao": "RJOZ", "name": "松山", "lon": 132.70, "lat": 33.83},
    {"icao": "RJOK", "name": "高知", "lon": 133.67, "lat": 33.55},
    {"icao": "RJFF", "name": "福岡", "lon": 130.45, "lat": 33.59},
    {"icao": "RJFT", "name": "熊本", "lon": 130.86, "lat": 32.84},
    {"icao": "RJFM", "name": "宮崎", "lon": 131.45, "lat": 31.88},
    {"icao": "RJFK", "name": "鹿児島", "lon": 130.72, "lat": 31.80},
    {"icao": "ROAH", "name": "那覇", "lon": 127.65, "lat": 26.20},
]


# ==============================================================================
# 共通描画ユーティリティ
# ==============================================================================
def get_jma_radar_cmap():
  """気象庁レーダー降水強度用のカラーマップと境界値を返す"""
  radar_colors = [
      (1, 1, 1, 0.0),  # 0 mm/h (完全透明)
      "#a0d2ff",  # 1 - 5
      "#0041ff",  # 5 - 10
      "#0096ff",  # 10 - 20
      "#ffff00",  # 20 - 30
      "#ff9900",  # 30 - 50
      "#ff2800",  # 50 - 80
      "#b40068",  # 80 mm/h 以上
  ]
  levels = [0.0, 1.0, 5.0, 10.0, 20.0, 30.0, 50.0, 80.0, 200.0]
  cmap = ListedColormap(radar_colors)
  norm = BoundaryNorm(levels, len(radar_colors))
  return cmap, norm, levels


# ==============================================================================
# 1. MSM 地上データ取得 (客観解析 FT=0 / 瞬時値のみ)
# ==============================================================================
def download_msm_grib(dt_str: str, raw_dir: str = RAW_DIR) -> str:
  """MSMの地上GRIB2ファイルをダウンロードする。"""
  os.makedirs(raw_dir, exist_ok=True)

  candidates = [
      f"Z__C_RJTD_{dt_str}0000_MSM_GPV_Rjp_Lsurf_FH00-15_grib2.bin",
      f"Z__C_RJTD_{dt_str}0000_MSM_GPV_Rjp_Lsurf_FH00-33_grib2.bin",
      f"Z__C_RJTD_{dt_str}0000_MSM_GPV_Rjp_Lsurf_FH00-39_grib2.bin",
      f"Z__C_RJTD_{dt_str}0000_MSM_GPV_Rjp_Lsurf_FH00-72_grib2.bin",
  ]
  base_url = f"http://database.rish.kyoto-u.ac.jp/arch/jmadata/data/gpv/original/{dt_str[:4]}/{dt_str[4:6]}/{dt_str[6:8]}"

  for fname in candidates:
    local_path = os.path.join(raw_dir, fname)
    if os.path.exists(local_path):
      return local_path

  for fname in candidates:
    url = f"{base_url}/{fname}"
    local_path = os.path.join(raw_dir, fname)
    print(f"[MSM] 取得試行: {fname}")
    try:
      urllib.request.urlretrieve(url, local_path)
      print(f"[MSM] ダウンロード完了: {fname}")
      return local_path
    except urllib.error.HTTPError as e:
      if e.code == 404:
        continue
      raise e
    except Exception as e:
      print(f"ダウンロードエラー: {e}")
      continue

  raise FileNotFoundError(
      f"MSMのGRIB2ファイルが見つかりません (対象時刻: {dt_str})"
  )


def get_msm_surface(
    dt_str: str,
    extent: Optional[Tuple[float, float, float, float]] = None,
    cache_dir: str = SURF_CACHE_DIR,
    raw_dir: str = RAW_DIR,
) -> Optional[xr.Dataset]:
  """MSM地上客観解析 (FT=0) データを取得・整形して返す関数"""
  os.makedirs(cache_dir, exist_ok=True)

  if len(dt_str) > 10:
    dt_str = dt_str[:10]

  ext_str = (
      f"{extent[0]}_{extent[1]}_{extent[2]}_{extent[3]}" if extent else "all"
  )
  cache_file = os.path.join(cache_dir, f"msm_surf_{dt_str}_{ext_str}.nc")

  if os.path.exists(cache_file):
    print(f"[MSM] キャッシュから読み込み: {cache_file}")
    return xr.open_dataset(cache_file)

  grib_path = download_msm_grib(dt_str, raw_dir=raw_dir)
  print(f"[MSM] GRIB2解析中...")

  try:
    ds = xr.open_dataset(
        grib_path,
        engine="cfgrib",
        backend_kwargs={"filter_by_keys": {"stepType": "instant"}},
    )
  except Exception as e:
    print(f"GRIB2オープンエラー: {e}")
    return None

  lon_key = "longitude" if "longitude" in ds.coords else "lon"
  lat_key = "latitude" if "latitude" in ds.coords else "lat"

  if lon_key != "longitude" or lat_key != "latitude":
    ds = ds.rename({lon_key: "longitude", lat_key: "latitude"})
    lon_key, lat_key = "longitude", "latitude"

  if extent is not None:
    lon_min, lon_max, lat_min, lat_max = extent
    lats = ds[lat_key].values
    if len(lats) > 1 and lats[0] > lats[-1]:
      ds = ds.sel({
          lon_key: slice(lon_min, lon_max),
          lat_key: slice(lat_max, lat_min),
      })
    else:
      ds = ds.sel({
          lon_key: slice(lon_min, lon_max),
          lat_key: slice(lat_min, lat_max),
      })

  for p_key in ["prmsl", "PRMSL", "msl", "MSL"]:
    if p_key in ds:
      if np.nanmean(ds[p_key].values) > 2000:
        ds[p_key] = ds[p_key] / 100.0
      ds[p_key].attrs["units"] = "hPa"
      break

  ds_clean = ds.squeeze()
  drop_candidates = [
      "step",
      "valid_time",
      "surface",
      "time",
      "heightAboveGround",
  ]
  vars_to_drop = [
      c
      for c in drop_candidates
      if c in ds_clean.coords and c not in ds_clean.dims
  ]
  if vars_to_drop:
    ds_clean = ds_clean.drop_vars(vars_to_drop)

  try:
    ds_clean.to_netcdf(cache_file)
    print(f"[MSM] NetCDF4キャッシュ保存完了: {cache_file}")
  except Exception as e:
    print(f"[MSM] キャッシュ保存スキップ ({e})")

  return ds_clean


# ==============================================================================
# 2. 解析雨量 (RA: 1kmメッシュ・1時間積算降水量)
# ==============================================================================
def download_ra_grib(dt_str: str, raw_dir: str = RAW_DIR) -> str:
  """解析雨量 (1kmメッシュ・1時間雨量) ファイルをダウンロードする。"""
  os.makedirs(raw_dir, exist_ok=True)
  yyyy, mm, dd, hh = dt_str[:4], dt_str[4:6], dt_str[6:8], dt_str[8:10]

  local_filename = f"ra1h_{dt_str}.grib2"
  local_path = os.path.join(raw_dir, local_filename)

  if os.path.exists(local_path) and os.path.getsize(local_path) > 0:
    return local_path

  file_candidates = [
      f"ra1h_{dt_str}.bin",
      f"ra1h_{dt_str}.grib2",
      f"ra1h_{yyyy}{mm}{dd}_{hh}.bin",
      f"Z__C_RJTD_{dt_str}0000_SRD_GPV_Rjp_Ggis1km_Prr1h_grib2.bin",
      f"{dt_str}.bin",
  ]

  base_urls = [
      f"http://database.rish.kyoto-u.ac.jp/arch/jmadata/data/gpv/original/{yyyy}/{mm}/{dd}",
      f"http://database.rish.kyoto-u.ac.jp/arch/jmadata/data/radar/ra1h/{yyyy}/{mm}/{dd}",
      f"http://database.rish.kyoto-u.ac.jp/arch/jmadata/data/radar/{yyyy}/{mm}/{dd}",
  ]

  for base_url in base_urls:
    for fname in file_candidates:
      url = f"{base_url}/{fname}"
      try:
        req = urllib.request.Request(url, method="HEAD")
        with urllib.request.urlopen(req, timeout=5) as resp:
          if resp.status == 200:
            print(f"[RA] ダウンロード中: {url}")
            urllib.request.urlretrieve(url, local_path)
            print(f"[RA] ダウンロード完了: {local_path}")
            return local_path
      except Exception:
        continue

  raise FileNotFoundError(
      f"解析雨量(RA) のファイルが見つかりませんでした (対象時刻: {dt_str})"
  )


def get_ra_dataset(
    dt_str: str,
    extent: Optional[Tuple[float, float, float, float]] = None,
    cache_dir: str = SURF_CACHE_DIR,
    raw_dir: str = RAW_DIR,
) -> Optional[xr.Dataset]:
  """解析雨量(1km)を取得し、指定領域をNetCDF4にキャッシュして返す。"""
  os.makedirs(cache_dir, exist_ok=True)

  ext_str = (
      f"{extent[0]}_{extent[1]}_{extent[2]}_{extent[3]}" if extent else "all"
  )
  cache_file = os.path.join(cache_dir, f"ra_1km_{dt_str}_{ext_str}.nc")

  if os.path.exists(cache_file):
    print(f"[RA] キャッシュから読み込み: {cache_file}")
    return xr.open_dataset(cache_file)

  grib_path = download_ra_grib(dt_str, raw_dir=raw_dir)
  print(f"[RA] GRIB2解析中...")

  try:
    ds = xr.open_dataset(grib_path, engine="cfgrib")
  except Exception as e:
    print(f"GRIB2オープンエラー: {e}")
    return None

  var_name = list(ds.data_vars.keys())[0]
  ds = ds.rename({var_name: "tp"})
  ds["tp"].attrs["units"] = "mm/h"
  ds["tp"].attrs["long_name"] = "1-hour Accumulated Precipitation"

  if extent:
    lon_min, lon_max, lat_min, lat_max = extent
    lats = ds["latitude"].values
    if len(lats) > 1 and lats[0] > lats[-1]:
      ds = ds.sel(
          latitude=slice(lat_max, lat_min), longitude=slice(lon_min, lon_max)
      )
    else:
      ds = ds.sel(
          latitude=slice(lat_min, lat_max), longitude=slice(lon_min, lon_max)
      )

  drop_coords = [
      c for c in ["step", "valid_time", "surface"] if c in ds.coords
  ]
  ds_clean = ds.reset_coords(drop_coords, drop=True)

  try:
    ds_clean.to_netcdf(cache_file)
    print(f"[RA] NetCDF4キャッシュ保存完了: {cache_file}")
  except Exception as e:
    print(f"[RA] キャッシュ保存スキップ ({e})")

  return ds_clean


# ==============================================================================
# 3. 全国合成レーダー (Radar: 1kmメッシュ・wgrib2経由)
# ==============================================================================
def download_radar_grib(dt_str: str, raw_dir: str = RAW_DIR) -> str:
  """RISHから全国合成レーダーGPV (1kmメッシュ) をダウンロードする。"""
  os.makedirs(raw_dir, exist_ok=True)
  if len(dt_str) == 10:
    dt_str += "00"

  yyyy, mm, dd = dt_str[:4], dt_str[4:6], dt_str[6:8]
  filename = f"Z__C_RJTD_{dt_str}00_RDR_JMAGPV_Ggis1km_Prr05lv_ANAL_grib2.bin"
  local_path = os.path.join(raw_dir, filename)

  if os.path.exists(local_path) and os.path.getsize(local_path) > 0:
    return local_path

  url = f"http://database.rish.kyoto-u.ac.jp/arch/jmadata/data/jma-radar/synthetic/original/{yyyy}/{mm}/{dd}/{filename}"
  print(f"[Radar] 取得中: {url}")

  try:
    urllib.request.urlretrieve(url, local_path)
    print(f"[Radar] ダウンロード完了: {filename}")
    return local_path
  except urllib.error.HTTPError:
    raise FileNotFoundError(
        f"合成レーダーファイルが見つかりません (404 Not Found): {url}"
    )


def get_radar_dataset(
    dt_str: str,
    extent: Optional[Tuple[float, float, float, float]] = None,
    cache_dir: str = SURF_CACHE_DIR,
    raw_dir: str = RAW_DIR,
) -> Optional[xr.Dataset]:
  """全国合成レーダー(1km)を wgrib2 で NetCDF に変換し、指定領域をキャッシュする。"""
  os.makedirs(cache_dir, exist_ok=True)
  if len(dt_str) == 10:
    dt_str += "00"

  ext_str = (
      f"{extent[0]}_{extent[1]}_{extent[2]}_{extent[3]}" if extent else "all"
  )
  cache_file = os.path.join(cache_dir, f"radar_1km_{dt_str}_{ext_str}.nc")

  if os.path.exists(cache_file):
    print(f"[Radar] キャッシュから読み込み: {cache_file}")
    return xr.open_dataset(cache_file)

  grib_path = download_radar_grib(dt_str, raw_dir=raw_dir)
  temp_nc = os.path.join(raw_dir, f"temp_radar_{dt_str}.nc")

  print(f"[Radar] wgrib2でNetCDFに変換中...")
  try:
    subprocess.run(
        ["wgrib2", grib_path, "-netcdf", temp_nc],
        check=True,
        capture_output=True,
        text=True,
    )
  except subprocess.CalledProcessError as e:
    raise RuntimeError(f"wgrib2による変換に失敗しました:\n{e.stderr}")

  ds = xr.open_dataset(temp_nc)
  var_name = list(ds.data_vars.keys())[0]
  ds = ds.rename({var_name: "tp"})
  ds["tp"].attrs["units"] = "mm/h"
  ds["tp"].attrs["long_name"] = "Radar Precipitation Intensity"

  if "lat" in ds.coords:
    ds = ds.rename({"lat": "latitude", "lon": "longitude"})

  if extent:
    lon_min, lon_max, lat_min, lat_max = extent
    if ds.latitude.values[0] > ds.latitude.values[-1]:
      ds = ds.sel(
          latitude=slice(lat_max, lat_min), longitude=slice(lon_min, lon_max)
      )
    else:
      ds = ds.sel(
          latitude=slice(lat_min, lat_max), longitude=slice(lon_min, lon_max)
      )

  try:
    ds.to_netcdf(cache_file)
    print(f"[Radar] NetCDF4キャッシュ保存完了: {cache_file}")
  except Exception as e:
    print(f"[Radar] キャッシュ保存スキップ ({e})")

  if os.path.exists(temp_nc):
    os.remove(temp_nc)

  return ds


# ==============================================================================
# 4. アメダス実況Parquet読み込み
# ==============================================================================
# surface_utils.py 内の load_amedas_parquet（完全UTC統一版）


def load_amedas_parquet(
    base_dir: str, target_utc: datetime, extent: list
) -> pd.DataFrame:
  """指定日時(UTC)のアメダスParquetデータを読み込み、指定領域でフィルタリングして返します。"""
  if not base_dir:
    return pd.DataFrame()

  part_path = (
      Path(base_dir)
      / f"year={target_utc.year}"
      / f"month={target_utc.month:02d}"
  )
  if not part_path.exists():
    print(f"[AMeDAS] ディレクトリが見つかりません: {part_path}")
    return pd.DataFrame()

  df = pd.read_parquet(part_path)

  # タイムゾーンを aware (UTC) に統一
  dt_series = pd.to_datetime(df["datetime"])
  if dt_series.dt.tz is None:
    dt_series = dt_series.dt.tz_localize(timezone.utc)
  else:
    dt_series = dt_series.dt.tz_convert(timezone.utc)

  # aware 同士で照合
  sub = df[dt_series == target_utc].copy()
  if sub.empty:
    available_dts = dt_series[dt_series <= target_utc]
    if not available_dts.empty:
      fallback_dt = available_dts.max()
      sub = df[dt_series == fallback_dt].copy()
      print(
          f"[AMeDAS INFO] {target_utc.strftime('%H:%M UTC')} は未着のため直近の"
          f" {fallback_dt.strftime('%H:%M UTC')} を使用します"
      )
    else:
      print(
          f"[AMeDAS] 該当時刻(UTC)のデータが見つかりません:"
          f" {target_utc.strftime('%Y-%m-%d %H:%M UTC')}"
      )
      return pd.DataFrame()

  sub["prec_val_safe"] = pd.to_numeric(
      sub.get("p1h_val", 0.0), errors="coerce"
  ).fillna(0.0)
  sub["temp_val_safe"] = pd.to_numeric(
      sub.get("temp_val", np.nan), errors="coerce"
  )
  sub["wspd_safe"] = pd.to_numeric(
      sub.get("wind_spd_val", np.nan), errors="coerce"
  )
  sub["wind_u_safe"] = pd.to_numeric(
      sub.get("wind_u", np.nan), errors="coerce"
  )
  sub["wind_v_safe"] = pd.to_numeric(
      sub.get("wind_v", np.nan), errors="coerce"
  )

  mask = (
      (sub["lon"] >= extent[0])
      & (sub["lon"] <= extent[1])
      & (sub["lat"] >= extent[2])
      & (sub["lat"] <= extent[3])
  )
  return sub[mask].copy().reset_index(drop=True)

# ==============================================================================
# 5. 地上メソ解析図（MSM海面更正気圧＋アメダス）
# ==============================================================================
def plot_surface_meso_analysis(
    dt_str: str,
    amedas_dir: str,
    extent: Optional[Tuple[float, float, float, float]] = None,
    title_suffix: str = "",
    save_path: Optional[str] = None,
):
  """地上メソ解析図（MSM海面更正気圧＋アメダス実況）作成関数"""
  ext = extent if extent is not None else REGION_EXTENTS["関東"]

  ds_msm = get_msm_surface(dt_str, extent=ext)
  if ds_msm is None:
    print(f"MSMデータの取得に失敗しました: {dt_str}")
    return

  lon_key = "longitude" if "longitude" in ds_msm.coords else "lon"
  lat_key = "latitude" if "latitude" in ds_msm.coords else "lat"
  p_key = next(
      (k for k in ["prmsl", "PRMSL", "msl", "MSL"] if k in ds_msm), None
  )

  lons_msm = ds_msm[lon_key].values
  lats_msm = ds_msm[lat_key].values

  mslp = np.squeeze(ds_msm[p_key].values)
  while mslp.ndim > 2:
    mslp = mslp[0]

  if lons_msm.ndim == 1:
    lons_2d, lats_2d = np.meshgrid(lons_msm, lats_msm)
  else:
    lons_2d, lats_2d = np.squeeze(lons_msm), np.squeeze(lats_msm)

# 1. UTC のタイムゾーン情報 (tzinfo) を明示的に付与
  dt_utc = datetime.strptime(dt_str[:10], "%Y%m%d%H").replace(tzinfo=timezone.utc)

# 2. astimezone で安全に JST へ変換
  target_jst = dt_utc.astimezone(timezone(timedelta(hours=9)))
# ※ zoneinfo をお使いの場合は dt_utc.astimezone(ZoneInfo("Asia/Tokyo"))

# 3. aware 同士で安全に照合される
  df_amedas = load_amedas_parquet(amedas_dir, dt_utc, ext)

  fig, ax = plt.subplots(
      figsize=(9, 8), dpi=150, subplot_kw={"projection": ccrs.PlateCarree()}
  )
  ax.set_extent(ext, crs=ccrs.PlateCarree())

  ax.add_feature(
      cfeature.COASTLINE.with_scale("10m"),
      edgecolor="#333333",
      linewidth=0.9,
      zorder=3,
  )

  p_min, p_max = int(np.floor(np.nanmin(mslp))), int(np.ceil(np.nanmax(mslp)))
  levels_minor = np.arange(p_min - (p_min % 2), p_max + 2, 2)
  levels_major = np.arange(p_min - (p_min % 4), p_max + 4, 4)

  ax.contour(
      lons_2d,
      lats_2d,
      mslp,
      levels=levels_minor,
      colors="#1b3c73",
      linewidths=0.7,
      linestyles="--",
      alpha=0.8,
      zorder=2,
  )
  cs = ax.contour(
      lons_2d,
      lats_2d,
      mslp,
      levels=levels_major,
      colors="#1b3c73",
      linewidths=1.3,
      zorder=2,
  )

  labels = ax.clabel(
      cs, fmt="%d", inline=True, fontsize=9, colors="#1b3c73", zorder=3
  )
  for label in labels:
    label.set_path_effects(
        [PathEffects.withStroke(linewidth=2.5, foreground="white")]
    )

  if not df_amedas.empty:
    valid_amedas = df_amedas[
        df_amedas["temp_val_safe"].notna()
        & (df_amedas["temp_val_safe"] > -50.0)
    ].copy()

    sc = ax.scatter(
        valid_amedas["lon"],
        valid_amedas["lat"],
        c=valid_amedas["temp_val_safe"],
        cmap="coolwarm",
        vmin=18,
        vmax=30,
        s=28,
        edgecolors="#333333",
        linewidth=0.6,
        zorder=4,
    )

    for _, row in valid_amedas.iterrows():
      txt = ax.text(
          row["lon"] + 0.02,
          row["lat"] + 0.02,
          f"{row['temp_val_safe']:.1f}",
          transform=ccrs.PlateCarree(),
          fontsize=7.5,
          fontweight="bold",
          color="#111111",
          zorder=5,
      )
      txt.set_path_effects(
          [PathEffects.withStroke(linewidth=2.0, foreground="white")]
      )

    if (
        "wind_u_safe" in valid_amedas.columns
        and "wind_v_safe" in valid_amedas.columns
    ):
      u_kt = valid_amedas["wind_u_safe"].values * 1.94384
      v_kt = valid_amedas["wind_v_safe"].values * 1.94384
      ax.barbs(
          valid_amedas["lon"].values,
          valid_amedas["lat"].values,
          u_kt,
          v_kt,
          length=4.5,
          linewidth=0.7,
          color="#1a1a1a",
          zorder=6,
      )

    cbar = plt.colorbar(
        sc, ax=ax, orientation="horizontal", pad=0.06, shrink=0.6, aspect=25
    )
    cbar.set_label("AMeDAS Surface Temperature [℃]", fontsize=8.5)

  gl = ax.gridlines(
      draw_labels=True,
      linestyle=":",
      color="#666666",
      alpha=0.6,
      linewidth=0.8,
      zorder=3,
  )
  gl.top_labels = False
  gl.right_labels = False
  gl.xlabel_style = {"size": 8.5, "rotation": 0}
  gl.ylabel_style = {"size": 8.5}

  plt.title(
        f"Quick-Look Meso Analysis (Radar + AMeDAS) [{reg_label}]\n"
        f"Valid: {dt_utc.strftime('%Y-%m-%d %H:%M UTC')} / {target_jst.strftime('%m/%d %H:%M JST')}",
        fontsize=10.5,
        pad=12,
  )

  if save_path:
    plt.savefig(save_path, bbox_inches="tight")
  plt.show()


# ==============================================================================
# 6. 速報メソ実況図（全国合成レーダー＋アメダス）
# ==============================================================================
from matplotlib.colors import BoundaryNorm, ListedColormap
import math
from PIL import Image
import requests
import io
import numpy as np

def deg2num(lat_deg: float, lon_deg: float, zoom: int) -> Tuple[int, int]:
    """経緯度から気象庁標準 Slippy Map タイル座標(X, Y)を正確に算出"""
    lat_rad = math.radians(lat_deg)
    n = 2.0**zoom
    xtile = int((lon_deg + 180.0) / 360.0 * n)
    ytile = int((1.0 - math.log(math.tan(lat_rad) + (1 / math.cos(lat_rad))) / math.pi) / 2.0 * n)
    return xtile, ytile

def num2deg(xtile: int, ytile: int, zoom: int) -> Tuple[float, float]:
    """タイル座標の北西角(NW)の経緯度(lat, lon)を算出"""
    n = 2.0**zoom
    lon_deg = xtile / n * 360.0 - 180.0
    lat_rad = math.atan(math.sinh(math.pi * (1.0 - 2.0 * ytile / n)))
    return math.degrees(lat_rad), lon_deg

def fetch_nowcast_raster_extent(
    extent: tuple,
    basetime: str,
    validtime: str,
    zoom: int = 6,
):
  """指定領域の降水ナウキャストタイルを取得し、モザイク画像・EPSG:3857範囲・有効ピクセル数を返す"""
  lon_min, lon_max, lat_min, lat_max = extent
  x_nw, y_nw = deg2num(lat_max, lon_min, zoom)
  x_se, y_se = deg2num(lat_min, lon_max, zoom)

  x_start, x_end = min(x_nw, x_se), max(x_nw, x_se)
  y_start, y_end = min(y_nw, y_se), max(y_nw, y_se)

  w_tiles = x_end - x_start + 1
  h_tiles = y_end - y_start + 1
  mosaic = Image.new("RGBA", (w_tiles * 256, h_tiles * 256), (0, 0, 0, 0))

  total_echo_pixels = 0
  headers = {"User-Agent": "WeatherQuickLook/1.0"}

  for x in range(x_start, x_end + 1):
    for y in range(y_start, y_end + 1):
      url = f"https://www.jma.go.jp/bosai/jmatile/data/nowc/{basetime}/none/{validtime}/surf/hrpns/{zoom}/{x}/{y}.png"
      try:
        r = requests.get(url, timeout=5, headers=headers)
        if r.status_code == 200 and len(r.content) > 100:
          tile = Image.open(io.BytesIO(r.content)).convert("RGBA")
          arr = np.array(tile)
          total_echo_pixels += int(np.sum(arr[:, :, 3] > 0))
          mosaic.paste(tile, ((x - x_start) * 256, (y - y_start) * 256))
      except Exception:
        continue

  # EPSG:3857 (球面メルカトル) のメートル座標系で外枠を計算
  origin_shift = 2 * math.pi * 6378137.0 / 2.0  # 約 20037508.34 m
  tile_size = 2 * origin_shift / (2.0**zoom)

  x_min_m = -origin_shift + x_start * tile_size
  x_max_m = -origin_shift + (x_end + 1) * tile_size
  y_max_m = origin_shift - y_start * tile_size
  y_min_m = origin_shift - (y_end + 1) * tile_size

  img_extent_3857 = [x_min_m, x_max_m, y_min_m, y_max_m]

  return mosaic, img_extent_3857, total_echo_pixels


def get_latest_common_time(amedas_dir: str) -> Tuple[datetime, str, str]:
  """【同期用】ナウキャスト実況とAMeDASの最新時刻を比較し、両者が揃う共通時刻を返す"""
  url = "https://www.jma.go.jp/bosai/jmatile/data/nowc/targetTimes_N1.json"
  try:
    res = requests.get(url, timeout=5, headers={"User-Agent": "Mozilla/5.0"})
    data = res.json()
    actual_entries = [
        e for e in data if e.get("basetime") == e.get("validtime")
    ]
  except Exception as e:
    print(f"[Nowcast 警告] targetTimes取得失敗: {e}")
    actual_entries = []

  # 1. ナウキャスト最新時刻の決定
  if len(actual_entries) >= 3:
    safe_entry = actual_entries[1]
  elif len(actual_entries) > 0:
    safe_entry = actual_entries[0]
  else:
    safe_utc = datetime.now(timezone.utc) - timedelta(minutes=10)
    safe_entry = {
        "basetime": safe_utc.strftime("%Y%m%d%H%M00"),
        "validtime": safe_utc.strftime("%Y%m%d%H%M00"),
    }

  # ★ aware (UTC) としてパース
  nowcast_utc = datetime.strptime(
      str(safe_entry["basetime"])[:12], "%Y%m%d%H%M"
  ).replace(tzinfo=timezone.utc)

  # 2. ローカルAMeDASの最新時刻取得
  amedas_latest_utc = nowcast_utc
  part_path = (
      Path(amedas_dir)
      / f"year={nowcast_utc.year}"
      / f"month={nowcast_utc.month:02d}"
  )

  if part_path.exists():
    try:
      df = pd.read_parquet(part_path, columns=["datetime"])
      dt_series = pd.to_datetime(df["datetime"])
      if dt_series.dt.tz is None:
        dt_series = dt_series.dt.tz_localize(timezone.utc)
      else:
        dt_series = dt_series.dt.tz_convert(timezone.utc)
      amedas_latest_utc = dt_series.max().to_pydatetime()
    except Exception:
      pass

  # 3. 両方揃っている時刻を採用し、tzinfo=timezone.utc を付与
  common_utc = min(nowcast_utc, amedas_latest_utc)
  common_utc = common_utc.replace(
      minute=(common_utc.minute // 5) * 5,
      second=0,
      microsecond=0,
      tzinfo=timezone.utc,  # ← 必須
  )
  common_str = common_utc.strftime("%Y%m%d%H%M00")

  return common_utc, common_str, common_str


def plot_nowcast_surface_quicklook(
    dt_str: Optional[str] = None,
    amedas_dir: str = "./data/amedas_parquet/",
    extent: Union[str, Tuple[float, float, float, float], list] = "関東",
    save_path: Optional[str] = None,
):
  if isinstance(extent, str):
    if extent not in REGION_EXTENTS:
      raise ValueError(f"未定義の地域名: {extent}")
    target_extent = tuple(REGION_EXTENTS[extent])
    reg_label = REGION_EN.get(extent, extent)
  else:
    target_extent = tuple(extent)
    reg_label = "Custom"

  # 1. 時刻の正規化（インデントを関数直下の 2 スペースに修正）
  if dt_str is None:
    dt_utc, basetime, validtime = get_latest_common_time(amedas_dir)
  else:
    raw_digits = "".join(filter(str.isdigit, str(dt_str)))
    if len(raw_digits) < 10:
      raise ValueError(f"無効な日時指定です: {dt_str}")
    dt_utc = datetime.strptime(
        raw_digits[:12].ljust(12, "0"), "%Y%m%d%H%M"
    ).replace(tzinfo=timezone.utc)
    basetime = dt_utc.strftime("%Y%m%d%H%M00")
    validtime = basetime

  # 2. 現在時刻との比較（aware同士）および JST 変換
  now_utc = datetime.now(timezone.utc)
  is_past = (now_utc - dt_utc).total_seconds() > 3.5 * 3600
  target_jst = dt_utc.astimezone(ZoneInfo("Asia/Tokyo"))

  fig, ax = plt.subplots(
      figsize=(10, 9), dpi=150, subplot_kw={"projection": ccrs.PlateCarree()}
  )
  ax.set_extent(target_extent, crs=ccrs.PlateCarree())
  ax.add_feature(
      cfeature.COASTLINE.with_scale("10m"),
      edgecolor="#222222",
      linewidth=1.0,
      zorder=4,
  )

  # 3. ナウキャスト面データ取得
  mosaic_img, img_extent, px_count = fetch_nowcast_raster_extent(
      target_extent, basetime=basetime, validtime=validtime, zoom=8
  )

  # 4. 降水エコーの描画分岐（重複していた now_utc の再取得を削除）
  if not is_past:
    ax.imshow(
        np.array(mosaic_img),
        extent=img_extent,
        transform=ccrs.epsg(3857),
        origin="upper",
        zorder=3,
        interpolation="nearest",
    )
  else:
    print(
        "[INFO] 指定時刻がナウキャスト保持期限切れのため、RISH全国合成レーダーを使用します。"
    )
    minute_30 = (dt_utc.minute // 30) * 30
    rish_utc = dt_utc.replace(minute=minute_30, second=0, microsecond=0)
    rish_dt_str = rish_utc.strftime("%Y%m%d%H%M")
    print(f"[Radar] RISH要求時刻を30分間隔に補正: {rish_dt_str}")

    try:
      ds_radar = get_radar_dataset(rish_dt_str, extent=target_extent)
      if ds_radar is not None:
        cmap, norm, levels = get_jma_radar_cmap()
        ax.contourf(
            ds_radar["longitude"].values,
            ds_radar["latitude"].values,
            np.squeeze(ds_radar["tp"].values),
            levels=levels,
            cmap=cmap,
            norm=norm,
            extend="max",
            zorder=3,
        )
    except Exception as e:
      print(f"[Radar 警告] レーダーデータ取得失敗: {e}")

  # 5. カラーバー
  jma_colors = [
      "#a0d2ff",
      "#218cff",
      "#0041ff",
      "#faf500",
      "#ff9900",
      "#ff2800",
      "#b40068",
  ]
  levels = [1, 5, 10, 20, 30, 50, 80]
  cmap_legend = ListedColormap(jma_colors)
  norm_legend = BoundaryNorm(levels, cmap_legend.N, extend="max")
  sm = plt.cm.ScalarMappable(cmap=cmap_legend, norm=norm_legend)
  sm.set_array([])

  cbar = plt.colorbar(
      sm,
      ax=ax,
      orientation="horizontal",
      pad=0.06,
      shrink=0.65,
      aspect=28,
      ticks=levels,
      extend="max",
  )
  cbar.set_label("Precipitation Intensity [mm/h]", fontsize=9)
  cbar.ax.set_xticklabels(["1", "5", "10", "20", "30", "50", "80"], fontsize=8)

  # 6. アメダス実況プロット
  df_amedas = load_amedas_parquet(amedas_dir, dt_utc, list(target_extent))
  if not df_amedas.empty:
    valid_amedas = df_amedas[
        df_amedas["temp_val_safe"].notna()
        & (df_amedas["temp_val_safe"] > -50.0)
    ]
    ax.scatter(
        valid_amedas["lon"],
        valid_amedas["lat"],
        transform=ccrs.PlateCarree(),
        s=14,
        color="#333333",
        edgecolors="white",
        linewidth=0.5,
        zorder=5,
    )

    for _, row in valid_amedas.iterrows():
      txt = ax.text(
          row["lon"] + 0.02,
          row["lat"] - 0.03,
          f"{row['temp_val_safe']:.1f}",
          transform=ccrs.PlateCarree(),
          fontsize=7.5,
          fontweight="bold",
          color="#111111",
          zorder=6,
      )
      txt.set_path_effects(
          [PathEffects.withStroke(linewidth=2.0, foreground="white")]
      )

    if (
        "wind_u_safe" in df_amedas.columns
        and "wind_v_safe" in df_amedas.columns
    ):
      u_kt, v_kt = (
          df_amedas["wind_u_safe"].values * 1.94384,
          df_amedas["wind_v_safe"].values * 1.94384,
      )
      ax.barbs(
          df_amedas["lon"].values,
          df_amedas["lat"].values,
          u_kt,
          v_kt,
          transform=ccrs.PlateCarree(),
          length=4.8,
          linewidth=0.8,
          color="#111111",
          zorder=6,
      )

  # 7. グリッドとタイトルの描画
  gl = ax.gridlines(
      draw_labels=True,
      linestyle=":",
      color="#777777",
      alpha=0.6,
      linewidth=0.8,
      zorder=4,
  )
  gl.top_labels, gl.right_labels = False, False
  gl.xlabel_style, gl.ylabel_style = (
      {"size": 8.5, "rotation": 0},
      {"size": 8.5},
  )

  plt.title(
      f"Quick-Look Meso Analysis (Radar/Nowcast + AMeDAS) [{reg_label}]\n"
      f"Valid: {dt_utc.strftime('%Y-%m-%d %H:%M UTC')} /"
      f" {target_jst.strftime('%m/%d %H:%M JST')}",
      fontsize=10.5,
      pad=12,
  )

  if save_path:
    plt.savefig(save_path, bbox_inches="tight")
  plt.show()