"""
upper_utils.py
==============
気象庁MSM気圧面客観解析GPV（GRIB2）からの等高度面データ抽出、および
指定気圧面の等高度線（ジオポテンシャル高度）・等温線コンターマップ描画モジュール。

提供関数一覧
------------
1. get_isobaric_dataset(dt, level_hpa=500)
   - 指定日時のMSM気圧面GRIB2をダウンロードし、指定気圧面(hPa)の気温(t)・高度(gh)をマージした xarray.Dataset を返却。
2. plot_isobaric_map(data, level_hpa=None, region="全国", target_point=None, target_name="", gh_step=60, temp_step=3, gh_sigma=0.0, temp_sigma=1.2)
   - 指定気圧面の等高度線（青実線）および等温線（赤破線）を地図上にオーバーレイ描画。平滑化や特定地点プロットに対応。

使用例
------
from datetime import datetime
from upper_utils import get_isobaric_dataset, plot_isobaric_map

dt = datetime(2026, 8, 13, 12)

# 1. 500 hPa 面の取得と描画（寒気・トラフ解析）
ds_500 = get_isobaric_dataset(dt, level_hpa=500)
if ds_500 is not None:
    plot_isobaric_map(
        ds_500,
        level_hpa=500,
        region="全国",
        target_point=(38.26, 140.90),
        target_name="Sendai",
        gh_step=20,
        temp_step=3,
        temp_sigma=1.5
    )

# 2. 850 hPa 面の取得と描画（下層暖気移流解析）
ds_850 = get_isobaric_dataset(dt, level_hpa=850)
if ds_850 is not None:
    plot_isobaric_map(
        ds_850,
        level_hpa=850,
        region="全国",
        gh_step=30,
        temp_step=2,
        temp_sigma=1.5
    )
"""

import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter
import xarray as xr

from sounding_utils import download_rish_msm_p

# ==============================================================================
# 地域プリセット定義
# ==============================================================================
REGION_EXTENTS = {
    "宮城": [140.0, 141.8, 37.7, 39.1],
    "東北": [138.5, 142.5, 36.8, 41.6],
    "関東": [138.3, 141.0, 34.8, 37.2],
    "全国": [128.0, 146.0, 30.0, 46.0],
}


# ==============================================================================
# 1. 指定気圧面データの取得
# ==============================================================================
def get_isobaric_dataset(
    dt: datetime, level_hpa: int = 500
) -> Optional[xr.Dataset]:
    """指定日時のMSM気圧面GRIB2をダウンロードし、指定高度面(hPa)のxarray Datasetを返す"""
    file_path = download_rish_msm_p(dt.year, dt.month, dt.day, dt.hour)
    if not file_path or not os.path.exists(file_path):
        print(f"データファイルが見つかりません: {dt}")
        return None

    try:
        ds_t = xr.open_dataset(
            file_path,
            engine="cfgrib",
            backend_kwargs={
                "filter_by_keys": {
                    "typeOfLevel": "isobaricInhPa",
                    "level": level_hpa,
                    "shortName": "t",
                }
            },
        )
        ds_gh = xr.open_dataset(
            file_path,
            engine="cfgrib",
            backend_kwargs={
                "filter_by_keys": {
                    "typeOfLevel": "isobaricInhPa",
                    "level": level_hpa,
                    "shortName": "gh",
                }
            },
        )
        ds = xr.merge([ds_t, ds_gh], compat="override")
        return ds
    except Exception as e:
        print(f"GRIB2抽出エラー (level={level_hpa}hPa): {e}")
        return None


# ==============================================================================
# 2. 等高度線・等温線マップの描画
# ==============================================================================
def plot_isobaric_map(
    data: xr.Dataset,
    level_hpa: Optional[int] = None,
    region: Union[str, list, tuple] = "全国",
    target_point: Optional[Tuple[float, float]] = None,
    target_name: str = "",
    gh_step: int = 60,
    temp_step: int = 3,
    gh_sigma: float = 0.0,
    temp_sigma: float = 1.2,
):
    """指定気圧面の等高度線・等温線を描画する関数"""
    if data is None:
        print("描画データが存在しません。")
        return

    # 1. 存在キーの集合を安全に作成
    if hasattr(data, "variables"):
        all_keys = list(data.variables.keys())
    elif hasattr(data, "keys"):
        all_keys = list(data.keys())
    else:
        all_keys = dir(data)

    # 2. 座標の抽出 & 2次元化
    if "lon" in all_keys:
        lon = np.asarray(data["lon"].values).squeeze()
    elif "longitude" in all_keys:
        lon = np.asarray(data["longitude"].values).squeeze()
    else:
        raise KeyError(f"経度データが見つかりません。利用可能: {all_keys}")

    if "lat" in all_keys:
        lat = np.asarray(data["lat"].values).squeeze()
    elif "latitude" in all_keys:
        lat = np.asarray(data["latitude"].values).squeeze()
    else:
        raise KeyError(f"緯度データが見つかりません。利用可能: {all_keys}")

    # 3. 高度データの抽出
    raw_gh = None
    for k in ["gh", "z", "hgt", "HGT", "geopotential_height"]:
        if k in all_keys:
            raw_gh = np.asarray(data[k].values).squeeze()
            break
    if raw_gh is None:
        raise KeyError(f"高度データが見つかりません。利用可能: {all_keys}")

    # 4. 気温データの抽出
    raw_temp = None
    for k in ["t", "tmp", "temp", "TMP", "temperature"]:
        if k in all_keys:
            raw_temp = np.asarray(data[k].values).squeeze()
            break
    if raw_temp is None:
        raise KeyError(f"気温データが見つかりません。利用可能: {all_keys}")

    # 多次元が残っている場合の安全対策 (time, lat, lon) -> (lat, lon)
    if raw_gh.ndim == 3:
        raw_gh = raw_gh[0]
    if raw_temp.ndim == 3:
        raw_temp = raw_temp[0]

    # ケルビン (K) -> 摂氏 (℃)
    if np.nanmean(raw_temp) > 100:
        raw_temp = raw_temp - 273.15

    # 5. ガウシアン平滑化の適用
    gh_to_plot = (
        gaussian_filter(raw_gh, sigma=gh_sigma) if gh_sigma > 0 else raw_gh
    )
    temp_to_plot = (
        gaussian_filter(raw_temp, sigma=temp_sigma)
        if temp_sigma > 0
        else raw_temp
    )

    # 6. 地図描画設定
    fig = plt.figure(figsize=(10, 9))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())

    extent = (
        REGION_EXTENTS.get(region, REGION_EXTENTS["全国"])
        if isinstance(region, str)
        else region
    )
    ax.set_extent(extent, crs=ccrs.PlateCarree())
    ax.coastlines(resolution="10m", color="black", linewidth=0.8)
    ax.add_feature(
        cfeature.BORDERS, linestyle=":", edgecolor="gray", linewidth=0.5
    )

    gl = ax.gridlines(
        crs=ccrs.PlateCarree(),
        draw_labels=True,
        linewidth=0.5,
        color="gray",
        alpha=0.5,
        linestyle=":",
    )
    gl.top_labels = False
    gl.right_labels = False

    # 7. 等高度線（青・実線）
    levels_gh = np.arange(0, 16000, gh_step)
    cs_gh = ax.contour(
        lon,
        lat,
        gh_to_plot,
        levels=levels_gh,
        colors="blue",
        linewidths=1.2,
        transform=ccrs.PlateCarree(),
    )
    ax.clabel(cs_gh, fmt="%dm", inline=True, fontsize=9, inline_spacing=2)

    # 8. 等温線（赤・破線）
    levels_temp = np.arange(-50, 40, temp_step)
    cs_temp = ax.contour(
        lon,
        lat,
        temp_to_plot,
        levels=levels_temp,
        colors="red",
        linestyles="dashed",
        linewidths=1.0,
        transform=ccrs.PlateCarree(),
    )
    ax.clabel(cs_temp, fmt="%d℃", inline=True, fontsize=9, inline_spacing=2)

    # 9. 着目地点プロット
    if target_point is not None:
        t_lat, t_lon = target_point
        ax.plot(
            t_lon,
            t_lat,
            marker="o",
            color="black",
            markersize=6,
            transform=ccrs.PlateCarree(),
        )
        if target_name:
            ax.text(
                t_lon + 0.15,
                t_lat + 0.05,
                target_name,
                fontsize=10,
                fontweight="bold",
                transform=ccrs.PlateCarree(),
            )

    # 10. タイトル設定（引数で渡された数値を最優先）
    level_str = (
        str(level_hpa)
        if level_hpa is not None
        else getattr(data, "level", "")
    )

    valid_time = getattr(data, "valid_time", None)
    time_str = (
        valid_time.strftime("%Y-%m-%d %H:%M UTC")
        if hasattr(valid_time, "strftime")
        else "2026-08-13 12:00 UTC"
    )

    ax.set_title(
        f"MSM Isobaric Surface Map: {level_str} hPa ({time_str})", fontsize=12
    )

    plt.tight_layout()
    plt.show()