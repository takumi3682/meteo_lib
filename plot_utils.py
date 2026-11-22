"""
plot_utils.py
=============
気象データ（AMeDAS、GPV、レーダー、地上/高層解析）の可視化・グラフ作成および
地図ベースレイヤーのセットアップを行う共通プロットユーティリティモジュール。

提供関数一覧
------------
1. plot_amedas_map(target_time, element="p1h", region="japan", show_wind_vectors=False, show_values=False, save_fig=False, base_dir=None)
   - AMeDAS観測値（降水量・気温・風）の空間分布マップを描画。
2. plot_amedas_precipitation_and_wind(df, xlim=None, ylim_p10m=None, ylim_cumsum=None, ...)
   - AMeDAS時系列データから上段に風向風速矢羽根、下段に10分降水量(棒)＋累積降水量(折れ線)を描画。
3. plot_amedas_temp_and_pressure(df, xlim=None, ylim_temp=None, ylim_press=None, ...)
   - AMeDAS時系列データから上段に風向風速矢羽根、下段に気温(折れ線)＋海面更正気圧(折れ線)を描画。
4. setup_map_axis(ax, extent) / setup_weather_map(ax, region="全国")
   - Cartopy GeoAxes に対する範囲設定、海岸線、国境、グリッド線の共通セットアップ。
5. get_element_style(elem) / get_jma_precipitation_cmap()
   - 気象庁指針準拠のカラーマップ、境界値（BoundaryNorm）、カラーバー用目盛りを返却。
6. to_2d_array(da) / smooth_field(data, sigma=1.2)
   - 2次元格子データの安全抽出およびガウシアン平滑化ヘルパー。
7. setup_time_axis(ax, ...) / plot_wind_barbs_subplot(ax_wind, df_wind) / plot_target_point(ax, point, name)
   - 時系列軸フォーマットや特定地点プロットなどの描画補助部品。

使用例
------
from datetime import datetime
from plot_utils import (
    plot_amedas_map,
    plot_amedas_precipitation_and_wind,
    get_jma_precipitation_cmap,
    setup_map_axis,
    REGION_EXTENTS,
)

# 1. AMeDAS空間分布図の描画
plot_amedas_map(
    datetime(2026, 8, 13, 12),
    element="p1h",
    region="関東",
    show_values=True,
    show_wind_vectors=True,
)

# 2. AMeDAS時系列グラフの描画
plot_amedas_precipitation_and_wind(
    df_station,
    xlim=(datetime(2026, 8, 13), datetime(2026, 8, 15)),
    ylim_p10m=(0, 30),
)
"""

from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import warnings

import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.io import DownloadWarning
import japanize_matplotlib
import matplotlib.colors as mcolors
from matplotlib.colors import BoundaryNorm, ListedColormap
import matplotlib.dates as mdates
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter
from metpy.calc import wind_components
from metpy.plots import StationPlot, current_weather, sky_cover
from metpy.units import units
from scipy.interpolate import RBFInterpolator
from scipy.ndimage import gaussian_filter

warnings.filterwarnings("ignore", category=DownloadWarning)

# ==============================================================================
# 地域プリセット定義（日本語・英語統一）
# ==============================================================================
REGION_EXTENTS = {
    "japan": [128.0, 146.0, 30.0, 46.0],
    "全国": [128.0, 146.0, 30.0, 46.0],
    "tohoku": [138.5, 142.5, 36.8, 41.6],
    "東北": [138.5, 142.5, 36.8, 41.6],
    "東北・関東": [138.0, 142.5, 34.8, 41.6],
    "miyagi": [140.0, 141.8, 37.7, 39.1],
    "宮城": [140.0, 141.8, 37.7, 39.1],
    "kanto": [138.3, 141.0, 34.8, 37.2],
    "関東": [138.3, 141.0, 34.8, 37.2],
    "hokuriku": [135.5, 139.9, 35.3, 38.6],
    "北陸": [135.5, 139.9, 35.3, 38.6],
    "kyushu": [129.0, 132.5, 30.8, 34.2],
    "九州": [129.0, 132.5, 30.8, 34.2],
}


# ==============================================================================
# 1. 地図ベースレイヤー・補助描画関数
# ==============================================================================
def setup_map_axis(ax: plt.Axes, extent: Union[List[float], Tuple[float, ...]]) -> Any:
    """地図の範囲設定、海岸線、国境線、グリッド線を一括設定"""
    ax.set_extent(extent, crs=ccrs.PlateCarree())
    ax.coastlines(resolution="10m", color="black", linewidth=0.8)
    ax.add_feature(
        cfeature.BORDERS, linestyle="-", edgecolor="black", linewidth=0.6
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
    return gl


def setup_weather_map(ax: plt.Axes, region: str = "全国") -> Any:
    """領域名から地図属性（海岸線・グリッド）を一括セットアップ"""
    extent = REGION_EXTENTS.get(region, REGION_EXTENTS["全国"])
    return setup_map_axis(ax, extent)


def plot_target_point(ax: plt.Axes, point: Tuple[float, float], name: str = ""):
    """着目地点のプロットと名称ラベル描画"""
    lat, lon = point
    ax.plot(
        lon,
        lat,
        marker="o",
        color="black",
        markersize=6,
        transform=ccrs.PlateCarree(),
    )
    if name:
        ax.text(
            lon + 0.15,
            lat + 0.05,
            name,
            fontsize=10,
            fontweight="bold",
            transform=ccrs.PlateCarree(),
        )


def add_contour_labels(cs: Any, fmt: str = "%d", fontsize: int = 9, inline_spacing: int = 5):
    """等値線ラベル（clabel）の書式・配置を一括調整"""
    plt.clabel(
        cs,
        fmt=fmt,
        fontsize=fontsize,
        inline=True,
        inline_spacing=inline_spacing,
    )


def to_2d_array(da: Any) -> np.ndarray:
    """DataArray/ndarray から安全に (lat, lon) の2次元配列を抽出"""
    arr = np.asarray(da.values if hasattr(da, "values") else da).squeeze()
    if arr.ndim == 3:
        arr = arr[0]
    return arr


def smooth_field(data: np.ndarray, sigma: float = 1.2) -> np.ndarray:
    """2次元格子データのガウシアン平滑化"""
    return gaussian_filter(data, sigma=sigma)


# ==============================================================================
# 2. 気象庁カラーマップ定義
# ==============================================================================
def get_jma_precipitation_cmap() -> Tuple[ListedColormap, BoundaryNorm, List[float]]:
    """気象庁標準の降水強度カラーマップとNormを返す"""
    levels = [0.1, 1, 5, 10, 20, 30, 50, 80]
    colors = [
        "#A0D2FF",
        "#0041FF",
        "#FAF500",
        "#FF9900",
        "#FF2800",
        "#B40068",
        "#A000FF",
        "#4B0082",
    ]
    cmap = mcolors.ListedColormap(colors)
    norm = mcolors.BoundaryNorm(levels, len(colors), extend="max")
    return cmap, norm, levels


def get_element_style(elem: str) -> Tuple[ListedColormap, BoundaryNorm, List[float], str, List[float]]:
    """降水量・気温・風速の気象庁準拠カラーマップ・境界・目盛りを返す"""
    if elem.startswith("p"):
        bounds = [0.0, 1.0, 5.0, 10.0, 20.0, 30.0, 50.0, 80.0, 150.0]
        rgbs = [
            (242 / 255, 242 / 255, 255 / 255),
            (160 / 255, 210 / 255, 255 / 255),
            (33 / 255, 140 / 255, 255 / 255),
            (0 / 255, 65 / 255, 255 / 255),
            (250 / 255, 245 / 255, 0 / 255),
            (255 / 255, 153 / 255, 0 / 255),
            (255 / 255, 40 / 255, 0 / 255),
            (180 / 255, 0 / 255, 104 / 255),
        ]
        label = "Precipitation (mm)"
        ticks = [0, 1, 5, 10, 20, 30, 50, 80]
    elif elem == "temp":
        bounds = [
            -30.0, -5.0, 0.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 50.0,
        ]
        rgbs = [
            (0 / 255, 32 / 255, 128 / 255),
            (0 / 255, 65 / 255, 255 / 255),
            (0 / 255, 150 / 255, 255 / 255),
            (185 / 255, 235 / 255, 255 / 255),
            (255 / 255, 255 / 255, 240 / 255),
            (255 / 255, 255 / 255, 150 / 255),
            (250 / 255, 245 / 255, 0 / 255),
            (255 / 255, 153 / 255, 0 / 255),
            (255 / 255, 40 / 255, 0 / 255),
            (180 / 255, 0 / 255, 104 / 255),
        ]
        label = "Temperature (℃)"
        ticks = [-5, 0, 5, 10, 15, 20, 25, 30, 35]
    elif elem in ["wind", "wind_spd"]:
        bounds = [0.0, 5.0, 10.0, 15.0, 20.0, 25.0, 60.0]
        rgbs = [
            (160 / 255, 210 / 255, 255 / 255),
            (0 / 255, 65 / 255, 255 / 255),
            (250 / 255, 245 / 255, 0 / 255),
            (255 / 255, 153 / 255, 0 / 255),
            (255 / 255, 40 / 255, 0 / 255),
            (180 / 255, 0 / 255, 104 / 255),
        ]
        label = "Wind Speed (m/s)"
        ticks = [0, 5, 10, 15, 20, 25]
    else:
        raise ValueError(f"未対応の気象要素です: {elem}")

    cmap = ListedColormap(rgbs)
    norm = BoundaryNorm(bounds, cmap.N)
    return cmap, norm, bounds, label, ticks


# ==============================================================================
# 3. AMeDAS 時系列プロット関数
# ==============================================================================
def setup_time_axis(
    ax: plt.Axes,
    xlim: Optional[Tuple[datetime, datetime]] = None,
    x_major_hours: int = 6,
    x_minor_hours: int = 1,
):
    """時系列グラフの横軸フォーマット・グリッド設定を一括適用"""
    if xlim:
        ax.set_xlim(xlim)
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=x_major_hours))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d\n%H:%M"))
    ax.xaxis.set_minor_locator(mdates.HourLocator(interval=x_minor_hours))
    ax.grid(True, which="major", linestyle="--", alpha=0.5)
    ax.grid(True, which="minor", linestyle=":", alpha=0.25)


def plot_wind_barbs_subplot(ax_wind: plt.Axes, df_wind: pd.DataFrame):
    """上段の風向風速矢羽根サブプロットを描画"""
    ax_wind.barbs(
        df_wind["datetime"],
        np.zeros(len(df_wind)),
        df_wind["wind_u"],
        df_wind["wind_v"],
        length=5.5,
        sizes=dict(emptybarb=0.05, spacing=0.2, height=0.5),
        barbcolor="navy",
        flagcolor="navy",
    )
    ax_wind.set_ylim(-1, 1)
    ax_wind.set_yticks([])
    ax_wind.set_ylabel(
        "風向\n風速", fontsize=9, rotation=0, labelpad=30, va="center"
    )
    ax_wind.grid(True, axis="x", linestyle="--", alpha=0.4)


def plot_amedas_precipitation_and_wind(
    df: pd.DataFrame,
    xlim: Optional[Tuple[datetime, datetime]] = None,
    ylim_p10m: Optional[Tuple[float, float]] = None,
    ylim_cumsum: Optional[Tuple[float, float]] = None,
    x_major_hours: int = 6,
    x_minor_hours: int = 1,
    p10m_step: float = 5.0,
    cumsum_step: float = 20.0,
):
    """AMeDAS 降水量（10分・累積）および風向風速の時系列グラフを描画"""
    df_plot = df.copy().sort_values("datetime")
    df_plot["cumulative_precip"] = df_plot["p10m_val"].fillna(0).cumsum()
    df_wind = df_plot.iloc[::3].copy()

    fig, (ax_wind, ax_precip) = plt.subplots(
        nrows=2,
        ncols=1,
        figsize=(14, 7),
        sharex=True,
        gridspec_kw={"height_ratios": [1, 5]},
    )

    plot_wind_barbs_subplot(ax_wind, df_wind)

    bar_width = timedelta(minutes=8)
    ax_precip.bar(
        df_plot["datetime"],
        df_plot["p10m_val"],
        width=bar_width,
        color="#3182bd",
        alpha=0.8,
        label="10分降水量 (mm)",
        align="center",
    )
    ax_precip.set_ylabel("10分降水量 (mm)", color="#3182bd", fontsize=11)
    ax_precip.tick_params(axis="y", labelcolor="#3182bd")

    if ylim_p10m:
        ax_precip.set_ylim(ylim_p10m)
    else:
        ax_precip.set_ylim(bottom=0)
    if p10m_step:
        ax_precip.yaxis.set_major_locator(ticker.MultipleLocator(p10m_step))

    ax_cumsum = ax_precip.twinx()
    ax_cumsum.plot(
        df_plot["datetime"],
        df_plot["cumulative_precip"],
        color="#de2d26",
        linewidth=2.0,
        label="累積降水量 (mm)",
    )
    ax_cumsum.set_ylabel("累積降水量 (mm)", color="#de2d26", fontsize=11)
    ax_cumsum.tick_params(axis="y", labelcolor="#de2d26")

    if ylim_cumsum:
        ax_cumsum.set_ylim(ylim_cumsum)
    else:
        ax_cumsum.set_ylim(bottom=0)
    if cumsum_step:
        ax_cumsum.yaxis.set_major_locator(ticker.MultipleLocator(cumsum_step))

    setup_time_axis(
        ax_precip,
        xlim=xlim
        if xlim
        else (df_plot["datetime"].min(), df_plot["datetime"].max()),
        x_major_hours=x_major_hours,
        x_minor_hours=x_minor_hours,
    )

    stn_name = (
        df_plot["stn_name"].iloc[0] if "stn_name" in df_plot.columns else ""
    )
    stn_id = df_plot["stn_id"].iloc[0] if "stn_id" in df_plot.columns else ""
    fig.suptitle(
        f"気象観測時系列グラフ: {stn_name} ({stn_id})", fontsize=13, y=0.98
    )

    lines1, labels1 = ax_precip.get_legend_handles_labels()
    lines2, labels2 = ax_cumsum.get_legend_handles_labels()
    ax_precip.legend(lines1 + lines2, labels1 + labels2, loc="upper left")

    plt.tight_layout()
    plt.show()


def plot_amedas_temp_and_pressure(
    df: pd.DataFrame,
    xlim: Optional[Tuple[datetime, datetime]] = None,
    ylim_temp: Optional[Tuple[float, float]] = None,
    ylim_press: Optional[Tuple[float, float]] = None,
    x_major_hours: int = 6,
    x_minor_hours: int = 1,
    temp_step: float = 5.0,
    press_step: float = 10.0,
):
    """AMeDAS 気温・気圧および風向風速の時系列グラフを描画"""
    df_plot = df.copy().sort_values("datetime")
    df_wind = df_plot.iloc[::3].copy()

    fig, (ax_wind, ax_temp) = plt.subplots(
        nrows=2,
        ncols=1,
        figsize=(14, 7),
        sharex=True,
        gridspec_kw={"height_ratios": [1, 4]},
    )

    plot_wind_barbs_subplot(ax_wind, df_wind)

    ax_temp.plot(
        df_plot["datetime"],
        df_plot["temp_val"],
        color="#d95f02",
        linewidth=1.5,
        label="気温 (°C)",
    )
    ax_temp.set_ylabel("気温 (°C)", color="#d95f02", fontsize=11)
    ax_temp.tick_params(axis="y", labelcolor="#d95f02")

    if ylim_temp:
        ax_temp.set_ylim(ylim_temp)
    if temp_step:
        ax_temp.yaxis.set_major_locator(ticker.MultipleLocator(temp_step))

    ax_press = ax_temp.twinx()
    ax_press.plot(
        df_plot["datetime"],
        df_plot["normal_press_val"],
        color="#3182bd",
        linewidth=1.5,
        label="海面更正気圧(hPa)",
    )
    ax_press.set_ylabel("海面更正気圧（hPa）", color="#1f78b4", fontsize=11)
    ax_press.tick_params(axis="y", labelcolor="#1f78b4")

    if ylim_press:
        ax_press.set_ylim(ylim_press)
    if press_step:
        ax_press.yaxis.set_major_locator(ticker.MultipleLocator(press_step))

    setup_time_axis(
        ax_temp,
        xlim=xlim
        if xlim
        else (df_plot["datetime"].min(), df_plot["datetime"].max()),
        x_major_hours=x_major_hours,
        x_minor_hours=x_minor_hours,
    )

    stn_name = (
        df_plot["stn_name"].iloc[0] if "stn_name" in df_plot.columns else ""
    )
    stn_id = df_plot["stn_id"].iloc[0] if "stn_id" in df_plot.columns else ""
    fig.suptitle(
        f"気象観測時系列グラフ（気温・気圧）: {stn_name} ({stn_id})",
        fontsize=13,
        y=0.98,
    )

    lines1, labels1 = ax_temp.get_legend_handles_labels()
    lines2, labels2 = ax_press.get_legend_handles_labels()
    ax_temp.legend(lines1 + lines2, labels1 + labels2, loc="upper left")

    plt.tight_layout()
    plt.show()


# ==============================================================================
# 4. AMeDAS 空間分布マップ描画関数
# ==============================================================================
def plot_amedas_map(
    target_time: Union[datetime, str],
    element: str = "p1h",
    region: Union[str, List[float], Tuple[float, ...]] = "japan",
    show_wind_vectors: bool = False,
    show_values: bool = False,
    save_fig: bool = False,
    base_dir: Optional[Path] = None,
):
    """AMeDAS 観測値の空間分布マップを描画"""
    from amedas_utils import get_amedas_snapshot

    extent = (
        REGION_EXTENTS.get(region, REGION_EXTENTS["japan"])
        if isinstance(region, str)
        else region
    )
    region_name = region if isinstance(region, str) else "custom"

    df_valid, dt_obj = (
        get_amedas_snapshot(
            target_time, element=element, extent=extent, base_dir=base_dir
        )
        if base_dir
        else get_amedas_snapshot(target_time, element=element, extent=extent)
    )
    if df_valid is None:
        return

    cmap, norm, bounds, label, ticks = get_element_style(element)
    val_col = f"{element}_val"

    fig = plt.figure(figsize=(10, 8), dpi=150)
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
    ax.set_extent(extent, crs=ccrs.PlateCarree())

    ax.add_feature(cfeature.LAND, facecolor="#F0F0F0")
    ax.add_feature(
        cfeature.OCEAN, facecolor=(135 / 255, 169 / 255, 210 / 255, 0.4)
    )
    ax.add_feature(cfeature.COASTLINE, linewidth=0.7, edgecolor="#555555")
    ax.add_feature(
        cfeature.BORDERS, linestyle=":", linewidth=0.5, edgecolor="#777777"
    )

    gl = ax.gridlines(
        draw_labels=True,
        linewidth=0.4,
        color="gray",
        alpha=0.5,
        linestyle="--",
    )
    gl.top_labels = False
    gl.right_labels = False

    df_sorted = df_valid.sort_values(val_col, ascending=True)
    sizes = (
        np.clip(df_sorted[val_col] * 1.5 + 20, 20, 160)
        if element.startswith("p")
        else 45
    )

    sc = ax.scatter(
        df_sorted["lon"],
        df_sorted["lat"],
        c=df_sorted[val_col],
        s=sizes,
        cmap=cmap,
        norm=norm,
        alpha=0.9,
        edgecolors="#333333",
        linewidths=0.5,
        transform=ccrs.PlateCarree(),
        zorder=4,
    )

    if show_wind_vectors and "wind_u" in df_sorted.columns:
        wind_valid = df_sorted[
            (df_sorted["wind_spd_flag"] == 0) & (df_sorted["wind_spd_val"] > 0.5)
        ]
        ax.quiver(
            wind_valid["lon"].values,
            wind_valid["lat"].values,
            wind_valid["wind_u"].values,
            wind_valid["wind_v"].values,
            color="#222222",
            scale=120,
            width=0.004,
            headwidth=4,
            headlength=5,
            transform=ccrs.PlateCarree(),
            zorder=5,
        )

    if show_values:
        for _, row in df_sorted.iterrows():
            val = row[val_col]
            if element.startswith("p"):
                if val < 0.5:
                    continue
                text_str = f"{val:.0f}" if val >= 10 else f"{val:.1f}"
            else:
                text_str = f"{val:.1f}"

            ax.text(
                row["lon"] + 0.03,
                row["lat"] + 0.03,
                text_str,
                fontsize=8,
                fontweight="bold",
                color="#111111",
                va="bottom",
                ha="left",
                clip_on=True,
                transform=ccrs.PlateCarree(),
                path_effects=[
                    pe.withStroke(linewidth=2.5, foreground="white")
                ],
                zorder=6,
            )

    cbar = plt.colorbar(
        sc, ax=ax, orientation="vertical", pad=0.03, shrink=0.75, ticks=ticks
    )
    cbar.set_label(label, fontsize=9)

    elem_names = {
        "p1h": "1-Hour Precipitation",
        "p3h": "3-Hour Precipitation",
        "p24h": "24-Hour Precipitation",
        "temp": "Temperature",
        "wind_spd": "Wind Speed & Vector",
    }
    jst_time_str = dt_obj.strftime("%Y-%m-%d %H:%M JST")
    ax.set_title(
        f"JMA AMeDAS {elem_names.get(element, element)}\n{jst_time_str}",
        fontsize=12,
        fontweight="bold",
    )

    if save_fig:
        time_str = dt_obj.strftime("%Y%m%d%H%M00")
        out_name = f"{element}_{region_name}_{time_str}.png"
        plt.savefig(out_name, bbox_inches="tight")
        print(f"保存完了: {out_name}")

    plt.show()


def setup_map_axis(extent: list, figsize=(13, 12), dpi=200):
    """地上解析用の地図軸を初期化し、白抜き海マスクとグリッドを設定します。"""
    fig = plt.figure(figsize=figsize, dpi=dpi)
    proj = ccrs.PlateCarree()
    ax = fig.add_subplot(1, 1, 1, projection=proj)
    ax.set_extent(extent, crs=proj)

    # 海マスク (白抜き)
    ax.add_feature(
        cfeature.OCEAN.with_scale("10m"),
        facecolor="#ffffff",
        edgecolor="none",
        zorder=2,
    )
    ax.add_feature(
        cfeature.COASTLINE.with_scale("10m"),
        edgecolor="#333333",
        linewidth=0.9,
        zorder=3,
    )
    ax.add_feature(
        cfeature.BORDERS.with_scale("10m"),
        linestyle=":",
        edgecolor="#666666",
        zorder=3,
    )

    gl = ax.gridlines(
        draw_labels=True,
        dms=True,
        x_inline=False,
        y_inline=False,
        alpha=0.3,
        zorder=3,
    )
    gl.top_labels = False
    gl.right_labels = False

    return fig, ax, proj


def plot_amedas_temp_shade(
    ax,
    proj,
    df_amedas: pd.DataFrame,
    extent: list,
    is_regional: bool,
    sigma=1.2,
    vmin: float = 10.0,
    vmax: float = 40.0,
    cmap: str = "coolwarm",
):
    """アメダス実況気温からRBF空間補間 + ガウシアン平滑化を行いカラーシェードを描画します。"""
    if df_amedas.empty:
        return None
    valid_temp = df_amedas.dropna(subset=["temp_val_safe", "lon", "lat"])
    if len(valid_temp) < 5:
        return None

    gx_t, gy_t = np.meshgrid(
        np.linspace(extent[0] - 1.0, extent[1] + 1.0, 250),
        np.linspace(extent[2] - 1.0, extent[3] + 1.0, 250),
    )
    pts_obs = np.column_stack(
        [valid_temp["lon"].values, valid_temp["lat"].values]
    )
    vals_obs = valid_temp["temp_val_safe"].values

    rbf = RBFInterpolator(
        pts_obs, vals_obs, kernel="thin_plate_spline", smoothing=0.1
    )
    pts_grid = np.column_stack([gx_t.ravel(), gy_t.ravel()])
    temp_grid = rbf(pts_grid).reshape(gx_t.shape)

    if sigma > 0:
        temp_grid = gaussian_filter(temp_grid, sigma=sigma)

    step = 1.0 if is_regional else 2.0
    levels = np.arange(vmin, vmax + step * 0.5, step)

    cs = ax.contourf(
        gx_t,
        gy_t,
        temp_grid,
        levels=levels,
        cmap=cmap,
        alpha=0.75,
        transform=proj,
        extend="both",
        zorder=1,
    )
    cbar = plt.colorbar(
        cs, ax=ax, orientation="horizontal", pad=0.04, shrink=0.55
    )
    cbar.set_label("AMeDAS Observed Surface Temperature (℃)", fontsize=9)
    return cs


def plot_msm_isobars(ax, proj, ds_msm, is_regional: bool, sigma=1.0):
    """MSM客観解析の海面気圧等圧線（主等圧線・補助破線）を平滑化して描画します。"""
    if ds_msm is None or "prmsl" not in ds_msm:
        return

    prmsl = np.squeeze(ds_msm["prmsl"].values)
    if prmsl.ndim == 3:
        prmsl = prmsl[0]
    if sigma > 0:
        prmsl = gaussian_filter(prmsl, sigma=sigma)

    lons_m = ds_msm["longitude"].values
    lats_m = ds_msm["latitude"].values
    gx_p, gy_p = (
        np.meshgrid(lons_m, lats_m) if lons_m.ndim == 1 else (lons_m, lats_m)
    )

    sub_step = 1 if is_regional else 2
    main_step = 2 if is_regional else 4

    ax.contour(
        gx_p,
        gy_p,
        prmsl,
        levels=np.arange(900, 1060, sub_step),
        colors="steelblue",
        linewidths=0.7,
        linestyles="--",
        alpha=0.75,
        transform=proj,
        zorder=4,
    )
    iso = ax.contour(
        gx_p,
        gy_p,
        prmsl,
        levels=np.arange(900, 1060, main_step),
        colors="navy",
        linewidths=1.4,
        linestyles="-",
        transform=proj,
        zorder=4,
    )
    ax.clabel(iso, fmt="%d", fontsize=9, inline=True, use_clabeltext=True)


def plot_amedas_barbs_and_obs(
    ax,
    proj,
    df_amedas: pd.DataFrame,
    is_regional: bool,
    show_wind=True,
    show_obs=True,
):
    """アメダスの風矢羽・降水マーカー・気温値を描画します。"""
    if df_amedas.empty:
        return

    # 風矢羽
    if show_wind:
        valid_wind = df_amedas.dropna(
            subset=["wind_u_safe", "wind_v_safe", "wspd_safe", "lon", "lat"]
        )
        valid_wind = valid_wind[valid_wind["wspd_safe"] >= 0.5]
        if not valid_wind.empty:
            u_kt = valid_wind["wind_u_safe"].values * 1.94384
            v_kt = valid_wind["wind_v_safe"].values * 1.94384

            sp_wind = StationPlot(
                ax,
                valid_wind["lon"].values,
                valid_wind["lat"].values,
                clip_on=True,
                transform=proj,
                fontsize=6,
                zorder=5,
            )
            sp_wind.plot_barb(
                u_kt,
                v_kt,
                length=4.2,
                barbcolor="#34495e",
                flagcolor="#34495e",
                barb_increments={"half": 2.5, "full": 5.0, "flag": 25.0},
                zorder=5,
            )

    # 降水点マーカー & 気温値
    if show_obs:
        rain_pts = df_amedas[df_amedas["prec_val_safe"] >= 1.0]
        if not rain_pts.empty:
            ax.scatter(
                rain_pts["lon"],
                rain_pts["lat"],
                s=28,
                color="dodgerblue",
                edgecolor="darkblue",
                linewidth=0.6,
                alpha=0.9,
                transform=proj,
                zorder=6,
                label="AMeDAS Rain (≥1mm)",
            )

        step = 1 if is_regional else 3
        for _, row in df_amedas.iloc[::step].iterrows():
            t_val = row.get("temp_val_safe")
            if pd.notnull(t_val):
                ax.text(
                    row["lon"],
                    row["lat"] - 0.035,
                    f"{t_val:.1f}" if is_regional else f"{t_val:.0f}",
                    fontsize=5.2 if is_regional else 4.2,
                    color="#111111",
                    alpha=0.9,
                    fontweight="bold",
                    ha="center",
                    va="top",
                    transform=proj,
                    zorder=6,
                    clip_on=True,
                    bbox=dict(
                        boxstyle="round,pad=0.08",
                        fc="white",
                        ec="none",
                        alpha=0.55,
                    ),
                )


def plot_synop_metar_station(
    ax, proj, sub_obs: pd.DataFrame, show_station_name: bool = False
):
    """METAR / SYNOP の国際式実況 StationPlot を描画します。"""
    if sub_obs.empty:
        return

    stationplot = StationPlot(
        ax,
        sub_obs["lon"].values,
        sub_obs["lat"].values,
        clip_on=True,
        transform=proj,
        fontsize=8.0,
        zorder=7,
    )
    stationplot.plot_parameter(
        "NW",
        sub_obs["temp_c"].values,
        color="crimson",
        fontweight="bold",
        formatter=lambda v: f"{v:.1f}",
        bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="none", alpha=0.65),
    )
    stationplot.plot_parameter(
        "SW",
        sub_obs["dewpoint_c"].values,
        color="forestgreen",
        fontweight="bold",
        formatter=lambda v: f"{v:.1f}",
        bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="none", alpha=0.65),
    )

    press_text = [
        f"{int(round((p % 100) * 10)):03d}"
        if (np.isfinite(p) and 850 <= p <= 1100)
        else ""
        for p in sub_obs["mslp"]
    ]
    stationplot.plot_text(
        "NE",
        press_text,
        color="black",
        fontsize=8.0,
        bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="none", alpha=0.65),
    )
    stationplot.plot_symbol(
        "C",
        sub_obs["cloud_cover"].fillna(0).astype(int).values,
        sky_cover,
        color="black",
    )
    stationplot.plot_symbol(
        "W",
        sub_obs["weather_code"].fillna(0).astype(int).values,
        current_weather,
        color="blue",
        fontsize=10.0,
    )

    if show_station_name:
        station_labels = [
            f"{row['station']}" + ("(M)" if row["source"] == "METAR" else "(S)")
            for _, row in sub_obs.iterrows()
        ]
        stationplot.plot_text(
            "SE", station_labels, color="#333333", fontsize=6.5, fontweight="bold"
        )

    w_spd = sub_obs["wind_spd_kt"].fillna(0.0).values * units.knots
    w_dir = sub_obs["wind_dir_deg"].fillna(0.0).values * units.degree
    u, v = wind_components(w_spd, w_dir)
    stationplot.plot_barb(
        u.to("knots").magnitude,
        v.to("knots").magnitude,
        length=5.5,
        barbcolor="navy",
        flagcolor="navy",
        zorder=8,
    )