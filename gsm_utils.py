from __future__ import annotations

import datetime as dt
import os
from pathlib import Path
import urllib.error
import urllib.request

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
import metpy.calc as mpcalc
from metpy.units import units
import numpy as np
from scipy.ndimage import (
    gaussian_filter,
    maximum_filter,
    minimum_filter,
    uniform_filter,
)
import xarray as xr

from .surface_utils import CACHE_DIR, RAW_DIR

RISH_GSM_BASE_URL = "http://database.rish.kyoto-u.ac.jp/arch/jmadata/data/gpv/original"
GSM_CACHE_DIR = os.path.join(CACHE_DIR, "gsm")
os.makedirs(GSM_CACHE_DIR, exist_ok=True)

# ==============================================================================
# 1. GSM データ取得・抽出・キャッシュ管理
# ==============================================================================
def build_gsm_filename_and_url(init_time_utc: dt.datetime, fcst_hours: int) -> tuple[str, str]:
    """初期時刻(UTC)と予報時間(6時間刻み)からRISHのGSMファイル名とURLを生成する。"""
    if fcst_hours % 6 != 0 or fcst_hours < 0:
        raise ValueError("予報時間(fcst_hours)は0以上の6時間刻みで指定してください。")

    days, hours = divmod(fcst_hours, 24)
    fcst_str = f"{days:02d}{hours:02d}"

    y_str = init_time_utc.strftime("%Y")
    m_str = init_time_utc.strftime("%m")
    d_str = init_time_utc.strftime("%d")
    hr_str = init_time_utc.strftime("%H0000")

    fname = f"Z__C_RJTD_{y_str}{m_str}{d_str}{hr_str}_GSM_GPV_Rgl_FD{fcst_str}_grib2.bin"
    url = f"{RISH_GSM_BASE_URL}/{y_str}/{m_str}/{d_str}/{fname}"
    return fname, url


def fetch_gsm_grib2(
    init_time_utc: dt.datetime,
    fcst_hours: int,
    save_dir: Path | str = RAW_DIR,
    overwrite: bool = False,
) -> Path:
    """京大RISHアーカイブからGSM GRIB2ファイルをダウンロードする（既存時はスキップ）。"""
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    fname, url = build_gsm_filename_and_url(init_time_utc, fcst_hours)
    dest_path = save_path / fname

    if dest_path.exists() and dest_path.stat().st_size > 0 and not overwrite:
        return dest_path

    try:
        urllib.request.urlretrieve(url, dest_path)
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"GSMデータのダウンロードに失敗しました ({e.code}): {url}") from e

    return dest_path


def get_gsm_fields(
    init_time_utc: dt.datetime,
    fcst_hours: int,
    lon_bounds: tuple[float, float] = (85.0, 205.0),
    lat_bounds: tuple[float, float] = (-60.0, 70.0),
    sigma: float = 1.0,
    cache_dir: Path | str = GSM_CACHE_DIR,
    raw_dir: Path | str = RAW_DIR,
) -> dict[str, np.ndarray | dt.datetime]:
    """GSMデータから海面更正気圧(hPa)と500/850hPa気温(℃)を抽出し、NetCDFキャッシュを利用して返す。"""
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)

    ini_str = init_time_utc.strftime("%Y%m%d%H")
    ext_str = f"{lon_bounds[0]}_{lon_bounds[1]}_{lat_bounds[0]}_{lat_bounds[1]}"
    nc_file = cache_path / f"gsm_{ini_str}_ft{fcst_hours:03d}_{ext_str}.nc"

    if nc_file.exists():
        with xr.open_dataset(nc_file) as ds_cached:
            prmsl_hpa = ds_cached["prmsl"].values
            t500_c = ds_cached["t500"].values
            t850_c = ds_cached["t850"].values
            lon = ds_cached["longitude"].values
            lat = ds_cached["latitude"].values
    else:
        grib_path = fetch_gsm_grib2(init_time_utc, fcst_hours, save_dir=raw_dir)
        lon_min, lon_max = lon_bounds
        lat_min, lat_max = lat_bounds

        with xr.open_dataset(
            grib_path,
            engine="cfgrib",
            backend_kwargs={
                "filter_by_keys": {"typeOfLevel": "meanSea", "shortName": "prmsl"},
                "indexpath": "",
            },
        ) as ds_sfc, xr.open_dataset(
            grib_path,
            engine="cfgrib",
            backend_kwargs={
                "filter_by_keys": {
                    "typeOfLevel": "isobaricInhPa",
                    "numberOfPoints": 259920,
                    "shortName": "t",  # 気温のみに限定（湿度とのレベル数衝突を回避）
                },
                "indexpath": "",
            },
        ) as ds_upr:

            lat_desc = bool(ds_upr.latitude[0] > ds_upr.latitude[-1])
            lat_slice = slice(lat_max, lat_min) if lat_desc else slice(lat_min, lat_max)
            lon_slice = slice(lon_min, lon_max)

            prmsl_sub = ds_sfc["prmsl"].sel(latitude=lat_slice, longitude=lon_slice)
            t500_sub = ds_upr["t"].sel(isobaricInhPa=500, latitude=lat_slice, longitude=lon_slice)
            t850_sub = ds_upr["t"].sel(isobaricInhPa=850, latitude=lat_slice, longitude=lon_slice)

            prmsl_hpa = (prmsl_sub.values * units.Pa).to(units.hPa).magnitude
            t500_c = (t500_sub.values * units.K).to(units.degC).magnitude
            t850_c = (t850_sub.values * units.K).to(units.degC).magnitude
            lon = t500_sub.longitude.values
            lat = t500_sub.latitude.values

        # 軽量NetCDF4としてキャッシュ保存（メモリ節約）
        ds_out = xr.Dataset(
            {
                "prmsl": (["latitude", "longitude"], prmsl_hpa, {"units": "hPa"}),
                "t500": (["latitude", "longitude"], t500_c, {"units": "degC"}),
                "t850": (["latitude", "longitude"], t850_c, {"units": "degC"}),
            },
            coords={"latitude": lat, "longitude": lon},
        )
        ds_out.to_netcdf(nc_file)
        # gsm_utils.py の get_gsm_fields() 内、ds_out.to_netcdf(nc_file) の直後に追記
        if not Path("/Users").exists():  # Linux(Streamlit Cloud)環境では生GRIB2を削除して容量節約
            grib_path.unlink(missing_ok=True)

    if sigma > 0:
        prmsl_hpa = gaussian_filter(prmsl_hpa, sigma=sigma)
        t500_c = gaussian_filter(t500_c, sigma=sigma)
        t850_c = gaussian_filter(t850_c, sigma=sigma)

    valid_dt = init_time_utc + dt.timedelta(hours=fcst_hours)

    return {
        "lon": lon,
        "lat": lat,
        "prmsl": prmsl_hpa,
        "t500": t500_c,
        "t850": t850_c,
        "init_time": init_time_utc,
        "valid_time": valid_dt,
    }


# ==============================================================================
# 2. 作図ユーティリティ・天気図描画関数
# ==============================================================================
def draw_mslp_contours(
    ax: plt.Axes,
    lon: np.ndarray,
    lat: np.ndarray,
    prmsl: np.ndarray,
    crs_data: ccrs.CRS,
) -> None:
    """地上気圧（海面更正気圧）の等圧線を描画する（4hPa間隔・20hPa太線）。"""
    clevs_all = np.arange(940, 1064, 4)
    clevs_major = np.arange(940, 1064, 20)
    clevs_minor = np.setdiff1d(clevs_all, clevs_major)

    ax.contour(
        lon, lat, prmsl, levels=clevs_minor,
        colors="dimgray", linewidths=0.6, linestyles="solid", transform=crs_data, zorder=3,
    )
    cs_major = ax.contour(
        lon, lat, prmsl, levels=clevs_major,
        colors="black", linewidths=1.2, linestyles="solid", transform=crs_data, zorder=4,
    )
    clbls = ax.clabel(cs_major, fmt="%d", fontsize=8, inline=True)
    plt.setp(clbls, path_effects=[path_effects.withStroke(linewidth=2, foreground="white")])


def draw_temp_panel(
    fig: plt.Figure,
    ax: plt.Axes,
    cax: plt.Axes,
    lon: np.ndarray,
    lat: np.ndarray,
    temp_c: np.ndarray,
    prmsl: np.ndarray,
    panel_title: str,
    level_hpa: int,
    snow_levels: list[float],
    clevs_temp: np.ndarray,
    cflevs_temp: np.ndarray,
    crs_data: ccrs.CRS,
    extent: tuple[float, float, float, float] = (117.0, 156.0, 18.0, 61.0),
) -> None:
    """指定等圧面の気温カラーシェード・等温線・雪目安線・地上気圧を重ねて描画する。"""
    ax.set_title(panel_title, loc="left", fontsize=10.5, fontweight="bold", pad=6)
    ax.add_feature(cfeature.COASTLINE.with_scale("50m"), linewidth=0.6, edgecolor="darkgreen", zorder=5)
    ax.add_feature(cfeature.BORDERS.with_scale("50m"), linewidth=0.5, linestyle=":", edgecolor="darkgreen", zorder=5)
    ax.set_extent(extent, crs=crs_data)

    # 寒気(青)〜暖気(赤)のカラーマップ (RdYlBu_r)
    norm = mpl.colors.BoundaryNorm(cflevs_temp, 256)
    cf = ax.contourf(
        lon, lat, temp_c, levels=cflevs_temp, norm=norm,
        cmap="RdYlBu_r", transform=crs_data, extend="both", alpha=0.55, zorder=1,
    )
    # 塗りは3℃刻み、目盛りラベルは6℃刻みで表示
    cbar = fig.colorbar(cf, cax=cax, ticks=cflevs_temp[::2], orientation="vertical")
    cbar.set_label(f"Temperature at {level_hpa}hPa ($^\\circ$C)", fontsize=9.5)
    cbar.ax.tick_params(labelsize=8.5)
    cbar.minorticks_off()

    cs_temp = ax.contour(
        lon, lat, temp_c, levels=clevs_temp,
        colors="navy", linewidths=0.5, linestyles="dashed", transform=crs_data, zorder=2,
    )
    clbls_t = ax.clabel(cs_temp, fmt="%d", fontsize=7)
    plt.setp(clbls_t, path_effects=[path_effects.withStroke(linewidth=1.5, foreground="white")])

    ax.contour(
        lon, lat, temp_c, levels=snow_levels,
        colors=["purple", "blue"], linewidths=2.8, linestyles="solid", alpha=0.85, transform=crs_data, zorder=3,
    )

    draw_mslp_contours(ax, lon, lat, prmsl, crs_data)

    ax.gridlines(
        xlocs=np.arange(100, 180, 10),
        ylocs=np.arange(10, 90, 10),
        linestyle="dashed", color="green", linewidth=0.4, alpha=0.5,
    )


def plot_gsm_snow_chart(
    dt_str: str,
    fcst_hours: int = 0,
    season: str = "winter",
    extent: tuple[float, float, float, float] = (117.0, 156.0, 18.0, 61.0),
    save_path: str | Path | None = None,
    return_fig: bool = False,
) -> plt.Figure | None:
    """GSMデータから地上気圧＋500/850hPa気温（雪目安）の天気図を生成する。

    Args:
        dt_str: 初期時刻(UTC)の文字列（例: "2025020400"）
        fcst_hours: 予報時間（6時間刻み、デフォルト 0）
        season: "winter" または "summer"
        extent: 描画範囲 (lon_min, lon_max, lat_min, lat_max)
        save_path: 画像保存パス（指定時のみ保存）
        return_fig: True の場合、plt.close() せず Figure オブジェクトを返す（Streamlit st.pyplot 用）
    """
    init_dt = dt.datetime.strptime(dt_str[:10], "%Y%m%d%H")
    fields = get_gsm_fields(init_dt, fcst_hours=fcst_hours, sigma=1.0)

    dtm_ini: dt.datetime = fields["init_time"]
    dtm_fcst: dt.datetime = fields["valid_time"]

    fig = plt.figure(figsize=(210 / 25.4, 294 / 25.4), dpi=150)
    fig.text(0.10, 0.97, "JMA GSM (0.5 deg) - MSLP & Upper Temperature", fontsize=13, fontweight="bold")
    fig.text(0.10, 0.95, f"Initial : {dtm_ini.strftime('%Y/%m/%d %H')} UTC", fontsize=11)
    fig.text(0.10, 0.93, f"Forecast: {dtm_fcst.strftime('%Y/%m/%d %H')} UTC (FT+{fcst_hours}h)", fontsize=11)

    crs_map = ccrs.NorthPolarStereo(central_longitude=140.0, true_scale_latitude=60.0)
    crs_data = ccrs.PlateCarree()

# 冬季レンジを実態に合わせて最適化（塗り: 3℃刻み / 目盛り: 6℃刻み）
    cflevs_500 = np.arange(-48, 3, 3) if season == "winter" else np.arange(-24, 6, 3)
    cflevs_850 = np.arange(-30, 15, 3) if season == "winter" else np.arange(-12, 30, 3)

    # パネルタイトルが重ならないよう上下の余白をわずかに調整
    ax01 = fig.add_axes([0.08, 0.51, 0.76, 0.38], projection=crs_map)
    cb01 = fig.add_axes([0.86, 0.51, 0.015, 0.38])
    ax02 = fig.add_axes([0.08, 0.05, 0.76, 0.38], projection=crs_map)
    cb02 = fig.add_axes([0.86, 0.05, 0.015, 0.38])

    draw_temp_panel(
        fig, ax01, cb01,
        fields["lon"], fields["lat"], fields["t500"], fields["prmsl"],
        panel_title="(a) 500 hPa Temp. (Purple: -36°C, Blue: -30°C) & MSLP (4 hPa)",
        level_hpa=500,
        snow_levels=[-36.0, -30.0],
        clevs_temp=np.arange(-54, 15, 3),
        cflevs_temp=cflevs_500,
        crs_data=crs_data,
        extent=extent,
    )

    draw_temp_panel(
        fig, ax02, cb02,
        fields["lon"], fields["lat"], fields["t850"], fields["prmsl"],
        panel_title="(b) 850 hPa Temp. (Purple: -12°C, Blue: -6°C) & MSLP (4 hPa)",
        level_hpa=850,
        snow_levels=[-12.0, -6.0],
        clevs_temp=np.arange(-45, 30, 3),
        cflevs_temp=cflevs_850,
        crs_data=crs_data,
        extent=extent,
    )

    if save_path:
        out = Path(save_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, transparent=False, bbox_inches="tight")

    if return_fig:
        return fig

    plt.show()
    plt.close(fig)
    return None


# ==============================================================================
# 3. 地上要素（MSLP・10m風・前6時間降水量）のデータ抽出・作図拡張
# ==============================================================================
def get_jma_prec_cmap() -> LinearSegmentedColormap:
    """気象庁風の降水量カラーマップを生成する。"""
    jmacolors = np.array(
        [
            [242, 242, 255, 255],
            [160, 210, 255, 255],
            [33, 140, 255, 255],
            [0, 65, 255, 255],
            [250, 245, 0, 255],
            [255, 153, 0, 255],
            [255, 40, 0, 255],
            [180, 0, 104, 255],
        ],
        dtype=np.float64,
    ) / 255.0
    return LinearSegmentedColormap.from_list("jmacmap", colors=jmacolors)


def _extract_single_surface_grib(
    init_time_utc: dt.datetime,
    fcst_hours: int,
    lon_bounds: tuple[float, float],
    lat_bounds: tuple[float, float],
    raw_dir: Path | str,
) -> dict[str, np.ndarray]:
    """単一時刻のGSM GRIB2から地上要素（prmsl, u10, v10, tp）を抽出する内部関数。"""
    grib_path = fetch_gsm_grib2(init_time_utc, fcst_hours, save_dir=raw_dir)
    lon_min, lon_max = lon_bounds
    lat_min, lat_max = lat_bounds

    # 1. 海面更正気圧 (meanSea)
    with xr.open_dataset(
        grib_path,
        engine="cfgrib",
        backend_kwargs={
            "filter_by_keys": {"typeOfLevel": "meanSea", "shortName": "prmsl"},
            "indexpath": "",
        },
    ) as ds_msl:
        lat_desc = bool(ds_msl.latitude[0] > ds_msl.latitude[-1])
        lat_slice = slice(lat_max, lat_min) if lat_desc else slice(lat_min, lat_max)
        lon_slice = slice(lon_min, lon_max)

        prmsl_sub = ds_msl["prmsl"].sel(latitude=lat_slice, longitude=lon_slice)
        prmsl_hpa = (prmsl_sub.values * units.Pa).to(units.hPa).magnitude
        lon = prmsl_sub.longitude.values
        lat = prmsl_sub.latitude.values

    # 2. 地上10m風 (heightAboveGround = 10)
    with xr.open_dataset(
        grib_path,
        engine="cfgrib",
        backend_kwargs={
            "filter_by_keys": {"typeOfLevel": "heightAboveGround", "level": 10},
            "indexpath": "",
        },
    ) as ds_wnd:
        u10_sub = ds_wnd["u10"].sel(latitude=lat_slice, longitude=lon_slice)
        v10_sub = ds_wnd["v10"].sel(latitude=lat_slice, longitude=lon_slice)
        u10_kt = (u10_sub.values * units("m/s")).to(units.knots).magnitude
        v10_kt = (v10_sub.values * units("m/s")).to(units.knots).magnitude

    # 3. 積算降水量 (surface / accum)
    with xr.open_dataset(
        grib_path,
        engine="cfgrib",
        backend_kwargs={
            "filter_by_keys": {
                "typeOfLevel": "surface",
                "stepType": "accum",
            },
            "indexpath": "",
        },
    ) as ds_tp:
        # 変数名が環境(eccodesのバージョン)により 'tp' または 'unknown' 等になる場合に対応
        tp_var = "tp" if "tp" in ds_tp.data_vars else list(ds_tp.data_vars.keys())[0]
        tp_sub = ds_tp[tp_var].sel(latitude=lat_slice, longitude=lon_slice)
        tp_mm = np.squeeze(tp_sub.values)

    if not Path("/Users").exists():
        grib_path.unlink(missing_ok=True)

    return {
        "lon": lon,
        "lat": lat,
        "prmsl": prmsl_hpa,
        "u10": u10_kt,
        "v10": v10_kt,
        "tp": tp_mm,
    }


def get_gsm_surface_fields(
    init_time_utc: dt.datetime,
    fcst_hours: int,
    lon_bounds: tuple[float, float] = (85.0, 205.0),
    lat_bounds: tuple[float, float] = (-60.0, 70.0),
    sigma: float = 1.0,
    cache_dir: Path | str = GSM_CACHE_DIR,
    raw_dir: Path | str = RAW_DIR,
) -> dict[str, np.ndarray | dt.datetime]:
    """GSMデータから海面更正気圧(hPa)・10m風(kt)・前6時間積算降水量(mm)を抽出し、NetCDFキャッシュを利用して返す。"""
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)

    ini_str = init_time_utc.strftime("%Y%m%d%H")
    ext_str = f"{lon_bounds[0]}_{lon_bounds[1]}_{lat_bounds[0]}_{lat_bounds[1]}"
    nc_file = cache_path / f"gsm_sfc_{ini_str}_ft{fcst_hours:03d}_{ext_str}.nc"

    if nc_file.exists():
        with xr.open_dataset(nc_file) as ds_cached:
            prmsl_hpa = ds_cached["prmsl"].values
            u10_kt = ds_cached["u10"].values
            v10_kt = ds_cached["v10"].values
            prcp_6h = ds_cached["prcp_6h"].values
            lon = ds_cached["longitude"].values
            lat = ds_cached["latitude"].values
    else:
        curr = _extract_single_surface_grib(init_time_utc, fcst_hours, lon_bounds, lat_bounds, raw_dir)
        lon = curr["lon"]
        lat = curr["lat"]
        prmsl_hpa = curr["prmsl"]
        u10_kt = curr["u10"]
        v10_kt = curr["v10"]

        if fcst_hours >= 6:
            prev = _extract_single_surface_grib(init_time_utc, fcst_hours - 6, lon_bounds, lat_bounds, raw_dir)
            prcp_6h = np.maximum(curr["tp"] - prev["tp"], 0.0)
        else:
            prcp_6h = np.zeros_like(prmsl_hpa)

        ds_out = xr.Dataset(
            {
                "prmsl": (["latitude", "longitude"], prmsl_hpa, {"units": "hPa"}),
                "u10": (["latitude", "longitude"], u10_kt, {"units": "knots"}),
                "v10": (["latitude", "longitude"], v10_kt, {"units": "knots"}),
                "prcp_6h": (["latitude", "longitude"], prcp_6h, {"units": "mm"}),
            },
            coords={"latitude": lat, "longitude": lon},
        )
        ds_out.to_netcdf(nc_file)

    if sigma > 0:
        prmsl_hpa = gaussian_filter(prmsl_hpa, sigma=sigma)
        u10_kt = gaussian_filter(u10_kt, sigma=sigma)
        v10_kt = gaussian_filter(v10_kt, sigma=sigma)
        prcp_6h = gaussian_filter(prcp_6h, sigma=sigma)

    wspd_kt = mpcalc.wind_speed(u10_kt * units.knots, v10_kt * units.knots).magnitude
    valid_dt = init_time_utc + dt.timedelta(hours=fcst_hours)

    return {
        "lon": lon,
        "lat": lat,
        "prmsl": prmsl_hpa,
        "u10": u10_kt,
        "v10": v10_kt,
        "wspd10": wspd_kt,
        "prcp_6h": prcp_6h,
        "init_time": init_time_utc,
        "valid_time": valid_dt,
    }


def draw_mslp_extrema(
    ax: plt.Axes,
    lon: np.ndarray,
    lat: np.ndarray,
    prmsl: np.ndarray,
    crs_data: ccrs.CRS,
    extent: tuple[float, float, float, float] | None = None,
    neighborhood_size: int = 16,
    prominence_hpa: float = 0.8,
) -> None:
    """描画図郭（Axes）内の海面更正気圧の相対的な極大(H)・極小(L)を検出しプロットする。"""
    lon_2d, lat_2d = np.meshgrid(lon, lat) if lon.ndim == 1 else (lon, lat)

    # 周囲の空間平均との相対差のみで判定（恣意的な絶対値閾値は使用しない）
    local_mean = uniform_filter(prmsl, size=neighborhood_size, mode="nearest")

    is_maxima = (prmsl == maximum_filter(prmsl, size=neighborhood_size)) & (
        (prmsl - local_mean) >= prominence_hpa
    )
    is_minima = (prmsl == minimum_filter(prmsl, size=neighborhood_size)) & (
        (local_mean - prmsl) >= prominence_hpa
    )

    stroke = [path_effects.Stroke(linewidth=2, foreground="white"), path_effects.Normal()]

    def _is_inside_axes(x_lon: float, y_lat: float, margin_axes: float = 0.03) -> bool:
        """経緯度をAxes正規化座標(0〜1)に変換し、図郭の内側にあるか判定する。"""
        proj_pt = ax.projection.transform_point(x_lon, y_lat, crs_data)
        axes_pt = ax.transAxes.inverted().transform(ax.transData.transform(proj_pt))
        return (margin_axes <= axes_pt[0] <= 1.0 - margin_axes) and (
            margin_axes <= axes_pt[1] <= 1.0 - margin_axes
        )

    for i, j in np.argwhere(is_maxima):
        if _is_inside_axes(lon_2d[i, j], lat_2d[i, j]):
            ax.text(
                lon_2d[i, j], lat_2d[i, j], "H",
                transform=crs_data, clip_on=True, color="blue",
                va="center", ha="center", weight="bold", fontsize=12,
                path_effects=stroke, zorder=6,
            )

    for i, j in np.argwhere(is_minima):
        if _is_inside_axes(lon_2d[i, j], lat_2d[i, j]):
            ax.text(
                lon_2d[i, j], lat_2d[i, j], "L",
                transform=crs_data, clip_on=True, color="red",
                va="center", ha="center", weight="bold", fontsize=12,
                path_effects=stroke, zorder=6,
            )

def plot_gsm_surface_chart(
    dt_str: str,
    fcst_hours: int = 6,
    extent: tuple[float, float, float, float] = (117.0, 156.0, 18.0, 61.0),
    save_path: str | Path | None = None,
    return_fig: bool = False,
) -> plt.Figure | None:
    """GSMデータから地上10m風（流線・矢羽）と地上気圧(H/L)＋前6時間降水量の天気図を生成する。

    Args:
        dt_str: 初期時刻(UTC)の文字列（例: "2024060100"）
        fcst_hours: 予報時間（6時間刻み、デフォルト 6）
        extent: 描画範囲 (lon_min, lon_max, lat_min, lat_max)
        save_path: 画像保存パス（指定時のみ保存、拡張子でPDF/PNG自動判別）
        return_fig: True の場合、plt.close() せず Figure オブジェクトを返す（Streamlit st.pyplot 用）
    """
    init_dt = dt.datetime.strptime(dt_str[:10], "%Y%m%d%H")
    fields = get_gsm_surface_fields(init_dt, fcst_hours=fcst_hours, sigma=1.0)

    dtm_ini: dt.datetime = fields["init_time"]
    dtm_fcst: dt.datetime = fields["valid_time"]
    lon = fields["lon"]
    lat = fields["lat"]

    fig = plt.figure(figsize=(210 / 25.4, 294 / 25.4), dpi=150)
    fig.text(0.10, 0.97, "JMA GSM (0.5 deg) - Surface Wind, MSLP & 6h Precipitation", fontsize=13, fontweight="bold")
    fig.text(0.10, 0.95, f"Initial : {dtm_ini.strftime('%Y/%m/%d %H')} UTC", fontsize=11)
    fig.text(0.10, 0.93, f"Forecast: {dtm_fcst.strftime('%Y/%m/%d %H')} UTC (FT+{fcst_hours}h)", fontsize=11)

    crs_map = ccrs.NorthPolarStereo(central_longitude=140.0, true_scale_latitude=60.0)
    crs_data = ccrs.PlateCarree()

    ax01 = fig.add_axes([0.08, 0.51, 0.76, 0.38], projection=crs_map)
    cb01 = fig.add_axes([0.86, 0.51, 0.015, 0.38])
    ax02 = fig.add_axes([0.08, 0.05, 0.76, 0.38], projection=crs_map)
    cb02 = fig.add_axes([0.86, 0.05, 0.015, 0.38])

    # --------------------------------------------------------------------------
    # パネル(a): 地上10m風（矢羽根 + 流線）
    # --------------------------------------------------------------------------
    ax01.set_title("(a) 10m Surface Wind (Barbs & Streamlines, kt)", loc="left", fontsize=10.5, fontweight="bold", pad=6)
    ax01.add_feature(cfeature.NaturalEarthFeature("physical", "land", "50m", edgecolor="none", facecolor="green"), alpha=0.1, zorder=1)
    ax01.add_feature(cfeature.COASTLINE.with_scale("50m"), linewidth=0.6, edgecolor="darkgreen", zorder=5)
    ax01.add_feature(cfeature.BORDERS.with_scale("50m"), linewidth=0.5, linestyle=":", edgecolor="darkgreen", zorder=5)
    ax01.set_extent(extent, crs=crs_data)

    clevs_wndsp = np.array([0, 10, 20, 30, 40, 50])
    norm_wnd = mpl.colors.Normalize(vmin=clevs_wndsp.min(), vmax=clevs_wndsp.max())

    # xarrayのlatは降順(北→南)の場合があるため、streamplot用に昇順(南→北)へ整形
    if lat[0] > lat[-1]:
        lat_asc = lat[::-1]
        u10_asc = fields["u10"][::-1, :]
        v10_asc = fields["v10"][::-1, :]
        wspd_asc = fields["wspd10"][::-1, :]
    else:
        lat_asc, u10_asc, v10_asc, wspd_asc = lat, fields["u10"], fields["v10"], fields["wspd10"]

    ax01.barbs(
        lon[::5], lat_asc[::5], u10_asc[::5, ::5], v10_asc[::5, ::5],
        length=4.2, transform=crs_data, sizes=dict(emptybarb=0.04), linewidth=0.45, pivot="middle", zorder=3,
    )
    st_plot = ax01.streamplot(
        lon, lat_asc, u10_asc, v10_asc,
        density=[1.0, 1.0], color=wspd_asc, transform=crs_data,
        linewidth=1.0, cmap="jet", norm=norm_wnd, zorder=4,
    )
    cbar1 = fig.colorbar(st_plot.lines, cax=cb01, ticks=clevs_wndsp, orientation="vertical")
    cbar1.set_label("10m Wind Speed (kt)", fontsize=9.5)
    cbar1.ax.tick_params(labelsize=8.5)
    cbar1.minorticks_off()

    ax01.gridlines(
        xlocs=np.arange(100, 180, 10), ylocs=np.arange(10, 90, 10),
        linestyle="dashed", color="green", linewidth=0.4, alpha=0.5,
    )

    # --------------------------------------------------------------------------
    # パネル(b): 海面更正気圧(H/L) ＋ 前6時間積算降水量
    # --------------------------------------------------------------------------
    ax02.set_title("(b) MSLP (4 hPa, H/L) & 6h Accumulated Precipitation (mm)", loc="left", fontsize=10.5, fontweight="bold", pad=6)
    ax02.add_feature(cfeature.NaturalEarthFeature("physical", "land", "50m", edgecolor="none", facecolor="green"), alpha=0.1, zorder=1)
    ax02.add_feature(cfeature.COASTLINE.with_scale("50m"), linewidth=0.6, edgecolor="darkgreen", zorder=5)
    ax02.add_feature(cfeature.BORDERS.with_scale("50m"), linewidth=0.5, linestyle=":", edgecolor="darkgreen", zorder=5)
    ax02.set_extent(extent, crs=crs_data)

    jmacmap = get_jma_prec_cmap()
    clevs_prcp = np.array([0.1, 1, 5, 10, 20, 30, 50, 80])
    norm_prcp = mpl.colors.BoundaryNorm([0.1, 1, 5, 10, 20, 30, 50, 80, 100], 256)

    cf = ax02.contourf(
        lon, lat, fields["prcp_6h"], levels=clevs_prcp, norm=norm_prcp,
        cmap=jmacmap, transform=crs_data, extend="max", zorder=2,
    )
    ax02.contour(
        lon, lat, fields["prcp_6h"], levels=[0.1],
        colors="black", transform=crs_data, linewidths=0.25, linestyles="dashed", zorder=2,
    )
    cbar2 = fig.colorbar(cf, cax=cb02, ticks=clevs_prcp, orientation="vertical")
    cbar2.set_label("6h Precipitation (mm)", fontsize=9.5)
    cbar2.ax.tick_params(labelsize=8.5)
    cbar2.minorticks_off()

    # 既存の共通関数で等圧線を描画し、極値(H/L)を追加
    clevs_mslp = np.arange(940, 1064, 4)
    cs = ax02.contour(
        lon, lat, fields["prmsl"], levels=clevs_mslp,
        colors="black", transform=crs_data,
        linewidths=[1.25, 0.75, 0.75, 0.75, 0.75], zorder=3,
    )
    clbls = ax02.clabel(cs, clevs_mslp[::2], fmt="%d", fontsize=8, inline=True)
    plt.setp(clbls, path_effects=[path_effects.withStroke(linewidth=2, foreground="white")])

    draw_mslp_extrema(
        ax02,
        lon,
        lat,
        fields["prmsl"],
        crs_data,
        extent=extent,
        neighborhood_size=16,
        prominence_hpa=0.8,
    )
    
    ax02.gridlines(
        xlocs=np.arange(100, 180, 10), ylocs=np.arange(10, 90, 10),
        linestyle="dashed", color="green", linewidth=0.4, alpha=0.5,
    )

    if save_path:
        out = Path(save_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, transparent=False, bbox_inches="tight")

    if return_fig:
        return fig

    plt.show()
    plt.close(fig)
    return None
