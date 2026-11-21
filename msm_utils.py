from __future__ import annotations

import datetime as dt
import os
from pathlib import Path
import urllib.error
import urllib.request

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib as mpl
import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
from metpy.units import units
import numpy as np
from scipy.ndimage import gaussian_filter
import xarray as xr

from .surface_utils import CACHE_DIR, RAW_DIR

RISH_MSM_BASE_URL = "http://database.rish.kyoto-u.ac.jp/arch/jmadata/data/gpv/original"
MSM_CACHE_DIR = os.path.join(CACHE_DIR, "msm")
os.makedirs(MSM_CACHE_DIR, exist_ok=True)


# ==============================================================================
# 1. MSM データ取得・抽出・キャッシュ管理
# ==============================================================================
def get_msm_fh_block(fcst_hours: int, level_type: str = "Lsurf") -> str:
    """予報時間(FT)と種別(Lsurf / L-pall)に対応するRISH MSMのFHブロック文字列を返す。"""
    if fcst_hours < 0 or fcst_hours > 78:
        raise ValueError("MSMの予報時間(fcst_hours)は 0〜78 の範囲で指定してください。")

    if level_type == "L-pall":
        if fcst_hours % 3 != 0:
            raise ValueError("MSM高層(L-pall)の予報時間は3時間刻みで指定してください。")
        if fcst_hours <= 15:
            return "00-15"
        if fcst_hours <= 33:
            return "18-33"
        if fcst_hours <= 39:
            return "36-39"
        if fcst_hours <= 51:
            return "42-51"
        return "54-78"
    else:
        if fcst_hours <= 15:
            return "00-15"
        if fcst_hours <= 33:
            return "16-33"
        if fcst_hours <= 39:
            return "34-39"
        if fcst_hours <= 51:
            return "40-51"
        return "52-78"


def fetch_msm_grib2(
    init_time_utc: dt.datetime,
    fcst_hours: int,
    level_type: str = "Lsurf",
    save_dir: Path | str = RAW_DIR,
    overwrite: bool = False,
) -> Path:
    """京大RISHアーカイブからMSM GRIB2ファイル(Lsurf または L-pall)をダウンロードする。"""
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    fh_block = get_msm_fh_block(fcst_hours, level_type=level_type)
    y_str = init_time_utc.strftime("%Y")
    m_str = init_time_utc.strftime("%m")
    d_str = init_time_utc.strftime("%d")
    dt_str = init_time_utc.strftime("%Y%m%d%H")

    fname = f"Z__C_RJTD_{dt_str}0000_MSM_GPV_Rjp_{level_type}_FH{fh_block}_grib2.bin"
    url = f"{RISH_MSM_BASE_URL}/{y_str}/{m_str}/{d_str}/{fname}"
    dest_path = save_path / fname

    if dest_path.exists() and dest_path.stat().st_size > 0 and not overwrite:
        return dest_path

    try:
        urllib.request.urlretrieve(url, dest_path)
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"MSM({level_type})のダウンロードに失敗しました ({e.code}): {url}") from e

    return dest_path


def get_msm_fields(
    init_time_utc: dt.datetime,
    fcst_hours: int,
    lon_bounds: tuple[float, float] = (120.5, 149.5),
    lat_bounds: tuple[float, float] = (22.5, 47.5),
    sigma_mslp: float = 2.0,
    sigma_temp: float = 1.0,
    cache_dir: Path | str = MSM_CACHE_DIR,
    raw_dir: Path | str = RAW_DIR,
) -> dict[str, np.ndarray | dt.datetime]:
    """MSMから海面更正気圧(hPa)と500/850hPa気温(℃)を抽出し、NetCDFキャッシュを利用して返す。"""
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)

    ini_str = init_time_utc.strftime("%Y%m%d%H")
    ext_str = f"{lon_bounds[0]}_{lon_bounds[1]}_{lat_bounds[0]}_{lat_bounds[1]}"
    nc_file = cache_path / f"msm_{ini_str}_ft{fcst_hours:02d}_{ext_str}.nc"

    if nc_file.exists():
        with xr.open_dataset(nc_file) as ds_cached:
            prmsl_hpa = ds_cached["prmsl"].values
            lon_sfc = ds_cached["lon_sfc"].values
            lat_sfc = ds_cached["lat_sfc"].values
            t500_c = ds_cached["t500"].values
            t850_c = ds_cached["t850"].values
            lon_upr = ds_cached["lon_upr"].values
            lat_upr = ds_cached["lat_upr"].values
    else:
        grib_sfc = fetch_msm_grib2(init_time_utc, fcst_hours, level_type="Lsurf", save_dir=raw_dir)
        grib_upr = fetch_msm_grib2(init_time_utc, fcst_hours, level_type="L-pall", save_dir=raw_dir)

        lon_min, lon_max = lon_bounds
        lat_min, lat_max = lat_bounds
        target_step = np.timedelta64(fcst_hours, "h")

        # 1. 地上（海面更正気圧: 0.05度格子）
        with xr.open_dataset(
            grib_sfc,
            engine="cfgrib",
            backend_kwargs={
                "filter_by_keys": {"typeOfLevel": "meanSea", "shortName": "prmsl"},
                "indexpath": "",
            },
        ) as ds_sfc:
            da_p = ds_sfc["prmsl"]
            if "step" in da_p.dims:
                da_p = da_p.sel(step=target_step)
            lat_desc = bool(da_p.latitude[0] > da_p.latitude[-1])
            lat_slice = slice(lat_max, lat_min) if lat_desc else slice(lat_min, lat_max)
            da_p = da_p.sel(latitude=lat_slice, longitude=slice(lon_min, lon_max))

            prmsl_hpa = (da_p.values * units.Pa).to(units.hPa).magnitude
            lon_sfc = da_p.longitude.values
            lat_sfc = da_p.latitude.values

        # 2. 高層（500hPa / 850hPa 気温: 0.1度格子）
        with xr.open_dataset(
            grib_upr,
            engine="cfgrib",
            backend_kwargs={
                "filter_by_keys": {"typeOfLevel": "isobaricInhPa", "shortName": "t"},
                "indexpath": "",
            },
        ) as ds_upr:
            da_t = ds_upr["t"]
            if "step" in da_t.dims:
                da_t = da_t.sel(step=target_step)
            lat_desc_u = bool(da_t.latitude[0] > da_t.latitude[-1])
            lat_slice_u = slice(lat_max, lat_min) if lat_desc_u else slice(lat_min, lat_max)
            da_t = da_t.sel(latitude=lat_slice_u, longitude=slice(lon_min, lon_max))

            t500_c = (da_t.sel(isobaricInhPa=500).values * units.K).to(units.degC).magnitude
            t850_c = (da_t.sel(isobaricInhPa=850).values * units.K).to(units.degC).magnitude
            lon_upr = da_t.longitude.values
            lat_upr = da_t.latitude.values

        # 軽量NetCDF4としてキャッシュ保存（地上と高層で格子間隔が異なるため座標を分けて保持）
        ds_out = xr.Dataset(
            {
                "prmsl": (["lat_sfc", "lon_sfc"], prmsl_hpa, {"units": "hPa"}),
                "t500": (["lat_upr", "lon_upr"], t500_c, {"units": "degC"}),
                "t850": (["lat_upr", "lon_upr"], t850_c, {"units": "degC"}),
            },
            coords={
                "lat_sfc": lat_sfc,
                "lon_sfc": lon_sfc,
                "lat_upr": lat_upr,
                "lon_upr": lon_upr,
            },
        )
        ds_out.to_netcdf(nc_file)

        if not Path("/Users").exists():
            grib_sfc.unlink(missing_ok=True)
            grib_upr.unlink(missing_ok=True)

    # MSMは高解像度のため、地形由来の細かな等圧線の波打ちを抑える平滑化を適用
    if sigma_mslp > 0:
        prmsl_hpa = gaussian_filter(prmsl_hpa, sigma=sigma_mslp)
    if sigma_temp > 0:
        t500_c = gaussian_filter(t500_c, sigma=sigma_temp)
        t850_c = gaussian_filter(t850_c, sigma=sigma_temp)

    valid_dt = init_time_utc + dt.timedelta(hours=fcst_hours)

    return {
        "lon_sfc": lon_sfc,
        "lat_sfc": lat_sfc,
        "lon_upr": lon_upr,
        "lat_upr": lat_upr,
        "prmsl": prmsl_hpa,
        "t500": t500_c,
        "t850": t850_c,
        "init_time": init_time_utc,
        "valid_time": valid_dt,
    }


# ==============================================================================
# 2. 作図関数（メソ天気図）
# ==============================================================================
def draw_msm_mslp_contours(
    ax: plt.Axes,
    lon: np.ndarray,
    lat: np.ndarray,
    prmsl: np.ndarray,
    crs_data: ccrs.CRS,
) -> None:
    """MSM地上気圧の等圧線を描画する（2hPa間隔・10hPa太線）。"""
    clevs_all = np.arange(940, 1062, 2)
    clevs_major = np.arange(940, 1062, 10)
    clevs_minor = np.setdiff1d(clevs_all, clevs_major)

    ax.contour(
        lon, lat, prmsl, levels=clevs_minor,
        colors="dimgray", linewidths=0.55, linestyles="solid", transform=crs_data, zorder=3,
    )
    cs_major = ax.contour(
        lon, lat, prmsl, levels=clevs_major,
        colors="black", linewidths=1.2, linestyles="solid", transform=crs_data, zorder=4,
    )
    clbls = ax.clabel(cs_major, fmt="%d", fontsize=8, inline=True, inline_spacing=2)
    plt.setp(clbls, path_effects=[path_effects.withStroke(linewidth=2, foreground="white")])


def draw_msm_temp_panel(
    fig: plt.Figure,
    ax: plt.Axes,
    cax: plt.Axes,
    fields: dict,
    temp_key: str,
    panel_title: str,
    level_hpa: int,
    snow_levels: list[float],
    clevs_temp: np.ndarray,
    cflevs_temp: np.ndarray,
    crs_data: ccrs.CRS,
    extent: tuple[float, float, float, float] = (122.0, 148.0, 24.0, 46.5),
) -> None:
    """MSM指定等圧面の気温・雪目安線・地上気圧(2hPa間隔)を重ねて描画する。"""
    ax.set_title(panel_title, loc="left", fontsize=10.5, fontweight="bold", pad=6)
    ax.add_feature(cfeature.COASTLINE.with_scale("50m"), linewidth=0.7, edgecolor="darkgreen", zorder=5)
    ax.add_feature(cfeature.BORDERS.with_scale("50m"), linewidth=0.5, linestyle=":", edgecolor="darkgreen", zorder=5)
    ax.set_extent(extent, crs=crs_data)

    lon_u, lat_u = fields["lon_upr"], fields["lat_upr"]
    temp_c = fields[temp_key]

    norm = mpl.colors.BoundaryNorm(cflevs_temp, 256)
    cf = ax.contourf(
        lon_u, lat_u, temp_c, levels=cflevs_temp, norm=norm,
        cmap="RdYlBu_r", transform=crs_data, extend="both", alpha=0.55, zorder=1,
    )
    cbar = fig.colorbar(cf, cax=cax, ticks=cflevs_temp[::2], orientation="vertical")
    cbar.set_label(f"Temperature at {level_hpa}hPa ($^\\circ$C)", fontsize=9.5)
    cbar.ax.tick_params(labelsize=8.5)
    cbar.minorticks_off()

    cs_temp = ax.contour(
        lon_u, lat_u, temp_c, levels=clevs_temp,
        colors="navy", linewidths=0.5, linestyles="dashed", transform=crs_data, zorder=2,
    )
    clbls_t = ax.clabel(cs_temp, fmt="%d", fontsize=7)
    plt.setp(clbls_t, path_effects=[path_effects.withStroke(linewidth=1.5, foreground="white")])

    ax.contour(
        lon_u, lat_u, temp_c, levels=snow_levels,
        colors=["purple", "blue"], linewidths=2.8, linestyles="solid", alpha=0.85, transform=crs_data, zorder=3,
    )

    draw_msm_mslp_contours(ax, fields["lon_sfc"], fields["lat_sfc"], fields["prmsl"], crs_data)

    ax.gridlines(
        xlocs=np.arange(120, 155, 5),
        ylocs=np.arange(20, 50, 5),
        linestyle="dashed", color="green", linewidth=0.4, alpha=0.5,
    )


def plot_msm_snow_chart(
    dt_str: str,
    fcst_hours: int = 0,
    season: str = "winter",
    extent: tuple[float, float, float, float] = (122.0, 148.0, 24.0, 46.5),
    save_path: str | Path | None = None,
    return_fig: bool = False,
) -> plt.Figure | None:
    """MSMデータから地上気圧(2hPa)＋500/850hPa気温（雪目安）の天気図を生成する。"""
    init_dt = dt.datetime.strptime(dt_str[:10], "%Y%m%d%H")
    fields = get_msm_fields(init_dt, fcst_hours=fcst_hours)

    dtm_ini: dt.datetime = fields["init_time"]
    dtm_fcst: dt.datetime = fields["valid_time"]

    fig = plt.figure(figsize=(210 / 25.4, 294 / 25.4), dpi=150)
    fig.text(0.10, 0.97, "JMA MSM (5 km) - MSLP & Upper Temperature", fontsize=13, fontweight="bold")
    fig.text(0.10, 0.95, f"Initial : {dtm_ini.strftime('%Y/%m/%d %H')} UTC", fontsize=11)
    fig.text(0.10, 0.93, f"Forecast: {dtm_fcst.strftime('%Y/%m/%d %H')} UTC (FT+{fcst_hours}h)", fontsize=11)

    crs_map = ccrs.LambertConformal(central_longitude=137.0, central_latitude=36.0)
    crs_data = ccrs.PlateCarree()

    ax01 = fig.add_axes([0.08, 0.51, 0.76, 0.38], projection=crs_map)
    cb01 = fig.add_axes([0.86, 0.51, 0.015, 0.38])
    ax02 = fig.add_axes([0.08, 0.05, 0.76, 0.38], projection=crs_map)
    cb02 = fig.add_axes([0.86, 0.05, 0.015, 0.38])

    cflevs_500 = np.arange(-48, 3, 3) if season == "winter" else np.arange(-24, 6, 3)
    cflevs_850 = np.arange(-30, 15, 3) if season == "winter" else np.arange(-12, 30, 3)

    draw_msm_temp_panel(
        fig, ax01, cb01, fields,
        temp_key="t500",
        panel_title="(a) 500 hPa Temp. (Purple: -36°C, Blue: -30°C) & MSLP (2 hPa)",
        level_hpa=500,
        snow_levels=[-36.0, -30.0],
        clevs_temp=np.arange(-54, 15, 3),
        cflevs_temp=cflevs_500,
        crs_data=crs_data,
        extent=extent,
    )

    draw_msm_temp_panel(
        fig, ax02, cb02, fields,
        temp_key="t850",
        panel_title="(b) 850 hPa Temp. (Purple: -12°C, Blue: -6°C) & MSLP (2 hPa)",
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

import datetime as dt
import os
from pathlib import Path
import urllib.error
import urllib.request

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib as mpl
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

from .gsm_utils import get_jma_prec_cmap
from .surface_utils import CACHE_DIR, RAW_DIR

RISH_MSM_BASE_URL = "http://database.rish.kyoto-u.ac.jp/arch/jmadata/data/gpv/original"
MSM_CACHE_DIR = os.path.join(CACHE_DIR, "msm")
os.makedirs(MSM_CACHE_DIR, exist_ok=True)


def _get_msm_surface_fh_candidates(fcst_hours: int) -> list[str]:
    """予報時間(FT)に対応するMSM地上GRIB2ファイルのFHサフィックス候補を返す。"""
    if fcst_hours <= 15:
        return ["FH00-15"]
    elif fcst_hours <= 33:
        return ["FH18-33"]
    elif fcst_hours <= 39:
        # 年代や初期時刻により39hまでと51h/78hまでのファイル名が存在する
        return ["FH36-39", "FH36-51", "FH36-78"]
    elif fcst_hours <= 51:
        return ["FH36-51", "FH36-78"]
    else:
        return ["FH36-78"]


def fetch_msm_surface_grib2(
    init_time_utc: dt.datetime,
    fcst_hours: int,
    save_dir: Path | str = RAW_DIR,
    overwrite: bool = False,
) -> Path:
    """京大RISHアーカイブから該当FTを含むMSM地上(Lsurf) GRIB2ファイルをダウンロードする。"""
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    y_str = init_time_utc.strftime("%Y")
    m_str = init_time_utc.strftime("%m")
    d_str = init_time_utc.strftime("%d")
    hr_str = init_time_utc.strftime("%H0000")
    base_url = f"{RISH_MSM_BASE_URL}/{y_str}/{m_str}/{d_str}"

    fh_list = _get_msm_surface_fh_candidates(fcst_hours)

    # キャッシュ（既存ファイル）の確認
    for fh in fh_list:
        fname = f"Z__C_RJTD_{y_str}{m_str}{d_str}{hr_str}_MSM_GPV_Rjp_Lsurf_{fh}_grib2.bin"
        dest_path = save_path / fname
        if dest_path.exists() and dest_path.stat().st_size > 0 and not overwrite:
            return dest_path

    # ダウンロード試行
    for fh in fh_list:
        fname = f"Z__C_RJTD_{y_str}{m_str}{d_str}{hr_str}_MSM_GPV_Rjp_Lsurf_{fh}_grib2.bin"
        dest_path = save_path / fname
        url = f"{base_url}/{fname}"
        try:
            urllib.request.urlretrieve(url, dest_path)
            if dest_path.exists() and dest_path.stat().st_size > 0:
                return dest_path
        except urllib.error.HTTPError as e:
            if e.code == 404:
                continue
            raise RuntimeError(f"MSM地上データのダウンロードに失敗しました ({e.code}): {url}") from e

    raise FileNotFoundError(
        f"MSM地上GRIB2ファイルが見つかりません: {init_time_utc:%Y-%m-%d %H}UTC (FT+{fcst_hours}h)"
    )


def get_msm_surface_fields(
    init_time_utc: dt.datetime,
    fcst_hours: int,
    lon_bounds: tuple[float, float] = (120.0, 150.0),
    lat_bounds: tuple[float, float] = (22.4, 47.6),
    sigma: float = 1.5,
    cache_dir: Path | str = MSM_CACHE_DIR,
    raw_dir: Path | str = RAW_DIR,
) -> dict[str, np.ndarray | dt.datetime]:
    """MSM地上データから海面更正気圧(hPa)・10m風(kt)・前1時間降水量(mm)を抽出しNetCDFキャッシュで返す。"""
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)

    ini_str = init_time_utc.strftime("%Y%m%d%H")
    ext_str = f"{lon_bounds[0]}_{lon_bounds[1]}_{lat_bounds[0]}_{lat_bounds[1]}"
    nc_file = cache_path / f"msm_sfc_{ini_str}_ft{fcst_hours:03d}_{ext_str}.nc"

    if nc_file.exists():
        with xr.open_dataset(nc_file) as ds_cached:
            prmsl_hpa = ds_cached["prmsl"].values
            u10_kt = ds_cached["u10"].values
            v10_kt = ds_cached["v10"].values
            prcp_1h = ds_cached["prcp_1h"].values
            lon = ds_cached["longitude"].values
            lat = ds_cached["latitude"].values
    else:
        grib_path = fetch_msm_surface_grib2(init_time_utc, fcst_hours, save_dir=raw_dir)
        lon_min, lon_max = lon_bounds
        lat_min, lat_max = lat_bounds
        target_step = np.timedelta64(fcst_hours, "h")

        # 1. 海面更正気圧・10m風 (instant)
        with xr.open_dataset(
            grib_path,
            engine="cfgrib",
            backend_kwargs={
                "filter_by_keys": {"stepType": "instant"},
                "indexpath": "",
            },
        ) as ds_inst:
            if "step" in ds_inst.dims:
                ds_inst_step = ds_inst.sel(step=target_step)
            else:
                ds_inst_step = ds_inst

            lat_desc = bool(ds_inst_step.latitude[0] > ds_inst_step.latitude[-1])
            lat_slice = slice(lat_max, lat_min) if lat_desc else slice(lat_min, lat_max)
            lon_slice = slice(lon_min, lon_max)

            prmsl_var = "prmsl" if "prmsl" in ds_inst_step.data_vars else "msl"
            prmsl_sub = ds_inst_step[prmsl_var].sel(latitude=lat_slice, longitude=lon_slice)
            prmsl_hpa = (prmsl_sub.values * units.Pa).to(units.hPa).magnitude
            lon = prmsl_sub.longitude.values
            lat = prmsl_sub.latitude.values

            u10_sub = ds_inst_step["u10"].sel(latitude=lat_slice, longitude=lon_slice)
            v10_sub = ds_inst_step["v10"].sel(latitude=lat_slice, longitude=lon_slice)
            u10_kt = (u10_sub.values * units("m/s")).to(units.knots).magnitude
            v10_kt = (v10_sub.values * units("m/s")).to(units.knots).magnitude

        # 2. 前1時間積算降水量 (accum / FT=0の場合はゼロ配列)
        if fcst_hours == 0:
            prcp_1h = np.zeros_like(prmsl_hpa)
        else:
            with xr.open_dataset(
                grib_path,
                engine="cfgrib",
                backend_kwargs={
                    "filter_by_keys": {"typeOfLevel": "surface", "stepType": "accum"},
                    "indexpath": "",
                },
            ) as ds_acc:
                if "step" in ds_acc.dims:
                    ds_acc_step = ds_acc.sel(step=target_step)
                else:
                    ds_acc_step = ds_acc

                tp_var = "tp" if "tp" in ds_acc_step.data_vars else list(ds_acc_step.data_vars.keys())[0]
                tp_sub = ds_acc_step[tp_var].sel(latitude=lat_slice, longitude=lon_slice)
                prcp_1h = np.maximum(np.squeeze(tp_sub.values), 0.0)

        ds_out = xr.Dataset(
            {
                "prmsl": (["latitude", "longitude"], prmsl_hpa, {"units": "hPa"}),
                "u10": (["latitude", "longitude"], u10_kt, {"units": "knots"}),
                "v10": (["latitude", "longitude"], v10_kt, {"units": "knots"}),
                "prcp_1h": (["latitude", "longitude"], prcp_1h, {"units": "mm"}),
            },
            coords={"latitude": lat, "longitude": lon},
        )
        ds_out.to_netcdf(nc_file)
        if not Path("/Users").exists():
            grib_path.unlink(missing_ok=True)

    if sigma > 0:
        prmsl_hpa = gaussian_filter(prmsl_hpa, sigma=sigma)
        u10_kt = gaussian_filter(u10_kt, sigma=sigma)
        v10_kt = gaussian_filter(v10_kt, sigma=sigma)
        prcp_1h = gaussian_filter(prcp_1h, sigma=sigma * 0.6)

    wspd_kt = mpcalc.wind_speed(u10_kt * units.knots, v10_kt * units.knots).magnitude
    valid_dt = init_time_utc + dt.timedelta(hours=fcst_hours)

    return {
        "lon": lon,
        "lat": lat,
        "prmsl": prmsl_hpa,
        "u10": u10_kt,
        "v10": v10_kt,
        "wspd10": wspd_kt,
        "prcp_1h": prcp_1h,
        "init_time": init_time_utc,
        "valid_time": valid_dt,
    }


def plot_msm_surface_chart(
    dt_str: str,
    fcst_hours: int = 3,
    extent: tuple[float, float, float, float] = (122.0, 148.0, 24.0, 46.5),
    save_path: str | Path | None = None,
    return_fig: bool = False,
) -> plt.Figure | None:
    """MSMデータから地上10m風（流線・矢羽）と地上気圧(2hPa間隔・H/L)＋前1時間降水量の天気図を生成する。"""
    init_dt = dt.datetime.strptime(dt_str[:10], "%Y%m%d%H")
    fields = get_msm_surface_fields(init_dt, fcst_hours=fcst_hours, sigma=1.5)

    dtm_ini: dt.datetime = fields["init_time"]
    dtm_fcst: dt.datetime = fields["valid_time"]
    lon = fields["lon"]
    lat = fields["lat"]

    fig = plt.figure(figsize=(210 / 25.4, 294 / 25.4), dpi=150)
    fig.text(0.10, 0.97, "JMA MSM (5 km) - Surface Wind, MSLP & 1h Precipitation", fontsize=13, fontweight="bold")
    fig.text(0.10, 0.95, f"Initial : {dtm_ini.strftime('%Y/%m/%d %H')} UTC", fontsize=11)
    fig.text(0.10, 0.93, f"Forecast: {dtm_fcst.strftime('%Y/%m/%d %H')} UTC (FT+{fcst_hours}h)", fontsize=11)

    # MSM標準のランベルト正角円錐図法
    lon_center = (extent[0] + extent[1]) / 2.0
    crs_map = ccrs.LambertConformal(central_longitude=lon_center, central_latitude=35.0, standard_parallels=(30.0, 40.0))
    crs_data = ccrs.PlateCarree()

    ax01 = fig.add_axes([0.08, 0.51, 0.76, 0.38], projection=crs_map)
    cb01 = fig.add_axes([0.86, 0.51, 0.015, 0.38])
    ax02 = fig.add_axes([0.08, 0.05, 0.76, 0.38], projection=crs_map)
    cb02 = fig.add_axes([0.86, 0.05, 0.015, 0.38])

    # 描画領域の広さに応じて矢羽根の間引き間隔を自動調整（MSMは0.0625度格子）
    lon_span = extent[1] - extent[0]
    barb_step = max(6, int(lon_span * 0.75))

    # --------------------------------------------------------------------------
    # パネル(a): 地上10m風（矢羽根 + 流線）
    # --------------------------------------------------------------------------
    ax01.set_title("(a) 10m Surface Wind (Barbs & Streamlines, kt)", loc="left", fontsize=10.5, fontweight="bold", pad=6)
    ax01.add_feature(cfeature.NaturalEarthFeature("physical", "land", "10m", edgecolor="none", facecolor="green"), alpha=0.08, zorder=1)
    ax01.add_feature(cfeature.COASTLINE.with_scale("10m"), linewidth=0.6, edgecolor="darkgreen", zorder=5)
    ax01.add_feature(cfeature.BORDERS.with_scale("10m"), linewidth=0.5, linestyle=":", edgecolor="darkgreen", zorder=5)
    ax01.set_extent(extent, crs=crs_data)

    clevs_wndsp = np.array([0, 10, 20, 30, 40, 50])
    norm_wnd = mpl.colors.Normalize(vmin=clevs_wndsp.min(), vmax=clevs_wndsp.max())

    if lat[0] > lat[-1]:
        lat_asc = lat[::-1]
        u10_asc = fields["u10"][::-1, :]
        v10_asc = fields["v10"][::-1, :]
        wspd_asc = fields["wspd10"][::-1, :]
    else:
        lat_asc, u10_asc, v10_asc, wspd_asc = lat, fields["u10"], fields["v10"], fields["wspd10"]

    ax01.barbs(
        lon[::barb_step], lat_asc[::barb_step], u10_asc[::barb_step, ::barb_step], v10_asc[::barb_step, ::barb_step],
        length=4.2, transform=crs_data, sizes=dict(emptybarb=0.04), linewidth=0.45, pivot="middle", zorder=3,
    )
    st_plot = ax01.streamplot(
        lon, lat_asc, u10_asc, v10_asc,
        density=[1.1, 1.1], color=wspd_asc, transform=crs_data,
        linewidth=1.0, cmap="jet", norm=norm_wnd, zorder=4,
    )
    cbar1 = fig.colorbar(st_plot.lines, cax=cb01, ticks=clevs_wndsp, orientation="vertical")
    cbar1.set_label("10m Wind Speed (kt)", fontsize=9.5)
    cbar1.ax.tick_params(labelsize=8.5)
    cbar1.minorticks_off()

    ax01.gridlines(
        xlocs=np.arange(115, 160, 5), ylocs=np.arange(20, 55, 5),
        linestyle="dashed", color="green", linewidth=0.4, alpha=0.5,
    )

    # --------------------------------------------------------------------------
    # パネル(b): 海面更正気圧(2hPa間隔, H/L) ＋ 前1時間積算降水量
    # --------------------------------------------------------------------------
    ax02.set_title("(b) MSLP (2 hPa, H/L) & 1h Accumulated Precipitation (mm)", loc="left", fontsize=10.5, fontweight="bold", pad=6)
    ax02.add_feature(cfeature.NaturalEarthFeature("physical", "land", "10m", edgecolor="none", facecolor="green"), alpha=0.08, zorder=1)
    ax02.add_feature(cfeature.COASTLINE.with_scale("10m"), linewidth=0.6, edgecolor="darkgreen", zorder=5)
    ax02.add_feature(cfeature.BORDERS.with_scale("10m"), linewidth=0.5, linestyle=":", edgecolor="darkgreen", zorder=5)
    ax02.set_extent(extent, crs=crs_data)

    jmacmap = get_jma_prec_cmap()
    clevs_prcp = np.array([0.1, 1, 5, 10, 20, 30, 50, 80])
    norm_prcp = mpl.colors.BoundaryNorm([0.1, 1, 5, 10, 20, 30, 50, 80, 100], 256)

    cf = ax02.contourf(
        lon, lat, fields["prcp_1h"], levels=clevs_prcp, norm=norm_prcp,
        cmap=jmacmap, transform=crs_data, extend="max", zorder=2,
    )
    ax02.contour(
        lon, lat, fields["prcp_1h"], levels=[0.1],
        colors="black", transform=crs_data, linewidths=0.25, linestyles="dashed", zorder=2,
    )
    cbar2 = fig.colorbar(cf, cax=cb02, ticks=clevs_prcp, orientation="vertical")
    cbar2.set_label("1h Precipitation (mm)", fontsize=9.5)
    cbar2.ax.tick_params(labelsize=8.5)
    cbar2.minorticks_off()

    # MSM用等圧線 (2hPa間隔、10hPaごとに太線、4hPaごとにラベル)
    clevs_mslp = np.arange(940, 1062, 2)
    cs = ax02.contour(
        lon, lat, fields["prmsl"], levels=clevs_mslp,
        colors="black", transform=crs_data,
        linewidths=[1.25, 0.7, 0.7, 0.7, 0.7], zorder=3,
    )
    clbls = ax02.clabel(cs, clevs_mslp[::2], fmt="%d", fontsize=8, inline=True)
    plt.setp(clbls, path_effects=[path_effects.withStroke(linewidth=2, foreground="white")])

    # 高気圧(H)・低気圧(L)の極値検出（MSMは格子間隔が細かいため探索窓を広めに設定）
    lon_2d, lat_2d = np.meshgrid(lon, lat) if lon.ndim == 1 else (lon, lat)
    prmsl = fields["prmsl"]
    win_size = 75
    local_mean = uniform_filter(prmsl, size=win_size, mode="nearest")
    is_maxima = (prmsl == maximum_filter(prmsl, size=win_size)) & ((prmsl - local_mean) >= 0.6)
    is_minima = (prmsl == minimum_filter(prmsl, size=win_size)) & ((local_mean - prmsl) >= 1.0)

    stroke = [path_effects.Stroke(linewidth=2, foreground="white"), path_effects.Normal()]

    def _is_inside_axes(x_lon: float, y_lat: float, margin_axes: float = 0.04) -> bool:
        proj_pt = ax02.projection.transform_point(x_lon, y_lat, crs_data)
        axes_pt = ax02.transAxes.inverted().transform(ax02.transData.transform(proj_pt))
        return (margin_axes <= axes_pt[0] <= 1.0 - margin_axes) and (
            margin_axes <= axes_pt[1] <= 1.0 - margin_axes
        )

    for i, j in np.argwhere(is_maxima):
        if _is_inside_axes(lon_2d[i, j], lat_2d[i, j]):
            ax02.text(
                lon_2d[i, j], lat_2d[i, j], "H",
                transform=crs_data, clip_on=True, color="blue",
                va="center", ha="center", weight="bold", fontsize=12,
                path_effects=stroke, zorder=6,
            )

    for i, j in np.argwhere(is_minima):
        if _is_inside_axes(lon_2d[i, j], lat_2d[i, j]):
            ax02.text(
                lon_2d[i, j], lat_2d[i, j], "L",
                transform=crs_data, clip_on=True, color="red",
                va="center", ha="center", weight="bold", fontsize=12,
                path_effects=stroke, zorder=6,
            )

    ax02.gridlines(
        xlocs=np.arange(115, 160, 5), ylocs=np.arange(20, 55, 5),
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