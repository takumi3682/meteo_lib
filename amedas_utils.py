"""
amedas_utils.py
===============
防災Web API由来のAMeDAS Parquetアーカイブデータに対するアクセスおよび抽出用ユーティリティモジュール。

提供関数一覧
------------
1. get_station_timeseries(start_dt, end_dt, station_id=None, station_name=None, base_dir=None)
   - 指定期間・特定観測所の時系列データ（10分間隔）を抽出し、ソート済みの DataFrame として取得。
2. save_station_parquet(df, station_id, station_name, start_dt, end_dt, out_dir=None)
   - 抽出した特定地点の時系列 DataFrame を再利用可能な Parquet ファイルとして保存。
3. get_amedas_snapshot(target_time, element="p1h", extent=None, base_dir=None)
   - 指定日時の全観測点スナップショットを読み込み、品質フラグ正常値および指定領域でフィルタリングして取得。

使用例
------
from datetime import datetime
from amedas_utils import get_station_timeseries, save_station_parquet, get_amedas_snapshot

# 1. 特定地点の時系列抽出と保存
start = datetime(2026, 8, 13, 0)
end = datetime(2026, 8, 15, 0)
df_station = get_station_timeseries(start, end, station_id="45212")
if df_station is not None:
    save_station_parquet(df_station, "45212", "CHIBA", start, end)

# 2. 特定時刻の全国・地域スナップショット取得（マップ描画用）
target_dt = datetime(2026, 8, 13, 12, 0)
kanto_extent = [138.3, 141.0, 34.8, 37.2]
df_snap, dt_obj = get_amedas_snapshot(target_dt, element="temp", extent=kanto_extent)
"""

from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Tuple, Union
import pandas as pd

# reports/amedas_utils.py の先頭部分


def _find_base_dir() -> Path:
    """カレントおよび上位ディレクトリを遡って data/amedas_parquet を特定"""
    cur = Path.cwd().resolve()
    for parent in [cur] + list(cur.parents):
        cand = parent / "data" / "amedas_parquet"
        if cand.exists():
            return cand
    return cur / "data" / "amedas_parquet"


BASE_DIR = _find_base_dir()
OUT_DIR = BASE_DIR.parent / "processed" / "stations"


def get_station_timeseries(
    start_dt: datetime,
    end_dt: datetime,
    station_id: Optional[str] = None,
    station_name: Optional[str] = None,
    base_dir: Path = BASE_DIR,
) -> Optional[pd.DataFrame]:
    """指定期間・特定観測所の時系列データを抽出して結合

    Parameters
    ----------
    start_dt : datetime
        取得開始日時
    end_dt : datetime
        取得終了日時
    station_id : str, optional
        観測所番号 (例: "45212")
    station_name : str, optional
        観測所名 (例: "CHIBA")
    base_dir : Path, optional
        Parquetアーカイブのルートパス

    Returns
    -------
    pd.DataFrame or None
        時系列データフレーム（データが存在しない場合は None）
    """
    current_dt = start_dt
    records = []

    filter_cond = (
        [("stn_id", "==", str(station_id))]
        if station_id
        else [("stn_name", "==", str(station_name))]
    )

    while current_dt <= end_dt:
        time_str = current_dt.strftime("%Y%m%d%H%M00")
        file_path = (
            base_dir
            / f"year={current_dt.strftime('%Y')}"
            / f"month={current_dt.strftime('%m')}"
            / f"{time_str}.parquet"
        )
        if file_path.exists():
            df_snap = pd.read_parquet(file_path, filters=filter_cond)
            if not df_snap.empty:
                if (
                    "datetime" in df_snap.columns
                    and df_snap["datetime"].dt.tz is not None
                ):
                    df_snap["datetime"] = df_snap["datetime"].dt.tz_localize(
                        None
                    )
                records.append(df_snap)
        current_dt += timedelta(minutes=10)

    if not records:
        return None

    df_timeseries = pd.concat(records, ignore_index=True).sort_values(
        "datetime"
    )
    if "geometry" in df_timeseries.columns:
        df_timeseries = df_timeseries.drop(columns=["geometry"])

    return df_timeseries


def save_station_parquet(
    df: pd.DataFrame,
    station_id: str,
    station_name: str,
    start_dt: datetime,
    end_dt: datetime,
    out_dir: Path = OUT_DIR,
) -> Path:
    """抽出した時系列をParquetにキャッシュ保存

    Parameters
    ----------
    df : pd.DataFrame
        保存対象のデータフレーム
    station_id : str
        観測所番号
    station_name : str
        観測所名
    start_dt : datetime
        開始日時
    end_dt : datetime
        終了日時
    out_dir : Path, optional
        保存先ディレクトリ

    Returns
    -------
    Path
        保存されたParquetファイルのフルパス
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    s_str = start_dt.strftime("%Y%m%d%H")
    e_str = end_dt.strftime("%Y%m%d%H")
    save_path = (
        out_dir / f"amedas_{station_id}_{station_name}_{s_str}_{e_str}.parquet"
    )
    df.to_parquet(save_path, engine="pyarrow", compression="snappy", index=False)
    return save_path


def get_amedas_snapshot(
    target_time: Union[datetime, str],
    element: str = "p1h",
    extent: Optional[list] = None,
    base_dir: Path = BASE_DIR,
) -> Tuple[Optional[pd.DataFrame], datetime]:
    """指定時刻のAMeDASスナップショットを読み込み、品質フラグ・領域でフィルタリングして返す

    Parameters
    ----------
    target_time : datetime or str
        取得対象時刻 (datetime, 'YYYYMMDDHHMM00', 'YYYYMMDDHH' 等)
    element : str, default "p1h"
        対象気象要素 ("temp", "p1h", "p24h", "wind_spd" 等)
    extent : list of float, optional
        領域指定 [lon_min, lon_max, lat_min, lat_max]
    base_dir : Path, optional
        Parquetアーカイブのルートパス

    Returns
    -------
    tuple of (pd.DataFrame or None, datetime)
        フィルタリング済みデータフレームと正規化された datetime オブジェクト
    """
    if isinstance(target_time, datetime):
        dt_obj = target_time
    else:
        s = str(target_time)
        if len(s) == 10:
            s += "0000"
        elif len(s) == 12:
            s += "00"
        dt_obj = datetime.strptime(s[:14], "%Y%m%d%H%M00")

    time_str = dt_obj.strftime("%Y%m%d%H%M00")
    file_path = (
        base_dir
        / f"year={dt_obj.strftime('%Y')}"
        / f"month={dt_obj.strftime('%m')}"
        / f"{time_str}.parquet"
    )

    if not file_path.exists():
        print(f"[エラー] ファイルが存在しません: {file_path}")
        return None, dt_obj

    df = pd.read_parquet(file_path)

    # 表示範囲（extent）内のフィルタリング
    if extent:
        lon_min, lon_max, lat_min, lat_max = extent
        df = df[
            (df["lon"] >= lon_min - 0.2)
            & (df["lon"] <= lon_max + 0.2)
            & (df["lat"] >= lat_min - 0.2)
            & (df["lat"] <= lat_max + 0.2)
        ].copy()

    val_col = f"{element}_val"
    flag_col = f"{element}_flag"

    # 有効データ（品質フラグ0かつ非欠測）の抽出
    if flag_col in df.columns and val_col in df.columns:
        df_valid = df[(df[flag_col] == 0) & (df[val_col].notna())].copy()
    else:
        df_valid = df

    return df_valid, dt_obj