"""
防災Web APIの最新データを確認し、遡って全てのデータを取得する
"""
"""fetch_amedas_recent.py.

====================
気象庁 防災Web APIから最新48時間のアメダス観測データを差分取得し、
GeoParquet形式でローカルに同期・保存するモジュール。
"""

from datetime import datetime, timedelta
from pathlib import Path
import time
from typing import Optional
import geopandas as gpd
import numpy as np
import pandas as pd
import requests

# 取得対象の気象要素定義
ELEMENTS = {
    "precipitation10m": "p10m",
    "precipitation1h": "p1h",
    "precipitation3h": "p3h",
    "precipitation24h": "p24h",
    "temp": "temp",
    "humidity": "humidity",
    "wind": "wind_spd",
    "windDirection": "wind_dir",
    "gust": "gust",
    "sun10m": "sun10m",
    "sun1h": "sun1h",
    "snow10m": "snow10m",
    "snow6h": "snow6h",
}


def get_latest_amedas_datetime() -> datetime:
  """気象庁サーバーの latest_time.txt から最新の観測日時(JST)を取得する"""
  url = "https://www.jma.go.jp/bosai/amedas/data/latest_time.txt"
  res = requests.get(url, timeout=10)
  res.raise_for_status()
  return datetime.fromisoformat(res.text.strip())


def get_amedas_master() -> pd.DataFrame:
  """アメダステーブルから観測所マスター情報を取得する"""
  url = "https://www.jma.go.jp/bosai/amedas/const/amedastable.json"
  res = requests.get(url, timeout=10)
  res.raise_for_status()
  master = res.json()

  records = []
  for stn_id, info in master.items():
    lat = info["lat"][0] + info["lat"][1] / 60.0
    lon = info["lon"][0] + info["lon"][1] / 60.0
    records.append({
        "stn_id": str(stn_id),
        "stn_name": info.get("kjName", ""),
        "lat": float(lat),
        "lon": float(lon),
        "alt": float(info["alt"]) if info.get("alt") is not None else np.nan,
    })

  return pd.DataFrame(records).set_index("stn_id")


def fetch_and_save_parquet(
    target_time: datetime,
    master_df: pd.DataFrame,
    save_base_dir: str = "./data/amedas_parquet",
    overwrite: bool = False,
) -> Optional[Path]:
  """指定日時のアメダスデータを取得し、UTC基準のHive形式GeoParquetで保存する"""
  time_str = target_time.strftime("%Y%m%d%H%M00")

  # UTC基準の日時でパーティション保存
  target_utc = target_time.astimezone().astimezone(
      datetime.fromtimestamp(0).tzinfo
  )
  out_dir = (
      Path(save_base_dir)
      / f"year={target_time.strftime('%Y')}"
      / f"month={target_time.strftime('%m')}"
  )
  file_path = out_dir / f"{time_str}.parquet"

  if file_path.exists() and not overwrite:
    return file_path

  url = f"https://www.jma.go.jp/bosai/amedas/data/map/{time_str}.json"
  try:
    res = requests.get(url, timeout=10)
    if res.status_code == 404:
      return None
    res.raise_for_status()
    obs_json = res.json()
  except requests.exceptions.RequestException as e:
    print(f"[通信エラー] {time_str}: {e}")
    return None

  records = []
  for stn_id, vals in obs_json.items():
    row = {"stn_id": str(stn_id)}
    for elem_key, prefix in ELEMENTS.items():
      val_flag = vals.get(elem_key, [np.nan, 8])
      v = val_flag[0] if val_flag[0] is not None else np.nan
      f = val_flag[1] if len(val_flag) > 1 and val_flag[1] is not None else 8
      row[f"{prefix}_val"] = float(v) if not np.isnan(v) else np.nan
      row[f"{prefix}_flag"] = int(f)
    records.append(row)

  df_obs = pd.DataFrame(records).set_index("stn_id")

  deg = np.where(
      (df_obs["wind_dir_val"] > 0) & (df_obs["wind_dir_val"] <= 16),
      df_obs["wind_dir_val"] * 22.5,
      np.nan,
  )
  rad = np.deg2rad(deg)
  spd = df_obs["wind_spd_val"]
  df_obs["wind_dir_deg"] = deg.astype("float32")
  df_obs["wind_u"] = np.where(
      df_obs["wind_dir_val"] == 0, 0.0, -spd * np.sin(rad)
  ).astype("float32")
  df_obs["wind_v"] = np.where(
      df_obs["wind_dir_val"] == 0, 0.0, -spd * np.cos(rad)
  ).astype("float32")

  df_combined = master_df.join(df_obs, how="inner").reset_index()
  df_combined["datetime"] = target_time

  for col in df_combined.columns:
    if col.endswith("_flag"):
      df_combined[col] = df_combined[col].astype("int8")
    elif col.endswith("_val") or col in ["lat", "lon", "alt"]:
      df_combined[col] = df_combined[col].astype("float32")

  out_dir.mkdir(parents=True, exist_ok=True)
  gdf = gpd.GeoDataFrame(
      df_combined,
      geometry=gpd.points_from_xy(df_combined["lon"], df_combined["lat"]),
      crs="EPSG:4326",
  )
  gdf.to_parquet(file_path, compression="snappy", schema_version="1.0.0")
  return file_path


def sync_recent_amedas(
    save_base_dir: str = "./data/amedas_parquet",
    max_steps: int = 288,
    verbose: bool = True,
) -> datetime:
  """気象庁サーバーから最新48時間（最大288ステップ）を遡って差分同期する関数

  Returns
  -------
  datetime : 最新の観測日時 (JST)
  """
  df_master = get_amedas_master()
  latest_dt = get_latest_amedas_datetime()

  time_queue = [latest_dt - timedelta(minutes=10 * i) for i in range(max_steps)]

  success_count = 0
  skip_count = 0
  stop_404_count = 0

  for target_dt in time_queue:
    saved = fetch_and_save_parquet(
        target_dt,
        df_master,
        save_base_dir=save_base_dir,
        overwrite=False,
    )

    if saved:
      # 新規書き込みか既存スキップかを更新時刻で簡易判定
      if saved.stat().st_mtime > time.time() - 2:
        if verbose:
          print(f"[新規保存] {target_dt.strftime('%m/%d %H:%M')}")
        success_count += 1
        time.sleep(0.2)
      else:
        skip_count += 1
      stop_404_count = 0
    else:
      stop_404_count += 1
      if stop_404_count >= 3:
        break

  if verbose:
    print(f"同期結果: 新規 {success_count} 件 / スキップ {skip_count} 件")

  return latest_dt


if __name__ == "__main__":
  sync_recent_amedas()