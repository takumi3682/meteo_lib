# obs_utils.py
from datetime import datetime, timedelta
import io
import re
import urllib.parse
import urllib.request
import numpy as np
import pandas as pd

# -------------------------------------------------------------
# 全国主要気象官署マスターテーブル (SYNOP: WMO 47xxx)
# -------------------------------------------------------------
SYNOP_STATIONS = {
    # 北海道
    "47401": {"name": "Wakkanai", "lat": 45.41, "lon": 141.68},
    "47409": {"name": "Abashiri", "lat": 44.02, "lon": 144.28},
    "47412": {"name": "Sapporo", "lat": 43.06, "lon": 141.33},
    "47418": {"name": "Kushiro", "lat": 42.98, "lon": 144.38},
    "47420": {"name": "Nemuro", "lat": 43.33, "lon": 145.58},
    "47430": {"name": "Hakodate", "lat": 41.82, "lon": 140.75},
    # 東北
    "47575": {"name": "Aomori", "lat": 40.82, "lon": 140.77},
    "47582": {"name": "Akita", "lat": 39.72, "lon": 140.10},
    "47584": {"name": "Morioka", "lat": 39.70, "lon": 141.17},
    "47590": {"name": "Sendai", "lat": 38.26, "lon": 140.90},
    "47598": {"name": "Onahama", "lat": 36.95, "lon": 140.90},
    "47588": {"name": "Yamagata", "lat": 38.25, "lon": 140.34},
    "47595": {"name": "Fukushima", "lat": 37.75, "lon": 140.47},
    # 北陸・甲信
    "47604": {"name": "Niigata", "lat": 37.92, "lon": 139.05},
    "47605": {"name": "Kanazawa", "lat": 36.59, "lon": 136.63},
    "47607": {"name": "Toyama", "lat": 36.70, "lon": 137.20},
    "47616": {"name": "Fukui", "lat": 36.05, "lon": 136.22},
    "47636": {"name": "Nagano", "lat": 36.66, "lon": 138.19},
    "47638": {"name": "Kofu", "lat": 35.66, "lon": 138.55},
    # 関東
    "47615": {"name": "Utsunomiya", "lat": 36.55, "lon": 139.87},
    "47624": {"name": "Maebashi", "lat": 36.38, "lon": 139.06},
    "47629": {"name": "Mito", "lat": 36.38, "lon": 140.47},
    "47648": {"name": "Choshi", "lat": 35.73, "lon": 140.86},
    "47656": {"name": "Shizuoka", "lat": 34.98, "lon": 138.40},
    "47662": {"name": "Tokyo", "lat": 35.69, "lon": 139.75},
    "47675": {"name": "Oshima", "lat": 34.78, "lon": 139.36},
    "47670": {"name": "Yokohama", "lat": 35.44, "lon": 139.65},
    # 東海・近畿・西日本
    "47636": {"name": "Nagoya", "lat": 35.17, "lon": 136.97},
    "47759": {"name": "Kyoto", "lat": 35.01, "lon": 135.73},
    "47772": {"name": "Osaka", "lat": 34.68, "lon": 135.52},
    "47765": {"name": "Hiroshima", "lat": 34.40, "lon": 132.46},
    "47761": {"name": "Matsue", "lat": 35.45, "lon": 133.07},
    "47891": {"name": "Takamatsu", "lat": 34.32, "lon": 134.05},
    "47887": {"name": "Matsuyama", "lat": 33.84, "lon": 132.78},
    "47893": {"name": "Kochi", "lat": 33.56, "lon": 133.55},
    "47807": {"name": "Fukuoka", "lat": 33.58, "lon": 130.38},
    "47817": {"name": "Nagasaki", "lat": 32.73, "lon": 129.87},
    "47819": {"name": "Kumamoto", "lat": 32.81, "lon": 130.71},
    "47827": {"name": "Kagoshima", "lat": 31.55, "lon": 130.55},
    "47830": {"name": "Miyazaki", "lat": 31.93, "lon": 131.42},
    "47936": {"name": "Naha", "lat": 26.21, "lon": 127.69},
}


def fetch_ogimet_synop(target_utc: datetime) -> pd.DataFrame:
    """OGIMETから日本の地上官署SYNOPデータを取得してパースします。"""
    ts_str = target_utc.strftime("%Y%m%d%H")
    url = f"https://www.ogimet.com/cgi-bin/getsynop?block=47&begin={ts_str}00&end={ts_str}00&state=Japan"
    req = urllib.request.Request(
        url, headers={"User-Agent": "WeatherAnalysis/1.0"}
    )
    try:
        with urllib.request.urlopen(req, timeout=12) as res:
            text = res.read().decode("utf-8", errors="ignore")
    except Exception:
        return pd.DataFrame()

    records = []
    for line in text.split("\n"):
        parts = line.strip().split(",")
        if len(parts) >= 7 and parts[0] in SYNOP_STATIONS:
            stn_id, raw_synop = parts[0], parts[6]
        elif (
            len(line.split()) >= 6
            and line.split()[0].isdigit()
            and line.split()[0] in SYNOP_STATIONS
        ):
            stn_id, raw_synop = line.split()[0], line
        else:
            continue

        tokens = raw_synop.split("=")[0].split()
        if stn_id not in tokens:
            continue

        # 地点番号 (例: "47590") の位置を基準にする
        stn_idx = tokens.index(stn_id)
        body = tokens[stn_idx + 1 :]  # 地点番号より後ろのトークン列
        if len(body) < 3:
            continue

        rec = {
            "station": SYNOP_STATIONS[stn_id]["name"],
            "lat": float(SYNOP_STATIONS[stn_id]["lat"]),
            "lon": float(SYNOP_STATIONS[stn_id]["lon"]),
            "temp_c": np.nan,
            "dewpoint_c": np.nan,
            "mslp": np.nan,
            "wind_spd_kt": np.nan,
            "wind_dir_deg": np.nan,
            "cloud_cover": 0,
            "weather_code": 0,
            "source": "SYNOP",
        }

        # body[0] は iRiXhVV（スキップ）
        # body[1] は Nddff（全雲量・風向・風速）
        nddff = body[1]
        if len(nddff) == 5:
            if nddff[0] in "012345678":
                rec["cloud_cover"] = int(nddff[0])
            try:
                wdir = int(nddff[1:3])
                wspd = int(nddff[3:5])
                if wdir <= 36 or wdir == 99:
                    rec["wind_dir_deg"] = 0.0 if wdir == 99 else float(wdir * 10)
                    rec["wind_spd_kt"] = float(wspd * 1.94384)
            except ValueError:
                pass

        # body[2] 以降が 1snTTT(気温), 2snTdTdTd(露点), 3...(現地気圧), 4PPPP(海面気圧)
        for tok in body[2:]:
            if tok in ["333", "555"]:
                break
            if len(tok) != 5 or not tok[1:].isdigit():
                continue

            group = tok[0]
            if group == "1" and tok[1] in "01" and pd.isna(rec["temp_c"]):
                sign = -1.0 if tok[1] == "1" else 1.0
                rec["temp_c"] = sign * int(tok[2:5]) / 10.0
            elif group == "2" and tok[1] in "01" and pd.isna(rec["dewpoint_c"]):
                sign = -1.0 if tok[1] == "1" else 1.0
                rec["dewpoint_c"] = sign * int(tok[2:5]) / 10.0
            elif group == "4" and pd.isna(rec["mslp"]):
                val = int(tok[1:5]) / 10.0
                rec["mslp"] = val + 1000.0 if val < 500.0 else val + 900.0
            elif group == "7" and rec["weather_code"] == 0:
                rec["weather_code"] = int(tok[1:3])

        # 気温 (1snTTT) の位置を探す（iRiXhVV と Nddff の後ろにあるため index >= 2 から検索）
        t_idx = -1
        for idx in range(2, len(tokens)):
            tok = tokens[idx]
            if tok in ["333", "555"]:
                break
            if re.match(r"^1[01]\d{3}$", tok):
                t_idx = idx
                sign = -1.0 if tok[1] == "1" else 1.0
                rec["temp_c"] = sign * int(tok[2:5]) / 10.0
                break

        # 気温の直前が Nddff (全雲量・風向・風速)
        if t_idx >= 1:
            tok = tokens[t_idx - 1]
            if len(tok) == 5 and tok[0] in "0123456789/":
                if tok[0] in "012345678":
                    rec["cloud_cover"] = int(tok[0])
                try:
                    wdir = int(tok[1:3])
                    wspd = int(tok[3:5])
                    if wdir <= 36 or wdir == 99:
                        rec["wind_dir_deg"] = 0.0 if wdir == 99 else float(wdir * 10)
                        rec["wind_spd_kt"] = float(wspd * 1.94384)
                except ValueError:
                    pass

        # 気温より後ろのトークンから露点(2)・海面気圧(4)・現在天気(7)を抽出
        if t_idx != -1:
            for tok in tokens[t_idx + 1 :]:
                if tok in ["333", "555"]:
                    break
                if len(tok) != 5:
                    continue
                if re.match(r"^2[01]\d{3}$", tok) and pd.isna(rec["dewpoint_c"]):
                    sign = -1.0 if tok[1] == "1" else 1.0
                    rec["dewpoint_c"] = sign * int(tok[2:5]) / 10.0
                elif re.match(r"^4\d{4}$", tok) and pd.isna(rec["mslp"]):
                    val = int(tok[1:5]) / 10.0
                    rec["mslp"] = val + 1000.0 if val < 500.0 else val + 900.0
                elif re.match(r"^7\d{4}$", tok) and rec["weather_code"] == 0:
                    rec["weather_code"] = int(tok[1:3])

        for tok in tokens:
            if tok in ["333", "555"]:
                break
            if re.match(r"^2[01]\d{3}$", tok) and pd.isna(rec["dewpoint_c"]):
                sign = -1.0 if tok[1] == "1" else 1.0
                rec["dewpoint_c"] = sign * int(tok[2:5]) / 10.0
            elif (
                re.match(r"^4\d{4}$", tok)
                and not tok.startswith("47")
                and pd.isna(rec["mslp"])
            ):
                val = int(tok[1:5]) / 10.0
                rec["mslp"] = val + 1000.0 if val < 500.0 else val + 900.0
            elif re.match(r"^7\d{4}$", tok) and rec["weather_code"] == 0:
                rec["weather_code"] = int(tok[1:3])

        if pd.notnull(rec["temp_c"]):
            records.append(rec)

    df = pd.DataFrame(records)
    return (
        df.groupby("station").last().reset_index(drop=False)
        if not df.empty
        else pd.DataFrame()
    )


def fetch_iem_metar(target_utc: datetime) -> pd.DataFrame:
    """Iowa Mesonetから日本のMETAR実況データを取得します。"""
    sts = (target_utc - timedelta(minutes=45)).strftime("%Y-%m-%dT%H:%M:%SZ")
    ets = (target_utc + timedelta(minutes=45)).strftime("%Y-%m-%dT%H:%M:%SZ")
    url = f"https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py?network=JP__ASOS&data=all&tz=Etc/UTC&format=onlycomma&latlon=yes&elev=yes&missing=null&trace=null&sts={sts}&ets={ets}"
    req = urllib.request.Request(
        url, headers={"User-Agent": "WeatherAnalysis/1.0"}
    )
    try:
        with urllib.request.urlopen(req, timeout=12) as res:
            csv_data = res.read().decode("utf-8")
    except Exception:
        return pd.DataFrame()

    df = pd.read_csv(io.StringIO(csv_data), na_values=["null", "M", ""])
    if df.empty or "valid" not in df.columns:
        return pd.DataFrame()

    df["valid"] = pd.to_datetime(df["valid"], utc=True)
    target_ts_utc = (
        pd.Timestamp(target_utc, tz="UTC")
        if target_utc.tzinfo is None
        else pd.Timestamp(target_utc).tz_convert("UTC")
    )
    df["time_diff"] = (df["valid"] - target_ts_utc).abs()

    df = df.sort_values("time_diff").groupby("station").first().reset_index()

    df_out = pd.DataFrame()
    df_out["station"] = df["station"].astype(str)
    df_out["lat"] = pd.to_numeric(df["lat"], errors="coerce")
    df_out["lon"] = pd.to_numeric(df["lon"], errors="coerce")
    df_out["temp_c"] = (
        (pd.to_numeric(df["tmpf"], errors="coerce") - 32.0) * 5.0 / 9.0
    ).round(1)
    df_out["dewpoint_c"] = (
        (pd.to_numeric(df["dwpf"], errors="coerce") - 32.0) * 5.0 / 9.0
    ).round(1)

    mslp = pd.to_numeric(df["mslp"], errors="coerce").values
    if "alti" in df.columns:
        alti_hpa = pd.to_numeric(df["alti"], errors="coerce").values * 33.8639
        mslp = np.where(np.isnan(mslp), alti_hpa, mslp)
    df_out["mslp"] = mslp
    df_out["wind_spd_kt"] = pd.to_numeric(df["sknt"], errors="coerce")
    df_out["wind_dir_deg"] = pd.to_numeric(df["drct"], errors="coerce")

    sky_map = {
        "CLR": 0,
        "SKC": 0,
        "FEW": 2,
        "SCT": 4,
        "BKN": 6,
        "OVC": 8,
        "VV": 8,
    }
    df_out["cloud_cover"] = [
        max(
            [
                sky_map.get(row.get(f"skyc{i}"), 0)
                for i in range(1, 5)
                if pd.notnull(row.get(f"skyc{i}"))
            ]
            or [0]
        )
        for _, row in df.iterrows()
    ]
    wx_map = {"RA": 61, "-RA": 60, "+RA": 65, "SN": 71, "TS": 95, "FG": 45}
    df_out["weather_code"] = [
        wx_map.get(str(code).split()[0], 0) if pd.notnull(code) else 0
        for code in df["wxcodes"]
    ]
    df_out["source"] = "METAR"
    return df_out


def get_synop_metar_combined(
    target_utc: datetime, extent: list, min_distance_deg: float = 0.3
) -> pd.DataFrame:
    """SYNOPとMETARを取得・統合し、重複・近傍地点を間引いたDataFrameを返します。"""
    df_synop = fetch_ogimet_synop(target_utc)
    df_metar = fetch_iem_metar(target_utc)
    combined = pd.concat([df_synop, df_metar], ignore_index=True)
    if combined.empty:
        return pd.DataFrame()

    for col in [
        "lat",
        "lon",
        "temp_c",
        "dewpoint_c",
        "mslp",
        "wind_spd_kt",
        "wind_dir_deg",
    ]:
        combined[col] = pd.to_numeric(combined[col], errors="coerce")

    mask = (
        (combined["lon"] >= extent[0])
        & (combined["lon"] <= extent[1])
        & (combined["lat"] >= extent[2])
        & (combined["lat"] <= extent[3])
        & combined["temp_c"].notnull()
    )
    sub = combined[mask].sort_values(by="source", ascending=False).reset_index(
        drop=True
    )

    selected_indices = []
    coords = []
    for idx, row in sub.iterrows():
        lat, lon = row["lat"], row["lon"]
        if not coords:
            coords.append((lat, lon))
            selected_indices.append(idx)
            continue
        dists = [np.hypot(lat - c[0], lon - c[1]) for c in coords]
        if min(dists) >= min_distance_deg:
            coords.append((lat, lon))
            selected_indices.append(idx)

    return sub.iloc[selected_indices].copy().reset_index(drop=True)