import io
import math
import os
from datetime import datetime, timezone, timedelta
import requests
from PIL import Image, ImageDraw, ImageFont

# 定義済み領域 [lon_min, lon_max, lat_min, lat_max]
REGION_EXTENTS = {
    "宮城": [140.2, 141.8, 37.7, 39.1],
    "東北": [138.5, 142.5, 36.8, 41.6],
    "東北・関東": [138.0, 142.5, 34.8, 41.6],
    "関東": [138.4, 141.0, 34.8, 37.2],
    "北陸": [135.5, 139.9, 35.3, 38.6],
    "全国": [128.0, 146.0, 30.0, 46.0],
}

# 領域ごとのデフォルトZoom
REGION_DEFAULT_ZOOM = {
    "宮城": 10,
    "東北": 8,
    "東北・関東": 7,
    "関東": 9,
    "北陸": 8,
    "全国": 5,
}

def deg2num(lat_deg: float, lon_deg: float, zoom: int) -> tuple[int, int]:
    lat_rad = math.radians(lat_deg)
    n = 2.0 ** zoom
    xtile = int((lon_deg + 180.0) / 360.0 * n)
    ytile = int((1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n)
    return (xtile, ytile)

def get_latest_basetime() -> str:
    url = "https://www.jma.go.jp/bosai/jmatile/data/risk/targetTimes.json"
    headers = {"User-Agent": "Mozilla/5.0"}
    res = requests.get(url, headers=headers, timeout=10)
    res.raise_for_status()
    return res.json()[0]["basetime"]

def parse_jst_to_utc_basetime(jst_time_str: str) -> str:
    clean_str = jst_time_str.replace("-", "").replace(" ", "").replace(":", "")
    if len(clean_str) == 12:
        dt_jst = datetime.strptime(clean_str, "%Y%m%d%H%M")
    elif len(clean_str) == 14:
        dt_jst = datetime.strptime(clean_str, "%Y%m%d%H%M%S")
    else:
        raise ValueError("日時は 'YYYY-MM-DD HH:MM' または 'YYYYMMDDHHMM' 形式で指定してください。")

    jst_tz = timezone(timedelta(hours=9))
    dt_jst = dt_jst.replace(tzinfo=jst_tz)
    dt_utc = dt_jst.astimezone(timezone.utc)
    return dt_utc.strftime("%Y%m%d%H%M00")

def fetch_tile(url: str) -> Image.Image:
    headers = {"User-Agent": "Mozilla/5.0"}
    res = requests.get(url, headers=headers, timeout=10)
    if res.status_code == 200:
        return Image.open(io.BytesIO(res.content)).convert("RGBA")
    elif res.status_code == 404:
        return Image.new("RGBA", (256, 256), (0, 0, 0, 0))
    res.raise_for_status()

def draw_label(image: Image.Image, basetime: str, element: str, region_name: str = "") -> Image.Image:
    utc_dt = datetime.strptime(basetime, "%Y%m%d%H%M%S").replace(tzinfo=timezone.utc)
    jst_dt = utc_dt.astimezone(timezone(timedelta(hours=9)))
    jst_str = jst_dt.strftime("%Y-%m-%d %H:%M JST")

    element_name_map = {
        "land": "土砂キキクル",
        "inund": "浸水キキクル",
        "flood": "洪水キキクル",
    }
    el_str = element_name_map.get(element, element)
    label_text = f" {el_str} [{region_name}] | {jst_str} "

    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype("Hiragino Sans.ttc", 16)
    except:
        font = ImageFont.load_default()

    box_x0, box_y0 = 10, 10
    box_x1, box_y1 = box_x0 + 360, box_y0 + 32
    draw.rectangle([box_x0, box_y0, box_x1, box_y1], fill=(0, 0, 0, 200))
    draw.text((box_x0 + 8, box_y0 + 6), label_text, fill=(255, 255, 255, 255), font=font)
    return image

def build_composite_image(
    lat_min: float, lat_max: float,
    lon_min: float, lon_max: float,
    zoom: int,
    element: str,
    basetime: str,
    region_name: str = ""
) -> Image.Image:
    x_min, y_min = deg2num(lat_max, lon_min, zoom)
    x_max, y_max = deg2num(lat_min, lon_max, zoom)

    width_tiles = x_max - x_min + 1
    height_tiles = y_max - y_min + 1

    canvas_bg = Image.new("RGBA", (width_tiles * 256, height_tiles * 256), (255, 255, 255, 255))
    canvas_overlay = Image.new("RGBA", (width_tiles * 256, height_tiles * 256), (0, 0, 0, 0))

    for i, x in enumerate(range(x_min, x_max + 1)):
        for j, y in enumerate(range(y_min, y_max + 1)):
            # 1. 地理院淡色地図
            bg_url = f"https://cyberjapandata.gsi.go.jp/xyz/pale/{zoom}/{x}/{y}.png"
            bg_tile = fetch_tile(bg_url)
            canvas_bg.paste(bg_tile, (i * 256, j * 256))

            # 2. キキクル透過タイル（検証済みURL）
            kikikuru_url = f"https://www.jma.go.jp/bosai/jmatile/data/risk/{basetime}/none/{basetime}/surf/{element}/{zoom}/{x}/{y}.png"
            ov_tile = fetch_tile(kikikuru_url)
            canvas_overlay.paste(ov_tile, (i * 256, j * 256))

    composite = Image.alpha_composite(canvas_bg, canvas_overlay)
    return draw_label(composite, basetime, element, region_name)

def save_kikikuru_snapshot(
    time_jst: str,
    region: str = "北陸",
    element: str = "land",
    zoom: int = None,
    output_dir: str = "output_images"
) -> str:
    os.makedirs(output_dir, exist_ok=True)

    if isinstance(region, str):
        if region not in REGION_EXTENTS:
            raise KeyError(f"未定義の領域名です: {region}")
        lon_min, lon_max, lat_min, lat_max = REGION_EXTENTS[region]
        region_name = region
        if zoom is None:
            zoom = REGION_DEFAULT_ZOOM.get(region, 8)
    else:
        lon_min, lon_max, lat_min, lat_max = region
        region_name = "custom"
        if zoom is None:
            zoom = 8

    basetime_utc = parse_jst_to_utc_basetime(time_jst)
    clean_time_tag = time_jst.replace("-", "").replace(" ", "_").replace(":", "")

    img = build_composite_image(
        lat_min=lat_min,
        lat_max=lat_max,
        lon_min=lon_min,
        lon_max=lon_max,
        zoom=zoom,
        element=element,
        basetime=basetime_utc,
        region_name=region_name
    )

    filename = f"{element}_{region_name}_{clean_time_tag}.png"
    out_path = os.path.join(output_dir, filename)
    img.convert("RGB").save(out_path, "PNG", optimize=True)
    print(f"[{region_name} | z{zoom}] 保存完了: {out_path}")
    return out_path