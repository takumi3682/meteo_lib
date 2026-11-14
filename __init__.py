# meteo_lib/__init__.py
"""
meteo_lib package

方針:
- import meteo_lib は常に成功させる
- 環境変数に依存する paths は「必要になったときだけ」読み込む
"""

# 環境変数不要なものは普通に公開してOK
from .core import getBlock, get_HourlyData, get_10minData

# paths は遅延ロードする（環境変数が未設定でも import meteo_lib は通す）
_PATH_VARS = {"METDATA", "JMA_DATA", "JMA_GPV_DATA", "ERA5_DATA", "CACHE_DIR"}

def __getattr__(name: str):
    """
    Python 3.7+ の遅延属性解決。
    meteo_lib.METDATA などにアクセスされた瞬間だけ paths を import する。
    """
    if name in _PATH_VARS:
        from . import paths as _p  # ここで初めて環境変数チェックが走る
        return getattr(_p, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

def __dir__():
    # 補完や dir(meteo_lib) を少し親切にする
    return sorted(list(globals().keys()) + list(_PATH_VARS))

__all__ = [
    # 遅延公開（paths側）
    "METDATA", "JMA_DATA", "JMA_GPV_DATA", "ERA5_DATA", "CACHE_DIR",
    # 即時公開（core側）
    "getBlock", "get_HourlyData", "get_10minData",
]
