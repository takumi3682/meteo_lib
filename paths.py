from __future__ import annotations

import os
from pathlib import Path


def _env(key: str, default: str | None = None) -> str | None:
    """環境変数を取得（無ければ default。未設定でも例外にはしない）"""
    v = os.environ.get(key, default)
    if v is None:
        return None
    v = str(v).strip()
    return v if v else None


def _find_upwards(start: Path, marker: str) -> Path | None:
    """start から親方向に marker を探して、見つかったディレクトリを返す"""
    p = start.resolve()
    for d in [p] + list(p.parents):
        if (d / marker).exists():
            return d
    return None


def _pick_existing_root(candidates: str | None) -> Path | None:
    """':' 区切りの候補から、存在する最初のディレクトリを返す"""
    if not candidates:
        return None
    for s in candidates.split(":"):
        s = s.strip()
        if not s:
            continue
        d = Path(s).expanduser()
        if d.exists():
            return d
    return None


def get_project_root(marker: str = "project_paths.py") -> Path | None:
    """
    いまの作業場所（CWD）から project_paths.py を上方向に探してプロジェクトルートを返す。
    """
    return _find_upwards(Path.cwd(), marker)


def get_metdata_root() -> Path:
    """
    データルート決定（恒久版）
    優先順位:
      1) METDATA_ROOT（単一の明示ルート）
      2) プロジェクト配下 data/（project_paths.py がある場所）
      3) METDATA_CANDIDATES（互換用の複数候補）
    """
    # 1) 明示指定（単一）
    if (root := _env("METDATA_ROOT")):
        p = Path(root).expanduser()
        if not p.exists():
            raise RuntimeError(f"METDATA_ROOT is set but does not exist: {p}")
        return p

    # 2) プロジェクト配下 data/
    proj = get_project_root("project_paths.py")
    if proj is not None:
        data_dir = proj / "data"
        if data_dir.exists():
            return data_dir

    # 3) 互換（複数候補）
    p = _pick_existing_root(_env("METDATA_CANDIDATES"))
    if p is not None:
        return p

    # どれも無い場合：エラーはここで初めて出す
    raise RuntimeError(
        "METDATA root could not be determined.\n"
        "Fix one of the following:\n"
        "  - Run from inside a project that contains 'project_paths.py' and 'data/'\n"
        "  - Set environment variable METDATA_ROOT\n"
        "  - Set environment variable METDATA_CANDIDATES (colon-separated paths)\n"
        f"Current working directory: {Path.cwd()}"
    )


# ===== 公開パス（ここから下が利用側の import 対象） =====
METDATA = get_metdata_root()

# 下位ディレクトリは、必要なら環境変数で上書きできる
JMA_DATA      = Path(_env("JMA_DATA",      str(METDATA / "JMA"))).expanduser()
JMA_GPV_DATA  = Path(_env("JMA_GPV_DATA",  str(METDATA / "JMA_GPV"))).expanduser()
ERA5_DATA     = Path(_env("ERA5_DATA",     str(METDATA / "ERA5"))).expanduser()
CACHE_DIR     = Path(_env("CACHE_DIR",     str(METDATA / "CACHE"))).expanduser()
