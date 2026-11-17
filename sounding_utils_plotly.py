"""
sounding_utils_plotly.py
========================
sounding_utils.py の出力を受け取り、
Plotly で本格的な Skew-T Log-P 図（風羽付き）を構築・描画するモジュール。
"""

from typing import Any, Dict, Optional
import numpy as np
import plotly.graph_objects as go


def skew_transform(t_val, p_val, rotation: float = 45):
    """指定された傾斜角（度）に合わせたスキュード座標変換（45度で係数40.0基準）"""
    skew_factor = 40.0 * np.tan(np.radians(rotation))
    return t_val + skew_factor * np.log(1000.0 / p_val)


def _add_background_grids(fig: go.Figure, rotation: float = 45):
    """MetPy標準に忠実な背景グリッド線（等温線、0℃破線、乾燥断熱線、湿潤断熱線）を描画"""
    p_grid = np.linspace(1050, 100, 120)

    # 1. 等温線 (rotation度傾斜)
    for t_val in range(-80, 51, 10):
        t_arr = np.full_like(p_grid, t_val, dtype=float)
        if t_val == 0:
            fig.add_trace(go.Scatter(
                x=skew_transform(t_arr, p_grid, rotation),
                y=p_grid,
                mode="lines",
                line=dict(color="cyan", width=1.5, dash="dash"),
                hoverinfo="skip",
                showlegend=False
            ))
        else:
            fig.add_trace(go.Scatter(
                x=skew_transform(t_arr, p_grid, rotation),
                y=p_grid,
                mode="lines",
                line=dict(color="rgba(180, 180, 180, 0.4)", width=1),
                hoverinfo="skip",
                showlegend=False
            ))

    # 2. 乾燥断熱線
    kappa = 0.286
    for theta in range(-30, 201, 20):
        theta_k = theta + 273.15
        t_c = theta_k * (p_grid / 1000.0) ** kappa - 273.15
        fig.add_trace(go.Scatter(
            x=skew_transform(t_c, p_grid, rotation),
            y=p_grid,
            mode="lines",
            line=dict(color="rgba(165, 42, 42, 0.25)", width=1, dash="dash"),
            hoverinfo="skip",
            showlegend=False
        ))

    # 3. 湿潤断熱線
    Rd, Cp, L, eps = 287.05, 1005.0, 2.501e6, 0.622
    dp = p_grid[1] - p_grid[0]
    
    for t_base in [-20, -10, 0, 10, 20, 28, 36]:
        t_prof = []
        t_curr = t_base + 273.15
        for p_val in p_grid:
            t_prof.append(t_curr - 273.15)
            tc = t_curr - 273.15
            es = 6.1078 * 10.0 ** (7.5 * tc / (tc + 237.3))
            rs = max(0.0, eps * es / (p_val - es))
            num = (Rd * t_curr / (p_val * 100.0)) * (1.0 + (L * rs) / (Rd * t_curr))
            den = Cp + (L ** 2 * rs * eps) / (Rd * t_curr ** 2)
            dtdp = num / den * 100.0
            t_curr += dtdp * dp

        fig.add_trace(go.Scatter(
            x=skew_transform(np.array(t_prof), p_grid, rotation),
            y=p_grid,
            mode="lines",
            line=dict(color="rgba(34, 139, 34, 0.35)", width=1, dash="dot"),
            hoverinfo="skip",
            showlegend=False
        ))


def _add_wind_barbs(fig: go.Figure, p_arr, u_arr, v_arr, x_barb=41.5):
    """図の右端に気象標準の風羽（Wind Barbs）を描画"""
    if u_arr is None or v_arr is None:
        return

    # 縦軸基底基準線（右端のバーブ用縦軸）
    fig.add_vline(x=x_barb, line_width=1, line_color="black")

    # プロット間隔の間引き（概ね50hPaごと、あるいは主要指定気圧面）
    target_p = [1000, 950, 925, 900, 850, 800, 700, 600, 500, 400, 300, 250, 200, 150, 100]
    selected_indices = []
    for tp in target_p:
        diffs = np.abs(p_arr - tp)
        min_idx = np.argmin(diffs)
        if diffs[min_idx] <= 25 and min_idx not in selected_indices:
            selected_indices.append(min_idx)

    # 描画用バッチ座標
    bx, by = [], []

    # 風羽描画パラメータ
    shaft_len = 3.5  # 軸の長さ (X軸スケール単位)
    barb_len = 2.2   # 羽の長さ
    
    for idx in selected_indices:
        p_val = p_arr[idx]
        u_val = u_arr[idx]
        v_val = v_arr[idx]
        if np.isnan(u_val) or np.isnan(v_val):
            continue

        # ノット換算 (m/s の場合は約 1.94384)
        spd_kt = np.hypot(u_val, v_val) * 1.94384
        
        # 静穏 (Calm: < 2.5 kt) -> 二重円または小円マーカー
        if spd_kt < 2.5:
            fig.add_shape(
                type="circle",
                xref="x", yref="y",
                x0=x_barb - 0.7, x1=x_barb + 0.7,
                y0=p_val * 1.025, y1=p_val * 0.975,
                line=dict(color="black", width=1.2)
            )
            continue

        # 気象学的風向角（風が吹いてくる方向、上向き0度、時計回り）
        # 数学角 theta (ラジアン): 矢柄の伸びる向き (基点から風上へ)
        angle_rad = np.arctan2(v_val, u_val)

        # 軸（シャフト）の先端座標
        x0, y0 = x_barb, p_val
        x_end = x0 - shaft_len * np.cos(angle_rad)
        # 対数軸のためY方向の比率は比率倍として計算
        y_factor = np.exp(-shaft_len * 0.04 * np.sin(angle_rad))
        y_end = y0 * y_factor

        # 軸線を追加
        bx.extend([x0, x_end, None])
        by.extend([y0, y_end, None])

        # 羽（Barbs / Flags）の分解
        rem_spd = spd_kt + 2.5  # 四捨五入
        n_flags = int(rem_spd // 50)
        rem_spd %= 50
        n_barbs = int(rem_spd // 10)
        rem_spd %= 10
        n_half = int(rem_spd // 5)

        # 羽が出る向きの角度（軸に対して約70度左）
        barb_rad = angle_rad + np.radians(110)

        # 羽を先端側から順に配置
        curr_pos = 0.0
        step = 0.18

        # 50kt 旗 (三角フラグ)
        for _ in range(n_flags):
            pos1 = 1.0 - curr_pos
            pos2 = 1.0 - (curr_pos + step * 1.3)
            curr_pos += step * 1.5
            
            x1 = x0 + (x_end - x0) * pos1
            y1 = y0 * (y_end / y0) ** pos1
            x2 = x0 + (x_end - x0) * pos2
            y2 = y0 * (y_end / y0) ** pos2
            
            xf = x1 - barb_len * np.cos(barb_rad)
            yf = y1 * np.exp(-barb_len * 0.04 * np.sin(barb_rad))
            
            # 三角旗の閉曲線
            bx.extend([x1, xf, x2, None])
            by.extend([y1, yf, y2, None])

        # 10kt 長羽
        for _ in range(n_barbs):
            pos = 1.0 - curr_pos
            curr_pos += step
            x_b = x0 + (x_end - x0) * pos
            y_b = y0 * (y_end / y0) ** pos
            xf = x_b - barb_len * np.cos(barb_rad)
            yf = y_b * np.exp(-barb_len * 0.04 * np.sin(barb_rad))
            bx.extend([x_b, xf, None])
            by.extend([y_b, yf, None])

        # 5kt 短羽
        for _ in range(n_half):
            pos = 1.0 - curr_pos
            curr_pos += step
            x_b = x0 + (x_end - x0) * pos
            y_b = y0 * (y_end / y0) ** pos
            xf = x_b - (barb_len * 0.5) * np.cos(barb_rad)
            yf = y_b * np.exp(-(barb_len * 0.5) * 0.04 * np.sin(barb_rad))
            bx.extend([x_b, xf, None])
            by.extend([y_b, yf, None])

    # まとめて1つのトレースとして描画
    if bx:
        fig.add_trace(go.Scatter(
            x=bx, y=by,
            mode="lines",
            line=dict(color="black", width=1.2),
            hoverinfo="skip",
            showlegend=False
        ))


def create_skewt_figure(
    sounding: Optional[Dict[str, Any]],
    title_info: str = "",
    rotation: float = 45
) -> go.Figure:
    """sounding_utils.get_sounding() のプロファイル辞書を受け取り Plotly Figure を返す"""
    fig = go.Figure()

    if sounding is None:
        fig.add_annotation(
            text="No Sounding Data Available",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=16, color="red")
        )
        return fig

# データの抽出と数値への変換
# データの抽出と数値への変換（mask_100 を外して元に戻す）
    p = sounding["p"].to("hPa").magnitude
    T = sounding["T"].to("degC").magnitude
    Td = sounding["Td"].to("degC").magnitude

    # 風データの取得 (m/s または knots)
    u = sounding.get("u")
    v = sounding.get("v")
    u_mag = u.to("m/s").magnitude if u is not None else None
    v_mag = v.to("m/s").magnitude if v is not None else None

    idx = sounding["indices"]
    parcel = idx.get("parcel_prof")
    parcel_T = parcel.to("degC").magnitude if parcel is not None else None
    
    # 背景線の描画
    _add_background_grids(fig, rotation=rotation)

    # 1. 状態曲線のプロット
    fig.add_trace(go.Scatter(
        x=skew_transform(T, p, rotation),
        y=p,
        mode="lines",
        name="Temperature (T)",
        line=dict(color="red", width=2.5),
        hovertemplate="Pressure: %{customdata[0]:.1f} hPa<br>Temp: %{customdata[1]:.1f} °C<extra></extra>",
        customdata=np.stack([p, T], axis=-1)
    ))

    fig.add_trace(go.Scatter(
        x=skew_transform(Td, p, rotation),
        y=p,
        mode="lines",
        name="Dewpoint (Td)",
        line=dict(color="green", width=2.5),
        hovertemplate="Pressure: %{customdata[0]:.1f} hPa<br>Dewpoint: %{customdata[1]:.1f} °C<extra></extra>",
        customdata=np.stack([p, Td], axis=-1)
    ))

    # 2. パーセルプロファイルのプロット
    if parcel_T is not None:
        p_valid = sounding["p_valid"].to("hPa").magnitude
        fig.add_trace(go.Scatter(
            x=skew_transform(parcel_T, p_valid, rotation),
            y=p_valid,
            mode="lines",
            name="Parcel Profile",
            line=dict(color="black", width=1.5, dash="dash"),
            hoverinfo="skip"
        ))

# 3. 特徴気圧面（LCL, LFC, EL）の水平線とラベル（凡例＋破線上アノテーション）
    level_styles = [
        ("lcl_p", "navy", "LCL", "dot"),
        ("lfc_p", "darkorange", "LFC", "dash"),
        ("el_p", "crimson", "EL", "dashdot"),
    ]
    for key, color, label_prefix, dash_style in level_styles:
        val = idx.get(key)
        if val is not None and not np.isnan(val.magnitude):
            p_val = float(val.to("hPa").magnitude)
            label_text = f"{label_prefix} ({p_val:.0f} hPa)"

            # (1) 水平破線を描画し、右上の凡例（Legend）にもラベルを表示する
            fig.add_trace(
                go.Scatter(
                    x=[-100, 100],
                    y=[p_val, p_val],
                    mode="lines",
                    line=dict(color=color, width=1.2, dash=dash_style),
                    name=label_text,
                    showlegend=True,
                    hovertemplate=f"{label_text}<extra></extra>",
                )
            )

            # (2) 破線上の左端付近に直接ラベルを表示（対数軸のため y には np.log10(p_val) を指定）
            fig.add_annotation(
                x=0.02,
                y=np.log10(p_val),
                xref="paper",
                yref="y",
                text=label_text,
                showarrow=False,
                xanchor="left",
                yanchor="bottom",
                font=dict(size=10, color=color),
                bgcolor="rgba(255, 255, 255, 0.7)",
            )

    # rotation に応じた X軸範囲と風羽位置の自動調整
    if rotation == 0:
        x_range = [-75, 48]   # 上空の -70℃ 付近まで見切れずに表示
        x_barb = 43.0
    else:
        x_range = [-40, 46]   # 従来の Skew-T 用範囲
        x_barb = 41.5

    # 4. 風羽（Wind Barbs）の描画
    _add_wind_barbs(fig, p, u_mag, v_mag, x_barb=x_barb)

    # 5. 安定度指数テキストボックス
    def fmt(val, unit_str, precision=0):
        if val is None or np.isnan(val.magnitude):
            return "--"
        return f"{val.magnitude:.{precision}f} {unit_str}"

    info_lines = [
        "<b>Indices & Parameters</b>",
        f"CAPE:  {fmt(idx.get('cape'), 'J/kg', 0)}",
        f"CIN:   {fmt(idx.get('cin'), 'J/kg', 0)}",
        f"PWAT:  {fmt(idx.get('pwat'), 'mm', 1)}",
        "--------------------",
        f"SSI:   {fmt(idx.get('ssi'), '°C', 1)}",
        f"LI:    {fmt(idx.get('li'), '°C', 1)}",
        f"K-Idx: {fmt(idx.get('k_idx'), '°C', 1)}",
        f"TT:    {fmt(idx.get('tt'), '°C', 1)}",
    ]

    fig.add_annotation(
        text="<br>".join(info_lines),
        xref="paper", yref="paper",
        x=0.02, y=0.98,
        showarrow=False,
        align="left",
        bgcolor="rgba(255, 255, 255, 0.85)",
        bordercolor="gray",
        borderwidth=1,
        borderpad=6,
        font=dict(family="monospace", size=11, color="black")
    )

    # 6. レイアウト設定
    fig.update_layout(
        title=dict(text=f"Interactive Sounding Profile: {title_info}", font=dict(size=14)),
        xaxis=dict(
            title="Temperature (°C)",
            range=x_range,
            dtick=10,
            zeroline=False,
            gridcolor="rgba(200, 200, 200, 0.2)"
        ),
        yaxis=dict(
            title="Pressure (hPa)",
            type="log",
#            autorange="reversed",
            range=[np.log10(1050), np.log10(100)],
            tickvals=[1000, 850, 700, 500, 400, 300, 250, 200, 150, 100],
            ticktext=["1000", "850", "700", "500", "400", "300", "250", "200", "150", "100"],
            gridcolor="rgba(200, 200, 200, 0.2)"
        ),
        legend=dict(x=0.72, y=0.98, bgcolor="rgba(255,255,255,0.8)"),
        width=820,
        height=850,
        template="plotly_white"
    )

    return fig


def plot_sounding(
    sounding: Optional[Dict[str, Any]],
    title_info: str = "",
    rotation: float = 45
) -> go.Figure:
    """sounding_utils.plot_sounding と互換性を持たせるためのラッパー関数"""
    return create_skewt_figure(sounding, title_info=title_info, rotation=rotation)