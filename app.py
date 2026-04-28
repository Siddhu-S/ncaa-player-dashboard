from shiny import App, ui, reactive, render
from shinywidgets import output_widget, render_widget
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from pathlib import Path

from data_engine import (
    load_data, load_d1_data,
    POS_COLOR, POS_LABEL, POSITIONS, CLASSES, height_str,
)

HERE = Path(__file__).parent

D2 = load_data(str(HERE / "data.csv"),            id_prefix="d2p")
D1 = load_d1_data(str(HERE / "mbb_with_pca.csv"), id_prefix="d1p")

d2_df          = D2["df"];  d2_conferences = D2["conferences"]
d2_league_avg  = D2["league_avg"];  d2_similar_to = D2["similar_to"]
D2_TOTAL       = len(d2_df)

d1_df          = D1["df"];  d1_conferences = D1["conferences"]
d1_league_avg  = D1["league_avg"];  d1_similar_to = D1["similar_to"]
D1_TOTAL       = len(d1_df)


# ─────────────────────────────────────────────────────────────────────────
# SHARED UI HELPERS
# ─────────────────────────────────────────────────────────────────────────

def stat_box(lbl, val, avg):
    delta = float(val) - float(avg)
    sign  = "+" if delta >= 0 else ""
    cls   = "up" if delta > 0.001 else ("down" if delta < -0.001 else "")
    return ui.div({"class": "stat-cell"},
                  ui.div(str(val), class_="num"),
                  ui.div(lbl,      class_="lbl"),
                  ui.div(f"{sign}{delta:.1f} vs avg", class_=f"delta {cls}"))

def bar_row(lbl, pv, av, mx, fmt=None):
    fmt = fmt or (lambda v: f"{v:.2f}")
    wp  = min(100.0, (pv / mx) * 100) if mx else 0.0
    wa  = min(100.0, (av / mx) * 100) if mx else 0.0
    return ui.div({"class": "cmp-row"},
                  ui.div(lbl, class_="lbl"),
                  ui.div({"class": "cmp-bar"},
                         ui.div({"class": "player-mark", "style": f"left:0;width:{wp:.1f}%"}),
                         ui.div({"class": "avg-mark",    "style": f"left:{wa:.1f}%"})),
                  ui.div(fmt(pv), class_="val"))

def bio_item(label, value, mono=False):
    return ui.div({"class": "bio-item"},
                  ui.div(label, class_="k"),
                  ui.div(value, class_="v mono" if mono else "v"))


def make_detail_modal(player_id, df, league_avg, similar_to_fn, division_label, watchlist):
    row  = df[df["id"] == player_id].iloc[0]
    sims = similar_to_fn(player_id, n_sim=5)
    pc   = POS_COLOR.get(row["pos"], "#888")
    sim_input = "d1_select_similar" if division_label == "D-I" else "d2_select_similar"
    ppg_max = 30 if division_label == "D-I" else 32
    starred  = player_id in watchlist
    star_icon  = "\u2605" if starred else "\u2606"
    star_label = "Remove from watchlist" if starred else "Add to watchlist"
    star_style = "color:var(--accent);" if starred else "color:var(--ink-3);"

    statline = [
        stat_box("MIN", f"{row['mpg']:.1f}", league_avg["mpg"]),
        stat_box("PTS", f"{row['ppg']:.1f}", league_avg["ppg"]),
        stat_box("REB", f"{row['rpg']:.1f}", league_avg["rpg"]),
        stat_box("AST", f"{row['apg']:.1f}", league_avg["apg"]),
        stat_box("STL", f"{row['spg']:.2f}", league_avg["spg"]),
        stat_box("BLK", f"{row['bpg']:.2f}", league_avg["bpg"]),
        stat_box("FG%", f"{row['fg']*100:.1f}", league_avg["fg"] * 100),
        stat_box("3P%", f"{row['tp']*100:.1f}", league_avg["tp"] * 100),
    ]
    bar_defs = [
        ("PPG", row["ppg"], league_avg["ppg"], ppg_max, None),
        ("RPG", row["rpg"], league_avg["rpg"], 14,      None),
        ("APG", row["apg"], league_avg["apg"], 12,      None),
        ("SPG", row["spg"], league_avg["spg"], 4,       None),
        ("BPG", row["bpg"], league_avg["bpg"], 4,       None),
        ("3P%", row["tp"],  league_avg["tp"],  0.55, lambda v: f"{v*100:.1f}%"),
        ("TS%", row["ts"],  league_avg["ts"],  0.75, lambda v: f"{v*100:.1f}%"),
    ]
    bars = [bar_row(l, pv, av, mx, fmt) for l, pv, av, mx, fmt in bar_defs]

    sim_rows = []
    for i, s in enumerate(sims):
        sc = POS_COLOR.get(s["pos"], "#888")
        sim_rows.append(
            ui.div(
                {"class": "sim-row",
                 "onclick": f"Shiny.setInputValue('{sim_input}','{s['id']}',{{priority:'event'}})"},
                ui.div(f"{i+1:02d}", class_="sim-rank"),
                ui.div(
                    ui.div(s["name"], class_="nm"),
                    ui.div(ui.span(s["pos"], class_="pos-badge",
                                   style=f"color:{sc};border-color:{sc}"),
                           ui.span(s["team"]),
                           ui.span(f"· {s['cls']}", style="color:var(--ink-3)"),
                           class_="meta"),
                    class_="sim-main"),
                ui.div(f"{s['similarity']*100:.0f}",
                       ui.span("%", style="font-size:11px;color:var(--ink-3)"),
                       ui.span("match", class_="sim-lbl"),
                       class_="sim-pct")))

    body = ui.div(
        {"id": "detail-body"},
        ui.div({"class": "detail-col"},
               ui.div(
                   {"class": "player-name-row"},
                   ui.div(row["name"], class_="player-name"),
                   ui.tags.button(
                       {"class": "star-btn",
                        "title": star_label,
                        "style": star_style,
                        "onclick": f"Shiny.setInputValue('toggle_watchlist','{player_id}',{{priority:'event'}})"},
                       star_icon)),
               ui.div(ui.span({"class": "team-dot", "style": f"background:{pc}"}),
                      f"{row['team']} · {row['confName']}", class_="player-team"),
               ui.div({"class": "bio-grid"},
                      bio_item("Division", division_label),
                      bio_item("Position", row["pos"]),
                      bio_item("Class",    row["cls"]),
                      bio_item("Height",   height_str(int(row["heightIn"])), mono=True),
                      bio_item("Games",    str(int(row["gp"])), mono=True),
                      bio_item("Min/G",    f"{row['mpg']:.1f}", mono=True))),
        ui.div({"class": "detail-col"},
               ui.div("Season Statline ", ui.span("2025–26", class_="sub"),
                      class_="col-title"),
               ui.div({"class": "statline"}, *statline),
               ui.div("vs. League Average ",
                      ui.span(f"unweighted mean, all {division_label} players", class_="sub"),
                      class_="col-title"),
               *bars,
               ui.div(ui.tags.b("Bar", style="color:var(--ink-2)"),
                      " = player value.  ",
                      ui.tags.b("Tick", style="color:var(--ink-2)"),
                      " = league mean.", class_="bar-note")),
        ui.div({"class": "detail-col"},
               ui.div("Most Similar Players ",
                      ui.span("Euclidean dist. over PC1–PC4 (z-scored)", class_="sub"),
                      class_="col-title"),
               *sim_rows))

    return ui.modal(body,
                    title=ui.HTML(f"Player Profile <b>· {row['name']}</b> "
                                  f'<span class="div-badge">{division_label}</span>'),
                    easy_close=True, size="xl", footer=None)


# ─────────────────────────────────────────────────────────────────────────
# SCATTER HELPERS
# ─────────────────────────────────────────────────────────────────────────

HOVER_TPL = (
    "<b>%{customdata[0]}</b><br>"
    "%{customdata[1]} · %{customdata[2]} · %{customdata[3]}<br>"
    "%{customdata[4]:.1f} PPG · %{customdata[5]:.1f} RPG · %{customdata[6]:.1f} APG"
    "<extra></extra>"
)

def cdata(d):
    return list(zip(d["name"], d["pos"], d["team"], d["cls"],
                    d["ppg"],  d["rpg"], d["apg"],  d["id"]))

def build_traces(plot_df, selected_id, dimmed_pos, dot_size=9.5, dot_opacity=0.78):
    traces = []
    for pos in POSITIONS:
        sub  = plot_df[plot_df["pos"] == pos]
        if sub.empty: continue
        alpha = 0.06 if pos in dimmed_pos else dot_opacity
        rest  = sub[sub["id"] != selected_id] if selected_id else sub
        sel   = sub[sub["id"] == selected_id] if selected_id else sub.iloc[0:0]
        if not rest.empty:
            traces.append(go.Scatter(
                x=rest["PC1"], y=rest["PC2"], mode="markers",
                marker=dict(size=dot_size, color=POS_COLOR[pos],
                            opacity=alpha, line=dict(width=0)),
                customdata=cdata(rest), hovertemplate=HOVER_TPL,
                name=POS_LABEL[pos], showlegend=False))
        if not sel.empty:
            r = sel.iloc[0]
            traces.append(go.Scatter(
                x=[r["PC1"]], y=[r["PC2"]], mode="markers",
                marker=dict(size=dot_size+16, color="rgba(0,0,0,0)",
                            line=dict(color="#c8a84b", width=1.5)),
                hoverinfo="skip", showlegend=False))
            traces.append(go.Scatter(
                x=[r["PC1"]], y=[r["PC2"]], mode="markers",
                marker=dict(size=dot_size+4, color=POS_COLOR[pos],
                            opacity=1.0, line=dict(color="#0f1623", width=1.8)),
                customdata=[cdata(sel)[0]], hovertemplate=HOVER_TPL,
                showlegend=False))
    return traces

def build_layout(_plot_df):
    axis = dict(gridcolor="rgba(0,0,0,0)", zeroline=True,
                zerolinecolor="#1e2d47", zerolinewidth=1.2,
                tickfont=dict(size=9, family="JetBrains Mono, monospace", color="#4a6080"),
                linecolor="#1e2d47", linewidth=1)
    tf = dict(size=10, family="JetBrains Mono, monospace", color="#4a6080")
    return go.Layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#0f1623",
        margin=dict(l=64, r=18, t=16, b=60),
        xaxis=dict(title="← Component 1 →", title_font=tf, **axis),
        yaxis=dict(title="← Component 2 →", title_font=tf, **axis),
        hoverlabel=dict(bgcolor="#1a2540", bordercolor="#c8a84b",
                        font=dict(family="JetBrains Mono, monospace",
                                  size=11.5, color="#c8d4e8")),
        hovermode="closest", dragmode="pan",
        font=dict(family="Inter, sans-serif"), clickmode="event")

def legend_html(dimmed_pos):
    parts = []
    for pos in POSITIONS:
        cls = "legend-item dim" if pos in dimmed_pos else "legend-item"
        col = POS_COLOR[pos]
        parts.append(
            f'<div class="{cls}" onclick="Shiny.setInputValue(\'toggle_dim\','
            f'\'{pos}\',{{priority:\'event\'}})">'
            f'<span class="swatch" style="background:{col}"></span>'
            f'<span>{pos} · {POS_LABEL[pos]}</span></div>')
    parts.append('<span class="legend-hint"></span>')
    return "".join(parts)


# ─────────────────────────────────────────────────────────────────────────
# SIDEBAR / PLOT AREA BUILDERS
# ─────────────────────────────────────────────────────────────────────────

def make_sidebar(prefix, df, conferences):
    mpg_max = max(38,  int(df["mpg"].max())  + 2)
    ppg_max = max(28,  int(df["ppg"].max())  + 2)
    apg_max = max(9,   int(df["apg"].max())  + 1)
    efg_max = round(max(0.80, float(df["efg"].max())), 2)
    tp_max  = round(max(0.60, float(df["tp"].max())),  2)
    ato_max = round(max(6.0,  float(df["ast_tov"].max())), 1)
    h_min   = max(60, int(df["heightIn"].min()))
    h_max   = max(87, int(df["heightIn"].max()))
    conf_choices = {c["conf"]: c["confName"]
                    for c in sorted(conferences, key=lambda x: x["confName"])}

    return ui.div(
        {"class": "sidebar"},
        ui.div("Filters", class_="sb-title"),
        ui.div(ui.div("Search by name", class_="sb-section-head"),
               ui.input_text(f"{prefix}_q", None, placeholder="e.g. Marcus Jackson"),
               class_="sb-section"),
        ui.div(ui.div(ui.span("Position"),
                      ui.tags.button("clear", class_="clear-btn",
                          onclick=f"Shiny.setInputValue('{prefix}_clear_pos',Math.random())"),
                      class_="sb-section-head"),
               ui.input_checkbox_group(f"{prefix}_positions", None,
                                       choices={p: p for p in POSITIONS}),
               class_="sb-section"),
        ui.div(ui.div(ui.span("Class"),
                      ui.tags.button("clear", class_="clear-btn",
                          onclick=f"Shiny.setInputValue('{prefix}_clear_cls',Math.random())"),
                      class_="sb-section-head"),
               ui.input_checkbox_group(f"{prefix}_classes", None,
                                       choices={c: c for c in CLASSES}),
               class_="sb-section"),
        ui.div(ui.div(ui.span("Conference"),
                      ui.tags.button("clear", class_="clear-btn",
                          onclick=f"Shiny.setInputValue('{prefix}_clear_conf',Math.random())"),
                      class_="sb-section-head"),
               ui.input_checkbox_group(f"{prefix}_confs", None, choices=conf_choices),
               class_="sb-section"),
        ui.div(ui.div("Team", class_="sb-section-head"),
               ui.input_select(f"{prefix}_team", None,
                   choices=["All teams"] + sorted(df["team"].unique().tolist())),
               class_="sb-section"),
        ui.div(ui.div("Minutes per game", class_="sb-section-head"),
               ui.input_slider(f"{prefix}_mpg", None, min=10, max=mpg_max,
                               value=[10, mpg_max], step=1, post=" min"),
               class_="sb-section"),
        ui.div(ui.div("Points per game", class_="sb-section-head"),
               ui.input_slider(f"{prefix}_ppg_range", None, min=0, max=ppg_max,
                               value=[0, ppg_max], step=1, post=" pts"),
               class_="sb-section"),
        ui.div(ui.div("eFG%", class_="sb-section-head"),
               ui.input_slider(f"{prefix}_efg", None, min=0.0, max=efg_max,
                               value=[0.0, efg_max], step=0.01),
               class_="sb-section"),
        ui.div(ui.div("3P%", class_="sb-section-head"),
               ui.input_slider(f"{prefix}_tp_range", None, min=0.0, max=tp_max,
                               value=[0.0, tp_max], step=0.01),
               class_="sb-section"),
        ui.div(ui.div("3P Share", class_="sb-section-head"),
               ui.input_slider(f"{prefix}_three_share", None, min=0.0, max=1.0,
                               value=[0.0, 1.0], step=0.01),
               class_="sb-section"),
        ui.div(ui.div("Assists per game", class_="sb-section-head"),
               ui.input_slider(f"{prefix}_apg_range", None, min=0.0, max=float(apg_max),
                               value=[0.0, float(apg_max)], step=0.1),
               class_="sb-section"),
        ui.div(ui.div("AST / TOV ratio", class_="sb-section-head"),
               ui.input_slider(f"{prefix}_ast_tov", None, min=0.0, max=ato_max,
                               value=[0.0, ato_max], step=0.1),
               class_="sb-section"),
        ui.div(ui.div("Height", class_="sb-section-head"),
               ui.input_slider(f"{prefix}_height", None, min=h_min, max=h_max,
                               value=[h_min, h_max], step=1),
               class_="sb-section"),
        ui.div({"class": "sb-count"},
               ui.span("Showing", class_="lbl"),
               ui.output_text(f"{prefix}_filter_count")),
    )


def make_plot_area(prefix):
    return ui.div(
        {"class": "plot-area"},
        ui.div({"class": "plot-toolbar"},
               ui.div(ui.HTML(""), class_="plot-headline"),
               ui.output_ui(f"{prefix}_plot_meta")),
        ui.div({"class": "legend-bar"},
               ui.output_ui(f"{prefix}_legend_ui")),
        ui.div({"class": "scatter-wrap"},
               output_widget(f"{prefix}_scatter")),
    )


# ─────────────────────────────────────────────────────────────────────────
# APP UI
# ─────────────────────────────────────────────────────────────────────────

app_ui = ui.page_fluid(
    ui.tags.head(
        ui.tags.link(rel="stylesheet",
            href="https://fonts.googleapis.com/css2?family=Source+Serif+4:ital,opsz,wght@0,8..60,400;0,8..60,500;0,8..60,600;0,8..60,700;1,8..60,400;1,8..60,500&family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap"),
        ui.include_css(str(HERE / "www" / "styles.css"), method="inline"),
        ui.tags.style("""
            /* ── Tab bar ─────────────────────────────────────── */
            #tab-switcher {
                display:flex; gap:4px; align-items:center;
                padding:0 28px; background:var(--bg);
                border-bottom:1px solid var(--rule);
                height:38px; flex-shrink:0;
            }
            .tab-btn {
                font-family:var(--sans); font-size:11px; font-weight:600;
                letter-spacing:.10em; text-transform:uppercase;
                color:var(--ink-3); background:none; border:none;
                border-bottom:2px solid transparent;
                padding:0 14px; height:38px; cursor:pointer;
                transition:color .15s, border-color .15s;
            }
            .tab-btn:hover     { color:var(--ink-2); }
            .tab-btn.active-d2 { color:var(--accent); border-bottom-color:var(--accent); }
            .tab-btn.active-d1 { color:#4a9eed;       border-bottom-color:#4a9eed; }
            .tab-btn.active-wl { color:#7cc47a;        border-bottom-color:#7cc47a; }
            .tab-sep { width:1px; height:16px; background:var(--rule-2); margin:0 4px; }

            /* watchlist badge on tab button */
            .wl-badge {
                display:inline-block; background:#7cc47a; color:#0f1623;
                font-size:9px; font-weight:800; font-family:var(--mono);
                border-radius:8px; padding:1px 5px; margin-left:5px;
                vertical-align:middle; line-height:14px;
            }

            /* star button inside modal */
            .player-name-row {
                display:flex; align-items:flex-start; gap:10px; margin-bottom:3px;
            }
            .star-btn {
                background:none; border:none; cursor:pointer;
                font-size:24px; line-height:1; padding:2px 0 0 0;
                flex-shrink:0; transition:color .15s, transform .1s;
            }
            .star-btn:hover { transform:scale(1.2); }

            /* watchlist tab layout */
            .wl-shell {
                display:flex; flex-direction:column; height:100%; overflow:hidden;
            }
            .wl-header {
                display:flex; align-items:baseline; gap:14px;
                padding:14px 28px 10px; border-bottom:1px solid var(--rule);
                flex-shrink:0;
            }
            .wl-title {
                font-family:var(--serif); font-size:20px; font-weight:600;
            }
            .wl-empty {
                flex:1; display:flex; flex-direction:column;
                align-items:center; justify-content:center;
                color:var(--ink-3); font-family:var(--mono); font-size:12px;
                gap:10px;
            }
            .wl-empty .wl-star { font-size:36px; opacity:.3; }
            .wl-grid {
                flex:1; overflow-y:auto;
                display:grid;
                grid-template-columns:repeat(auto-fill, minmax(300px, 1fr));
                gap:12px; padding:18px 24px; align-content:start;
            }
            .wl-card {
                background:var(--bg-2); border:1px solid var(--rule-2);
                padding:14px 16px; display:flex; flex-direction:column; gap:8px;
                cursor:pointer; transition:border-color .15s;
                position:relative;
            }
            .wl-card:hover { border-color:var(--ink-2); }
            .wl-card-name {
                font-family:var(--serif); font-size:16px; font-weight:600;
                line-height:1.1; padding-right:28px;
            }
            .wl-card-meta {
                font-size:11px; color:var(--ink-2);
                display:flex; gap:6px; align-items:center;
            }
            .wl-card-stats {
                display:grid; grid-template-columns:repeat(4,1fr);
                gap:4px 0; margin-top:4px;
                border-top:1px solid var(--rule); padding-top:8px;
            }
            .wl-stat { display:flex; flex-direction:column; }
            .wl-stat .n { font-family:var(--serif); font-size:15px; font-weight:600; }
            .wl-stat .l {
                font-size:8.5px; letter-spacing:.1em; text-transform:uppercase;
                color:var(--ink-3); margin-top:1px;
            }
            .wl-remove {
                position:absolute; top:10px; right:10px;
                background:none; border:none; cursor:pointer;
                color:var(--ink-3); font-size:16px; line-height:1; padding:2px;
                transition:color .15s;
            }
            .wl-remove:hover { color:var(--accent); }

            /* ── Tab panels ──────────────────────────────────────
               KEY TRICK: both panels stay in layout flow at full
               size so Plotly measures real px on startup.
               The inactive panel is clipped to h=0 + overflow:hidden
               so slider tooltips (position:absolute children) cannot
               escape and bleed onto the active tab.            */
            #tab-content {
                flex:1; overflow:hidden;
                display:flex; flex-direction:column;
            }
            .tab-panel {
                /* collapsed but still in flow — Plotly can size itself */
                flex:0; height:0; overflow:hidden;
                display:flex; flex-direction:column;
                min-height:0;
            }
            .tab-panel.active {
                flex:1; height:auto; overflow:hidden;
            }

            /* ── Body / sidebar / plot shared layout ─────────── */
            .body-grid {
                display:grid; grid-template-columns:220px 1fr;
                flex:1; overflow:hidden; height:100%;
            }
            .sidebar {
                overflow-y:auto; border-right:1px solid var(--rule);
                padding:16px 14px 32px;
                /* match original #sidebar styling */
                background:var(--bg);
            }
            .plot-area    { display:flex; flex-direction:column; overflow:hidden; }
            .plot-toolbar {
                display:flex; justify-content:space-between; align-items:center;
                padding:8px 18px 4px; border-bottom:1px solid var(--rule); flex-shrink:0;
            }
            .legend-bar {
                padding:6px 18px; border-bottom:1px solid var(--rule);
                display:flex; gap:12px; flex-shrink:0;
            }
            .scatter-wrap { flex:1; overflow:hidden; }

            /* ── Sidebar inputs — apply to both tabs via .sidebar class ── */
            .sidebar .form-control,
            .sidebar .selectize-input,
            .sidebar select {
                border:1px solid var(--rule-2) !important; border-radius:0 !important;
                background:var(--bg) !important; color:var(--ink) !important;
                font-family:var(--sans) !important; font-size:12.5px !important;
                box-shadow:none !important; padding:6px 8px !important;
            }
            .sidebar .irs--shiny .irs-bar    { background:var(--accent) !important; border-top:none; border-bottom:none; }
            .sidebar .irs--shiny .irs-handle { background:var(--bg) !important; border:2px solid var(--accent) !important; border-radius:50% !important; }
            .sidebar .irs--shiny .irs-line   { background:var(--rule-2) !important; border:none; }
            .sidebar .irs--shiny .irs-from,
            .sidebar .irs--shiny .irs-to,
            .sidebar .irs--shiny .irs-single { background:var(--accent) !important; color:#0f1623 !important; font-family:var(--mono); font-size:10px; font-weight:700; border-radius:0 !important; }
            .sidebar .irs--shiny .irs-min,
            .sidebar .irs--shiny .irs-max   { font-family:var(--mono); font-size:9.5px; color:var(--ink-2); }

            /* ── Checkbox groups — hide native box, style label as compact row ── */
            .sidebar .shiny-input-container { margin-bottom:0; }

            /* Hide the actual checkbox input */
            .sidebar .checkbox input[type="checkbox"] { display:none !important; }

            /* All checkbox labels: compact, monospace, small */
            .sidebar .checkbox label {
                display:flex !important; align-items:center !important;
                gap:6px !important; cursor:pointer;
                font-family:var(--mono) !important; font-size:10.5px !important;
                font-weight:400 !important; color:var(--ink-2) !important;
                padding:3px 2px !important; margin:0 !important;
                border:none !important; background:transparent !important;
                border-bottom:1px dotted var(--rule);
                white-space:nowrap; overflow:hidden; text-overflow:ellipsis;
                max-width:200px; line-height:1.3;
                transition:color .1s;
            }
            .sidebar .checkbox label:hover { color:var(--ink) !important; }

            /* Checked state — bold, full ink */
            .sidebar .checkbox input[type="checkbox"]:checked + span,
            .sidebar .checkbox input[type="checkbox"]:checked ~ span {
                color:var(--ink) !important; font-weight:700 !important;
            }
            /* Fake checkbox dot before label text */
            .sidebar .checkbox label::before {
                content:""; display:inline-block; flex-shrink:0;
                width:7px; height:7px; border-radius:50%;
                border:1px solid var(--rule-2); background:transparent;
                transition:background .1s, border-color .1s;
            }
            .sidebar .checkbox input[type="checkbox"]:checked ~ label::before,
            .sidebar .checkbox:has(input:checked) label::before {
                background:var(--accent); border-color:var(--accent);
            }

            /* Stack checkboxes vertically */
            .sidebar .shiny-options-group {
                display:flex !important; flex-direction:column !important;
                gap:0 !important; flex-wrap:nowrap !important;
            }
            .sidebar .checkbox {
                display:block !important; width:100%; margin:0 !important;
            }

            /* Conference list — slightly smaller since names are longer */
            .sidebar [id$="_confs"] .checkbox label {
                font-size:10px !important;
                max-width:195px !important;
            }

            /* ── D-I blue accent ──────────────────────────────── */
            #d1-tab .accent { color:#4a9eed; }

            /* ── Division badge in modal ──────────────────────── */
            .div-badge {
                font-size:10px; font-weight:700; letter-spacing:.12em;
                text-transform:uppercase; border-radius:3px;
                padding:2px 7px; margin-left:6px;
                background:var(--bg-2); color:var(--ink-2); vertical-align:middle;
            }
        """),
        ui.tags.script("""
            function switchTab(tab) {
                document.querySelectorAll('.tab-panel').forEach(function(p) {
                    p.classList.remove('active');
                });
                document.querySelectorAll('.tab-btn').forEach(function(b) {
                    b.classList.remove('active-d1','active-d2','active-wl');
                });
                document.getElementById(tab+'-tab').classList.add('active');
                document.getElementById('btn-'+tab).classList.add('active-'+tab);

                // Plotly needs a resize after its container goes from h=0 to full height.
                // Two rAF calls ensure the browser has finished layout before we measure.
                requestAnimationFrame(function() {
                    requestAnimationFrame(function() {
                        var panel = document.getElementById(tab+'-tab');
                        if (!panel) return;
                        panel.querySelectorAll('.js-plotly-plot').forEach(function(el) {
                            if (window.Plotly) Plotly.Plots.resize(el);
                        });
                    });
                });
            }
        """),
    ),

    ui.div({"id": "atlas-shell"},

        ui.div({"id": "masthead"},
            ui.div({"class": "mast-left"},
                   ui.div(ui.HTML('NCAA Men\'s Basketball <span class="dot"></span> 2025–26'),
                          class_="kicker"),
                   ui.div(ui.HTML("Player <em>Dashboard</em>"), class_="atlas-title")),
            ui.div({"class": "mast-meta"},
                   ui.div(ui.div(str(D1_TOTAL),               class_="mast-stat-num"),
                          ui.div("D-I Players",               class_="mast-stat-lbl"), class_="mast-stat"),
                   ui.div(ui.div(str(d1_df["team"].nunique()), class_="mast-stat-num"),
                          ui.div("D-I Teams",                 class_="mast-stat-lbl"), class_="mast-stat"),
                   ui.div(ui.div(str(D2_TOTAL),               class_="mast-stat-num"),
                          ui.div("D-II Players",              class_="mast-stat-lbl"), class_="mast-stat"),
                   ui.div(ui.div(str(d2_df["team"].nunique()), class_="mast-stat-num"),
                          ui.div("D-II Teams",                class_="mast-stat-lbl"), class_="mast-stat")),
        ),

        ui.div({"id": "tab-switcher"},
               ui.tags.button("Division I",  id="btn-d1", class_="tab-btn",
                              onclick="switchTab('d1')"),
               ui.div({"class": "tab-sep"}),
               ui.tags.button("Division II", id="btn-d2", class_="tab-btn active-d2",
                              onclick="switchTab('d2')"),
               ui.div({"class": "tab-sep"}),
               ui.tags.button(
                   ui.HTML('Watchlist <span id="wl-badge" class="wl-badge" style="display:none">0</span>'),
                   id="btn-wl", class_="tab-btn",
                   onclick="switchTab('wl')")),

        ui.div({"id": "tab-content"},
            ui.div({"id": "d2-tab", "class": "tab-panel active"},
                   ui.div({"class": "body-grid"},
                          make_sidebar("d2", d2_df, d2_conferences),
                          make_plot_area("d2"))),
            ui.div({"id": "d1-tab", "class": "tab-panel"},
                   ui.div({"class": "body-grid"},
                          make_sidebar("d1", d1_df, d1_conferences),
                          make_plot_area("d1"))),

            ui.div({"id": "wl-tab", "class": "tab-panel"},
                   ui.div({"class": "wl-shell"},
                          ui.div({"class": "wl-header"},
                                 ui.div("Watchlist", class_="wl-title"),
                                 ui.output_text("wl_count")),
                          ui.output_ui("watchlist_ui"))),
        ),
    ),

    ui.output_ui("d2_modal_trigger"),
    ui.output_ui("d1_modal_trigger"),
)


# ─────────────────────────────────────────────────────────────────────────
# SERVER
# ─────────────────────────────────────────────────────────────────────────

def server(input, output, session):

    d2_sel      = reactive.Value(None)   # highlighted dot on scatter
    d2_dim      = reactive.Value(set())
    d1_sel      = reactive.Value(None)
    d1_dim      = reactive.Value(set())
    watchlist   = reactive.Value(set())  # set of player IDs (cross-division)
    # Modal signal: a (player_id, counter) tuple. Counter increments on every
    # click so the same player can be re-opened without ReactiveValue skipping
    # the invalidation (which happens when you set the same value twice).
    modal_req   = reactive.Value(None)

    d2_fig = go.FigureWidget()
    d1_fig = go.FigureWidget()

    # ── Watchlist toggle (fired by star button in modal) ──────────────────
    @reactive.effect
    @reactive.event(input.toggle_watchlist)
    def _toggle_watchlist():
        pid  = input.toggle_watchlist()
        curr = set(watchlist.get())
        curr.discard(pid) if pid in curr else curr.add(pid)
        watchlist.set(curr)
        # Re-open the modal with updated star state by setting modal_req.
        # _open_modal will read watchlist fresh (it's not isolated there).
        import random
        modal_req.set((pid, random.random()))

    # ── legend dim ────────────────────────────────────────────────────────
    @reactive.effect
    @reactive.event(input.toggle_dim)
    def _d2_toggle_dim():
        pos  = input.toggle_dim()
        curr = set(d2_dim.get())
        curr.discard(pos) if pos in curr else curr.add(pos)
        d2_dim.set(curr)

    @reactive.effect
    @reactive.event(input.toggle_dim)
    def _d1_toggle_dim():
        pos  = input.toggle_dim()
        curr = set(d1_dim.get())
        curr.discard(pos) if pos in curr else curr.add(pos)
        d1_dim.set(curr)

    # ═══════════════════════════════════════════════════════════════════════
    # D-II
    # ═══════════════════════════════════════════════════════════════════════

    @reactive.effect
    @reactive.event(input.d2_clear_pos)
    def _d2_clear_pos():
        ui.update_checkbox_group("d2_positions", selected=[])

    @reactive.effect
    @reactive.event(input.d2_clear_cls)
    def _d2_clear_cls():
        ui.update_checkbox_group("d2_classes", selected=[])

    @reactive.effect
    @reactive.event(input.d2_clear_conf)
    def _d2_clear_conf():
        ui.update_checkbox_group("d2_confs", selected=[])
        ui.update_select("d2_team", selected="All teams")

    @reactive.effect
    @reactive.event(input.d2_select_similar)
    def _d2_select_similar():
        sid = input.d2_select_similar()
        if sid:
            d2_sel.set(sid)
            ui.modal_remove()
            import random
            modal_req.set((sid, random.random()))

    @reactive.calc
    def d2_filtered():
        d = d2_df.copy()
        q = (input.d2_q() or "").strip().lower()
        if q: d = d[d["name"].str.lower().str.contains(q, na=False)]
        ps = list(input.d2_positions() or [])
        if ps: d = d[d["pos"].isin(ps)]
        cs = list(input.d2_classes() or [])
        if cs: d = d[d["cls"].isin(cs)]
        xs = list(input.d2_confs() or [])
        if xs: d = d[d["conf"].isin(xs)]
        t = input.d2_team()
        if t and t != "All teams": d = d[d["team"] == t]
        lo, hi = input.d2_mpg();         d = d[(d["mpg"]        >= lo) & (d["mpg"]        <= hi)]
        lo, hi = input.d2_ppg_range();   d = d[(d["ppg"]        >= lo) & (d["ppg"]        <= hi)]
        lo, hi = input.d2_efg();         d = d[(d["efg"]        >= lo) & (d["efg"]        <= hi)]
        lo, hi = input.d2_tp_range();    d = d[(d["tp"]         >= lo) & (d["tp"]         <= hi)]
        lo, hi = input.d2_three_share(); d = d[(d["three_share"] >= lo) & (d["three_share"] <= hi)]
        lo, hi = input.d2_apg_range();   d = d[(d["apg"]        >= lo) & (d["apg"]        <= hi)]
        lo, hi = input.d2_ast_tov();     d = d[(d["ast_tov"]    >= lo) & (d["ast_tov"]    <= hi)]
        lo, hi = input.d2_height();      d = d[(d["heightIn"]   >= lo) & (d["heightIn"]   <= hi)]
        return d

    @reactive.calc
    def d2_plot_df():
        ids = set(d2_filtered()["id"])
        sid = d2_sel.get()
        if sid: ids.add(sid)
        return d2_df[d2_df["id"].isin(ids)]

    @output
    @render.text
    def d2_filter_count():
        return f"{len(d2_filtered())} / {D2_TOTAL}"

    @output
    @render.ui
    def d2_legend_ui():
        return ui.HTML(legend_html(d2_dim.get()))

    @output
    @render.ui
    def d2_plot_meta():
        sid = d2_sel.get()
        if sid is not None:
            row = d2_df[d2_df["id"] == sid]
            if not row.empty:
                return ui.div(ui.HTML(f'<span class="accent">●</span> {row.iloc[0]["name"]} selected'), class_="plot-meta")
        return ui.div("Hover a dot for details · click to expand", class_="plot-meta")

    @render_widget
    def d2_scatter():
        return d2_fig

    @reactive.effect
    def _d2_sync():
        traces = build_traces(d2_plot_df(), d2_sel.get(), d2_dim.get())
        layout = build_layout(d2_plot_df())
        with d2_fig.batch_update():
            d2_fig.data = []
            for t in traces: d2_fig.add_trace(t)
            d2_fig.update_layout(layout)
        for trace in d2_fig.data:
            if hasattr(trace, "customdata") and trace.customdata is not None and len(trace.customdata):
                trace.on_click(_d2_clicked)

    def _d2_clicked(trace, points, selector):
        if not points or not points.point_inds: return
        cd = trace.customdata[points.point_inds[0]]
        if cd is not None and len(cd) >= 8:
            pid = str(cd[7])
            d2_sel.set(pid)
            # Use a tuple with a random token so ReactiveValue always invalidates,
            # even when the same player is clicked twice in a row.
            import random
            modal_req.set((pid, random.random()))

    # Single effect that opens the modal for whichever player was just clicked.
    # Reacts to modal_req (a (pid, token) tuple) so it fires on every click,
    # even repeated clicks on the same player.
    @reactive.effect
    @reactive.event(modal_req)
    def _open_modal():
        req = modal_req.get()
        if not req: return
        pid, _ = req
        wl = watchlist.get()
        if pid.startswith("d1"):
            df_, la_, sf_, div_ = d1_df, d1_league_avg, d1_similar_to, "D-I"
        else:
            df_, la_, sf_, div_ = d2_df, d2_league_avg, d2_similar_to, "D-II"
        row = df_[df_["id"] == pid]
        if row.empty: return
        ui.modal_show(make_detail_modal(pid, df_, la_, sf_, div_, wl))

    # Keep these output stubs so Shiny doesn't complain about missing output IDs
    # declared in the UI (ui.output_ui("d2_modal_trigger") etc.)
    @output
    @render.ui
    def d2_modal_trigger():
        return ui.div()

    # ═══════════════════════════════════════════════════════════════════════
    # D-I
    # ═══════════════════════════════════════════════════════════════════════

    @reactive.effect
    @reactive.event(input.d1_clear_pos)
    def _d1_clear_pos():
        ui.update_checkbox_group("d1_positions", selected=[])

    @reactive.effect
    @reactive.event(input.d1_clear_cls)
    def _d1_clear_cls():
        ui.update_checkbox_group("d1_classes", selected=[])

    @reactive.effect
    @reactive.event(input.d1_clear_conf)
    def _d1_clear_conf():
        ui.update_checkbox_group("d1_confs", selected=[])
        ui.update_select("d1_team", selected="All teams")

    @reactive.effect
    @reactive.event(input.d1_select_similar)
    def _d1_select_similar():
        sid = input.d1_select_similar()
        if sid:
            d1_sel.set(sid)
            ui.modal_remove()
            import random
            modal_req.set((sid, random.random()))

    @reactive.calc
    def d1_filtered():
        d = d1_df.copy()
        q = (input.d1_q() or "").strip().lower()
        if q: d = d[d["name"].str.lower().str.contains(q, na=False)]
        ps = list(input.d1_positions() or [])
        if ps: d = d[d["pos"].isin(ps)]
        cs = list(input.d1_classes() or [])
        if cs: d = d[d["cls"].isin(cs)]
        xs = list(input.d1_confs() or [])
        if xs: d = d[d["conf"].isin(xs)]
        t = input.d1_team()
        if t and t != "All teams": d = d[d["team"] == t]
        lo, hi = input.d1_mpg();         d = d[(d["mpg"]        >= lo) & (d["mpg"]        <= hi)]
        lo, hi = input.d1_ppg_range();   d = d[(d["ppg"]        >= lo) & (d["ppg"]        <= hi)]
        lo, hi = input.d1_efg();         d = d[(d["efg"]        >= lo) & (d["efg"]        <= hi)]
        lo, hi = input.d1_tp_range();    d = d[(d["tp"]         >= lo) & (d["tp"]         <= hi)]
        lo, hi = input.d1_three_share(); d = d[(d["three_share"] >= lo) & (d["three_share"] <= hi)]
        lo, hi = input.d1_apg_range();   d = d[(d["apg"]        >= lo) & (d["apg"]        <= hi)]
        lo, hi = input.d1_ast_tov();     d = d[(d["ast_tov"]    >= lo) & (d["ast_tov"]    <= hi)]
        lo, hi = input.d1_height();      d = d[(d["heightIn"]   >= lo) & (d["heightIn"]   <= hi)]
        return d

    @reactive.calc
    def d1_plot_df():
        ids = set(d1_filtered()["id"])
        sid = d1_sel.get()
        if sid: ids.add(sid)
        return d1_df[d1_df["id"].isin(ids)]

    @output
    @render.text
    def d1_filter_count():
        return f"{len(d1_filtered())} / {D1_TOTAL}"

    @output
    @render.ui
    def d1_legend_ui():
        return ui.HTML(legend_html(d1_dim.get()))

    @output
    @render.ui
    def d1_plot_meta():
        sid = d1_sel.get()
        if sid is not None:
            row = d1_df[d1_df["id"] == sid]
            if not row.empty:
                return ui.div(ui.HTML(f'<span class="accent">●</span> {row.iloc[0]["name"]} selected'), class_="plot-meta")
        return ui.div("Hover a dot for details · click to expand", class_="plot-meta")

    @render_widget
    def d1_scatter():
        return d1_fig

    @reactive.effect
    def _d1_sync():
        traces = build_traces(d1_plot_df(), d1_sel.get(), d1_dim.get())
        layout = build_layout(d1_plot_df())
        with d1_fig.batch_update():
            d1_fig.data = []
            for t in traces: d1_fig.add_trace(t)
            d1_fig.update_layout(layout)
        for trace in d1_fig.data:
            if hasattr(trace, "customdata") and trace.customdata is not None and len(trace.customdata):
                trace.on_click(_d1_clicked)

    def _d1_clicked(trace, points, selector):
        if not points or not points.point_inds: return
        cd = trace.customdata[points.point_inds[0]]
        if cd is not None and len(cd) >= 8:
            pid = str(cd[7])
            d1_sel.set(pid)
            import random
            modal_req.set((pid, random.random()))

    @output
    @render.ui
    def d1_modal_trigger():
        return ui.div()


    # ── Watchlist outputs ─────────────────────────────────────────────────

    @output
    @render.text
    def wl_count():
        n = len(watchlist.get())
        return f"{n} player{'s' if n != 1 else ''}"

    @output
    @render.ui
    def watchlist_ui():
        wl  = watchlist.get()
        if not wl:
            return ui.div(
                ui.tags.script("var b=document.getElementById('wl-badge');if(b){b.style.display='none';}"),
                ui.div({"class": "wl-empty"},
                       ui.div("☆", class_="wl-star"),
                       ui.div("No players starred yet."),
                       ui.div("Open any player profile and click ☆ to add them here.",
                              style="color:var(--ink-3);max-width:280px;text-align:center;line-height:1.5")))

        cards = []
        for pid in wl:
            # Look up in whichever dataset this id belongs to
            if pid.startswith("d1"):
                df_, div_ = d1_df, "D-I"
            else:
                df_, div_ = d2_df, "D-II"
            row_ = df_[df_["id"] == pid]
            if row_.empty:
                continue
            r    = row_.iloc[0]
            pc_  = POS_COLOR.get(r["pos"], "#888")
            # Which modal input to fire on card click
            open_js = f"Shiny.setInputValue('{'d1' if pid.startswith('d1') else 'd2'}_select_similar','{pid}',{{priority:'event'}})"
            # Actually we want to open the modal directly — reuse the sel reactive
            open_js = f"Shiny.setInputValue('wl_open_player','{pid}',{{priority:'event'}})"
            cards.append(
                ui.div(
                    {"class": "wl-card", "onclick": open_js},
                    # Remove button (stops click propagating to card)
                    ui.tags.button(
                        {"class": "wl-remove",
                         "title": "Remove from watchlist",
                         "onclick": f"event.stopPropagation();Shiny.setInputValue('toggle_watchlist','{pid}',{{priority:'event'}})"},
                        "★"),
                    ui.div(r["name"], class_="wl-card-name"),
                    ui.div(
                        ui.span(r["pos"], class_="pos-badge",
                                style=f"color:{pc_};border-color:{pc_}"),
                        ui.span(r["team"]),
                        ui.span(f"· {r['cls']} · {div_}", style="color:var(--ink-3)"),
                        class_="wl-card-meta"),
                    ui.div({"class": "wl-card-stats"},
                           ui.div(ui.div(f"{r['ppg']:.1f}", class_="n"),
                                  ui.div("PPG", class_="l"), class_="wl-stat"),
                           ui.div(ui.div(f"{r['rpg']:.1f}", class_="n"),
                                  ui.div("RPG", class_="l"), class_="wl-stat"),
                           ui.div(ui.div(f"{r['apg']:.1f}", class_="n"),
                                  ui.div("APG", class_="l"), class_="wl-stat"),
                           ui.div(ui.div(f"{r['fg']*100:.0f}%", class_="n"),
                                  ui.div("FG%", class_="l"), class_="wl-stat")),
                ))
        n   = len(wl)
        vis = "inline-block" if n else "none"
        js  = f"var b=document.getElementById('wl-badge');if(b){{b.textContent='{n}';b.style.display='{vis}';}}"
        return ui.div(
            ui.tags.script(js),
            ui.div({"class": "wl-grid"}, *cards))

    # Open player modal from watchlist card click
    @reactive.effect
    @reactive.event(input.wl_open_player)
    def _wl_open_player():
        pid = input.wl_open_player()
        if not pid: return
        import random
        modal_req.set((pid, random.random()))


app = App(app_ui, server, static_assets=HERE / "www")