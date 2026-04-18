"""
CSV Plotter & Visualizer — a free, open-source web app for interactive CSV plotting.

Features:
  • Upload CSV (or paste a URL)
  • Multiple subplots, 2D and 3D, with multiple line groups each
  • MATLAB-style interactive toolbar: zoom, pan, box-zoom, autoscale, PNG download
  • Optional animation (frame-by-frame playback with slider)
  • Configurable titles, axis labels, colors, line styles
  • Everything runs in the browser — no install needed for users

Deploy: push this repo to GitHub -> share.streamlit.io -> "New app" -> done.
"""

from __future__ import annotations

import io
import os
import tempfile
from typing import List, Literal

import imageio.v2 as imageio
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

# AI agent is imported lazily inside handlers so the app still runs
# even if the google-genai package is unavailable.

# --------------------------------------------------------------------------- #
#  Page setup                                                                  #
# --------------------------------------------------------------------------- #

st.set_page_config(
    page_title="CSV Plotter",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
      .block-container { padding-top: 1.5rem; padding-bottom: 2rem; max-width: 1400px; }
      h1 { margin-bottom: 0.25rem; }
      .subtle { color: #888; font-size: 0.9rem; }
      div[data-testid="stExpander"] { border: 1px solid #2a2a2a; border-radius: 6px; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("📊 CSV Plotter")
st.markdown(
    '<p class="subtle">Upload a CSV, configure any number of interactive plots, '
    "zoom / pan / download like MATLAB. Free and open source.</p>",
    unsafe_allow_html=True,
)

# --------------------------------------------------------------------------- #
#  Data model (imported from models.py, shared with ai_agent.py)              #
# --------------------------------------------------------------------------- #

from models import LineGroup, SubplotConfig


# --------------------------------------------------------------------------- #
#  Session state init                                                          #
# --------------------------------------------------------------------------- #

if "df" not in st.session_state:
    st.session_state.df = None
if "filename" not in st.session_state:
    st.session_state.filename = None
if "subplots" not in st.session_state:
    st.session_state.subplots: List[SubplotConfig] = []

# --------------------------------------------------------------------------- #
#  Sidebar — data source + global controls                                     #
# --------------------------------------------------------------------------- #

with st.sidebar:
    st.header("1. Load data")
    source = st.radio(
        "Source", ["Upload file", "Paste URL", "Try sample data"],
        index=0, key="source_radio",
    )

    df = None
    if source == "Upload file":
        upload = st.file_uploader(
            "Data file",
            type=["csv", "tsv", "txt", "xlsx", "xls"],
            key="csv_uploader",
            help="Supports CSV, TSV, TXT, and Excel (.xlsx / .xls)",
        )
        if upload is not None:
            name = upload.name.lower()
            try:
                if name.endswith((".xlsx", ".xls")):
                    # Multi-sheet Excel — let the user pick a sheet
                    xls = pd.ExcelFile(upload)
                    if len(xls.sheet_names) > 1:
                        sheet = st.selectbox(
                            "Sheet", xls.sheet_names, key="xlsx_sheet",
                        )
                    else:
                        sheet = xls.sheet_names[0]
                    df = pd.read_excel(xls, sheet_name=sheet)
                    st.session_state.filename = f"{upload.name} [{sheet}]"
                else:
                    sep = "\t" if name.endswith((".tsv", ".txt")) else ","
                    df = pd.read_csv(upload, sep=sep)
                    st.session_state.filename = upload.name
            except Exception as e:
                st.error(f"Could not read file: {e}")
    elif source == "Paste URL":
        url = st.text_input("CSV URL", placeholder="https://example.com/data.csv", key="csv_url")
        if url and st.button("Fetch"):
            try:
                df = pd.read_csv(url)
                st.session_state.filename = url.rsplit("/", 1)[-1]
            except Exception as e:
                st.error(f"Could not fetch: {e}")
    else:  # sample
        if st.button("Load sample"):
            t = np.linspace(0, 20, 500)
            df = pd.DataFrame({
                "t": t,
                "sin": np.sin(t),
                "cos": np.cos(t),
                "damped": np.exp(-t / 10) * np.sin(t),
                "noise": np.random.normal(0, 0.15, t.size),
                "spiral_x": np.cos(t) * t / 5,
                "spiral_y": np.sin(t) * t / 5,
                "spiral_z": t / 2,
            })
            st.session_state.filename = "sample_data.csv"

    if df is not None:
        st.session_state.df = df

    if st.session_state.df is not None:
        st.success(f"Loaded **{st.session_state.filename}**  \n"
                   f"{len(st.session_state.df):,} rows × {len(st.session_state.df.columns)} cols")
        if st.button("Clear data", type="secondary"):
            st.session_state.df = None
            st.session_state.subplots = []
            st.rerun()

    st.divider()
    st.header("Display")
    layout_mode = st.radio(
        "Layout",
        ["One plot per row", "Grid (2 per row)", "Combined (shared figure)"],
        index=0,
        key="layout_radio",
    )
    plot_height = st.slider("Plot height (px)", 300, 900, 500, 50, key="plot_height_slider")
    theme_choice = st.radio(
        "Plot theme",
        ["Dark", "Light"],
        index=0,
        key="theme_radio",
        horizontal=True,
    )
    PLOTLY_TEMPLATE = "plotly_dark" if theme_choice == "Dark" else "plotly_white"
    st.session_state.plotly_template = PLOTLY_TEMPLATE

    download_format = st.selectbox(
        "Image download format",
        ["png", "svg", "jpeg", "webp"],
        index=0,
        key="download_format_select",
        help="Click the camera icon in any plot's toolbar to download in this format.",
    )

    st.divider()
    st.header("🤖 AI agent (optional)")
    st.caption(
        "Uses Google Gemini (free tier, 1000 req/day). "
        "[Get a free key →](https://aistudio.google.com/apikey)"
    )
    user_key_input = st.text_input(
        "Your Gemini API key (optional)",
        type="password",
        help="Leave blank to use the app's shared key (if configured). "
             "Your key is kept in session only.",
        key="user_api_key",
    )
    with st.expander("How to create a free Gemini API key", expanded=False):
        st.markdown(
            "1. Open https://aistudio.google.com/apikey\n"
            "2. Sign in with your Google account\n"
            "3. Click **Create API key**\n"
            "4. Copy the key and paste it above\n"
            "5. Keep it private and do not commit it to GitHub"
        )

    st.divider()
    with st.expander("About / deploy"):
        st.markdown(
            "- **Interactive toolbar** on every plot: box-zoom, pan, autoscale, "
            "download PNG (hover the chart, toolbar appears top-right).\n"
            "- Double-click a chart to reset zoom.\n"
            "- To deploy your own copy: fork the repo and push to "
            "[Streamlit Community Cloud](https://share.streamlit.io)."
        )

# --------------------------------------------------------------------------- #
#  Site-wide light-theme override                                              #
#  (Streamlit config.toml sets dark as the base; when the user picks Light,    #
#   we inject CSS that repaints the whole UI, not just the plots.)             #
# --------------------------------------------------------------------------- #

if st.session_state.get("theme_radio") == "Light":
    st.markdown(
        """
        <style>
        /* Main background + body text */
        [data-testid="stAppViewContainer"],
        [data-testid="stMain"],
        [data-testid="stHeader"] {
            background-color: #FFFFFF !important;
            color: #1E1E1E !important;
        }
        /* Sidebar */
        [data-testid="stSidebar"] {
            background-color: #F5F7FA !important;
        }
        [data-testid="stSidebar"] * {
            color: #1E1E1E !important;
        }
        [data-testid="stSidebar"] [data-baseweb="select"] * {
            color: #1E1E1E !important;
        }
        [data-testid="stSidebar"] [data-baseweb="radio"] * {
            color: #1E1E1E !important;
        }
        /* Top toolbar (Deploy / menu) */
        [data-testid="stToolbar"],
        [data-testid="stToolbar"] * {
            color: #1E1E1E !important;
            fill: #1E1E1E !important;
        }
        [data-testid="stToolbar"] button {
            background-color: #FFFFFF !important;
            border: 1px solid #D0D5DB !important;
        }
        [data-testid="stToolbar"] button:disabled {
            background-color: #F3F4F6 !important;
            color: #6B7280 !important;
            border-color: #D1D5DB !important;
        }
        /* All readable text in the main area */
        [data-testid="stAppViewContainer"] p,
        [data-testid="stAppViewContainer"] h1,
        [data-testid="stAppViewContainer"] h2,
        [data-testid="stAppViewContainer"] h3,
        [data-testid="stAppViewContainer"] h4,
        [data-testid="stAppViewContainer"] h5,
        [data-testid="stAppViewContainer"] label,
        [data-testid="stAppViewContainer"] li,
        .stMarkdown, .stMarkdown p, .stMarkdown li,
        .stCaption, small {
            color: #1E1E1E !important;
        }
        /* Subtle / muted text */
        .subtle, [data-testid="stCaptionContainer"] {
            color: #666 !important;
        }
        /* Text input, text area, number input */
        [data-testid="stTextInput"] input,
        [data-testid="stTextArea"] textarea,
        [data-testid="stNumberInput"] input,
        [data-testid="stChatInput"] textarea {
            background-color: #FFFFFF !important;
            color: #1E1E1E !important;
            border-color: #D0D5DB !important;
        }
        /* Select box + multiselect */
        [data-baseweb="select"] > div,
        [data-baseweb="select"] div[role="button"] {
            background-color: #FFFFFF !important;
            color: #1E1E1E !important;
            border-color: #D0D5DB !important;
        }
        [data-baseweb="tag"] {
            background-color: #E0E4E9 !important;
            color: #1E1E1E !important;
        }
        /* File uploader */
        [data-testid="stFileUploader"] section {
            background-color: #F7F9FC !important;
            border: 1px dashed #B0B6BE !important;
        }
        [data-testid="stFileUploader"] [data-testid="stFileUploaderDropzone"] {
            background-color: #F7F9FC !important;
            border: 1px dashed #B0B6BE !important;
        }
        [data-testid="stFileUploader"] [data-testid="stFileUploaderFile"] {
            background-color: #FFFFFF !important;
            border: 1px solid #D0D5DB !important;
        }
        [data-testid="stFileUploader"] [data-testid="stFileUploaderFile"] * {
            color: #1E1E1E !important;
            fill: #1E1E1E !important;
        }
        [data-testid="stFileUploader"] div {
            color: #1E1E1E !important;
        }
        [data-testid="stFileUploader"] svg {
            fill: #1E1E1E !important;
        }
        [data-testid="stFileUploader"] small,
        [data-testid="stFileUploader"] button {
            color: #1E1E1E !important;
        }
        /* Expanders */
        [data-testid="stExpander"] {
            background-color: #F7F9FC !important;
            border: 1px solid #E0E4E9 !important;
        }
        [data-testid="stExpander"] summary {
            background-color: #F7F9FC !important;
            border-bottom: 1px solid #E0E4E9 !important;
        }
        [data-testid="stExpander"] summary,
        [data-testid="stExpander"] details {
            color: #1E1E1E !important;
        }
        /* Metrics */
        [data-testid="stMetric"] {
            background-color: #F7F9FC !important;
        }
        [data-testid="stMetricValue"], [data-testid="stMetricLabel"] {
            color: #1E1E1E !important;
        }
        /* Buttons (default + primary) */
        .stButton button, [data-testid="stDownloadButton"] button {
            background-color: #FFFFFF !important;
            color: #1E1E1E !important;
            border: 1px solid #D0D5DB !important;
        }
        .stButton button:disabled, [data-testid="stDownloadButton"] button:disabled {
            background-color: #F3F4F6 !important;
            color: #6B7280 !important;
            border-color: #D1D5DB !important;
        }
        .stButton button[kind="primary"] {
            background-color: #4A9EFF !important;
            color: #FFFFFF !important;
            border-color: #4A9EFF !important;
        }
        .stButton button:hover {
            border-color: #4A9EFF !important;
        }
        /* Tabs */
        .stTabs [data-baseweb="tab-list"] {
            background-color: transparent !important;
            border-bottom: 1px solid #E0E4E9 !important;
        }
        .stTabs [data-baseweb="tab"] {
            color: #1E1E1E !important;
        }
        /* Alerts: info / success / warning / error */
        [data-testid="stAlert"] {
            color: #1E1E1E !important;
        }
        /* Data tables */
        [data-testid="stDataFrame"] {
            background-color: #FFFFFF !important;
        }
        /* Slider / radio / checkbox labels */
        [data-testid="stWidgetLabel"],
        [data-testid="stRadio"] label,
        [data-testid="stCheckbox"] label,
        [data-testid="stSlider"] label {
            color: #1E1E1E !important;
        }
        /* Checkbox / toggle visual contrast */
        [data-testid="stCheckbox"] input + div,
        [data-testid="stCheckbox"] div[role="checkbox"] {
            border-color: #AAB2BD !important;
            background-color: #FFFFFF !important;
        }
        [data-testid="stCheckbox"] input:checked + div,
        [data-testid="stCheckbox"] div[role="checkbox"][aria-checked="true"] {
            background-color: #4A9EFF !important;
            border-color: #4A9EFF !important;
        }
        [data-testid="stToggle"] [data-baseweb="switch"] > div {
            background-color: #D0D5DB !important;
        }
        [data-testid="stToggle"] [data-baseweb="switch"] input:checked + div {
            background-color: #4A9EFF !important;
        }
        [data-testid="stToggle"] label,
        [data-testid="stCheckbox"] label {
            color: #1E1E1E !important;
        }
        [data-testid="stToggle"] [data-baseweb="switch"] * {
            color: #1E1E1E !important;
        }
        /* Slider contrast in light mode */
        [data-testid="stSlider"] [data-baseweb="slider"] > div,
        [data-testid="stSlider"] [data-baseweb="slider"] * {
            color: #1E1E1E !important;
        }
        /* Code blocks */
        code, pre {
            background-color: #F0F2F6 !important;
            color: #1E1E1E !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

# --------------------------------------------------------------------------- #
#  Main area                                                                   #
# --------------------------------------------------------------------------- #

df = st.session_state.df

if df is None:
    st.info("👈 Load a CSV from the sidebar to get started, or try the sample data.")
    st.markdown(
        "### What this app does\n"
        "- Drop in a CSV and pick columns — get fully interactive plots\n"
        "- 2D line/scatter and 3D line/scatter with multiple series per subplot\n"
        "- MATLAB-style zoom, pan, box-zoom, autoscale, PNG export\n"
        "- Optional animation with a frame slider\n"
        "- Nothing is stored server-side; your data stays in the session"
    )
    st.stop()

# ----- Data preview ----- #
with st.expander(f"Preview — {st.session_state.filename}", expanded=False):
    st.dataframe(df.head(200), width="stretch", height=250)
    c1, c2, c3 = st.columns(3)
    c1.metric("Rows", f"{len(df):,}")
    c2.metric("Columns", len(df.columns))
    c3.metric("Numeric cols", len(df.select_dtypes(include="number").columns))

columns = list(df.columns)
numeric_cols = list(df.select_dtypes(include="number").columns)

# --------------------------------------------------------------------------- #
#  AI agent section                                                            #
# --------------------------------------------------------------------------- #

def _get_api_key() -> str | None:
    from ai_agent import resolve_api_key
    return resolve_api_key(st.session_state.get("user_api_key"))


def _subplots_summary_for_ai() -> str:
    if not st.session_state.subplots:
        return ""
    lines = []
    for i, c in enumerate(st.session_state.subplots, 1):
        grp_txt = " | ".join(
            f"X={g.x_col}, Y={g.y_cols}" + (f", Z={g.z_cols}" if g.z_cols else "")
            for g in c.groups
        )
        lines.append(f"{i}. [{c.plot_type}] '{c.title}' — {grp_txt}")
    return "\n".join(lines)


def _snapshot_for_undo(label: str):
    """Push current subplots onto the undo stack before a mutating operation."""
    import copy
    history = st.session_state.setdefault("undo_stack", [])
    history.append((label, copy.deepcopy(st.session_state.subplots)))
    # Cap history to last 15 actions
    if len(history) > 15:
        history.pop(0)


def _add_ai_plan_to_subplots(plan):
    """Apply an AI plan to the subplot list. Returns (added, skipped, notes)."""
    from ai_agent import ai_to_app_config
    _snapshot_for_undo("AI plots added")
    added, skipped = 0, 0
    all_notes: List[str] = []
    for ai_cfg in plan.subplots:
        app_cfg, notes = ai_to_app_config(ai_cfg, columns, numeric_cols)
        all_notes.extend(notes)
        if app_cfg is not None:
            st.session_state.subplots.append(app_cfg)
            added += 1
        else:
            skipped += 1
    return added, skipped, all_notes


def _flash(kind: str, text: str):
    """Queue a message to display on the next script run (survives st.rerun)."""
    st.session_state.setdefault("flash_queue", []).append((kind, text))


# Drain any flash messages queued from a previous run
for _kind, _text in st.session_state.pop("flash_queue", []):
    {"success": st.success, "warning": st.warning,
     "error": st.error, "info": st.info}.get(_kind, st.info)(_text)


st.header("2. Ask the AI (optional)")

api_key = _get_api_key()
if not api_key:
    st.info(
        "Enter your free Gemini API key in the sidebar to enable the AI agent, "
        "or skip straight to manual plot building below."
    )
else:
    tab_chat, tab_suggest, tab_summary = st.tabs(
        ["💬 Ask for plots", "💡 Suggest plots", "📋 Summarize data"]
    )

    def _handle_ai_result(plan, action_label: str):
        """Shared handler: add plots, show result summary, show diagnostics."""
        added, skipped, notes = _add_ai_plan_to_subplots(plan)
        with st.expander(f"AI reasoning ({action_label})", expanded=True):
            st.write(plan.reasoning)
        if added:
            st.success(f"✅ Added {added} plot(s)." +
                       (f" Skipped {skipped} (see details below)." if skipped else ""))
        else:
            st.error(
                f"AI returned {len(plan.subplots)} plot(s), but none survived validation. "
                "See 'Details' below — this usually means the AI picked columns "
                "that don't exist in your data."
            )
        if notes:
            with st.expander(f"Details ({len(notes)} messages)", expanded=not added):
                for n in notes:
                    st.text(f"• {n}")

    with tab_chat:
        st.caption(
            "Describe what you want to see. E.g. *'plot temperature vs time'*, "
            "*'compare A and B'*, *'show me anything interesting'*."
        )
        user_q = st.text_area(
            "Your request",
            placeholder="Plot column X versus column Y on a log scale...",
            key="ai_user_query",
            height=80,
        )
        if st.button("🚀 Generate plots", type="primary", disabled=not user_q.strip(),
                     key="ai_generate_btn"):
            with st.spinner("Thinking..."):
                try:
                    from ai_agent import answer_plot_request
                    plan = answer_plot_request(
                        df, user_q, api_key, _subplots_summary_for_ai()
                    )
                    _handle_ai_result(plan, "chat request")
                except Exception as e:
                    st.error(f"AI request failed: {e}")

    with tab_suggest:
        st.caption("Let the AI propose plots based on the data alone.")
        n_suggest = st.slider("How many?", 2, 6, 4, key="ai_n_suggest")
        if st.button("💡 Suggest plots", key="ai_suggest_btn"):
            with st.spinner("Exploring the data..."):
                try:
                    from ai_agent import suggest_plots
                    plan = suggest_plots(df, api_key, n=n_suggest)
                    _handle_ai_result(plan, "suggested plots")
                except Exception as e:
                    st.error(f"AI request failed: {e}")

    with tab_summary:
        st.caption("Get a plain-language summary of what's in the CSV.")
        if st.button("📋 Analyze dataset", key="ai_analyze_btn"):
            with st.spinner("Analyzing..."):
                try:
                    from ai_agent import summarize_dataset
                    summary = summarize_dataset(df, api_key)
                    st.session_state.last_summary = summary
                except Exception as e:
                    st.error(f"AI request failed: {e}")

        if "last_summary" in st.session_state:
            s = st.session_state.last_summary
            st.markdown(f"**Overview.** {s.overview}")
            st.markdown("**Observations:**")
            for obs in s.observations:
                st.markdown(f"- {obs}")
            st.markdown("**Questions you could ask:**")
            for q in s.suggested_questions:
                st.markdown(f"- *{q}*")

st.divider()

# ----- Subplot builder ----- #
st.header("3. Build plots manually")

bcol1, bcol2, bcol3, bcol4 = st.columns([1, 1, 1, 2])
if bcol1.button("➕ Add 2D subplot", width="stretch", key="add_2d_btn"):
    _snapshot_for_undo("Add 2D subplot")
    st.session_state.subplots.append(
        SubplotConfig(
            title=f"Plot {len(st.session_state.subplots) + 1}",
            plot_type="2D",
            chart_type="line",
            groups=[LineGroup(x_col=columns[0],
                              y_cols=[numeric_cols[0]] if numeric_cols else [])],
        )
    )
if bcol2.button("➕ Add 3D subplot", width="stretch", key="add_3d_btn"):
    _snapshot_for_undo("Add 3D subplot")
    st.session_state.subplots.append(
        SubplotConfig(
            title=f"Plot {len(st.session_state.subplots) + 1}",
            plot_type="3D",
            groups=[LineGroup(
                x_col=numeric_cols[0] if numeric_cols else columns[0],
                y_cols=[numeric_cols[1]] if len(numeric_cols) > 1 else [],
                z_cols=[numeric_cols[2]] if len(numeric_cols) > 2 else [],
            )],
        )
    )

undo_stack = st.session_state.get("undo_stack", [])
undo_label = f"↶ Undo ({undo_stack[-1][0]})" if undo_stack else "↶ Undo"
if bcol3.button(undo_label, width="stretch", disabled=not undo_stack,
                key="undo_btn"):
    _, prev = undo_stack.pop()
    st.session_state.subplots = prev
    _flash("info", "Undone.")
    st.rerun()

if bcol4.button("🗑 Remove all", width="stretch", key="remove_all_btn"):
    _snapshot_for_undo("Remove all")
    st.session_state.subplots = []
    st.rerun()

if not st.session_state.subplots:
    st.info("Add a 2D or 3D subplot to begin.")
    st.stop()

# ----- Per-subplot editor ----- #
for i, cfg in enumerate(st.session_state.subplots):
    with st.expander(f"**#{i + 1}** — {cfg.title}  ·  {cfg.plot_type}", expanded=True):
        c_title, c_mode, c_anim, c_del = st.columns([3, 2, 1, 1])
        cfg.title = c_title.text_input("Title", cfg.title, key=f"title_{i}")
        if cfg.plot_type == "2D":
            chart_opts = ["line", "scatter", "bar", "histogram"]
            current_chart = cfg.chart_type if cfg.chart_type in chart_opts else "line"
            cfg.chart_type = c_mode.selectbox(
                "2D chart",
                chart_opts,
                index=chart_opts.index(current_chart),
                key=f"chart_type_{i}",
            )
            if cfg.chart_type in ["line", "scatter"]:
                cfg.mode = c_mode.selectbox(
                    "Mode",
                    ["lines", "markers", "lines+markers"],
                    index=["lines", "markers", "lines+markers"].index(cfg.mode),
                    key=f"mode_{i}",
                )
                if cfg.chart_type == "scatter" and cfg.mode == "lines":
                    cfg.mode = "markers"
        else:
            cfg.mode = c_mode.selectbox(
                "Mode",
                ["lines", "markers", "lines+markers"],
                index=["lines", "markers", "lines+markers"].index(cfg.mode),
                key=f"mode_{i}",
            )
        cfg.animated = c_anim.toggle("Animate", cfg.animated, key=f"anim_{i}")
        if cfg.animated:
            axis_mode_label = c_anim.selectbox(
                "Anim axes",
                ["Autoscale", "Fixed full-range"],
                index=0 if cfg.animation_axis_mode == "auto" else 1,
                key=f"anim_axis_mode_{i}",
            )
            cfg.animation_axis_mode = "auto" if axis_mode_label == "Autoscale" else "fixed"
        if c_del.button("❌", key=f"del_{i}", help="Delete this subplot"):
            _snapshot_for_undo(f"Delete plot #{i+1}")
            st.session_state.subplots.pop(i)
            st.rerun()

        ax_c1, ax_c2, ax_c3, ax_c4 = st.columns(4)
        cfg.x_label = ax_c1.text_input("X label", cfg.x_label, key=f"xlab_{i}")
        cfg.y_label = ax_c2.text_input("Y label", cfg.y_label, key=f"ylab_{i}")
        if cfg.plot_type == "3D":
            cfg.z_label = ax_c3.text_input("Z label", cfg.z_label, key=f"zlab_{i}")
        else:
            cfg.log_x = ax_c3.checkbox("log X", cfg.log_x, key=f"logx_{i}")
            cfg.log_y = ax_c4.checkbox("log Y", cfg.log_y, key=f"logy_{i}")

        st.markdown("**Line groups**")
        for g_idx, grp in enumerate(cfg.groups):
            g_cols = st.columns([2, 4, 2, 0.8] if cfg.plot_type == "2D"
                                else [2, 3, 3, 0.8])
            grp.x_col = g_cols[0].selectbox(
                f"X", columns,
                index=columns.index(grp.x_col) if grp.x_col in columns else 0,
                key=f"x_{i}_{g_idx}",
            )
            grp.y_cols = g_cols[1].multiselect(
                "Y", numeric_cols,
                default=[c for c in grp.y_cols if c in numeric_cols],
                key=f"y_{i}_{g_idx}",
            )
            if cfg.plot_type == "3D":
                grp.z_cols = g_cols[2].multiselect(
                    "Z", numeric_cols,
                    default=[c for c in grp.z_cols if c in numeric_cols],
                    key=f"z_{i}_{g_idx}",
                )
            else:
                grp.name_prefix = g_cols[2].text_input(
                    "Label prefix", grp.name_prefix, key=f"pfx_{i}_{g_idx}",
                )
            if g_cols[-1].button("✕", key=f"delg_{i}_{g_idx}", help="Remove group"):
                cfg.groups.pop(g_idx)
                st.rerun()

        if st.button("+ Add group", key=f"addg_{i}"):
            cfg.groups.append(LineGroup(x_col=columns[0]))
            st.rerun()

# --------------------------------------------------------------------------- #
#  Plotting helpers                                                            #
# --------------------------------------------------------------------------- #

PALETTE = px.colors.qualitative.Plotly


def _color(i: int) -> str:
    return PALETTE[i % len(PALETTE)]


def _build_2d_traces(cfg: SubplotConfig, df: pd.DataFrame) -> List:
    traces, color_i = [], 0
    chart_type = getattr(cfg, "chart_type", "line")
    for grp in cfg.groups:
        for y in grp.y_cols:
            label = f"{grp.name_prefix}{y}" if grp.name_prefix else y
            if chart_type == "bar":
                traces.append(go.Bar(
                    x=df[grp.x_col], y=df[y], name=label,
                    marker=dict(color=_color(color_i)),
                ))
            elif chart_type == "histogram":
                traces.append(go.Histogram(
                    x=df[y], name=label,
                    marker=dict(color=_color(color_i)), opacity=0.75,
                    nbinsx=min(80, max(10, int(len(df) ** 0.5))),
                ))
            else:
                effective_mode = cfg.mode
                if chart_type == "scatter" and effective_mode == "lines":
                    effective_mode = "markers"
                traces.append(go.Scatter(
                    x=df[grp.x_col], y=df[y], mode=effective_mode, name=label,
                    line=dict(color=_color(color_i), width=2),
                    marker=dict(color=_color(color_i), size=5),
                ))
            color_i += 1
    return traces


def _build_3d_traces(cfg: SubplotConfig, df: pd.DataFrame) -> List[go.Scatter3d]:
    traces, color_i = [], 0
    for grp in cfg.groups:
        n = min(len(grp.y_cols), len(grp.z_cols))
        for k in range(n):
            y, z = grp.y_cols[k], grp.z_cols[k]
            x = grp.x_col
            label = f"{grp.name_prefix}{x}/{y}/{z}" if grp.name_prefix else f"{x}/{y}/{z}"
            traces.append(go.Scatter3d(
                x=df[x], y=df[y], z=df[z], mode=cfg.mode, name=label,
                line=dict(color=_color(color_i), width=3),
                marker=dict(color=_color(color_i), size=3),
            ))
            color_i += 1
    return traces


def _apply_2d_layout(fig: go.Figure, cfg: SubplotConfig, height: int):
    is_light = st.session_state.get("plotly_template", "plotly_dark") == "plotly_white"
    chart_type = getattr(cfg, "chart_type", "line")
    if chart_type == "histogram":
        default_x = cfg.x_label or (cfg.groups[0].y_cols[0] if cfg.groups and cfg.groups[0].y_cols else "Value")
        default_y = cfg.y_label or "Count"
    else:
        default_x = cfg.x_label or (cfg.groups[0].x_col if cfg.groups else "")
        default_y = cfg.y_label or ""
    fig.update_layout(
        title=cfg.title, height=height,
        xaxis_title=default_x,
        yaxis_title=default_y,
        template=st.session_state.get("plotly_template", "plotly_dark"),
        paper_bgcolor="#FFFFFF" if is_light else "#0E1117",
        plot_bgcolor="#FFFFFF" if is_light else "#0E1117",
        font=dict(color="#1E1E1E" if is_light else "#FAFAFA"),
        margin=dict(l=60, r=30, t=60, b=50),
        legend=dict(orientation="h", y=-0.15),
        barmode="group",
    )
    if cfg.log_x:
        fig.update_xaxes(type="log")
    if cfg.log_y:
        fig.update_yaxes(type="log")
    # Ensure first render is fully zoomed out to full data extents.
    fig.update_xaxes(autorange=True)
    fig.update_yaxes(autorange=True)


def _apply_3d_layout(fig: go.Figure, cfg: SubplotConfig, height: int):
    is_light = st.session_state.get("plotly_template", "plotly_dark") == "plotly_white"
    bg = "#FFFFFF" if is_light else "#0E1117"
    fig.update_layout(
        title=cfg.title, height=height,
        template=st.session_state.get("plotly_template", "plotly_dark"),
        paper_bgcolor=bg,
        font=dict(color="#1E1E1E" if is_light else "#FAFAFA"),
        margin=dict(l=0, r=0, t=50, b=0),
        scene=dict(
            xaxis=dict(title=cfg.x_label or "X", autorange=True),
            yaxis=dict(title=cfg.y_label or "Y", autorange=True),
            zaxis=dict(title=cfg.z_label or "Z", autorange=True),
            bgcolor=bg,
        ),
        legend=dict(orientation="h", y=-0.05),
    )


def build_static(cfg: SubplotConfig, df: pd.DataFrame, height: int) -> go.Figure:
    fig = go.Figure()
    if cfg.plot_type == "2D":
        for t in _build_2d_traces(cfg, df):
            fig.add_trace(t)
        _apply_2d_layout(fig, cfg, height)
    else:
        for t in _build_3d_traces(cfg, df):
            fig.add_trace(t)
        _apply_3d_layout(fig, cfg, height)
    return fig


def _apply_animation_axis_mode(
    fig: go.Figure,
    cfg: SubplotConfig,
    df_scope: pd.DataFrame,
    axis_mode: Literal["auto", "fixed"],
):
    def _col_range(cols: List[str]):
        vals = pd.concat([df_scope[c] for c in cols if c in df_scope.columns], axis=0)
        vals = pd.to_numeric(vals, errors="coerce").dropna()
        if vals.empty:
            return None
        lo, hi = float(vals.min()), float(vals.max())
        pad = (hi - lo) * 0.05 if hi > lo else max(abs(hi), 1.0) * 0.05
        return [lo - pad, hi + pad]

    chart_type = getattr(cfg, "chart_type", "line")

    if axis_mode == "fixed":
        all_x = [g.x_col for g in cfg.groups]
        all_y = [c for g in cfg.groups for c in g.y_cols]
        all_z = [c for g in cfg.groups for c in g.z_cols]
        # Histogram x-range should come from selected Y columns because bars are value bins.
        x_range = _col_range(all_y if chart_type == "histogram" else all_x)
        y_range = _col_range(all_y)
        z_range = _col_range(all_z) if cfg.plot_type == "3D" else None

        if cfg.plot_type == "2D":
            if x_range and not cfg.log_x:
                fig.update_xaxes(range=x_range, autorange=False)
            if y_range and not cfg.log_y and chart_type != "histogram":
                fig.update_yaxes(range=y_range, autorange=False)
        else:
            scene_update = {}
            if x_range:
                scene_update["xaxis"] = dict(range=x_range, autorange=False)
            if y_range:
                scene_update["yaxis"] = dict(range=y_range, autorange=False)
            if z_range:
                scene_update["zaxis"] = dict(range=z_range, autorange=False)
            if scene_update:
                fig.update_layout(scene=scene_update)
    else:
        if cfg.plot_type == "2D":
            if not cfg.log_x:
                fig.update_xaxes(autorange=True)
            if not cfg.log_y:
                fig.update_yaxes(autorange=True)


def build_animated(cfg: SubplotConfig, df: pd.DataFrame, height: int,
                   n_frames: int = 60,
                   axis_mode: Literal["auto", "fixed"] = "auto") -> go.Figure:
    """Frame-by-frame animation: each frame reveals data up to index k."""
    N = len(df)
    if N == 0:
        return build_static(cfg, df, height)
    n_frames = min(n_frames, N)
    step = max(1, N // n_frames)
    frame_ends = list(range(step, N + 1, step))
    if frame_ends[-1] != N:
        frame_ends.append(N)

    # initial frame: first segment
    first_df = df.iloc[:frame_ends[0]]
    fig = build_static(cfg, first_df, height)

    _apply_animation_axis_mode(fig, cfg, df, axis_mode)

    # build frames
    frames = []
    for k, end in enumerate(frame_ends):
        sub = df.iloc[:end]
        if cfg.plot_type == "2D":
            data = _build_2d_traces(cfg, sub)
        else:
            data = _build_3d_traces(cfg, sub)
        frames.append(go.Frame(data=data, name=str(k)))
    fig.frames = frames

    is_light = st.session_state.get("plotly_template", "plotly_dark") == "plotly_white"
    menu_bg = "#F0F2F6" if is_light else "#22252E"
    menu_fg = "#1E1E1E" if is_light else "#FAFAFA"
    fig.update_layout(
        updatemenus=[dict(
            type="buttons", showactive=False,
            x=0.02, y=1.12, xanchor="left", yanchor="top",
            bgcolor=menu_bg,
            bordercolor="#D0D5DB" if is_light else "#3A3F4B",
            borderwidth=1,
            font=dict(color=menu_fg),
            buttons=[
                dict(label="▶ Play", method="animate",
                     args=[None, {"frame": {"duration": 40, "redraw": True},
                                  "fromcurrent": True,
                                  "transition": {"duration": 0}}]),
                dict(label="⏸ Pause", method="animate",
                     args=[[None], {"frame": {"duration": 0, "redraw": False},
                                    "mode": "immediate",
                                    "transition": {"duration": 0}}]),
            ],
        )],
        sliders=[dict(
            active=0, y=0, x=0.1, len=0.85,
            bgcolor=menu_bg,
            bordercolor="#D0D5DB" if is_light else "#3A3F4B",
            borderwidth=1,
            currentvalue=dict(prefix="Frame: ", visible=True, font=dict(color=menu_fg)),
            steps=[dict(method="animate", label=str(k),
                        args=[[str(k)], {"frame": {"duration": 0, "redraw": True},
                                         "mode": "immediate"}])
                   for k in range(len(frames))],
        )],
    )
    return fig


def _animation_frame_ends(n_rows: int, max_frames: int) -> List[int]:
    if n_rows <= 0:
        return []
    n_frames = max(2, min(max_frames, n_rows))
    step = max(1, n_rows // n_frames)
    frame_ends = list(range(step, n_rows + 1, step))
    if frame_ends[-1] != n_rows:
        frame_ends.append(n_rows)
    return frame_ends


def _build_mp4_bytes(cfg: SubplotConfig, df: pd.DataFrame, height: int,
                     fps: int = 20, max_frames: int = 90,
                     axis_mode: Literal["auto", "fixed"] = "auto") -> bytes:
    """Render an animated subplot to MP4 bytes using Plotly image frames."""
    frame_ends = _animation_frame_ends(len(df), max_frames)
    if not frame_ends:
        raise ValueError("No data available to render animation.")

    width = 1200
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
            tmp_path = tmp.name

        with imageio.get_writer(
            tmp_path,
            format="FFMPEG",
            mode="I",
            fps=fps,
            codec="libx264",
            pixelformat="yuv420p",
        ) as writer:
            for end in frame_ends:
                fig = build_static(cfg, df.iloc[:end], height)
                _apply_animation_axis_mode(fig, cfg, df, axis_mode)
                png_bytes = fig.to_image(format="png", width=width, height=height, scale=1)
                writer.append_data(imageio.imread(io.BytesIO(png_bytes)))

        with open(tmp_path, "rb") as f:
            return f.read()
    except Exception as e:
        raise RuntimeError(
            f"Could not export MP4: {e}"
        ) from e
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass


def _build_gif_bytes(cfg: SubplotConfig, df: pd.DataFrame, height: int,
                     fps: int = 12, max_frames: int = 90,
                     axis_mode: Literal["auto", "fixed"] = "auto") -> bytes:
    frame_ends = _animation_frame_ends(len(df), max_frames)
    if not frame_ends:
        raise ValueError("No data available to render animation.")

    width = 960
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".gif", delete=False) as tmp:
            tmp_path = tmp.name

        with imageio.get_writer(tmp_path, mode="I", format="GIF", fps=fps, loop=0) as writer:
            for end in frame_ends:
                fig = build_static(cfg, df.iloc[:end], height)
                _apply_animation_axis_mode(fig, cfg, df, axis_mode)
                png_bytes = fig.to_image(format="png", width=width, height=height, scale=1)
                writer.append_data(imageio.imread(io.BytesIO(png_bytes)))

        with open(tmp_path, "rb") as f:
            return f.read()
    except Exception as e:
        raise RuntimeError(f"Could not export GIF: {e}") from e
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass


# --------------------------------------------------------------------------- #
#  Render                                                                      #
# --------------------------------------------------------------------------- #

st.header("4. Plots")


def _plotly_config(cfg_title: str) -> dict:
    """Per-plot config; uses the user's selected download format + filename."""
    safe_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in cfg_title)[:40]
    return {
        "displaylogo": False,
        "scrollZoom": True,
        "modeBarButtonsToAdd": ["drawline", "drawrect", "eraseshape"],
        "toImageButtonOptions": {
            "format": st.session_state.get("download_format_select", "png"),
            "scale": 2,
            "filename": safe_name or "plot",
        },
    }


def render_one(cfg: SubplotConfig, plot_idx: int):
    if not cfg.groups or all(not g.y_cols for g in cfg.groups):
        st.warning(f"'{cfg.title}' has no Y columns selected.")
        return
    fig = (build_animated(cfg, df, plot_height, axis_mode=cfg.animation_axis_mode)
           if cfg.animated else build_static(cfg, df, plot_height))
    st.plotly_chart(
        fig,
        width="stretch",
        config=_plotly_config(cfg.title),
        theme=None,
    )

    # Interactive HTML download (self-contained, preserves zoom/pan)
    html_str = fig.to_html(include_plotlyjs="cdn", full_html=True)
    safe_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in cfg.title)[:40]
    st.download_button(
        "⬇ Download interactive HTML",
        html_str,
        file_name=f"{safe_name or f'plot_{plot_idx}'}.html",
        mime="text/html",
        key=f"dl_html_{plot_idx}",
        help="Save this plot as an offline, interactive HTML file (zoom/pan works).",
    )

    if cfg.animated:
        with st.expander("⬇ Export animation", expanded=False):
            n_rows = len(df)
            r1, r2 = st.columns(2)
            start_row = r1.number_input(
                "Start row",
                min_value=1,
                max_value=max(1, n_rows),
                value=1,
                step=1,
                key=f"anim_start_{plot_idx}",
            )
            end_row = r2.number_input(
                "End row",
                min_value=1,
                max_value=max(1, n_rows),
                value=max(1, n_rows),
                step=1,
                key=f"anim_end_{plot_idx}",
            )

            if end_row <= start_row:
                st.warning("End row must be greater than start row.")
                return

            df_export = df.iloc[int(start_row) - 1:int(end_row)].copy()
            export_rows = len(df_export)

            q_col1, q_col2 = st.columns([2, 1])
            quality_preset = q_col1.selectbox(
                "Quality preset",
                ["Quick preview", "Balanced", "High quality"],
                index=1,
                key=f"anim_quality_{plot_idx}",
                help=(
                    "Quick preview: fastest, fewer frames. "
                    "Balanced: good default. "
                    "High quality: smoother output, larger files."
                ),
            )
            q_col2.caption(f"Rows selected: {export_rows}")

            preset_defaults = {
                "Quick preview": {"fps": 10, "frames": 40},
                "Balanced": {"fps": 16, "frames": 90},
                "High quality": {"fps": 24, "frames": 180},
            }

            fps_key = f"anim_fps_{plot_idx}"
            frames_key = f"anim_frames_{plot_idx}"
            preset_applied_key = f"anim_quality_applied_{plot_idx}"
            max_frames_cap = min(300, max(10, export_rows))

            # When preset changes, refresh slider defaults while still allowing manual overrides.
            if st.session_state.get(preset_applied_key) != quality_preset:
                st.session_state[fps_key] = preset_defaults[quality_preset]["fps"]
                st.session_state[frames_key] = min(preset_defaults[quality_preset]["frames"], max_frames_cap)
                st.session_state[preset_applied_key] = quality_preset
            else:
                # Keep any user-tuned values, but clamp to valid range for current selection.
                if frames_key in st.session_state:
                    st.session_state[frames_key] = max(10, min(max_frames_cap, st.session_state[frames_key]))
                if fps_key in st.session_state:
                    st.session_state[fps_key] = max(6, min(30, st.session_state[fps_key]))

            c1, c2, c3 = st.columns(3)
            fps = c1.slider("FPS", 6, 30, 16, key=f"anim_fps_{plot_idx}")
            max_frames = c2.slider(
                "Max frames",
                10,
                max_frames_cap,
                min(90, max(10, export_rows)),
                key=f"anim_frames_{plot_idx}",
            )
            export_format = c3.selectbox(
                "Format",
                ["MP4", "GIF"],
                index=0,
                key=f"anim_fmt_{plot_idx}",
            )

            axis_mode_label = st.selectbox(
                "Axis mode for export",
                ["Autoscale (moving axes)", "Fixed full-range (static axes)"],
                index=0 if cfg.animation_axis_mode == "auto" else 1,
                key=f"anim_export_axis_mode_{plot_idx}",
            )
            export_axis_mode = "auto" if "Autoscale" in axis_mode_label else "fixed"

            if st.button(f"Prepare {export_format}", key=f"prep_anim_{plot_idx}"):
                with st.spinner(f"Rendering {export_format}..."):
                    try:
                        if export_format == "MP4":
                            st.session_state[f"anim_bytes_{plot_idx}"] = _build_mp4_bytes(
                                cfg,
                                df_export,
                                plot_height,
                                fps=fps,
                                max_frames=max_frames,
                                axis_mode=export_axis_mode,
                            )
                            st.session_state[f"anim_mime_{plot_idx}"] = "video/mp4"
                            st.session_state[f"anim_ext_{plot_idx}"] = "mp4"
                        else:
                            st.session_state[f"anim_bytes_{plot_idx}"] = _build_gif_bytes(
                                cfg,
                                df_export,
                                plot_height,
                                fps=min(fps, 18),
                                max_frames=max_frames,
                                axis_mode=export_axis_mode,
                            )
                            st.session_state[f"anim_mime_{plot_idx}"] = "image/gif"
                            st.session_state[f"anim_ext_{plot_idx}"] = "gif"
                        st.success(
                            f"{export_format} is ready. Rows: {int(start_row)}-{int(end_row)} | "
                            f"Frames: up to {max_frames}"
                        )
                    except Exception as e:
                        st.error(
                            f"{e} Install extras with: pip install kaleido imageio imageio-ffmpeg"
                        )

            anim_bytes = st.session_state.get(f"anim_bytes_{plot_idx}")
            if anim_bytes:
                ext = st.session_state.get(f"anim_ext_{plot_idx}", "mp4")
                mime = st.session_state.get(f"anim_mime_{plot_idx}", "video/mp4")
                st.download_button(
                    f"⬇ Download animation as {ext.upper()}",
                    anim_bytes,
                    file_name=f"{safe_name or f'plot_{plot_idx}'}.{ext}",
                    mime=mime,
                    key=f"dl_anim_{plot_idx}",
                )


if layout_mode == "One plot per row":
    for i, cfg in enumerate(st.session_state.subplots):
        render_one(cfg, i)

elif layout_mode == "Grid (2 per row)":
    cols = st.columns(2)
    for i, cfg in enumerate(st.session_state.subplots):
        with cols[i % 2]:
            render_one(cfg, i)

else:  # Combined
    plots_2d = [c for c in st.session_state.subplots if c.plot_type == "2D"]
    plots_3d = [c for c in st.session_state.subplots if c.plot_type == "3D"]
    if plots_3d:
        st.info("3D subplots render individually in 'Combined' mode (Plotly limitation).")
        for i, cfg in enumerate(plots_3d):
            render_one(cfg, 1000 + i)  # offset idx to keep button keys unique
    if plots_2d:
        n = len(plots_2d)
        rows = (n + 1) // 2
        cols = 1 if n == 1 else 2
        fig = make_subplots(
            rows=rows, cols=cols,
            subplot_titles=[c.title for c in plots_2d],
        )
        for idx, cfg in enumerate(plots_2d):
            r, c = idx // cols + 1, idx % cols + 1
            for tr in _build_2d_traces(cfg, df):
                fig.add_trace(tr, row=r, col=c)
            chart_type = getattr(cfg, "chart_type", "line")
            if chart_type == "histogram":
                default_x = cfg.x_label or (cfg.groups[0].y_cols[0] if cfg.groups and cfg.groups[0].y_cols else "Value")
                default_y = cfg.y_label or "Count"
            else:
                default_x = cfg.x_label or cfg.groups[0].x_col
                default_y = cfg.y_label or ""
            fig.update_xaxes(title_text=default_x,
                             row=r, col=c,
                             type="log" if cfg.log_x else "linear")
            fig.update_yaxes(title_text=default_y, row=r, col=c,
                             type="log" if cfg.log_y else "linear")
        fig.update_layout(
            height=plot_height * rows,
            template=st.session_state.get("plotly_template", "plotly_dark"),
            paper_bgcolor="#FFFFFF" if st.session_state.get("plotly_template") == "plotly_white" else "#0E1117",
            plot_bgcolor="#FFFFFF" if st.session_state.get("plotly_template") == "plotly_white" else "#0E1117",
            font=dict(color="#1E1E1E" if st.session_state.get("plotly_template") == "plotly_white" else "#FAFAFA"),
            showlegend=True,
            margin=dict(l=60, r=30, t=60, b=50),
        )
        st.plotly_chart(
            fig,
            width="stretch",
            config=_plotly_config("combined"),
            theme=None,
        )

# --------------------------------------------------------------------------- #
#  Export row                                                                  #
# --------------------------------------------------------------------------- #

st.divider()
st.caption(
    "Each chart has a toolbar (top-right on hover) with zoom, pan, autoscale, and "
    "a camera icon that downloads in the format selected in the sidebar "
    f"(currently **{st.session_state.get('download_format_select', 'png').upper()}**). "
    "Double-click any chart to reset zoom. Use the HTML download for an interactive copy."
)

# Download the loaded data back as CSV
csv_buf = io.StringIO()
df.to_csv(csv_buf, index=False)
safe_base = (st.session_state.filename or "data.csv").split(" [")[0]  # drop sheet marker
st.download_button(
    "⬇ Download data as CSV",
    csv_buf.getvalue(),
    file_name=safe_base if safe_base.endswith(".csv") else f"{safe_base}.csv",
    mime="text/csv",
    key="dl_data_csv",
)
