"""
AI agent for the CSV Plotter.

Uses Google's Gemini API (free tier: Gemini 2.5 Flash-Lite, 1000 requests/day,
no credit card required). Returns *structured JSON* that maps onto the existing
SubplotConfig schema — no arbitrary code execution, safe for public deployment.

Get a free API key: https://aistudio.google.com/apikey

Alternative: set GEMINI_API_KEY env var, or add it to .streamlit/secrets.toml
as `GEMINI_API_KEY = "..."`.
"""

from __future__ import annotations

import json
import os
from typing import List, Literal, Optional

import pandas as pd
from pydantic import BaseModel, Field

# ---- Pydantic schemas: what we tell Gemini to return ----------------------- #

class AILineGroup(BaseModel):
    x_col: str = Field(description="Column name to use for the X axis")
    y_cols: List[str] = Field(
        default_factory=list,
        description="Column names to plot on the Y axis (must be numeric)",
    )
    z_cols: List[str] = Field(
        default_factory=list,
        description="For 3D plots only: column names for Z axis",
    )
    name_prefix: str = Field(default="", description="Optional label prefix for legend")


class AISubplotConfig(BaseModel):
    title: str = Field(description="Short, descriptive title for the plot")
    plot_type: Literal["2D", "3D"] = Field(description="2D for line/scatter, 3D for xyz")
    groups: List[AILineGroup] = Field(description="One or more line groups")
    mode: Literal["lines", "markers", "lines+markers"] = Field(default="lines")
    animated: bool = Field(default=False)
    x_label: str = Field(default="", description="X-axis label (leave blank to auto)")
    y_label: str = Field(default="", description="Y-axis label (leave blank to auto)")
    z_label: str = Field(default="")
    log_x: bool = Field(default=False)
    log_y: bool = Field(default=False)


class AIPlotPlan(BaseModel):
    """The full response the LLM returns."""
    reasoning: str = Field(
        description="2-4 sentence explanation of the choices made, "
                    "plain language for the end user"
    )
    subplots: List[AISubplotConfig] = Field(
        description="One or more subplots to render"
    )


class AISummary(BaseModel):
    overview: str = Field(description="1-2 sentence description of what the dataset contains")
    observations: List[str] = Field(
        description="3-6 bullet-point observations about the data "
                    "(patterns, ranges, oddities, potential relationships)"
    )
    suggested_questions: List[str] = Field(
        description="3-5 questions the user might want to ask about this data"
    )


# ---- Dataset fingerprint helpers ------------------------------------------- #

def dataset_fingerprint(df: pd.DataFrame, max_sample_rows: int = 5) -> str:
    """Small, LLM-friendly summary of the dataset.
    We deliberately don't send the whole CSV — just enough for the AI to reason.
    """
    lines = [f"Dataset shape: {len(df):,} rows × {len(df.columns)} columns"]
    lines.append("\n=== Columns ===")
    for col in df.columns:
        dtype = str(df[col].dtype)
        n_unique = df[col].nunique(dropna=True)
        n_na = df[col].isna().sum()
        info = f"  - '{col}' (dtype={dtype}, unique={n_unique}, missing={n_na})"
        if pd.api.types.is_numeric_dtype(df[col]):
            try:
                info += (f"  min={df[col].min():.4g}, max={df[col].max():.4g}, "
                         f"mean={df[col].mean():.4g}, std={df[col].std():.4g}")
            except Exception:
                pass
        lines.append(info)

    numeric = df.select_dtypes(include="number")
    if len(numeric.columns) >= 2:
        try:
            corr = numeric.corr().round(2)
            lines.append("\n=== Pairwise correlations (numeric columns) ===")
            lines.append(corr.to_string())
        except Exception:
            pass

    lines.append(f"\n=== First {min(max_sample_rows, len(df))} rows ===")
    lines.append(df.head(max_sample_rows).to_string(index=False))
    return "\n".join(lines)


# ---- Gemini client --------------------------------------------------------- #

DEFAULT_MODEL = "gemini-2.5-flash-lite"
FALLBACK_MODEL = "gemini-2.5-flash"


def resolve_api_key(user_key: Optional[str] = None) -> Optional[str]:
    """Priority: user-supplied > st.secrets > env var."""
    if user_key and user_key.strip():
        return user_key.strip()

    try:
        import streamlit as st
        if "GEMINI_API_KEY" in st.secrets:
            return st.secrets["GEMINI_API_KEY"]
    except Exception:
        pass

    return os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")


def _get_client(api_key: str):
    """Lazy import so the app still runs if google-genai isn't installed."""
    from google import genai
    return genai.Client(api_key=api_key)


def _call_gemini_structured(
    api_key: str,
    prompt: str,
    response_schema: type[BaseModel],
    model: str = DEFAULT_MODEL,
) -> BaseModel:
    client = _get_client(api_key)
    try:
        response = client.models.generate_content(
            model=model,
            contents=prompt,
            config={
                "response_mime_type": "application/json",
                "response_schema": response_schema,
                "temperature": 0.3,
            },
        )
    except Exception as e:
        # Rate-limited? Fall back to heavier free model.
        if model != FALLBACK_MODEL and any(
            s in str(e).lower() for s in ["429", "quota", "rate", "resource_exhausted"]
        ):
            return _call_gemini_structured(api_key, prompt, response_schema, FALLBACK_MODEL)
        raise

    # Prefer the SDK-parsed object; fall back to manual JSON parse.
    parsed = getattr(response, "parsed", None)
    if parsed is not None:
        return parsed
    return response_schema.model_validate_json(response.text)


# ---- Public API used by the Streamlit app ---------------------------------- #

def summarize_dataset(df: pd.DataFrame, api_key: str) -> AISummary:
    fp = dataset_fingerprint(df)
    prompt = (
        "You are a data analyst. Given the dataset summary below, produce:\n"
        "1. A 1-2 sentence overview of what the data appears to represent.\n"
        "2. 3-6 short observations (patterns, correlations, ranges, anomalies).\n"
        "3. 3-5 natural-language questions a user might want to ask.\n"
        "Base everything ONLY on the statistics and sample shown — do not invent "
        "domain facts.\n\n"
        f"{fp}"
    )
    return _call_gemini_structured(api_key, prompt, AISummary)


def suggest_plots(df: pd.DataFrame, api_key: str, n: int = 4) -> AIPlotPlan:
    fp = dataset_fingerprint(df)
    prompt = (
        f"You are a data visualization expert. Based on the dataset below, "
        f"propose {n} interesting plots that would reveal structure in the data. "
        "Prefer plots that highlight relationships, trends, or distributions. "
        "Use 2D unless three numeric variables genuinely belong on the same 3D axes. "
        "Use ONLY the exact column names from the 'Columns' section — never invent or rename. "
        "Y (and Z) columns MUST be numeric. X can be any column. "
        "Write a short 'reasoning' field (2-3 sentences) explaining your choices.\n\n"
        f"{fp}"
    )
    return _call_gemini_structured(api_key, prompt, AIPlotPlan)


def answer_plot_request(
    df: pd.DataFrame,
    user_request: str,
    api_key: str,
    existing_subplots_summary: str = "",
) -> AIPlotPlan:
    fp = dataset_fingerprint(df)
    existing_ctx = (
        f"\n=== Plots already on screen ===\n{existing_subplots_summary}\n"
        if existing_subplots_summary.strip() else ""
    )
    prompt = (
        "You are a plotting assistant. Translate the user's request into one or more "
        "plot configurations.\n"
        "Rules:\n"
        "- Use ONLY exact column names from the 'Columns' section. Never invent or rename.\n"
        "- Y and Z columns MUST be numeric.\n"
        "- Prefer 2D unless the user clearly needs 3D.\n"
        "- If the user says 'add' or 'also', the new plots are in addition to existing ones.\n"
        "- If the request is ambiguous, make a reasonable choice and explain it in 'reasoning'.\n"
        "- Give each plot a short, human-readable title.\n"
        "- Keep 'reasoning' to 2-4 sentences in plain language.\n\n"
        f"{fp}\n{existing_ctx}\n"
        f"=== User request ===\n{user_request}"
    )
    return _call_gemini_structured(api_key, prompt, AIPlotPlan)


# ---- Converters: AI schemas -> app schemas --------------------------------- #

def _fuzzy_match(wanted: str, pool: List[str]) -> str | None:
    """Case- and whitespace-tolerant column lookup. Returns the real column name
    from `pool` that matches `wanted`, or None if no reasonable match exists."""
    if not wanted:
        return None
    if wanted in pool:
        return wanted
    norm = lambda s: s.strip().lower().replace(" ", "_").replace("-", "_")
    target = norm(wanted)
    for col in pool:
        if norm(col) == target:
            return col
    # Prefix / substring fallback
    for col in pool:
        if norm(col).startswith(target) or target.startswith(norm(col)):
            return col
    return None


def ai_to_app_config(ai_cfg: AISubplotConfig, df_columns: List[str],
                     numeric_columns: List[str]):
    """Convert an AI-produced config into the app's SubplotConfig.

    Uses fuzzy column matching so minor name drift (case, spaces) from the LLM
    doesn't silently drop columns. Returns (config, diagnostic_notes).
    """
    # Import from models.py (NOT app.py — importing app would re-execute the
    # Streamlit script and cause 'duplicate widget key' errors).
    from models import LineGroup, SubplotConfig

    notes: List[str] = []

    def resolve_cols(cols, pool, label):
        resolved = []
        for c in cols:
            m = _fuzzy_match(c, pool)
            if m is None:
                notes.append(f"'{c}' ({label}) not found in the data — skipped")
            elif m != c:
                notes.append(f"'{c}' → '{m}' (name fuzzy-matched)")
                resolved.append(m)
            else:
                resolved.append(m)
        return resolved

    groups = []
    for g in ai_cfg.groups:
        x = _fuzzy_match(g.x_col, df_columns) or (df_columns[0] if df_columns else "")
        y = resolve_cols(g.y_cols, numeric_columns, "Y")
        z = resolve_cols(g.z_cols, numeric_columns, "Z")
        if ai_cfg.plot_type == "2D" and not y:
            notes.append(f"Plot '{ai_cfg.title}': no valid Y columns after matching — group dropped")
            continue
        if ai_cfg.plot_type == "3D" and (not y or not z):
            notes.append(f"Plot '{ai_cfg.title}': 3D needs both Y and Z — group dropped")
            continue
        groups.append(LineGroup(x_col=x, y_cols=y, z_cols=z,
                                name_prefix=g.name_prefix))

    if not groups:
        return None, notes

    cfg = SubplotConfig(
        title=ai_cfg.title or "AI-generated plot",
        plot_type=ai_cfg.plot_type,
        groups=groups,
        mode=ai_cfg.mode,
        animated=ai_cfg.animated,
        x_label=ai_cfg.x_label,
        y_label=ai_cfg.y_label,
        z_label=ai_cfg.z_label,
        log_x=ai_cfg.log_x,
        log_y=ai_cfg.log_y,
    )
    return cfg, notes
