# 📊 CSV Plotter with AI

A free, open-source web app for interactive CSV plotting. Upload a CSV, ask an
AI to generate plots in natural language, or build them manually with a
MATLAB-style interactive toolbar (zoom, pan, autoscale, PNG export).

Built with [Streamlit](https://streamlit.io) + [Plotly](https://plotly.com/python/)
+ [Google Gemini](https://ai.google.dev) (free tier).

## Features

### Plotting
- Upload CSV / TSV / URL, or use built-in sample data
- Multiple subplots — 2D line/scatter and 3D line/scatter
- Multiple line groups per subplot
- MATLAB-style toolbar on every chart: box-zoom, pan, autoscale, PNG download
- Animation mode with play/pause + frame slider
- Log axes, custom labels, 3 layout modes (stacked / grid / combined)

### AI agent (free)
- 💬 **Ask for plots** — *"plot temperature vs time on a log scale"*
- 💡 **Suggest plots** — AI proposes 3–6 interesting plots from the data alone
- 📋 **Summarize data** — plain-language overview, observations, suggested questions
- Powered by Google Gemini 2.5 Flash-Lite (free, 1000 req/day, no credit card)
- **Bring-your-own-key** supported — users paste their own free key for unlimited use
- AI returns structured JSON that maps to the plot schema — no code execution,
  safe for public deployment

## Run locally

```bash
git clone https://github.com/<you>/csv-plotter.git
cd csv-plotter
pip install -r requirements.txt
```

Optional — get a free Gemini key from https://aistudio.google.com/apikey, then:

```bash
export GEMINI_API_KEY="your-key-here"
streamlit run app.py
```

Or skip the key and enter it in the sidebar when the app opens. Without a
key the AI tabs are disabled, manual plotting still works.

## Deploy for free (3 minutes)

### Streamlit Community Cloud (easiest)

1. Push this folder to a **public** GitHub repo.
2. Go to https://share.streamlit.io → sign in → **New app** → pick the repo →
   main file `app.py` → **Deploy**.
3. **Set the shared Gemini key:** in your new app's dashboard →
   **Settings → Secrets** → paste:
   ```toml
   GEMINI_API_KEY = "your-key-here"
   ```
4. You get a URL like `https://<you>-csv-plotter.streamlit.app`.

Free tier: 1 always-on app, public, automatic redeploy on every `git push`.

### Hugging Face Spaces
Create a Space → SDK: Streamlit → upload the repo → add `GEMINI_API_KEY` in
the Space's **Settings → Variables and secrets**.

### Render / Railway / Fly.io
Free tiers. Needs a `Procfile`:
```
web: streamlit run app.py --server.port=$PORT --server.address=0.0.0.0
```

## Important notes on the free tier

**Gemini free tier as of April 2026**:

| Model | Requests/min | Requests/day |
|---|---|---|
| Gemini 2.5 Flash-Lite (default) | 15 | 1,000 |
| Gemini 2.5 Flash (auto-fallback) | 10 | 250 |

- Limits are **per Google Cloud project**, not per key.
- If your shared deployment is popular, the 1,000/day cap is shared across users.
- **BYOK pattern mitigates this:** each user can paste their own free key in
  the sidebar and get their own quota. App is designed around this.
- Daily quotas reset at midnight Pacific Time.
- No credit card required for the free tier.

**Privacy note to show your users:** the AI agent sends dataset metadata
(column names, dtypes, basic stats, 5 sample rows) and the user's natural
language question to Google. The full CSV is never uploaded anywhere except
the app session. Google's terms state that free-tier prompts may be used to
improve products; paid-tier prompts are not.

## Architecture — why structured output, not code execution?

Many "AI data analyst" apps let the LLM write Python and execute it. That's
dangerous on a public server — a malicious prompt can read the filesystem,
fork-bomb, mine crypto, exfiltrate data, etc. This app instead:

1. Sends the AI a small dataset fingerprint + the user's question
2. Forces the AI to return JSON matching a Pydantic schema
3. Renders the JSON using trusted, pre-existing plotting code

Result: the AI controls *what* to plot, never *how* it runs. Safe for public
deployment, cheaper on tokens, more reliable (schema-validated output),
post-hoc editable by the user.

## Roadmap (v2)

- MP4 / GIF video export (needs ffmpeg → Docker image on Render or HF Spaces)
- Save / load plot configurations as JSON
- Row range filtering and subsampling
- AI-assisted data cleaning (missing values, outlier flags)
- AI follow-up: "modify plot #2 to use log Y"
- Share link that encodes plot config in URL

## License

MIT.
