"""
FRE GCP v1.0 — Chart Generation Tool
======================================
Generates matplotlib charts from structured data extracted from documents.
The chart bytes are uploaded to GCS (session_previews/) and a V4 signed URL
is returned so the ADK web UI renders it inline without any local HTTP server.

Design
------
• Accepts JSON-encoded series so the LLM can construct it from extracted text.
• Dark-theme styling to match the ADK dev-UI.
• Stateless — no local file writes, no background HTTP server.
• Signed URLs are generated via IAM Credentials API (px-proxy compatible).

Note: do NOT add 'from __future__ import annotations' here. ADK uses
runtime type introspection on function signatures; that import turns
all annotations into lazy strings, which ADK cannot parse.
"""

import io
import json
import logging
import uuid

logger = logging.getLogger(__name__)

# Shared dark-theme colour palette
_COLORS = [
    "#7ec8e3", "#ff9f43", "#48dbfb", "#ff6b9d",
    "#a29bfe", "#55efc4", "#fdcb6e", "#e17055",
]


def _apply_dark_theme(fig, ax) -> None:
    """Apply a consistent dark background / white text style."""
    bg = "#1e1e2e"
    fig.patch.set_facecolor(bg)
    ax.set_facecolor(bg)
    ax.tick_params(colors="white", labelsize=9)
    ax.xaxis.label.set_color("white")
    ax.yaxis.label.set_color("white")
    ax.title.set_color("white")
    for spine in ax.spines.values():
        spine.set_edgecolor("#444466")
    ax.grid(True, color="#333355", linestyle="--", alpha=0.5, linewidth=0.7)


def generate_chart(
    chart_type: str,
    title: str,
    series_json: str,
    x_label: str = "",
    y_label: str = "",
    source_document: str = "",
) -> dict:
    """
    Generate a chart from data extracted from documents and return it as an inline Markdown image.

    Use this tool when the user asks to visualise, plot, draw, or chart any numerical
    data found in documents (e.g. force/displacement curves, cycle counts, failure
    statistics, test result comparisons).

    Parameters
    ----------
    chart_type      : Chart style. One of: line, bar, scatter, pie, histogram.
    title           : Descriptive chart title, e.g. "Force vs Load Cycles — EB-25_534".
    series_json     : JSON string encoding the data series.
                      For line/scatter: '[{"name":"Force","x":[0,50000,141660],"y":[0,9.8,10.3]}]'
                      For bar:          '[{"name":"Pass","value":12},{"name":"Fail","value":3}]'
                      For pie:          '[{"label":"Snap ring","value":3},{"label":"Wear","value":2}]'
                      For histogram:    '[{"name":"Cycles","values":[50000,80000,141660]}]'
    x_label         : X-axis label, e.g. "Load Cycles" or "Time (s)".
    y_label         : Y-axis label, e.g. "Force (kN)" or "Displacement (mm)".
    source_document : GCS URI of the source document (shown as caption).

    Returns
    -------
    Dict with keys:
      image_markdown (str) : Markdown with base64 PNG — include this verbatim in your reply.
      title          (str) : Chart title used.
      chart_type     (str) : Chart type used.
      source         (str) : Source document.
      error          (str) : Non-empty if generation failed; explain to the user.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")          # non-interactive, safe inside a server process
        import matplotlib.pyplot as plt
    except ImportError:
        return {
            "image_markdown": "",
            "title": title,
            "chart_type": chart_type,
            "source": source_document,
            "error": "matplotlib is not installed — run: pip install matplotlib",
        }

    try:
        series = json.loads(series_json)
    except json.JSONDecodeError as exc:
        return {
            "image_markdown": "",
            "title": title,
            "chart_type": chart_type,
            "source": source_document,
            "error": f"series_json is not valid JSON: {exc}",
        }

    fig, ax = plt.subplots(figsize=(8, 4))
    _apply_dark_theme(fig, ax)

    chart_type = chart_type.lower().strip()

    try:
        if chart_type in ("line", "scatter"):
            for i, s in enumerate(series):
                color = _COLORS[i % len(_COLORS)]
                y = s.get("y", [])
                x = s.get("x", list(range(len(y))))
                name = s.get("name", f"Series {i + 1}")
                if chart_type == "line":
                    ax.plot(x, y, label=name, color=color, linewidth=2,
                            marker="o", markersize=4)
                else:
                    ax.scatter(x, y, label=name, color=color, s=40, alpha=0.85)
            ax.legend(facecolor="#2d2d44", labelcolor="white",
                      edgecolor="#444466", fontsize=9)

        elif chart_type == "bar":
            names  = [s.get("name",  str(i))  for i, s in enumerate(series)]
            values = [s.get("value", 0)        for s    in series]
            bar_colors = [_COLORS[i % len(_COLORS)] for i in range(len(names))]
            bars = ax.bar(names, values, color=bar_colors,
                          edgecolor="#444466", linewidth=0.6)
            ax.bar_label(bars, fmt="%.1f", color="white", fontsize=9, padding=3)
            plt.xticks(rotation=30, ha="right", color="white", fontsize=9)

        elif chart_type == "pie":
            labels = [s.get("label", str(i)) for i, s in enumerate(series)]
            values = [s.get("value", 0)       for s    in series]
            pie_colors = [_COLORS[i % len(_COLORS)] for i in range(len(labels))]
            _, texts, autotexts = ax.pie(
                values, labels=labels, colors=pie_colors,
                autopct="%1.1f%%", startangle=90,
                textprops={"color": "white", "fontsize": 10},
            )
            for at in autotexts:
                at.set_color("white")
            ax.set_facecolor("#1e1e2e")
            ax.grid(False)

        elif chart_type == "histogram":
            all_values: list = []
            for s in series:
                all_values.extend(s.get("values", s.get("y", [])))
            ax.hist(all_values, bins=20, color=_COLORS[0],
                    edgecolor="#444466", alpha=0.85, linewidth=0.6)

        else:
            plt.close(fig)
            return {
                "image_markdown": "",
                "title": title,
                "chart_type": chart_type,
                "source": source_document,
                "error": (
                    f"Unsupported chart_type '{chart_type}'. "
                    "Choose: line, bar, scatter, pie, histogram"
                ),
            }

        ax.set_title(title, fontsize=13, fontweight="bold", pad=14)
        if x_label:
            ax.set_xlabel(x_label, fontsize=11)
        if y_label:
            ax.set_ylabel(y_label, fontsize=11)

        if source_document:
            short = source_document.rsplit("/", 1)[-1]
            fig.text(0.99, 0.01, f"Source: {short}",
                     ha="right", va="bottom", fontsize=7,
                     color="#888899", style="italic")

        plt.tight_layout()

        buf = io.BytesIO()
        fig.savefig(buf, format="jpeg", dpi=85, bbox_inches="tight",
                    facecolor=fig.get_facecolor(), quality=75)
        plt.close(fig)
        buf.seek(0)
        img_bytes = buf.read()

        # Upload to GCS session_previews/ and return a signed URL.
        import config
        from storage.gcs import upload_bytes as _upload, generate_signed_url as _sign
        bucket = config.GCS_BUCKET or "fre-cognitive-search-docs"
        safe_title = title[:40].replace(" ", "_").replace("/", "_").replace("\\", "_")
        fname = f"chart_{safe_title}_{uuid.uuid4().hex[:6]}.jpg"
        blob_path = f"session_previews/{fname}"
        _upload(img_bytes, bucket, blob_path, "image/jpeg")
        image_url = _sign(bucket, blob_path)
        logger.info("Chart uploaded to GCS and signed: %s", blob_path)

        image_markdown = f"![{title}]({image_url})"

        return {
            "image_markdown": image_markdown,
            "title": title,
            "chart_type": chart_type,
            "source": source_document,
            "error": "",
        }

    except Exception as exc:
        plt.close(fig)
        logger.error("generate_chart failed: %s", exc, exc_info=True)
        return {
            "image_markdown": "",
            "title": title,
            "chart_type": chart_type,
            "source": source_document,
            "error": str(exc),
        }


# ─────────────────────────────────────────────────────────────────────────────
# Statistical model fitting + what-if scenario tool
# ─────────────────────────────────────────────────────────────────────────────

def analyze_and_fit_data(
    x_values_json: str,
    y_values_json: str,
    x_label: str = "X",
    y_label: str = "Y",
    title: str = "Statistical Model Fit",
    scenario_x: str = "",
    scenario_label: str = "",
    source_document: str = "",
) -> dict:
    """
    Fit multiple statistical models to x/y data extracted from documents.
    Automatically selects the best model by R², generates a chart showing
    original data, the best-fit curve, and an optional what-if scenario.

    Use this tool when the user asks to:
      - Model how one variable relates to another ("fit a curve to this data")
      - Predict what happens if X changes to a new value
        ("what if load increases from 50kN to 75kN?")
      - Understand the statistical relationship between two quantities
      - Run scenario analysis on values found in documents

    Models tried (best by R² is selected automatically):
      Linear, Quadratic, Cubic, Exponential, Logarithmic, Power-law

    Parameters
    ----------
    x_values_json   : JSON array of x-axis values, e.g. "[0, 50000, 100000, 141660]"
    y_values_json   : JSON array of y-axis values, e.g. "[0, 9.8, 10.1, 10.3]"
    x_label         : X-axis label, e.g. "Load Cycles"
    y_label         : Y-axis label, e.g. "Force (kN)"
    title           : Chart title
    scenario_x      : Optional new x value as a string to predict, e.g. "75000"
                      Leave empty for no scenario prediction.
    scenario_label  : Label for the scenario marker, e.g. "What if: 75,000 cycles"
    source_document : GCS URI of the source document (shown as caption)

    Returns
    -------
    Dict with keys:
      image_markdown  (str)  : Markdown image — paste verbatim into reply
      best_model      (str)  : Name of best-fitting model, e.g. "Cubic"
      r_squared       (float): R² of best model (0–1, higher = better fit)
      equation        (str)  : Human-readable equation string
      scenario_result (str)  : Predicted value at scenario_x (if provided)
      model_summary   (str)  : All models ranked by R², pipe-separated
      error           (str)  : Non-empty if generation failed
    """
    try:
        import numpy as np
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        return {
            "image_markdown": "", "best_model": "", "r_squared": 0.0,
            "equation": "", "scenario_result": "", "model_summary": "",
            "error": f"Missing dependency: {exc} — run: pip install numpy matplotlib",
        }

    # ── Parse inputs ─────────────────────────────────────────────────────────
    try:
        x_arr = np.array(json.loads(x_values_json), dtype=float)
        y_arr = np.array(json.loads(y_values_json), dtype=float)
    except (json.JSONDecodeError, ValueError) as exc:
        return {
            "image_markdown": "", "best_model": "", "r_squared": 0.0,
            "equation": "", "scenario_result": "", "model_summary": "",
            "error": f"Invalid JSON input: {exc}",
        }

    if len(x_arr) != len(y_arr) or len(x_arr) < 2:
        return {
            "image_markdown": "", "best_model": "", "r_squared": 0.0,
            "equation": "", "scenario_result": "", "model_summary": "",
            "error": "x and y must have the same length and at least 2 points.",
        }

    # ── R² helper ────────────────────────────────────────────────────────────
    def _r2(y_actual: "np.ndarray", y_pred: "np.ndarray") -> float:
        ss_res = float(np.sum((y_actual - y_pred) ** 2))
        ss_tot = float(np.sum((y_actual - np.mean(y_actual)) ** 2))
        if ss_tot == 0:
            return 1.0 if ss_res < 1e-12 else 0.0
        return 1.0 - ss_res / ss_tot

    # ── Fit models ───────────────────────────────────────────────────────────
    models: dict = {}

    # Linear
    try:
        c = np.polyfit(x_arr, y_arr, 1)
        p = np.poly1d(c)
        models["Linear"] = {
            "r2": _r2(y_arr, p(x_arr)),
            "equation": f"y = {c[0]:.4g}x + {c[1]:.4g}",
            "predict": lambda xv, p=p: float(p(xv)),
            "curve": lambda xs, p=p: p(xs),
        }
    except Exception:
        pass

    # Quadratic
    try:
        c = np.polyfit(x_arr, y_arr, 2)
        p = np.poly1d(c)
        models["Quadratic"] = {
            "r2": _r2(y_arr, p(x_arr)),
            "equation": f"y = {c[0]:.4g}x\u00b2 + {c[1]:.4g}x + {c[2]:.4g}",
            "predict": lambda xv, p=p: float(p(xv)),
            "curve": lambda xs, p=p: p(xs),
        }
    except Exception:
        pass

    # Cubic
    try:
        c = np.polyfit(x_arr, y_arr, 3)
        p = np.poly1d(c)
        models["Cubic"] = {
            "r2": _r2(y_arr, p(x_arr)),
            "equation": f"y = {c[0]:.4g}x\u00b3 + {c[1]:.4g}x\u00b2 + {c[2]:.4g}x + {c[3]:.4g}",
            "predict": lambda xv, p=p: float(p(xv)),
            "curve": lambda xs, p=p: p(xs),
        }
    except Exception:
        pass

    # Exponential: y = a·e^(b·x)  via log-linearisation
    try:
        if np.all(y_arr > 0):
            c = np.polyfit(x_arr, np.log(y_arr), 1)
            b, log_a = float(c[0]), float(c[1])
            a = float(np.exp(log_a))
            y_pred = a * np.exp(b * x_arr)
            models["Exponential"] = {
                "r2": _r2(y_arr, y_pred),
                "equation": f"y = {a:.4g} \u00d7 e^({b:.4g}x)",
                "predict": lambda xv, a=a, b=b: float(a * np.exp(b * xv)),
                "curve": lambda xs, a=a, b=b: a * np.exp(b * xs),
            }
    except Exception:
        pass

    # Logarithmic: y = a·ln(x) + b
    try:
        if np.all(x_arr > 0):
            c = np.polyfit(np.log(x_arr), y_arr, 1)
            a, b = float(c[0]), float(c[1])
            models["Logarithmic"] = {
                "r2": _r2(y_arr, a * np.log(x_arr) + b),
                "equation": f"y = {a:.4g} \u00d7 ln(x) + {b:.4g}",
                "predict": lambda xv, a=a, b=b: float(a * np.log(xv) + b),
                "curve": lambda xs, a=a, b=b: a * np.log(xs) + b,
            }
    except Exception:
        pass

    # Power law: y = a·x^b  via log-log linearisation
    try:
        if np.all(x_arr > 0) and np.all(y_arr > 0):
            c = np.polyfit(np.log(x_arr), np.log(y_arr), 1)
            b, log_a = float(c[0]), float(c[1])
            a = float(np.exp(log_a))
            models["Power"] = {
                "r2": _r2(y_arr, a * np.power(x_arr, b)),
                "equation": f"y = {a:.4g} \u00d7 x^{b:.4g}",
                "predict": lambda xv, a=a, b=b: float(a * np.power(xv, b)),
                "curve": lambda xs, a=a, b=b: a * np.power(xs, b),
            }
    except Exception:
        pass

    if not models:
        return {
            "image_markdown": "", "best_model": "", "r_squared": 0.0,
            "equation": "", "scenario_result": "", "model_summary": "",
            "error": "No model could be fit to the provided data.",
        }

    # ── Pick best model ───────────────────────────────────────────────────────
    best_name = max(models, key=lambda k: models[k]["r2"])
    best = models[best_name]

    rankings = sorted(models.items(), key=lambda kv: kv[1]["r2"], reverse=True)
    model_summary = " | ".join(
        f"{name}: R\u00b2={info['r2']:.4f}" for name, info in rankings
    )

    # ── Scenario prediction ───────────────────────────────────────────────────
    scenario_result = ""
    scenario_xv = None
    if scenario_x.strip():
        try:
            scenario_xv = float(scenario_x.strip())
            pred_y = best["predict"](scenario_xv)
            lbl = scenario_label or f"{x_label} = {scenario_xv:,.2f}"
            scenario_result = (
                f"At {lbl}: predicted {y_label} = {pred_y:.4g}  "
                f"(model: {best_name}, R\u00b2={best['r2']:.4f})"
            )
        except Exception as exc:
            scenario_result = f"Scenario prediction failed: {exc}"

    # ── Build chart ───────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 4.5))
    _apply_dark_theme(fig, ax)

    # Smooth curve range — extend to scenario_x if beyond data
    x_max = max(float(x_arr.max()), scenario_xv) if scenario_xv is not None else float(x_arr.max())
    x_smooth = np.linspace(float(x_arr.min()), x_max, 400)
    try:
        y_smooth = best["curve"](x_smooth)
    except Exception:
        y_smooth = np.array([best["predict"](xv) for xv in x_smooth])

    # Original data + best-fit
    ax.scatter(x_arr, y_arr, color=_COLORS[0], s=55, zorder=5,
               label="Original data", alpha=0.9)
    ax.plot(x_smooth, y_smooth, color=_COLORS[1], linewidth=2.2, zorder=4,
            label=f"{best_name} fit (R\u00b2={best['r2']:.4f})")

    # Scenario marker
    if scenario_xv is not None:
        try:
            pred_y = best["predict"](scenario_xv)
            ax.axvline(x=scenario_xv, color=_COLORS[2], linewidth=1.5,
                       linestyle=":", alpha=0.75)
            ax.scatter([scenario_xv], [pred_y], color=_COLORS[2], s=80,
                       zorder=6, marker="D",
                       label=f"{scenario_label or 'Scenario'} \u2192 {pred_y:.4g}")
            ax.annotate(f"  {pred_y:.4g}", xy=(scenario_xv, pred_y),
                        color=_COLORS[2], fontsize=9, va="bottom")
        except Exception:
            pass

    ax.legend(facecolor="#2d2d44", labelcolor="white",
              edgecolor="#444466", fontsize=9)
    ax.set_title(title, fontsize=13, fontweight="bold", pad=14)
    if x_label:
        ax.set_xlabel(x_label, fontsize=11)
    if y_label:
        ax.set_ylabel(y_label, fontsize=11)
    if source_document:
        short = source_document.rsplit("/", 1)[-1]
        fig.text(0.99, 0.01, f"Source: {short}", ha="right", va="bottom",
                 fontsize=7, color="#888899", style="italic")

    # Equation annotation inside chart
    fig.text(0.02, 0.03, f"Equation: {best['equation']}",
             fontsize=8, color="#aaaacc", style="italic")

    plt.tight_layout()

    # ── Upload to GCS and return signed URL (same pattern as generate_chart) ─
    buf = io.BytesIO()
    fig.savefig(buf, format="jpeg", dpi=85, bbox_inches="tight",
                facecolor=fig.get_facecolor(), quality=75)
    plt.close(fig)
    buf.seek(0)
    img_bytes = buf.read()

    import config
    from storage.gcs import upload_bytes as _upload, generate_signed_url as _sign
    bucket = config.GCS_BUCKET or "fre-cognitive-search-docs"
    safe_title = title[:40].replace(" ", "_").replace("/", "_").replace("\\", "_")
    fname = f"fit_{safe_title}_{uuid.uuid4().hex[:6]}.jpg"
    blob_path = f"session_previews/{fname}"
    _upload(img_bytes, bucket, blob_path, "image/jpeg")
    image_url = _sign(bucket, blob_path)
    logger.info("Fit chart uploaded to GCS and signed: %s", blob_path)

    image_markdown = f"![{title}]({image_url})"

    return {
        "image_markdown": image_markdown,
        "best_model": best_name,
        "r_squared": round(best["r2"], 6),
        "equation": best["equation"],
        "scenario_result": scenario_result,
        "model_summary": model_summary,
        "error": "",
    }
