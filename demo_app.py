"""
Hosted portfolio demo: replays precomputed YOLO tracking + Gemini
AI-correction results for a couple of sample videos. No GPU, no API key,
no torch/ultralytics/google-genai needed at runtime - everything here is a
static file read from demo_data/, built ahead of time by build_demo_data.py.
"""

import json
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

DEMO_DATA_DIR = Path("demo_data")

st.set_page_config(
    page_title="Queue Intelligence — Live Demo",
    page_icon="🏪",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
    .main { background: linear-gradient(135deg, #1a1a1a 0%, #2d2d2d 100%); color: #ffffff; }
    h1, h2, h3 { background: linear-gradient(90deg, #ffde59, #ff914d);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text; }
    .step-card {
        background: rgba(255, 222, 89, 0.08); border: 1px solid rgba(255, 222, 89, 0.35);
        border-radius: 12px; padding: 18px; height: 100%;
    }
    .metric-box {
        background: rgba(255, 222, 89, 0.1); border: 2px solid rgba(255, 222, 89, 0.4);
        border-radius: 10px; padding: 16px; text-align: center;
    }
    .metric-box .value { font-size: 2rem; font-weight: 700; color: #ffde59; }
    .metric-box .label { color: #cfcfcf; font-size: 0.85rem; }
    .terminal {
        background: #0c0c0c; border: 1px solid #333; border-radius: 8px;
        padding: 16px; font-family: 'Consolas', 'Courier New', monospace;
        font-size: 0.85rem; line-height: 1.55; overflow-x: auto; white-space: pre-wrap;
    }
    .term-muted { color: #888; }
    .term-info { color: #7ec8ff; }
    .term-ok { color: #6cff9b; }
    .term-warn { color: #ffde59; }
    .verdict-merge { color: #6cff9b; font-weight: 700; }
    .verdict-separate { color: #ff914d; font-weight: 700; }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_manifest():
    manifest_path = DEMO_DATA_DIR / "manifest.json"
    if not manifest_path.exists():
        return []
    return json.loads(manifest_path.read_text())


@st.cache_data
def load_analysis(slug: str):
    return json.loads((DEMO_DATA_DIR / slug / "analysis.json").read_text())


@st.cache_data
def load_dwell_times(slug: str):
    csv_path = DEMO_DATA_DIR / slug / "dwell_times.csv"
    if not csv_path.exists():
        return None
    return pd.read_csv(csv_path)


def metric_box(value, label):
    st.markdown(
        f'<div class="metric-box"><div class="value">{value}</div>'
        f'<div class="label">{label}</div></div>',
        unsafe_allow_html=True,
    )


def render_terminal_log(analysis: dict):
    """A styled monospace block mirroring agent.py's real console output,
    so the AI's reasoning reads like watching the actual pipeline run."""
    lines = []
    corrections = analysis.get("corrections", [])

    for i, correction in enumerate(corrections, 1):
        event = correction["confusion_event"]
        ai = correction["ai_analysis"]
        same_person = ai.get("same_person", False)
        confidence = ai.get("confidence", 0.0)
        evidence = ai.get("visual_evidence", "No evidence provided")

        lines.append(f'<span class="term-muted">--- Confusion {i}/{len(corrections)} ---</span>')
        lines.append(f'<span class="term-info">Type:</span> {event["event_type"]}   '
                      f'<span class="term-info">Person ID:</span> {event["person_id"]}   '
                      f'<span class="term-info">Timestamp:</span> {event["timestamp"]:.1f}s')
        lines.append('<span class="term-info">📤 Uploading segment to Gemini 2.5 Flash...</span>')
        lines.append('<span class="term-ok">✓ Analysis complete</span>')
        verdict_class = "verdict-merge" if same_person else "verdict-separate"
        verdict_text = "SAME PERSON — merging dwell time" if same_person else "DIFFERENT PEOPLE — keeping times separate"
        lines.append(f'Verdict: <span class="{verdict_class}">{verdict_text}</span>   '
                      f'<span class="term-warn">Confidence: {confidence:.0%}</span>')
        lines.append(f'<span class="term-muted">Evidence:</span> {evidence}')
        lines.append("")

    st.markdown(f'<div class="terminal">{"<br>".join(lines)}</div>', unsafe_allow_html=True)


def render_correction_detail(analysis: dict, slug: str):
    corrections = analysis.get("corrections", [])
    if not corrections:
        st.info("No confusion events were flagged in this run — tracking was clean.")
        return

    for i, correction in enumerate(corrections, 1):
        event = correction["confusion_event"]
        ai = correction["ai_analysis"]
        same_person = ai.get("same_person", False)

        with st.expander(
            f"Confusion {i}: {event['event_type']} — Person {event['person_id']} "
            f"at {event['timestamp']:.1f}s "
            f"({'merged' if same_person else 'kept separate'}, "
            f"{ai.get('confidence', 0):.0%} confidence)"
        ):
            col_clip, col_reasoning = st.columns([1, 1])
            with col_clip:
                clip_path = DEMO_DATA_DIR / slug / correction["segment_path"]
                if clip_path.exists():
                    st.video(str(clip_path))
                else:
                    st.warning("Clip not available")
            with col_reasoning:
                st.markdown(f"**AI verdict:** {'Same person (ID switch merged)' if same_person else 'Different people (times kept separate)'}")
                st.markdown(f"**Confidence:** {ai.get('confidence', 0):.0%}")
                st.markdown(f"**Visual evidence:**")
                st.write(ai.get("visual_evidence", "—"))
                if ai.get("corrected_times"):
                    st.markdown("**Corrected times:**")
                    st.json(ai["corrected_times"])


# ==================== HEADER ====================
st.title("🏪 Queue Intelligence System")
st.markdown(
    "YOLOv8 tracks how long each customer spends in-frame, flags moments where "
    "tracking likely got confused, then a Gemini vision agent watches the actual "
    "clip and decides whether the dwell time needs correcting. "
    "**This page replays precomputed results — no live GPU or API calls.**"
)

manifest = load_manifest()
if not manifest:
    st.error(
        "No demo data found. Run `python run_tracking.py <video>` followed by "
        "`python build_demo_data.py <slug> <label> <video>` to generate it."
    )
    st.stop()

video_options = {m["label"]: m["slug"] for m in manifest}
selected_label = st.selectbox("Choose a sample video:", list(video_options.keys()))
slug = video_options[selected_label]

analysis = load_analysis(slug)
dwell_df = load_dwell_times(slug)
final_times = analysis.get("final_times", {})
corrections = analysis.get("corrections", [])

st.markdown("---")

# ==================== HOW IT WORKS ====================
st.markdown("### How it works")
c1, c2, c3 = st.columns(3)
with c1:
    st.markdown(
        '<div class="step-card"><h4>1. Track</h4>'
        'YOLOv8 detects and tracks every person per frame. Anyone inside a '
        'user-drawn staff zone is excluded — only customers count toward dwell time.'
        '</div>', unsafe_allow_html=True)
with c2:
    st.markdown(
        '<div class="step-card"><h4>2. Detect confusion</h4>'
        'Heuristics flag likely tracking mistakes: an ID that switches mid-track, '
        'a brief occlusion, or someone reappearing long after they left.'
        '</div>', unsafe_allow_html=True)
with c3:
    st.markdown(
        '<div class="step-card"><h4>3. AI correction</h4>'
        'Each flagged moment is clipped and sent to Gemini, which watches the '
        'actual video and decides whether to merge or separate the tracked IDs.'
        '</div>', unsafe_allow_html=True)

st.markdown("---")

# ==================== ZONE + METRICS ====================
col_zone, col_metrics = st.columns([3, 2])
with col_zone:
    st.markdown("### Customers vs. staff zone")
    zone_img = DEMO_DATA_DIR / slug / "zone_preview.jpg"
    if zone_img.exists():
        st.image(str(zone_img), use_container_width=True,
                  caption="Red = staff zone (excluded from customer tracking). Drawn once per camera angle.")
    else:
        st.info("No zone preview available for this video.")

with col_metrics:
    st.markdown("### At a glance")
    m1, m2 = st.columns(2)
    with m1:
        metric_box(len(final_times), "Customers tracked")
    with m2:
        metric_box(len(corrections), "Confusions flagged")
    m3, m4 = st.columns(2)
    with m3:
        merged = sum(1 for c in corrections if c["ai_analysis"].get("same_person"))
        metric_box(merged, "AI merged (ID switch)")
    with m4:
        avg_conf = (sum(c["ai_analysis"].get("confidence", 0) for c in corrections) / len(corrections)) if corrections else 0
        metric_box(f"{avg_conf:.0%}", "Avg. AI confidence")

st.markdown("---")

# ==================== TRACKED VIDEO ====================
st.markdown("### Tracked video")
tracked_video = DEMO_DATA_DIR / slug / "tracked_video.mp4"
if tracked_video.exists():
    st.video(str(tracked_video))
else:
    st.info("Tracked video not available for this demo.")

st.markdown("---")

# ==================== DWELL TIME: BEFORE / AFTER ====================
st.markdown("### Dwell time: before vs. after AI correction")

if final_times:
    rows = []
    for pid, data in final_times.items():
        rows.append({
            "Customer ID": pid,
            "Original (s)": round(data["original_time"], 1),
            "AI-Corrected (s)": round(data["corrected_time"], 1),
            "Δ (s)": round(data["corrected_time"] - data["original_time"], 1),
            "Corrections applied": len(data["corrections_applied"]),
        })
    df_times = pd.DataFrame(rows).sort_values("Original (s)", ascending=False)

    changed = df_times[df_times["Δ (s)"] != 0]
    if not changed.empty:
        fig = go.Figure()
        fig.add_bar(name="Original", x=changed["Customer ID"].astype(str), y=changed["Original (s)"],
                    marker_color="#888888")
        fig.add_bar(name="AI-Corrected", x=changed["Customer ID"].astype(str), y=changed["AI-Corrected (s)"],
                    marker_color="#ffde59")
        fig.update_layout(
            barmode="group", template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            height=350, margin=dict(l=10, r=10, t=30, b=10),
            title="Customers whose time changed after AI review",
        )
        st.plotly_chart(fig, use_container_width=True)

    st.dataframe(df_times, use_container_width=True, hide_index=True)
else:
    st.info("No tracking data available for this demo.")

st.markdown("---")

# ==================== AI REASONING: TERMINAL + CLIPS ====================
st.markdown("### AI correction log")
st.caption("What the pipeline actually printed while Gemini analyzed each flagged moment.")
render_terminal_log(analysis)

st.markdown("### Watch each correction")
render_correction_detail(analysis, slug)

st.markdown("---")
st.caption(
    "Built with YOLOv8 (Ultralytics) + Gemini 2.5 Flash. "
    "Source: github.com — see the full pipeline in queue_tracker.py, agent.py, and pipeline.py."
)
