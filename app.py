import cv2
import numpy as np
import requests
import streamlit as st
from io import BytesIO
from PIL import Image

from agents import run_agentic_pipeline
from tools import COLLAGE_LAYOUTS, execute_steps, make_collage


# page config

st.set_page_config(page_title="AI Photo Editor", page_icon="📸", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=DM+Sans:wght@300;400;500;600&display=swap');
html, body, [class*="css"]  { font-family: 'DM Sans', sans-serif; }
.stApp                       { background-color: #0c0c0f; color: #e4e2dd; }
.block-container             { padding: 1.5rem 2rem; max-width: 100%; }

.stTabs [data-baseweb="tab-list"] {
    background: #101013; border-radius: 10px; padding: 4px; gap: 4px; border: 1px solid #1e1e26; }
.stTabs [data-baseweb="tab"] {
    border-radius: 7px; color: #666; font-family: 'DM Mono', monospace;
    font-size: 0.78rem; letter-spacing: 0.05em; padding: 6px 18px; }
.stTabs [aria-selected="true"] { background: #1a1a22 !important; color: #aaee44 !important; }

.user-bubble {
    background: #181820; border: 1px solid #26263a; border-radius: 12px 12px 2px 12px;
    padding: 10px 14px; margin: 5px 0; font-size: 0.88rem; color: #e4e2dd; text-align: right; }
.agent-bubble {
    background: #111116; border: 1px solid #1c1c26; border-radius: 12px 12px 12px 2px;
    padding: 10px 14px; margin: 5px 0; font-size: 0.88rem; color: #a8a6a0; }
.agent-bubble strong { color: #aaee44; }

.section-label {
    font-family: 'DM Mono', monospace; font-size: 0.65rem; letter-spacing: 0.14em;
    color: #444; text-transform: uppercase; margin-bottom: 8px; }
.clarify-box {
    background: #13131e; border: 1px solid #26264a; border-left: 3px solid #aaee44;
    border-radius: 8px; padding: 12px 16px; margin: 8px 0; font-size: 0.86rem; color: #b8b6b0; }
.history-pill {
    display: inline-block; background: #161e0a; border: 1px solid #2a3a14; color: #aaee44;
    border-radius: 20px; padding: 2px 10px; font-family: 'DM Mono', monospace;
    font-size: 0.68rem; margin: 2px 3px; }

.stTextInput > div > div > input {
    background: #101014 !important; border: 1px solid #202030 !important;
    color: #e4e2dd !important; border-radius: 8px !important; }
.stTextInput > div > div > input:focus { border-color: #aaee44 !important; }
.stButton > button {
    background: #161e0a !important; border: 1px solid #2a3a14 !important; color: #aaee44 !important;
    border-radius: 7px !important; font-family: 'DM Mono', monospace !important;
    font-size: 0.76rem !important; padding: 6px 14px !important; }
.stButton > button[kind="primary"] {
    background: #aaee44 !important; color: #0c0c0f !important; border: none !important; font-weight: 600 !important; }
.stSlider > div > div > div { background: #aaee44 !important; }
hr  { border-color: #1a1a22 !important; }
h1  { font-family: 'DM Sans', sans-serif !important; font-weight: 300 !important; color: #e4e2dd !important; }
h2, h3 { font-family: 'DM Sans', sans-serif !important; font-weight: 400 !important; color: #c8c6c0 !important; }
</style>
""", unsafe_allow_html=True)


# session state

def init_state():
    defaults = {
        "original_image":      None,   # PIL Image - never mutated, used by reset tool
        "current_image":       None,   # PIL Image - live working state
        "image_snapshots":     [],     # stack of PIL Images - one pushed before each edit
        "edit_history":        [],     # list of str labels shown as pills
        "chat_messages":       [],     # list of {"role": "user"|"agent", "content": str}
        "pending_clarification": None, # dict or None - holds clarify object until user picks
        "_last_upload_name":   None,   # prevents re-processing on Streamlit rerenders
        "collage_images":      [],     # images loaded into the collage tab
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()


# state helpers

def push_snapshot():

    if st.session_state.current_image:
        st.session_state.image_snapshots.append(st.session_state.current_image.copy())

def add_agent_msg(content: str):
    st.session_state.chat_messages.append({"role": "agent", "content": content})

def apply_and_store(steps: list, message: str = ""):

    push_snapshot()
    new_img, applied = execute_steps(steps)
    st.session_state.current_image = new_img
    st.session_state.edit_history.extend(applied)
    add_agent_msg(message or f"Done: **{', '.join(applied)}**")

def load_new_image(img: Image.Image):

    st.session_state.original_image = img
    st.session_state.current_image  = img.copy()
    st.session_state.image_snapshots.clear()
    st.session_state.edit_history.clear()
    st.session_state.chat_messages.clear()
    st.session_state.pending_clarification = None

def fetch_image_from_url(url: str) -> Image.Image | None:
    try:
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        return Image.open(BytesIO(r.content)).convert("RGB")
    except Exception as e:
        st.error(f"Could not fetch image: {e}")
        return None


# UI chat panel

def render_chat_panel():
    st.markdown('<div class="section-label">✦ AI Agent</div>', unsafe_allow_html=True)

    # Render conversation history
    for msg in st.session_state.chat_messages:
        css = "user-bubble" if msg["role"] == "user" else "agent-bubble"
        st.markdown(f'<div class="{css}">{msg["content"]}</div>', unsafe_allow_html=True)

    # Clarification buttons — shown instead of input while pending
    if st.session_state.pending_clarification:
        c = st.session_state.pending_clarification
        st.markdown(f'<div class="clarify-box">❓ {c["question"]}</div>', unsafe_allow_html=True)
        opt_cols = st.columns(len(c["options"]))
        for i, (col, opt) in enumerate(zip(opt_cols, c["options"])):
            with col:
                if st.button(opt, key=f"clarify_{i}"):
                    apply_and_store(c["option_steps"][i], f"Applied **{opt}**.")
                    st.session_state.pending_clarification = None
                    st.rerun()
        return

    st.divider()

    # Chat input
    with st.form(key="chat_form", clear_on_submit=True):
        c1, c2 = st.columns([5, 1])
        with c1:
            user_input = st.text_input(
                "cmd",
                placeholder='e.g. "make it look vintage but a bit brighter"',
                label_visibility="collapsed",
            )
        with c2:
            submitted = st.form_submit_button("→", width="stretch")

    if submitted and user_input.strip():
        if st.session_state.current_image is None:
            st.warning("Upload an image first.")
            return

        msg = user_input.strip()
        st.session_state.chat_messages.append({"role": "user", "content": msg})

        with st.spinner("Agents working..."):
            resp = run_agentic_pipeline(msg)

        if resp["action"] == "execute":
            apply_and_store(resp["steps"], resp.get("message", ""))

        elif resp["action"] == "clarify":
            st.session_state.pending_clarification = resp
            add_agent_msg(resp.get("message", "Let me clarify before I proceed."))

        elif resp["action"] == "undo":
            if st.session_state.image_snapshots:
                st.session_state.current_image = st.session_state.image_snapshots.pop()
                undone = st.session_state.edit_history.pop() if st.session_state.edit_history else "last edit"
                add_agent_msg(f"Undid **{undone}**.")
            else:
                add_agent_msg("Nothing left to undo.")

        elif resp["action"] == "reset":
            st.session_state.current_image = st.session_state.original_image.copy()
            st.session_state.image_snapshots.clear()
            st.session_state.edit_history.clear()
            add_agent_msg("Reset to original image.")

        elif resp["action"] in ("info", "error"):
            prefix = "⚠️ " if resp["action"] == "error" else ""
            add_agent_msg(f"{prefix}{resp.get('message', '')}")

        st.rerun()

    # Quick-edit chips
    if st.session_state.current_image is not None:
        st.markdown('<div class="section-label" style="margin-top:14px">quick edits</div>',
                    unsafe_allow_html=True)
        chips = ["Vintage look", "Cinematic", "HDR effect",
                 "Make brighter", "More contrast", "Remove background"]
        for i, chip in enumerate(chips):
            if i % 3 == 0:
                chip_cols = st.columns(3)
            with chip_cols[i % 3]:
                if st.button(chip, key=f"chip_{i}", width="stretch"):
                    st.session_state.chat_messages.append({"role": "user", "content": chip})
                    with st.spinner("Applying..."):
                        resp = run_agentic_pipeline(chip)
                    if resp["action"] == "execute":
                        apply_and_store(resp["steps"], resp.get("message", ""))
                    elif resp["action"] == "clarify":
                        st.session_state.pending_clarification = resp
                        add_agent_msg(resp.get("message", ""))
                    st.rerun()


# image preview panel

def render_preview_panel():
    st.markdown('<div class="section-label">✦ Preview</div>', unsafe_allow_html=True)

    if st.session_state.current_image is None:
        st.markdown("""
        <div style="border:1px dashed #1e1e2a; border-radius:12px; padding:70px 20px;
             text-align:center; color:#2a2a36; font-family:'DM Mono',monospace; font-size:0.78rem;">
            NO IMAGE LOADED
        </div>""", unsafe_allow_html=True)
        return

    st.image(st.session_state.current_image, width="stretch")

    if st.session_state.edit_history:
        st.markdown('<div class="section-label" style="margin-top:10px">edit stack</div>',
                    unsafe_allow_html=True)
        pills = "".join(
            f'<span class="history-pill">{h}</span>'
            for h in st.session_state.edit_history
        )
        st.markdown(f'<div style="line-height:2.4">{pills}</div>', unsafe_allow_html=True)

    st.markdown("")
    c1, c2, c3 = st.columns(3)
    with c1:
        if st.button("↺ Undo", width="stretch"):
            if st.session_state.image_snapshots:
                st.session_state.current_image = st.session_state.image_snapshots.pop()
                if st.session_state.edit_history:
                    st.session_state.edit_history.pop()
                st.rerun()
    with c2:
        if st.button("✕ Reset", width="stretch"):
            st.session_state.current_image = st.session_state.original_image.copy()
            st.session_state.image_snapshots.clear()
            st.session_state.edit_history.clear()
            add_agent_msg("Reset to original.")
            st.rerun()
    with c3:
        buf = BytesIO()
        st.session_state.current_image.save(buf, format="PNG")
        st.download_button("⬇ Download", buf.getvalue(), "lens_edit.png",
                           "image/png", width="stretch")


# manual edit panel

def render_manual_tab():
    from tools import (
        tool_apply_vintage, tool_apply_cinematic, tool_apply_hdr,
        tool_apply_sepia, tool_apply_grayscale, tool_apply_cartoon,
        tool_apply_edge_detection, tool_adjust_brightness, tool_adjust_contrast,
        tool_adjust_saturation, tool_adjust_sharpness, tool_apply_warmth,
        tool_rotate_image, tool_apply_blur,
    )

    if st.session_state.current_image is None:
        st.info("Upload an image in the AI Editor tab first.")
        return

    col_ctrl, col_prev = st.columns([1, 2])

    with col_ctrl:
        st.markdown("#### Filter Presets")
        filt = st.selectbox("Filter", ["None", "Vintage", "Cinematic", "HDR",
                                        "Sepia", "Grayscale", "Cartoon", "Edge Detection"])
        st.markdown("#### Tone Adjustments")
        brightness = st.slider("Brightness",         -100, 100, 0)
        contrast   = st.slider("Contrast",           -100, 100, 0)
        saturation = st.slider("Saturation",         -100, 100, 0)
        sharpness  = st.slider("Sharpness",          -100, 100, 0)
        warmth     = st.slider("Warmth ←   → Cool",  -100, 100, 0)
        st.markdown("#### Transform")
        angle      = st.slider("Rotate °",  -180, 180, 0)
        blur_r     = st.slider("Blur radius",  0,  30, 0)

        if st.button("Apply All", type="primary", width="stretch"):
            img = st.session_state.current_image.copy()
            applied = []

            fmap = {
                "Vintage":        (tool_apply_vintage,        {}),
                "Cinematic":      (tool_apply_cinematic,      {}),
                "HDR":            (tool_apply_hdr,            {}),
                "Sepia":          (tool_apply_sepia,          {"intensity": 1.0}),
                "Grayscale":      (tool_apply_grayscale,      {}),
                "Cartoon":        (tool_apply_cartoon,        {}),
                "Edge Detection": (tool_apply_edge_detection, {}),
            }
            if filt in fmap:
                fn, p = fmap[filt]
                img = fn(img, **p)
                applied.append(filt)

            if brightness != 0: img = tool_adjust_brightness(img, brightness); applied.append(f"Brightness {brightness:+d}")
            if contrast   != 0: img = tool_adjust_contrast(img, contrast);     applied.append(f"Contrast {contrast:+d}")
            if saturation != 0: img = tool_adjust_saturation(img, saturation); applied.append(f"Saturation {saturation:+d}")
            if sharpness  != 0: img = tool_adjust_sharpness(img, sharpness);   applied.append(f"Sharpness {sharpness:+d}")
            if warmth     != 0: img = tool_apply_warmth(img, warmth);          applied.append(f"Warmth {warmth:+d}")
            if angle      != 0: img = tool_rotate_image(img, angle);           applied.append(f"Rotate {angle}°")
            if blur_r     >  0: img = tool_apply_blur(img, blur_r);            applied.append(f"Blur r={blur_r}")

            if applied:
                push_snapshot()
                st.session_state.current_image = img
                st.session_state.edit_history.extend([f"[manual] {a}" for a in applied])
                st.rerun()

    with col_prev:
        st.image(st.session_state.current_image, width="stretch")


# collage tab

def render_collage_tab():
    st.markdown("### Collage Maker")
    uploads = st.file_uploader("Upload images", type=["jpg", "jpeg", "png"],
                                accept_multiple_files=True, key="collage_up")
    if uploads:
        st.session_state.collage_images = [Image.open(f).convert("RGB") for f in uploads]

    images = st.session_state.collage_images
    if not images:
        st.info("Upload 2 or more images to create a collage.")
        return

    thumbs = st.columns(min(len(images), 5))
    for col, img in zip(thumbs, images):
        with col:
            st.image(img, width="stretch")

    st.divider()
    c1, c2, c3 = st.columns(3)
    with c1: layout_name = st.selectbox("Layout", list(COLLAGE_LAYOUTS.keys()))
    with c2: cell_w = st.slider("Cell width px",  200, 800, 400, step=50)
    with c3: cell_h = st.slider("Cell height px", 150, 600, 300, step=50)
    gap    = st.slider("Gap px", 0, 30, 6)
    bg_hex = st.color_picker("Background colour", "#0e0e12")
    bg     = (int(bg_hex[1:3], 16), int(bg_hex[3:5], 16), int(bg_hex[5:7], 16))

    layout = COLLAGE_LAYOUTS[layout_name]
    needed = layout["cols"] * layout["rows"]

    if len(images) < needed:
        st.warning(f"This layout needs {needed} images — you have {len(images)}.")
    else:
        if st.button("Generate Collage", type="primary"):
            with st.spinner("Building..."):
                collage = make_collage(images, layout["cols"], layout["rows"],
                                       cell_w, cell_h, gap, bg)
            st.image(collage, width="stretch")
            buf = BytesIO()
            collage.save(buf, format="PNG")
            st.download_button("⬇ Download Collage", buf.getvalue(), "collage.png", "image/png")


# panorama tab

def render_panorama_tab():
    st.markdown("### Panorama Creator")
    st.caption("Upload two or more overlapping images to stitch a panorama.")

    uploads = st.file_uploader("Upload overlapping images", type=["jpg", "jpeg", "png"],
                                accept_multiple_files=True, key="pano_up")
    if not uploads:
        return

    images = [cv2.imdecode(np.frombuffer(f.read(), np.uint8), 1) for f in uploads]
    thumbs = st.columns(min(len(images), 4))
    for col, img in zip(thumbs, images):
        with col:
            st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), width="stretch")

    if len(images) < 2:
        st.warning("Need at least 2 images.")
        return

    if st.button("Stitch Panorama", type="primary"):
        with st.spinner("Stitching..."):
            try:
                status, pano = cv2.Stitcher_create().stitch(images)
                if status == cv2.Stitcher_OK:
                    pano_rgb = cv2.cvtColor(pano, cv2.COLOR_BGR2RGB)
                    st.image(pano_rgb, width="stretch")
                    buf = BytesIO()
                    Image.fromarray(pano_rgb).save(buf, format="PNG")
                    st.download_button("⬇ Download", buf.getvalue(), "panorama.png", "image/png")
                else:
                    st.error("Stitching failed — make sure images have overlapping regions.")
            except Exception as e:
                st.error(f"Error: {e}")


# main app

st.markdown("# 📸 AI Photo Editor")
st.markdown(
    '<div style="color:#444; font-size:0.8rem; margin-top:-12px; margin-bottom:20px; '
    'font-family:\'DM Mono\',monospace; letter-spacing:0.08em;">AI-POWERED IMAGE EDITOR</div>',
    unsafe_allow_html=True,
)

tab_ai, tab_manual, tab_collage, tab_pano = st.tabs([
    "✦  AI Editor", "⊞  Manual", "⧉  Collage", "⟶  Panorama"
])

# AI editor 
with tab_ai:
    left, right = st.columns([1, 1], gap="large")

    with left:
        st.markdown('<div class="section-label">Image source</div>', unsafe_allow_html=True)
        src = st.radio("src", ["Upload", "URL"], horizontal=True, label_visibility="collapsed")

        if src == "Upload":
            uf = st.file_uploader("img", type=["jpg", "png", "jpeg"], label_visibility="collapsed")
            if uf and uf.name != st.session_state._last_upload_name:
                load_new_image(Image.open(uf).convert("RGB"))
                st.session_state._last_upload_name = uf.name
        else:
            url = st.text_input("URL", placeholder="https://...")
            if url and st.button("Load"):
                img = fetch_image_from_url(url)
                if img:
                    load_new_image(img)

        st.divider()
        render_chat_panel()

    with right:
        render_preview_panel()

# manual edit controls
with tab_manual:
    render_manual_tab()

# collage maker
with tab_collage:
    render_collage_tab()

# panorama creator
with tab_pano:
    render_panorama_tab()
