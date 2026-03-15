# 📸 AI-Powered Image Editor

Here is a conversational image editor built with Streamlit and powered by a multi-agent LLM pipeline running on Groq. Instead of navigating menus, you just describe what you want - *"make it look cinematic"*, *"remove the background"*, *"a bit brighter than that"* - and the agents handle the rest.

---

## Features

**AI Editor (natural language)**
- Chat-based editing - describe edits in plain English
- Multi-agent pipeline: Orchestrator - Edit Planner - Clarifier
- Ambiguous requests trigger clarification buttons rather than guessing
- True undo - every edit is snapshotted, roll back one step at a time

**Manual Controls**
- Filter presets: Vintage, Cinematic, HDR, Sepia, Grayscale, Cartoon, Edge Detection
- Tone sliders: Brightness, Contrast, Saturation, Sharpness, Warmth/Cool
- Transform: Rotate, Blur

**Collage Maker**
- 5 layout presets (side-by-side, stacked, 2×2 grid, 3×2 grid, etc.)
- Customisable cell size, gap, and background colour
- Download finished collage as PNG

**Panorama Creator**
- Upload 2+ overlapping images
- OpenCV stitching to produce an ultra-wide panorama
- Download as PNG

---

## Tech Stack

| Layer | Technology |
|---|---|
| UI | Streamlit |
| Agent LLM | Groq - `llama-3.3-70b-versatile` |
| Image processing | Pillow, OpenCV, NumPy |
| Background removal | rembg (U2Net) |
| Face detection | MediaPipe |

---

## Running Locally

**1. Clone and install**
```bash
git clone https://github.com/EESHAK02/image-editor.git
cd image-editor
pip install -r requirements.txt
```

**2. Add your key**

Create `.streamlit/secrets.toml`:
```toml
GROQ_API_KEY = "gsk_your_key_here"
```
Get your free API key from [console.groq.com](https://console.groq.com)

**3. Run**
```bash
streamlit run app.py
```

---

## Deploying to Streamlit Cloud

1. Push the repo to GitHub (`.streamlit/secrets.toml` is gitignored - safe to push)
2. Go to [share.streamlit.io](https://share.streamlit.io) -> New app -> connect your repo
3. Set main file to `app.py`
4. Under **Advanced settings - Secrets**, paste:
   ```toml
   GROQ_API_KEY = "gsk_your_key_here"
   ```
5. Click Deploy

---

## Adding a New Tool

1. Write a function in `tools.py`: `def tool_my_effect(img: Image.Image, param=default, **_) -> Image.Image`
2. Add one entry to `TOOL_REGISTRY` at the bottom of `tools.py`

The tool is automatically available to all agents - no other changes needed.