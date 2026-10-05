"""Kjentmann web app: upload a photo taken from the air, get your position.

Run with: streamlit run app.py
"""

from __future__ import annotations

import re
import tempfile
from pathlib import Path

import folium
import streamlit as st
import streamlit.components.v1 as components
from streamlit_folium import st_folium

from kjentmann.config import load_config
from kjentmann.locate import describe, geocode_candidates, locate_photo, result_map
from kjentmann.match import make_matcher

COORDS = re.compile(r"\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*")

ALTITUDES = {
    "Airliner on approach (about 3000 m)": 3000,
    "Small aircraft (about 1500 m)": 1500,
    "Helicopter (about 600 m)": 600,
    "Skydiver at exit (about 4000 m)": 4000,
}

st.set_page_config(page_title="Kjentmann", page_icon="🧭", layout="wide")
st.title("Kjentmann")
st.caption("Lost GPS? Take a photo straight down, and Kjentmann finds where you are.")


@st.cache_resource
def setup():
    cfg = load_config("config.yaml")
    return cfg, make_matcher(cfg.matcher, cfg.max_keypoints, cfg.upscale)


@st.cache_data(show_spinner=False)
def search_places(name: str):
    m = COORDS.fullmatch(name)
    if m:  # "59.58, 11.16" needs no lookup
        return [(float(m.group(1)), float(m.group(2)), name.strip())]
    return geocode_candidates(name, limit=5)


cfg, matcher = setup()
state = st.session_state
state.setdefault("pos", None)  # (lat, lon, label)

left, right = st.columns([1, 2])

with left:
    st.subheader("1. Your photo")
    upload = st.file_uploader(
        "Taken straight down from an aircraft, helicopter or drone",
        type=["jpg", "jpeg", "png"],
    )
    if upload:
        st.image(upload, width="stretch")

    st.subheader("2. Roughly where are you?")
    name = st.text_input("Place name", placeholder="e.g. Askim, Norway  or  59.58, 11.16")
    if name:
        try:
            hits = search_places(name)
        except Exception as e:  # network or service error
            hits = []
            st.error(f"Place search failed: {e}")
        if hits:
            labels = [h[2] for h in hits]
            pick = st.selectbox("Which one?", labels, index=0)
            lat, lon, label = hits[labels.index(pick)]
            if state.pos is None or state.pos[2] != label:
                state.pos = (lat, lon, label)
        else:
            st.warning("No place with that name. Try adding the country, or click on the map.")
    st.caption("Or click on the map to set your rough position.")

    st.subheader("3. How high?")
    choice = st.selectbox("Altitude", [*ALTITUDES, "Enter altitude"])
    altitude = (
        st.number_input("Altitude in metres", 100, 12000, 2000, step=100)
        if choice == "Enter altitude"
        else ALTITUDES[choice]
    )
    radius = st.slider(
        "How unsure are you of your position? (km)", 1, 15, int(cfg.locate_radius_km)
    )
    go = st.button(
        "Find my position", type="primary", disabled=not (upload and state.pos is not None)
    )

with right:
    if go:
        lat, lon, label = state.pos
        suffix = Path(upload.name).suffix or ".jpg"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as f:
            f.write(upload.getvalue())  # raw bytes: keeps EXIF (GPS, orientation)
            photo_path = Path(f.name)
        with st.spinner("Searching. The first search in a new area downloads a map (~30 s)..."):
            fix = locate_photo(
                cfg, matcher, photo_path, f"{lat},{lon}", float(altitude), float(radius)
            )
        if fix.found:
            st.success(f"Found: {fix.lat:.5f}, {fix.lon:.5f}  (± {fix.accuracy_m:.0f} m)")
        else:
            st.warning("Position not found. The dashed circle is only the area that was searched.")
        st.text(describe(fix))
        out = result_map(fix, cfg, cfg.data_dir / "locate" / "app_map.html", (lat, lon), radius)
        components.html(out.read_text(encoding="utf-8"), height=620)
        st.button("New search")
    else:
        center = state.pos[:2] if state.pos else (60.0, 10.0)
        m = folium.Map(location=center, zoom_start=11 if state.pos else 5)
        if state.pos:
            folium.Circle(
                state.pos[:2], radius=radius * 1000, color="#888888", dash_array="8 6", fill=False
            ).add_to(m)
            folium.Marker(state.pos[:2], tooltip=state.pos[2]).add_to(m)
        clicked = st_folium(m, height=620, use_container_width=True, key="picker")
        if clicked and clicked.get("last_clicked"):
            c = clicked["last_clicked"]
            new = (c["lat"], c["lng"], f"{c['lat']:.4f}, {c['lng']:.4f} (clicked)")
            moved = state.pos is None or abs(state.pos[0] - new[0]) + abs(state.pos[1] - new[1])
            if moved and moved > 1e-6:
                state.pos = new
                st.rerun()
        if state.pos:
            st.caption(f"Rough position: {state.pos[2]}. The dashed circle is the search area.")
