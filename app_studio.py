import io
import os
import hashlib
import json
import csv
import time
import zipfile
from datetime import datetime
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import streamlit as st
import tifffile
from PIL import Image
try:
    from streamlit_plotly_events import plotly_events
except Exception:
    plotly_events = None


st.set_page_config(page_title="CMYK Production Studio", layout="wide")

CHANNELS = ["cyan", "magenta", "yellow", "black"]
CHANNEL_INDEX = {name: idx for idx, name in enumerate(CHANNELS)}
PRESET_RECIPES = [
    {"name": "Print Green", "cmyk": [89, 33, 100, 27]},
    {"name": "Deep Green", "cmyk": [89, 0, 50, 59]},
    {"name": "Rich Red", "cmyk": [0, 100, 100, 0]},
    {"name": "Warm Yellow", "cmyk": [0, 8, 100, 0]},
    {"name": "Corporate Blue", "cmyk": [100, 70, 0, 10]},
]
APP_DATA_DIR = Path("studio_data")
APP_DATA_DIR.mkdir(exist_ok=True)
SETTINGS_STORE_PATH = APP_DATA_DIR / "settings_profiles.json"
LIBRARY_STORE_PATH = APP_DATA_DIR / "color_libraries.json"
HISTORY_STORE_PATH = APP_DATA_DIR / "job_history.json"


def load_json_store(path, default):
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def save_json_store(path, payload):
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
def normalize_to_uint8(arr):
    if arr.dtype == np.uint8:
        return arr
    if arr.dtype == np.uint16:
        return (arr / 257.0).round().astype(np.uint8)
    raise ValueError(f"Unsupported dtype: {arr.dtype}")


def extract_cmyk(arr):
    """
    Convert common TIFF layouts to H x W x 4 CMYK.
    Supported:
    - H x W x 4
    - 4 x H x W
    - H x W x 5  (drops 5th channel)
    - 5 x H x W  (drops 5th channel)
    """
    arr = normalize_to_uint8(arr)

    if arr.ndim != 3:
        raise ValueError(f"Unsupported TIFF shape: {arr.shape}")

    if arr.shape[2] in (4, 5):
        return arr[:, :, :4]

    if arr.shape[0] in (4, 5):
        return np.transpose(arr[:4, :, :], (1, 2, 0))

    raise ValueError(f"Cannot interpret TIFF as CMYK. Shape: {arr.shape}")


def save_cmyk_tiff_bytes(cmyk_array):
    buf = io.BytesIO()
    tifffile.imwrite(
        buf,
        cmyk_array.astype(np.uint8),
        photometric="separated",
        planarconfig="contig",
        compression="lzw",
        metadata=None,
    )
    buf.seek(0)
    return buf.getvalue()


def save_gray_tiff_bytes(gray_array):
    buf = io.BytesIO()
    tifffile.imwrite(
        buf,
        gray_array.astype(np.uint8),
        photometric="minisblack",
        compression="lzw",
        metadata=None,
    )
    buf.seek(0)
    return buf.getvalue()


def create_plate_preview(plate, white_threshold=245):
    preview = 255 - plate.astype(np.uint8)
    preview[preview >= white_threshold] = 255
    return preview


def save_preview_tiff_bytes(plate, white_threshold=245):
    preview = create_plate_preview(plate, white_threshold=white_threshold)
    return save_gray_tiff_bytes(preview), preview


@st.cache_data(show_spinner=False)
def read_uploaded_tiff_bytes(file_bytes, page_index=0):
    with tifffile.TiffFile(io.BytesIO(file_bytes)) as tif:
        page = tif.pages[page_index]
        arr = page.asarray()
        info = {
            "shape": page.shape,
            "dtype": str(page.dtype),
            "photometric": str(page.photometric),
            "compression": str(page.compression),
            "page_count": len(tif.pages),
        }
    return arr, info


@st.cache_data(show_spinner=False)
def get_tiff_overview(file_bytes):
    with tifffile.TiffFile(io.BytesIO(file_bytes)) as tif:
        return [f"Page {idx + 1} - {page.shape}" for idx, page in enumerate(tif.pages)]


def build_file_signature(uploaded_file):
    file_bytes = uploaded_file.getvalue()
    return hashlib.sha1(file_bytes).hexdigest(), file_bytes


def create_separations(cmyk):
    h, w, _ = cmyk.shape
    zeros = np.zeros((h, w), dtype=np.uint8)

    return {
        "cyan": np.stack([cmyk[:, :, 0], zeros, zeros, zeros], axis=2),
        "magenta": np.stack([zeros, cmyk[:, :, 1], zeros, zeros], axis=2),
        "yellow": np.stack([zeros, zeros, cmyk[:, :, 2], zeros], axis=2),
        "black": np.stack([zeros, zeros, zeros, cmyk[:, :, 3]], axis=2),
    }


def pil_from_gray(gray_array):
    return Image.fromarray(gray_array.astype(np.uint8), mode="L")


def pil_from_rgb(rgb_array):
    return Image.fromarray(rgb_array.astype(np.uint8), mode="RGB")


def build_zip(file_map):
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        for file_name, data in file_map.items():
            zf.writestr(file_name, data)
    zip_buffer.seek(0)
    return zip_buffer.getvalue()


def percent_to_uint8(value):
    return int(round(np.clip(value, 0, 100) * 255 / 100))


def uint8_to_percent(value):
    return int(round(np.clip(value, 0, 255) * 100 / 255))


def cmyk_percent_to_rgb(c, m, y, k):
    c = np.clip(c, 0, 100) / 100.0
    m = np.clip(m, 0, 100) / 100.0
    y = np.clip(y, 0, 100) / 100.0
    k = np.clip(k, 0, 100) / 100.0
    r = int(round(255 * (1 - c) * (1 - k)))
    g = int(round(255 * (1 - m) * (1 - k)))
    b = int(round(255 * (1 - y) * (1 - k)))
    return r, g, b


def rgb_to_cmyk_percent(r, g, b):
    r = np.clip(r, 0, 255) / 255.0
    g = np.clip(g, 0, 255) / 255.0
    b = np.clip(b, 0, 255) / 255.0
    k = 1.0 - max(r, g, b)

    if k >= 0.999999:
        return 0, 0, 0, 100

    denom = max(1.0 - k, 1e-12)
    c = (1.0 - r - k) / denom
    m = (1.0 - g - k) / denom
    y = (1.0 - b - k) / denom

    return tuple(
        int(round(np.clip(channel, 0.0, 1.0) * 100))
        for channel in (c, m, y, k)
    )


def normalize_hex_value(value):
    hex_value = value.strip().lstrip("#").upper()
    if len(hex_value) == 3:
        hex_value = "".join(char * 2 for char in hex_value)
    if len(hex_value) != 6 or any(char not in "0123456789ABCDEF" for char in hex_value):
        raise ValueError("HEX must be 3 or 6 hexadecimal characters, for example 00A651.")
    return f"#{hex_value}"


def hex_to_rgb(value):
    hex_value = normalize_hex_value(value)
    return tuple(int(hex_value[i : i + 2], 16) for i in (1, 3, 5))


def hex_to_cmyk_percent(value):
    return rgb_to_cmyk_percent(*hex_to_rgb(value))


def cmyk_percent_to_hex(c, m, y, k):
    return "#{:02X}{:02X}{:02X}".format(*cmyk_percent_to_rgb(c, m, y, k))


def cmyk_to_rgb_preview(cmyk_array):
    cmyk_float = cmyk_array.astype(np.float32) / 255.0
    c = cmyk_float[:, :, 0]
    m = cmyk_float[:, :, 1]
    y = cmyk_float[:, :, 2]
    k = cmyk_float[:, :, 3]

    rgb = np.stack(
        [
            255 * (1 - c) * (1 - k),
            255 * (1 - m) * (1 - k),
            255 * (1 - y) * (1 - k),
        ],
        axis=2,
    )
    return np.clip(rgb.round(), 0, 255).astype(np.uint8)


def build_hover_preview_data(cmyk_array, render_width):
    height, width, _ = cmyk_array.shape
    if width <= render_width:
        x_idx = np.arange(width, dtype=int)
        y_idx = np.arange(height, dtype=int)
    else:
        scaled_width = render_width
        scaled_height = max(1, int(round(height * (scaled_width / width))))
        x_idx = np.linspace(0, width - 1, scaled_width).round().astype(int)
        y_idx = np.linspace(0, height - 1, scaled_height).round().astype(int)

    preview_cmyk = cmyk_array[np.ix_(y_idx, x_idx)]
    full_rgb = cmyk_to_rgb_preview(cmyk_array)
    preview_rgb = np.array(
        Image.fromarray(full_rgb, mode="RGB").resize(
            (len(x_idx), len(y_idx)),
            Image.Resampling.LANCZOS,
        )
    )
    preview_percent = np.round(preview_cmyk.astype(np.float32) * 100.0 / 255.0).astype(np.uint8)

    source_x = np.broadcast_to(x_idx.reshape(1, -1), (len(y_idx), len(x_idx)))
    source_y = np.broadcast_to(y_idx.reshape(-1, 1), (len(y_idx), len(x_idx)))
    customdata = np.dstack([preview_percent, source_x, source_y]).astype(np.int32)

    return preview_rgb, customdata


def get_preview_display_width(cmyk_array):
    return int(min(1100, max(760, cmyk_array.shape[1])))


def create_cmyk_hover_figure(cmyk_array, display_width=None, render_width=None):
    if display_width is None:
        display_width = get_preview_display_width(cmyk_array)
    if render_width is None:
        render_width = min(cmyk_array.shape[1], max(display_width * 2, 1800))
    preview_rgb, customdata = build_hover_preview_data(cmyk_array, render_width=render_width)
    preview_height, preview_width = preview_rgb.shape[:2]

    fig = go.Figure(
        go.Image(
            z=preview_rgb,
            customdata=customdata,
            hovertemplate=(
                "C: %{customdata[0]}%<br>"
                "M: %{customdata[1]}%<br>"
                "Y: %{customdata[2]}%<br>"
                "K: %{customdata[3]}%<br>"
                "X: %{customdata[4]}<br>"
                "Y: %{customdata[5]}<extra></extra>"
            ),
        )
    )
    fig.update_layout(
        width=display_width,
        height=max(220, int(round(preview_height * (display_width / max(preview_width, 1))))),
        margin=dict(l=0, r=0, t=0, b=0),
        hoverlabel=dict(
            bgcolor="white",
            bordercolor="#2e7d32",
            font=dict(color="black", size=16),
        ),
    )
    fig.update_xaxes(showticklabels=False, showgrid=False, zeroline=False)
    fig.update_yaxes(showticklabels=False, showgrid=False, zeroline=False, scaleanchor="x")
    return fig


def create_target_color_separation(cmyk, target_cmyk_percent):
    target = np.array([percent_to_uint8(channel) for channel in target_cmyk_percent], dtype=np.uint8)
    cmyk_percent = np.round(cmyk.astype(np.float32) * 100.0 / 255.0).astype(np.uint8)
    match_mask = np.all(
        cmyk_percent == np.array(target_cmyk_percent, dtype=np.uint8).reshape(1, 1, 4),
        axis=2,
    )
    picked = np.zeros_like(cmyk, dtype=np.uint8)
    picked[match_mask] = target.astype(np.uint8)
    return picked, match_mask


def find_nearest_cmyk_recipe(cmyk, target_cmyk_percent):
    cmyk_percent = np.round(cmyk.astype(np.float32) * 100.0 / 255.0).astype(np.uint8)
    flat = cmyk_percent.reshape(-1, 4)
    unique_colors, counts = np.unique(flat, axis=0, return_counts=True)
    target = np.array(target_cmyk_percent, dtype=np.float32)
    distances = np.sqrt(np.mean((unique_colors.astype(np.float32) - target) ** 2, axis=1))
    best_index = int(np.argmin(distances))
    return unique_colors[best_index].tolist(), int(counts[best_index]), float(distances[best_index])


def find_top_cmyk_recipes(cmyk, limit=10):
    cmyk_percent = np.round(cmyk.astype(np.float32) * 100.0 / 255.0).astype(np.uint8)
    flat = cmyk_percent.reshape(-1, 4)
    unique_colors, counts = np.unique(flat, axis=0, return_counts=True)
    order = np.argsort(counts)[::-1][:limit]
    return [
        {
            "cmyk": unique_colors[idx].tolist(),
            "pixels": int(counts[idx]),
        }
        for idx in order
    ]


def create_multi_color_separation(cmyk, selected_recipes, match_mode="exact", spread=0, selection_mode="keep"):
    cmyk_percent = np.round(cmyk.astype(np.float32) * 100.0 / 255.0).astype(np.uint8)
    masks = []
    matched_recipes = []

    for recipe in selected_recipes:
        recipe_arr = np.array(recipe, dtype=np.uint8)
        if match_mode == "nearest":
            nearest_recipe, _nearest_count, _nearest_distance = find_nearest_cmyk_recipe(cmyk, recipe)
            matched_recipe = np.array(nearest_recipe, dtype=np.uint8)
            mask = np.all(cmyk_percent == matched_recipe.reshape(1, 1, 4), axis=2)
        elif match_mode == "spread":
            differences = np.abs(cmyk_percent.astype(np.int16) - recipe_arr.reshape(1, 1, 4).astype(np.int16))
            mask = np.max(differences, axis=2) <= int(spread)
            matched_recipe = recipe_arr
        else:
            mask = np.all(cmyk_percent == recipe_arr.reshape(1, 1, 4), axis=2)
            matched_recipe = recipe_arr

        masks.append(mask)
        matched_recipes.append(matched_recipe.tolist())

    if not masks:
        combined_mask = np.zeros(cmyk.shape[:2], dtype=bool)
    else:
        combined_mask = np.logical_or.reduce(masks)

    if selection_mode == "remove":
        combined_mask = ~combined_mask

    picked = np.zeros_like(cmyk, dtype=np.uint8)
    picked[combined_mask] = cmyk[combined_mask]
    return picked, combined_mask, matched_recipes


def build_csv_report_bytes(report_rows):
    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow(["label", "c", "m", "y", "k", "pixels"])
    for row in report_rows:
        writer.writerow(row)
    return buffer.getvalue().encode("utf-8")


def create_target_grayscale_preview(picked_cmyk, white_threshold):
    ink_strength = np.max(picked_cmyk, axis=2).astype(np.uint8)
    preview = 255 - ink_strength
    preview[preview >= white_threshold] = 255
    return preview


def write_outputs_to_disk(output_folder, file_map):
    os.makedirs(output_folder, exist_ok=True)
    for file_name, data in file_map.items():
        with open(os.path.join(output_folder, file_name), "wb") as handle:
            handle.write(data)


def set_picker_cmyk(c, m, y, k):
    st.session_state["picker_c"] = int(np.clip(c, 0, 100))
    st.session_state["picker_m"] = int(np.clip(m, 0, 100))
    st.session_state["picker_y"] = int(np.clip(y, 0, 100))
    st.session_state["picker_k"] = int(np.clip(k, 0, 100))
    hex_value = cmyk_percent_to_hex(
        st.session_state["picker_c"],
        st.session_state["picker_m"],
        st.session_state["picker_y"],
        st.session_state["picker_k"],
    )
    st.session_state["picker_hex"] = hex_value
    st.session_state["picker_visual"] = hex_value
    st.session_state["picker_hex_error"] = ""


def set_picker_cmyk_text(c, m, y, k):
    st.session_state["picker_c_draft"] = str(int(np.clip(c, 0, 100)))
    st.session_state["picker_m_draft"] = str(int(np.clip(m, 0, 100)))
    st.session_state["picker_y_draft"] = str(int(np.clip(y, 0, 100)))
    st.session_state["picker_k_draft"] = str(int(np.clip(k, 0, 100)))
    st.session_state["sync_cmyk_inputs_from_draft"] = True


def initialize_picker_state():
    if "picker_c" not in st.session_state:
        set_picker_cmyk(100, 0, 100, 0)
    else:
        st.session_state.setdefault(
            "picker_hex",
            cmyk_percent_to_hex(
                st.session_state["picker_c"],
                st.session_state["picker_m"],
                st.session_state["picker_y"],
                st.session_state["picker_k"],
            ),
        )
        st.session_state.setdefault("picker_visual", st.session_state["picker_hex"])
        st.session_state.setdefault("picker_hex_error", "")
    st.session_state.setdefault("picker_c_draft", str(st.session_state["picker_c"]))
    st.session_state.setdefault("picker_m_draft", str(st.session_state["picker_m"]))
    st.session_state.setdefault("picker_y_draft", str(st.session_state["picker_y"]))
    st.session_state.setdefault("picker_k_draft", str(st.session_state["picker_k"]))
    st.session_state.setdefault("picker_c_input", st.session_state["picker_c_draft"])
    st.session_state.setdefault("picker_m_input", st.session_state["picker_m_draft"])
    st.session_state.setdefault("picker_y_input", st.session_state["picker_y_draft"])
    st.session_state.setdefault("picker_k_input", st.session_state["picker_k_draft"])
    st.session_state.setdefault("sync_cmyk_inputs_from_draft", False)
    st.session_state.setdefault("show_cmyk_hover_inspector", False)
    st.session_state.setdefault("current_upload_signature", "")
    st.session_state.setdefault("standard_pipeline_result", None)
    st.session_state.setdefault("target_color_result", None)
    st.session_state.setdefault(
        "active_target_cmyk",
        (
            st.session_state["picker_c"],
            st.session_state["picker_m"],
            st.session_state["picker_y"],
            st.session_state["picker_k"],
        ),
    )


def sync_from_cmyk():
    set_picker_cmyk(
        st.session_state["picker_c"],
        st.session_state["picker_m"],
        st.session_state["picker_y"],
        st.session_state["picker_k"],
    )


def parse_cmyk_text_inputs():
    try:
        values = (
            int(str(st.session_state["picker_c_draft"]).strip()),
            int(str(st.session_state["picker_m_draft"]).strip()),
            int(str(st.session_state["picker_y_draft"]).strip()),
            int(str(st.session_state["picker_k_draft"]).strip()),
        )
    except ValueError:
        return None, "CMYK values must be whole numbers from 0 to 100."

    if any(value < 0 or value > 100 for value in values):
        return None, "CMYK values must stay between 0 and 100."

    return values, ""


def sync_from_hex():
    try:
        c, m, y, k = hex_to_cmyk_percent(st.session_state["picker_hex"])
        set_picker_cmyk(c, m, y, k)
        set_picker_cmyk_text(c, m, y, k)
    except ValueError as exc:
        st.session_state["picker_hex_error"] = str(exc)


def sync_from_visual_picker():
    st.session_state["picker_hex"] = st.session_state["picker_visual"]
    sync_from_hex()


def save_cmyk_input_draft():
    st.session_state["picker_c_draft"] = st.session_state["picker_c_input"]
    st.session_state["picker_m_draft"] = st.session_state["picker_m_input"]
    st.session_state["picker_y_draft"] = st.session_state["picker_y_input"]
    st.session_state["picker_k_draft"] = st.session_state["picker_k_input"]


def sync_cmyk_inputs_from_draft_if_needed():
    if st.session_state.get("sync_cmyk_inputs_from_draft"):
        st.session_state["picker_c_input"] = st.session_state["picker_c_draft"]
        st.session_state["picker_m_input"] = st.session_state["picker_m_draft"]
        st.session_state["picker_y_input"] = st.session_state["picker_y_draft"]
        st.session_state["picker_k_input"] = st.session_state["picker_k_draft"]
        st.session_state["sync_cmyk_inputs_from_draft"] = False


def reset_outputs_for_new_file(upload_signature):
    if st.session_state["current_upload_signature"] != upload_signature:
        st.session_state["current_upload_signature"] = upload_signature
        st.session_state["standard_pipeline_result"] = None
        st.session_state["target_color_result"] = None
        st.session_state["show_cmyk_hover_inspector"] = False


def initialize_studio_state():
    st.session_state.setdefault("studio_settings_store", load_json_store(SETTINGS_STORE_PATH, {}))
    st.session_state.setdefault(
        "studio_library_store",
        load_json_store(
            LIBRARY_STORE_PATH,
            {"recent": [], "favorites": [], "clients": {}, "presets": PRESET_RECIPES},
        ),
    )
    st.session_state.setdefault("studio_history_store", load_json_store(HISTORY_STORE_PATH, []))
    st.session_state.setdefault("studio_selected_colors", [])
    st.session_state.setdefault("studio_match_mode", "exact")
    st.session_state.setdefault("studio_selection_mode", "keep")
    st.session_state.setdefault("studio_spread", 6)
    st.session_state.setdefault("studio_operator", "")
    st.session_state.setdefault("studio_client", "")
    st.session_state.setdefault("studio_revision", "1")
    st.session_state.setdefault("studio_notes", "")
    st.session_state.setdefault("studio_last_job", None)
    st.session_state.setdefault("studio_batch_names", {})
    st.session_state.setdefault("studio_batch_order", {})


def add_recent_color(recipe):
    recent = st.session_state["studio_library_store"]["recent"]
    recent = [item for item in recent if item["cmyk"] != list(recipe)]
    recent.insert(0, {"name": "Recent", "cmyk": list(recipe)})
    st.session_state["studio_library_store"]["recent"] = recent[:12]
    save_json_store(LIBRARY_STORE_PATH, st.session_state["studio_library_store"])


def add_favorite_color(recipe, name="Favorite"):
    favorites = st.session_state["studio_library_store"]["favorites"]
    if not any(item["cmyk"] == list(recipe) for item in favorites):
        favorites.append({"name": name, "cmyk": list(recipe)})
        save_json_store(LIBRARY_STORE_PATH, st.session_state["studio_library_store"])


initialize_picker_state()
initialize_studio_state()


st.title("CMYK Production Studio")
st.caption(
    "Studio copy of the original app with batch workflow, saved settings, color libraries, reports, and advanced CMYK extraction controls."
)

with st.sidebar:
    st.header("Studio")
    st.session_state["studio_operator"] = st.text_input("Operator", value=st.session_state["studio_operator"])
    st.session_state["studio_client"] = st.text_input("Client", value=st.session_state["studio_client"])
    st.session_state["studio_revision"] = st.text_input("Revision", value=st.session_state["studio_revision"])
    st.session_state["studio_notes"] = st.text_area("Notes", value=st.session_state["studio_notes"], height=70)
    st.divider()
    st.header("Settings")
    white_threshold = st.slider(
        "White threshold for grayscale preview",
        min_value=0,
        max_value=255,
        value=245,
        help="Higher values force more light pixels to pure white in the grayscale output.",
    )
    compression_type = st.selectbox("TIFF compression", ["lzw", "deflate", "none"], index=0)
    save_to_disk = st.checkbox("Also save files to a local output folder", value=False)
    output_folder = st.text_input("Local output folder", value="output")
    st.session_state["studio_match_mode"] = st.selectbox(
        "Match mode",
        ["exact", "nearest", "spread"],
        index=["exact", "nearest", "spread"].index(st.session_state["studio_match_mode"]),
    )
    st.session_state["studio_selection_mode"] = st.selectbox(
        "Selection mode",
        ["keep", "remove"],
        index=0 if st.session_state["studio_selection_mode"] == "keep" else 1,
    )
    st.session_state["studio_spread"] = st.slider(
        "Color spread",
        min_value=0,
        max_value=40,
        value=int(st.session_state["studio_spread"]),
    )
    save_profile_name = st.text_input("Save settings profile", value="")
    if st.button("Save current settings", key="save_studio_profile") and save_profile_name.strip():
        st.session_state["studio_settings_store"][save_profile_name.strip()] = {
            "picker": {
                "c": st.session_state["picker_c_draft"],
                "m": st.session_state["picker_m_draft"],
                "y": st.session_state["picker_y_draft"],
                "k": st.session_state["picker_k_draft"],
            },
            "match_mode": st.session_state["studio_match_mode"],
            "selection_mode": st.session_state["studio_selection_mode"],
            "spread": st.session_state["studio_spread"],
            "selected_colors": st.session_state["studio_selected_colors"],
        }
        save_json_store(SETTINGS_STORE_PATH, st.session_state["studio_settings_store"])
    profile_options = [""] + sorted(st.session_state["studio_settings_store"].keys())
    selected_profile = st.selectbox("Load settings profile", profile_options)
    if st.button("Load settings", key="load_studio_profile") and selected_profile:
        profile = st.session_state["studio_settings_store"][selected_profile]
        st.session_state["picker_c_draft"] = profile["picker"]["c"]
        st.session_state["picker_m_draft"] = profile["picker"]["m"]
        st.session_state["picker_y_draft"] = profile["picker"]["y"]
        st.session_state["picker_k_draft"] = profile["picker"]["k"]
        st.session_state["picker_c_input"] = profile["picker"]["c"]
        st.session_state["picker_m_input"] = profile["picker"]["m"]
        st.session_state["picker_y_input"] = profile["picker"]["y"]
        st.session_state["picker_k_input"] = profile["picker"]["k"]
        st.session_state["studio_match_mode"] = profile["match_mode"]
        st.session_state["studio_selection_mode"] = profile["selection_mode"]
        st.session_state["studio_spread"] = profile["spread"]
        st.session_state["studio_selected_colors"] = profile["selected_colors"]
        st.rerun()

uploaded_files = st.file_uploader("Upload CMYK TIFF", type=["tif", "tiff"], accept_multiple_files=True)

if not uploaded_files:
    st.info("Upload one or more CMYK TIFF files to begin.")
else:
    batch_rows = []
    for idx, uploaded_file in enumerate(uploaded_files, start=1):
        signature, file_bytes = build_file_signature(uploaded_file)
        st.session_state["studio_batch_names"].setdefault(signature, Path(uploaded_file.name).stem)
        st.session_state["studio_batch_order"].setdefault(signature, idx)
        batch_rows.append(
            {
                "signature": signature,
                "file": uploaded_file,
                "bytes": file_bytes,
                "name": uploaded_file.name,
            }
        )

    batch_rows.sort(
        key=lambda item: (
            st.session_state["studio_batch_order"][item["signature"]],
            item["name"],
        )
    )

    st.markdown("**Batch Queue**")
    for idx, row in enumerate(batch_rows):
        q1, q2, q3, q4, q5 = st.columns([0.1, 0.42, 0.12, 0.12, 0.24])
        with q1:
            st.session_state["studio_batch_order"][row["signature"]] = st.number_input(
                "Order",
                min_value=1,
                max_value=max(1, len(batch_rows)),
                value=int(st.session_state["studio_batch_order"][row["signature"]]),
                key=f"studio_order_{row['signature']}",
                label_visibility="collapsed",
            )
        with q2:
            st.session_state["studio_batch_names"][row["signature"]] = st.text_input(
                "Job",
                value=st.session_state["studio_batch_names"][row["signature"]],
                key=f"studio_job_name_{row['signature']}",
                label_visibility="collapsed",
            )
        with q3:
            if st.button("Up", key=f"studio_up_{row['signature']}") and idx > 0:
                prev_signature = batch_rows[idx - 1]["signature"]
                st.session_state["studio_batch_order"][row["signature"]], st.session_state["studio_batch_order"][prev_signature] = (
                    st.session_state["studio_batch_order"][prev_signature],
                    st.session_state["studio_batch_order"][row["signature"]],
                )
                st.rerun()
        with q4:
            if st.button("Down", key=f"studio_down_{row['signature']}") and idx < len(batch_rows) - 1:
                next_signature = batch_rows[idx + 1]["signature"]
                st.session_state["studio_batch_order"][row["signature"]], st.session_state["studio_batch_order"][next_signature] = (
                    st.session_state["studio_batch_order"][next_signature],
                    st.session_state["studio_batch_order"][row["signature"]],
                )
                st.rerun()
        with q5:
            st.caption(row["name"])

    batch_rows.sort(
        key=lambda item: (
            st.session_state["studio_batch_order"][item["signature"]],
            item["name"],
        )
    )
    active_signature = st.selectbox(
        "Active job",
        options=[row["signature"] for row in batch_rows],
        format_func=lambda sig: st.session_state["studio_batch_names"][sig],
    )
    active_row = next(row for row in batch_rows if row["signature"] == active_signature)
    uploaded_file = active_row["file"]
    file_bytes = active_row["bytes"]
    original_stem = Path(st.session_state["studio_batch_names"][active_signature]).stem

    try:
        upload_signature = active_signature
        reset_outputs_for_new_file(upload_signature)

        page_options = get_tiff_overview(file_bytes)
        active_page_index = st.selectbox("TIFF page", options=list(range(len(page_options))), format_func=lambda idx: page_options[idx])
        arr, info = read_uploaded_tiff_bytes(file_bytes, active_page_index)
        cmyk = extract_cmyk(arr)
        preview_display_width = get_preview_display_width(cmyk)

        st.success("TIFF loaded successfully.")

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Shape", str(info["shape"]))
        col2.metric("Dtype", info["dtype"])
        col3.metric("Photometric", info["photometric"])
        col4.metric("Compression", info["compression"])

        standard_tab, target_tab = st.tabs(
            ["Standard CMYK Pipeline", "Selected Color Separation"]
        )

        with standard_tab:
            st.markdown("Generate the original 4-color separation set and grayscale plate previews.")

            if st.button("Run full CMYK pipeline", type="primary"):
                separations = create_separations(cmyk)
                separation_bytes = {}
                preview_bytes = {}
                preview_images = {}
                zip_files = {}

                for color, sep_array in separations.items():
                    separation_file = f"{original_stem}_{color}.tif"
                    preview_file = f"{original_stem}_{color}_plate_preview.tif"
                    separation_bytes[color] = save_cmyk_tiff_bytes(sep_array)

                    plate = sep_array[:, :, CHANNEL_INDEX[color]]
                    preview_tiff, preview_gray = save_preview_tiff_bytes(
                        plate,
                        white_threshold=white_threshold,
                    )
                    preview_bytes[color] = preview_tiff
                    preview_images[color] = preview_gray
                    zip_files[separation_file] = separation_bytes[color]
                    zip_files[preview_file] = preview_bytes[color]

                if save_to_disk:
                    write_outputs_to_disk(output_folder, zip_files)

                st.session_state["standard_pipeline_result"] = {
                    "upload_signature": upload_signature,
                    "original_stem": original_stem,
                    "separation_bytes": separation_bytes,
                    "preview_bytes": preview_bytes,
                    "preview_images": preview_images,
                    "zip_bytes": build_zip(zip_files),
                    "saved_to_disk": save_to_disk,
                    "output_folder": os.path.abspath(output_folder),
                }

            standard_result = st.session_state["standard_pipeline_result"]
            if standard_result and standard_result["upload_signature"] == upload_signature:
                st.subheader("Plate preview")
                preview_cols = st.columns(4)
                for idx, color in enumerate(CHANNELS):
                    with preview_cols[idx]:
                        st.markdown(f"**{color.title()}**")
                        st.image(
                            pil_from_gray(standard_result["preview_images"][color]),
                            use_container_width=True,
                        )

                st.subheader("Downloads")
                for color in CHANNELS:
                    c1, c2 = st.columns(2)
                    with c1:
                        st.download_button(
                            label=f"Download {color.title()} separation TIFF",
                            data=standard_result["separation_bytes"][color],
                            file_name=f"{standard_result['original_stem']}_{color}.tif",
                            mime="image/tiff",
                            key=f"sep_{color}",
                        )
                    with c2:
                        st.download_button(
                            label=f"Download {color.title()} plate preview TIFF",
                            data=standard_result["preview_bytes"][color],
                            file_name=f"{standard_result['original_stem']}_{color}_plate_preview.tif",
                            mime="image/tiff",
                            key=f"prev_{color}",
                        )

                st.download_button(
                    label="Download all outputs as ZIP",
                    data=standard_result["zip_bytes"],
                    file_name=f"{standard_result['original_stem']}_cmyk_outputs.zip",
                    mime="application/zip",
                )

                st.success("Standard CMYK pipeline completed.")
                if standard_result["saved_to_disk"]:
                    st.info(f"Files also saved to: {standard_result['output_folder']}")

        with target_tab:
            st.markdown(
                "Build a color job with one or more CMYK recipes. Use exact, nearest, or spread matching, then keep or remove the matched colors."
            )
            typed_target_cmyk, cmyk_input_error = parse_cmyk_text_inputs()
            current_target_cmyk = typed_target_cmyk or st.session_state["active_target_cmyk"]

            preview_rgb = cmyk_to_rgb_preview(cmyk)
            st.markdown("**TIFF Preview**")
            inspector_cols = st.columns([0.32, 0.68])
            with inspector_cols[0]:
                button_label = "On" if st.session_state["show_cmyk_hover_inspector"] else "Pick"
                if st.button(button_label, key="toggle_hover_inspector"):
                    st.session_state["show_cmyk_hover_inspector"] = (
                        not st.session_state["show_cmyk_hover_inspector"]
                    )
            with inspector_cols[1]:
                if st.session_state["show_cmyk_hover_inspector"]:
                    st.caption(
                        "Hover on the TIFF preview to see the CMYK percentages at that area."
                    )
                else:
                    st.caption(
                        "Press the button to turn the TIFF preview into a CMYK hover inspector."
                    )

            if st.session_state["show_cmyk_hover_inspector"]:
                inspector_figure = create_cmyk_hover_figure(cmyk, display_width=preview_display_width)
                if plotly_events is not None:
                    picked_points = plotly_events(
                        inspector_figure,
                        click_event=True,
                        hover_event=False,
                        select_event=False,
                        override_width=min(preview_display_width, 1200),
                        override_height=max(320, int(inspector_figure.layout.height)),
                        key=f"studio_pick_{upload_signature}_{active_page_index}",
                    )
                    if picked_points:
                        picked_point = picked_points[-1]
                        picked_x = int(round(picked_point["x"]))
                        picked_y = int(round(picked_point["y"]))
                        _hover_preview_rgb, hover_customdata = build_hover_preview_data(
                            cmyk,
                            render_width=min(cmyk.shape[1], max(preview_display_width * 2, 1800)),
                        )
                        if 0 <= picked_y < hover_customdata.shape[0] and 0 <= picked_x < hover_customdata.shape[1]:
                            picked = hover_customdata[picked_y, picked_x]
                            st.session_state["picker_c_draft"] = str(int(picked[0]))
                            st.session_state["picker_m_draft"] = str(int(picked[1]))
                            st.session_state["picker_y_draft"] = str(int(picked[2]))
                            st.session_state["picker_k_draft"] = str(int(picked[3]))
                            st.session_state["picker_c_input"] = st.session_state["picker_c_draft"]
                            st.session_state["picker_m_input"] = st.session_state["picker_m_draft"]
                            st.session_state["picker_y_input"] = st.session_state["picker_y_draft"]
                            st.session_state["picker_k_input"] = st.session_state["picker_k_draft"]
                            st.success(
                                f"Picked color: C {picked[0]} | M {picked[1]} | Y {picked[2]} | K {picked[3]}"
                            )
                else:
                    st.plotly_chart(
                        inspector_figure,
                        use_container_width=True,
                        config={"displayModeBar": True},
                    )
            else:
                st.image(
                    pil_from_rgb(preview_rgb),
                    width=preview_display_width,
                )

            st.markdown("**CMYK Input**")
            with st.form("target_cmyk_form", enter_to_submit=False):
                input_col1, input_col2, input_col3, input_col4 = st.columns(4)
                with input_col1:
                    st.text_input(
                        "C",
                        key="picker_c_input",
                    )
                with input_col2:
                    st.text_input(
                        "M",
                        key="picker_m_input",
                    )
                with input_col3:
                    st.text_input(
                        "Y",
                        key="picker_y_input",
                    )
                with input_col4:
                    st.text_input(
                        "K",
                        key="picker_k_input",
                    )

                add_selected_color_button = st.form_submit_button(
                    "Add color to selection list"
                )
                extract_selected_color = st.form_submit_button(
                    "Process selected color job",
                    type="primary",
                )

            if cmyk_input_error:
                st.warning(cmyk_input_error)

            preview_hex = cmyk_percent_to_hex(
                current_target_cmyk[0],
                current_target_cmyk[1],
                current_target_cmyk[2],
                current_target_cmyk[3],
            )
            st.markdown(
                f"""
                <div style="display: flex; align-items: center; gap: 10px; margin: 8px 0 12px 0;">
                    <div style="width: 34px; height: 34px; border-radius: 6px; border: 1px solid #d7d7d7; background: {preview_hex};"></div>
                    <div style="font-size: 14px;">Entered color preview</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.info(
                f"Current CMYK: C {current_target_cmyk[0]} | "
                f"M {current_target_cmyk[1]} | "
                f"Y {current_target_cmyk[2]} | "
                f"K {current_target_cmyk[3]}"
            )
            st.caption(
                "Typing in these boxes does not reload the page. Add one or more colors, then process the selected job."
            )

            preset_cols = st.columns(5)
            for idx, preset in enumerate(PRESET_RECIPES):
                with preset_cols[idx]:
                    if st.button(preset["name"], key=f"studio_preset_{idx}"):
                        st.session_state["picker_c_draft"] = str(preset["cmyk"][0])
                        st.session_state["picker_m_draft"] = str(preset["cmyk"][1])
                        st.session_state["picker_y_draft"] = str(preset["cmyk"][2])
                        st.session_state["picker_k_draft"] = str(preset["cmyk"][3])
                        st.session_state["picker_c_input"] = st.session_state["picker_c_draft"]
                        st.session_state["picker_m_input"] = st.session_state["picker_m_draft"]
                        st.session_state["picker_y_input"] = st.session_state["picker_y_draft"]
                        st.session_state["picker_k_input"] = st.session_state["picker_k_draft"]
                        st.rerun()

            action_cols = st.columns(4)
            if action_cols[0].button("Favorite current"):
                if typed_target_cmyk:
                    add_favorite_color(typed_target_cmyk, name="Favorite")
            if action_cols[1].button("Duplicate last job"):
                if st.session_state["studio_last_job"] is not None:
                    st.session_state["studio_selected_colors"] = json.loads(
                        json.dumps(st.session_state["studio_last_job"]["selected_colors"])
                    )
                    st.rerun()
            if action_cols[2].button("Clear current job"):
                st.session_state["studio_selected_colors"] = []
                st.rerun()
            if action_cols[3].button("Use current color only"):
                if typed_target_cmyk:
                    st.session_state["studio_selected_colors"] = [{"name": "Current", "cmyk": list(typed_target_cmyk)}]
                    st.rerun()

            library_col1, library_col2 = st.columns(2)
            with library_col1:
                st.markdown("**Recent colors**")
                for idx, item in enumerate(st.session_state["studio_library_store"]["recent"][:6]):
                    if st.button(
                        f"C {item['cmyk'][0]} M {item['cmyk'][1]} Y {item['cmyk'][2]} K {item['cmyk'][3]}",
                        key=f"recent_color_{idx}",
                    ):
                        st.session_state["picker_c_draft"] = str(item["cmyk"][0])
                        st.session_state["picker_m_draft"] = str(item["cmyk"][1])
                        st.session_state["picker_y_draft"] = str(item["cmyk"][2])
                        st.session_state["picker_k_draft"] = str(item["cmyk"][3])
                        st.session_state["picker_c_input"] = st.session_state["picker_c_draft"]
                        st.session_state["picker_m_input"] = st.session_state["picker_m_draft"]
                        st.session_state["picker_y_input"] = st.session_state["picker_y_draft"]
                        st.session_state["picker_k_input"] = st.session_state["picker_k_draft"]
                        st.rerun()
            with library_col2:
                st.markdown("**Favorites**")
                for idx, item in enumerate(st.session_state["studio_library_store"]["favorites"][:6]):
                    if st.button(
                        f"{item['name']} - C {item['cmyk'][0]} M {item['cmyk'][1]} Y {item['cmyk'][2]} K {item['cmyk'][3]}",
                        key=f"favorite_color_{idx}",
                    ):
                        st.session_state["picker_c_draft"] = str(item["cmyk"][0])
                        st.session_state["picker_m_draft"] = str(item["cmyk"][1])
                        st.session_state["picker_y_draft"] = str(item["cmyk"][2])
                        st.session_state["picker_k_draft"] = str(item["cmyk"][3])
                        st.session_state["picker_c_input"] = st.session_state["picker_c_draft"]
                        st.session_state["picker_m_input"] = st.session_state["picker_m_draft"]
                        st.session_state["picker_y_input"] = st.session_state["picker_y_draft"]
                        st.session_state["picker_k_input"] = st.session_state["picker_k_draft"]
                        st.rerun()

            if add_selected_color_button and not cmyk_input_error:
                st.session_state["studio_selected_colors"].append(
                    {
                        "name": f"Color {len(st.session_state['studio_selected_colors']) + 1}",
                        "cmyk": list(typed_target_cmyk),
                    }
                )
                add_recent_color(typed_target_cmyk)
                st.rerun()

            st.markdown("**Selected job colors**")
            for idx, item in enumerate(st.session_state["studio_selected_colors"]):
                sel1, sel2 = st.columns([0.8, 0.2])
                sel1.caption(
                    f"{item['name']} | C {item['cmyk'][0]} M {item['cmyk'][1]} Y {item['cmyk'][2]} K {item['cmyk'][3]}"
                )
                if sel2.button("Remove", key=f"remove_selected_color_{idx}"):
                    st.session_state["studio_selected_colors"].pop(idx)
                    st.rerun()

            if extract_selected_color:
                save_cmyk_input_draft()
                typed_target_cmyk, cmyk_input_error = parse_cmyk_text_inputs()
                if cmyk_input_error:
                    st.warning(cmyk_input_error)
                    st.stop()

                target_cmyk = typed_target_cmyk
                st.session_state["active_target_cmyk"] = target_cmyk
                selected_recipes = [item["cmyk"] for item in st.session_state["studio_selected_colors"]] or [list(target_cmyk)]
                started_at = time.perf_counter()
                picked_cmyk, match_mask, matched_recipes = create_multi_color_separation(
                    cmyk,
                    selected_recipes,
                    match_mode=st.session_state["studio_match_mode"],
                    spread=st.session_state["studio_spread"],
                    selection_mode=st.session_state["studio_selection_mode"],
                )
                processing_seconds = time.perf_counter() - started_at
                matched_pixels = int(match_mask.sum())
                matched_percent = (matched_pixels / match_mask.size) * 100 if match_mask.size else 0.0

                file_prefix = (
                    f"{original_stem}_page{active_page_index + 1}_"
                    f"{st.session_state['studio_selection_mode']}_{st.session_state['studio_match_mode']}"
                )

                if matched_pixels == 0:
                    nearest_cmyk, nearest_count, nearest_distance = find_nearest_cmyk_recipe(
                        cmyk,
                        target_cmyk,
                    )
                    st.session_state["target_color_result"] = {
                        "upload_signature": upload_signature,
                        "status": "no_match",
                        "warning": "No exact CMYK match was found in this TIFF for the selected recipe.",
                        "nearest_message": (
                            f"Nearest CMYK recipe in the TIFF: "
                            f"C {nearest_cmyk[0]} | M {nearest_cmyk[1]} | Y {nearest_cmyk[2]} | K {nearest_cmyk[3]} "
                            f"({nearest_count:,} pixels, distance {nearest_distance:.2f})."
                        ),
                    }
                else:
                    color_tiff_bytes = save_cmyk_tiff_bytes(picked_cmyk)
                    grayscale_preview = create_target_grayscale_preview(
                        picked_cmyk,
                        white_threshold=white_threshold,
                    )
                    grayscale_tiff_bytes = save_gray_tiff_bytes(grayscale_preview)
                    coverage_mask_bytes = save_gray_tiff_bytes(np.where(match_mask, 0, 255).astype(np.uint8))
                    report_rows = []
                    rounded_percent = np.round(cmyk.astype(np.float32) * 100.0 / 255.0).astype(np.uint8)
                    for idx, recipe in enumerate(matched_recipes):
                        recipe_mask = np.all(
                            rounded_percent == np.array(recipe, dtype=np.uint8).reshape(1, 1, 4),
                            axis=2,
                        )
                        report_rows.append(
                            [
                                f"color_{idx + 1}",
                                recipe[0],
                                recipe[1],
                                recipe[2],
                                recipe[3],
                                int(recipe_mask.sum()),
                            ]
                        )
                    csv_report_bytes = build_csv_report_bytes(report_rows)

                    output_files = {
                        f"{file_prefix}.tif": color_tiff_bytes,
                        f"{file_prefix}_grayscale.tif": grayscale_tiff_bytes,
                        f"{file_prefix}_coverage_mask.tif": coverage_mask_bytes,
                        f"{file_prefix}_report.csv": csv_report_bytes,
                    }

                    if save_to_disk:
                        write_outputs_to_disk(output_folder, output_files)

                    st.session_state["target_color_result"] = {
                        "upload_signature": upload_signature,
                        "status": "matched",
                        "color_preview": cmyk_to_rgb_preview(picked_cmyk),
                        "grayscale_preview": grayscale_preview,
                        "matched_pixels": matched_pixels,
                        "matched_percent": matched_percent,
                        "processing_seconds": processing_seconds,
                        "matched_recipes": matched_recipes,
                        "color_tiff_bytes": color_tiff_bytes,
                        "grayscale_tiff_bytes": grayscale_tiff_bytes,
                        "coverage_mask_bytes": coverage_mask_bytes,
                        "csv_report_bytes": csv_report_bytes,
                        "zip_bytes": build_zip(output_files),
                        "file_prefix": file_prefix,
                        "saved_to_disk": save_to_disk,
                        "output_folder": os.path.abspath(output_folder),
                    }
                    st.session_state["studio_last_job"] = {
                        "selected_colors": json.loads(json.dumps(st.session_state["studio_selected_colors"])),
                    }
                    st.session_state["studio_history_store"].append(
                        {
                            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            "file": uploaded_file.name,
                            "job_name": st.session_state["studio_batch_names"][upload_signature],
                            "page": active_page_index + 1,
                            "match_mode": st.session_state["studio_match_mode"],
                            "selection_mode": st.session_state["studio_selection_mode"],
                            "matched_pixels": matched_pixels,
                            "matched_area": round(matched_percent, 2),
                            "processing_seconds": round(processing_seconds, 3),
                            "operator": st.session_state["studio_operator"],
                            "client": st.session_state["studio_client"],
                        }
                    )
                    save_json_store(HISTORY_STORE_PATH, st.session_state["studio_history_store"][-200:])

            target_result = st.session_state["target_color_result"]
            if target_result and target_result["upload_signature"] == upload_signature:
                if target_result["status"] == "no_match":
                    st.warning(target_result["warning"])
                    st.info(target_result["nearest_message"])
                else:
                    st.info(
                        f"Processing time: {target_result['processing_seconds']:.3f}s | "
                        f"Match mode: {st.session_state['studio_match_mode']} | "
                        f"Selection mode: {st.session_state['studio_selection_mode']}"
                    )
                    preview_col, gray_col = st.columns(2)
                    with preview_col:
                        st.markdown("**Selected color separation**")
                        st.image(
                            pil_from_rgb(target_result["color_preview"]),
                            use_container_width=True,
                        )
                    with gray_col:
                        st.markdown("**Grayscale output**")
                        st.image(
                            pil_from_gray(target_result["grayscale_preview"]),
                            use_container_width=True,
                        )

                    stats_col1, stats_col2 = st.columns(2)
                    stats_col1.metric("Matched pixels", f"{target_result['matched_pixels']:,}")
                    stats_col2.metric("Matched area", f"{target_result['matched_percent']:.2f}%")

                    d1, d2 = st.columns(2)
                    with d1:
                        st.download_button(
                            label="Download selected color TIFF",
                            data=target_result["color_tiff_bytes"],
                            file_name=f"{target_result['file_prefix']}.tif",
                            mime="image/tiff",
                            key="download_target_color",
                        )
                    with d2:
                        st.download_button(
                            label="Download selected grayscale TIFF",
                            data=target_result["grayscale_tiff_bytes"],
                            file_name=f"{target_result['file_prefix']}_grayscale.tif",
                            mime="image/tiff",
                            key="download_target_gray",
                        )

                    d3, d4 = st.columns(2)
                    with d3:
                        st.download_button(
                            label="Download coverage mask TIFF",
                            data=target_result["coverage_mask_bytes"],
                            file_name=f"{target_result['file_prefix']}_coverage_mask.tif",
                            mime="image/tiff",
                            key="download_target_mask",
                        )
                    with d4:
                        st.download_button(
                            label="Download CSV report",
                            data=target_result["csv_report_bytes"],
                            file_name=f"{target_result['file_prefix']}_report.csv",
                            mime="text/csv",
                            key="download_target_report",
                        )

                    st.download_button(
                        label="Download selected-color outputs as ZIP",
                        data=target_result["zip_bytes"],
                        file_name=f"{target_result['file_prefix']}_outputs.zip",
                        mime="application/zip",
                    )

                    st.markdown("**Matched CMYK recipes**")
                    for idx, recipe in enumerate(target_result["matched_recipes"]):
                        st.caption(
                            f"{idx + 1}. C {recipe[0]} M {recipe[1]} Y {recipe[2]} K {recipe[3]}"
                        )

                    st.markdown("**Most-used CMYK recipes in this artwork**")
                    for item in find_top_cmyk_recipes(cmyk, limit=8):
                        st.caption(
                            f"C {item['cmyk'][0]} M {item['cmyk'][1]} Y {item['cmyk'][2]} K {item['cmyk'][3]} | {item['pixels']:,} px"
                        )

                    st.success("Selected color extraction completed.")
                    if target_result["saved_to_disk"]:
                        st.info(f"Files also saved to: {target_result['output_folder']}")

        with st.expander("Libraries And Reports", expanded=False):
            left_col, right_col = st.columns(2)
            with left_col:
                st.markdown("**Client library**")
                client_key = st.session_state["studio_client"] or "default_client"
                st.session_state["studio_library_store"]["clients"].setdefault(client_key, [])
                if st.button("Save selected job colors to client library"):
                    st.session_state["studio_library_store"]["clients"][client_key] = json.loads(
                        json.dumps(st.session_state["studio_selected_colors"])
                    )
                    save_json_store(LIBRARY_STORE_PATH, st.session_state["studio_library_store"])
                for idx, item in enumerate(st.session_state["studio_library_store"]["clients"][client_key][:10]):
                    if st.button(
                        f"{item['name']} - C {item['cmyk'][0]} M {item['cmyk'][1]} Y {item['cmyk'][2]} K {item['cmyk'][3]}",
                        key=f"client_library_color_{idx}",
                    ):
                        st.session_state["studio_selected_colors"].append(item)
                        st.rerun()
            with right_col:
                st.markdown("**Job history**")
                for item in reversed(st.session_state["studio_history_store"][-12:]):
                    st.caption(
                        f"{item['timestamp']} | {item['file']} | {item['job_name']} | "
                        f"{item['matched_pixels']:,} px | {item['matched_area']:.2f}% | {item['processing_seconds']:.3f}s"
                    )

            st.download_button(
                "Export color libraries JSON",
                data=json.dumps(st.session_state["studio_library_store"], indent=2).encode("utf-8"),
                file_name="studio_color_libraries.json",
                mime="application/json",
                key="download_studio_library_json",
            )

    except Exception as exc:
        st.error(f"Failed to process TIFF: {exc}")
