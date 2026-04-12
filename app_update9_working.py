import io
import os
import hashlib
import zipfile
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import streamlit as st
import tifffile
from PIL import Image


st.set_page_config(page_title="CMYK TIFF Separation Dashboard", layout="wide")

CHANNELS = ["cyan", "magenta", "yellow", "black"]
CHANNEL_INDEX = {name: idx for idx, name in enumerate(CHANNELS)}
MAX_TARGET_RECIPES = 2
DEFAULT_TARGET_RECIPE = (100, 0, 100, 0)
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


def read_uploaded_tiff(uploaded_file):
    uploaded_file.seek(0)
    with tifffile.TiffFile(uploaded_file) as tif:
        page = tif.pages[0]
        arr = page.asarray()
        info = {
            "shape": page.shape,
            "dtype": str(page.dtype),
            "photometric": str(page.photometric),
            "compression": str(page.compression),
        }
    return arr, info


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


def create_target_grayscale_preview(picked_cmyk, white_threshold):
    ink_strength = np.max(picked_cmyk, axis=2).astype(np.uint8)
    preview = 255 - ink_strength
    preview[preview >= white_threshold] = 255
    return preview


def analyze_target_recipes(cmyk, target_recipes):
    combined_picked_cmyk = np.zeros_like(cmyk, dtype=np.uint8)
    combined_removed_cmyk = cmyk.copy()
    combined_match_mask = np.zeros(cmyk.shape[:2], dtype=bool)
    recipe_stats = []

    for recipe_index, target_cmyk in enumerate(target_recipes, start=1):
        picked_cmyk, match_mask = create_target_color_separation(
            cmyk,
            target_cmyk,
        )
        recipe_matched_pixels = int(match_mask.sum())
        recipe_matched_percent = (
            (recipe_matched_pixels / match_mask.size) * 100 if match_mask.size else 0.0
        )
        recipe_stat = {
            "recipe_index": recipe_index,
            "target_cmyk": target_cmyk,
            "matched_pixels": recipe_matched_pixels,
            "matched_percent": recipe_matched_percent,
        }

        if recipe_matched_pixels == 0:
            nearest_cmyk, nearest_count, nearest_distance = find_nearest_cmyk_recipe(
                cmyk,
                target_cmyk,
            )
            recipe_stat["nearest_message"] = (
                f"Nearest CMYK recipe in the TIFF: "
                f"C {nearest_cmyk[0]} | M {nearest_cmyk[1]} | Y {nearest_cmyk[2]} | K {nearest_cmyk[3]} "
                f"({nearest_count:,} pixels, distance {nearest_distance:.2f})."
            )
        else:
            combined_picked_cmyk[match_mask] = picked_cmyk[match_mask]
            combined_removed_cmyk[match_mask] = 0
            combined_match_mask |= match_mask

        recipe_stats.append(recipe_stat)

    matched_pixels = int(combined_match_mask.sum())
    matched_percent = (
        (matched_pixels / combined_match_mask.size) * 100
        if combined_match_mask.size
        else 0.0
    )
    warning_message = (
        "No exact CMYK match was found in this TIFF for the selected recipe."
        if len(target_recipes) == 1
        else "No exact CMYK matches were found in this TIFF for the selected recipes."
    )

    return {
        "picked_cmyk": combined_picked_cmyk,
        "removed_cmyk": combined_removed_cmyk,
        "match_mask": combined_match_mask,
        "matched_pixels": matched_pixels,
        "matched_percent": matched_percent,
        "recipe_stats": recipe_stats,
        "warning_message": warning_message,
    }


def write_outputs_to_disk(output_folder, file_map):
    os.makedirs(output_folder, exist_ok=True)
    for file_name, data in file_map.items():
        with open(os.path.join(output_folder, file_name), "wb") as handle:
            handle.write(data)


def get_target_recipe_prefix(index):
    return "picker" if index == 1 else f"picker{index}"


def get_target_recipe_input_key(index, channel):
    return f"{get_target_recipe_prefix(index)}_{channel}_input"


def get_target_recipe_draft_key(index, channel):
    return f"{get_target_recipe_prefix(index)}_{channel}_draft"


def set_target_recipe_text(index, c, m, y, k):
    values = {"c": c, "m": m, "y": y, "k": k}
    for channel, value in values.items():
        st.session_state[get_target_recipe_draft_key(index, channel)] = str(
            int(np.clip(value, 0, 100))
        )
        st.session_state[get_target_recipe_input_key(index, channel)] = str(
            int(np.clip(value, 0, 100))
        )


def save_target_recipe_input_drafts():
    for index in range(1, MAX_TARGET_RECIPES + 1):
        for channel in ("c", "m", "y", "k"):
            input_key = get_target_recipe_input_key(index, channel)
            draft_key = get_target_recipe_draft_key(index, channel)
            if input_key in st.session_state:
                st.session_state[draft_key] = st.session_state[input_key]


def parse_target_recipe_text_input(index, allow_blank=False):
    raw_values = [
        str(st.session_state.get(get_target_recipe_draft_key(index, channel), "")).strip()
        for channel in ("c", "m", "y", "k")
    ]
    label = f"Separation {index}"

    if allow_blank and all(value == "" for value in raw_values):
        return None, ""

    if any(value == "" for value in raw_values):
        if allow_blank:
            return None, f"{label}: enter all C, M, Y, K values or leave the block blank."
        return None, "CMYK values must be whole numbers from 0 to 100."

    try:
        values = tuple(int(value) for value in raw_values)
    except ValueError:
        if index == 1:
            return None, "CMYK values must be whole numbers from 0 to 100."
        return None, f"{label}: CMYK values must be whole numbers from 0 to 100."

    if any(value < 0 or value > 100 for value in values):
        if index == 1:
            return None, "CMYK values must stay between 0 and 100."
        return None, f"{label}: CMYK values must stay between 0 and 100."

    return values, ""


def parse_target_recipe_text_inputs():
    recipes = []
    recipe_count = int(st.session_state.get("target_recipe_count", 1))
    for index in range(1, min(recipe_count, MAX_TARGET_RECIPES) + 1):
        recipe, error = parse_target_recipe_text_input(index, allow_blank=index > 1)
        if error:
            return None, error
        if recipe is not None:
            recipes.append(recipe)

    if not recipes:
        return None, "Enter at least one CMYK recipe."

    return recipes, ""


def build_target_file_prefix(original_stem, target_recipes, operation="picked"):
    recipe_tokens = [
        f"c{recipe[0]}_m{recipe[1]}_y{recipe[2]}_k{recipe[3]}"
        for recipe in target_recipes
    ]
    if len(recipe_tokens) == 1:
        return f"{original_stem}_{operation}_{recipe_tokens[0]}"
    return f"{original_stem}_{operation}_combined_{'__'.join(recipe_tokens)}"


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
        set_picker_cmyk(*DEFAULT_TARGET_RECIPE)
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
    st.session_state.setdefault("target_remove_result", None)
    st.session_state.setdefault(
        "active_target_cmyk",
        (
            st.session_state["picker_c"],
            st.session_state["picker_m"],
            st.session_state["picker_y"],
            st.session_state["picker_k"],
        ),
    )
    st.session_state.setdefault(
        "active_target_cmyks",
        [tuple(st.session_state["active_target_cmyk"])],
    )
    active_target_cmyks = [
        tuple(recipe)
        for recipe in st.session_state.get("active_target_cmyks", [])
        if recipe is not None
    ][:MAX_TARGET_RECIPES]
    if not active_target_cmyks:
        active_target_cmyks = [tuple(st.session_state["active_target_cmyk"])]
    st.session_state["active_target_cmyks"] = active_target_cmyks
    st.session_state.setdefault(
        "target_recipe_count",
        min(MAX_TARGET_RECIPES, max(1, len(active_target_cmyks))),
    )

    for index in range(1, MAX_TARGET_RECIPES + 1):
        if index <= len(active_target_cmyks):
            default_values = active_target_cmyks[index - 1]
            default_strings = {
                "c": str(default_values[0]),
                "m": str(default_values[1]),
                "y": str(default_values[2]),
                "k": str(default_values[3]),
            }
        else:
            default_strings = {"c": "", "m": "", "y": "", "k": ""}

        for channel in ("c", "m", "y", "k"):
            draft_key = get_target_recipe_draft_key(index, channel)
            input_key = get_target_recipe_input_key(index, channel)
            st.session_state.setdefault(draft_key, default_strings[channel])
            st.session_state.setdefault(input_key, st.session_state[draft_key])


def sync_from_cmyk():
    set_picker_cmyk(
        st.session_state["picker_c"],
        st.session_state["picker_m"],
        st.session_state["picker_y"],
        st.session_state["picker_k"],
    )


def parse_cmyk_text_inputs():
    return parse_target_recipe_text_input(1, allow_blank=False)


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
    save_target_recipe_input_drafts()


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
        st.session_state["target_remove_result"] = None
        st.session_state["show_cmyk_hover_inspector"] = False


initialize_picker_state()


st.title("CMYK TIFF Separation Dashboard")
st.caption(
    "Upload one CMYK TIFF, generate the 4 standard CMYK separations, extract only the selected CMYK color, or remove selected CMYK colors to white and download both color and grayscale TIFF files."
)

with st.sidebar:
    st.header("Settings")
    white_threshold = st.slider(
        "White threshold for grayscale preview",
        min_value=0,
        max_value=255,
        value=245,
        help="Higher values force more light pixels to pure white in the grayscale output.",
    )
    save_to_disk = st.checkbox("Also save files to a local output folder", value=False)
    output_folder = st.text_input("Local output folder", value="output")

uploaded_file = st.file_uploader("Upload CMYK TIFF", type=["tif", "tiff"])

if uploaded_file is None:
    st.info("Upload a CMYK TIFF to begin.")
else:
    original_stem = Path(uploaded_file.name).stem

    try:
        upload_signature, file_bytes = build_file_signature(uploaded_file)
        reset_outputs_for_new_file(upload_signature)

        arr, info = read_uploaded_tiff(io.BytesIO(file_bytes))
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
                "Pick one or two target CMYK recipes. You can either keep only those colors and remove all others, or remove those colors from the TIFF and replace them with white."
            )
            typed_target_recipes, cmyk_input_error = parse_target_recipe_text_inputs()
            current_target_recipes = typed_target_recipes or st.session_state["active_target_cmyks"]

            preview_rgb = cmyk_to_rgb_preview(cmyk)
            st.markdown("**TIFF Preview**")
            inspector_cols = st.columns([0.32, 0.68])
            with inspector_cols[0]:
                button_label = "Inspect on" if st.session_state["show_cmyk_hover_inspector"] else "Inspect off"
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
                st.plotly_chart(
                    create_cmyk_hover_figure(cmyk, display_width=preview_display_width),
                    use_container_width=True,
                    config={"displayModeBar": False},
                )
            else:
                st.image(
                    pil_from_rgb(preview_rgb),
                    width=preview_display_width,
                )

            st.markdown("**CMYK Input**")
            add_button_cols = st.columns([0.28, 0.72])
            with add_button_cols[0]:
                if st.session_state["target_recipe_count"] < MAX_TARGET_RECIPES:
                    if st.button("+ Add separation", key="add_target_recipe"):
                        st.session_state["target_recipe_count"] = MAX_TARGET_RECIPES
                else:
                    st.caption("Maximum 2 separations.")

            with st.form("target_cmyk_form", enter_to_submit=False):
                for recipe_index in range(1, st.session_state["target_recipe_count"] + 1):
                    st.markdown(f"**Separation {recipe_index}**")
                    input_col1, input_col2, input_col3, input_col4 = st.columns(4)
                    with input_col1:
                        st.text_input(
                            "C",
                            key=get_target_recipe_input_key(recipe_index, "c"),
                        )
                    with input_col2:
                        st.text_input(
                            "M",
                            key=get_target_recipe_input_key(recipe_index, "m"),
                        )
                    with input_col3:
                        st.text_input(
                            "Y",
                            key=get_target_recipe_input_key(recipe_index, "y"),
                        )
                    with input_col4:
                        st.text_input(
                            "K",
                            key=get_target_recipe_input_key(recipe_index, "k"),
                        )

                    if recipe_index == 2:
                        st.caption(
                            "Optional second recipe. Leave all four fields blank to keep single-recipe behavior."
                        )

                action_col1, action_col2 = st.columns(2)
                with action_col1:
                    extract_selected_color = st.form_submit_button(
                        "Extract selected CMYK color",
                        type="primary",
                    )
                with action_col2:
                    remove_selected_color = st.form_submit_button(
                        "Remove selected CMYK color",
                    )

            if cmyk_input_error:
                st.warning(cmyk_input_error)

            preview_cols = st.columns(len(current_target_recipes))
            for recipe_index, recipe in enumerate(current_target_recipes, start=1):
                preview_hex = cmyk_percent_to_hex(
                    recipe[0],
                    recipe[1],
                    recipe[2],
                    recipe[3],
                )
                with preview_cols[recipe_index - 1]:
                    st.markdown(
                        f"""
                        <div style="display: flex; align-items: center; gap: 10px; margin: 8px 0 12px 0;">
                            <div style="width: 34px; height: 34px; border-radius: 6px; border: 1px solid #d7d7d7; background: {preview_hex};"></div>
                            <div style="font-size: 14px;">Separation {recipe_index} preview</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                    st.caption(
                        f"C {recipe[0]} | M {recipe[1]} | Y {recipe[2]} | K {recipe[3]}"
                    )

            st.caption(
                "Typing in these boxes does not reload the page. The values are applied only when you click the button."
            )

            if extract_selected_color or remove_selected_color:
                save_target_recipe_input_drafts()
                typed_target_recipes, cmyk_input_error = parse_target_recipe_text_inputs()
                if cmyk_input_error:
                    st.warning(cmyk_input_error)
                    st.stop()

                target_recipes = typed_target_recipes
                st.session_state["active_target_cmyk"] = target_recipes[0]
                st.session_state["active_target_cmyks"] = target_recipes
                analysis = analyze_target_recipes(cmyk, target_recipes)
                result_key = "target_color_result" if extract_selected_color else "target_remove_result"
                result_label = "picked" if extract_selected_color else "removed"
                output_cmyk = analysis["picked_cmyk"] if extract_selected_color else analysis["removed_cmyk"]
                file_prefix = build_target_file_prefix(
                    original_stem,
                    target_recipes,
                    operation=result_label,
                )

                if analysis["matched_pixels"] == 0:
                    st.session_state[result_key] = {
                        "upload_signature": upload_signature,
                        "status": "no_match",
                        "warning": analysis["warning_message"],
                        "target_recipes": target_recipes,
                        "recipe_stats": analysis["recipe_stats"],
                    }
                else:
                    color_tiff_bytes = save_cmyk_tiff_bytes(output_cmyk)
                    grayscale_preview = create_target_grayscale_preview(
                        output_cmyk,
                        white_threshold=white_threshold,
                    )
                    grayscale_tiff_bytes = save_gray_tiff_bytes(grayscale_preview)

                    output_files = {
                        f"{file_prefix}.tif": color_tiff_bytes,
                        f"{file_prefix}_grayscale.tif": grayscale_tiff_bytes,
                    }

                    if save_to_disk:
                        write_outputs_to_disk(output_folder, output_files)

                    st.session_state[result_key] = {
                        "upload_signature": upload_signature,
                        "status": "matched",
                        "target_recipes": target_recipes,
                        "recipe_stats": analysis["recipe_stats"],
                        "color_preview": cmyk_to_rgb_preview(output_cmyk),
                        "grayscale_preview": grayscale_preview,
                        "matched_pixels": analysis["matched_pixels"],
                        "matched_percent": analysis["matched_percent"],
                        "color_tiff_bytes": color_tiff_bytes,
                        "grayscale_tiff_bytes": grayscale_tiff_bytes,
                        "zip_bytes": build_zip(output_files),
                        "file_prefix": file_prefix,
                        "saved_to_disk": save_to_disk,
                        "output_folder": os.path.abspath(output_folder),
                    }

            target_result = st.session_state["target_color_result"]
            if target_result and target_result["upload_signature"] == upload_signature:
                st.markdown("**Per-recipe match info**")
                for recipe_stat in target_result["recipe_stats"]:
                    recipe = recipe_stat["target_cmyk"]
                    recipe_label = (
                        f"Separation {recipe_stat['recipe_index']}: "
                        f"C {recipe[0]} | M {recipe[1]} | Y {recipe[2]} | K {recipe[3]}"
                    )
                    if recipe_stat["matched_pixels"] > 0:
                        st.info(
                            f"{recipe_label} matched {recipe_stat['matched_pixels']:,} pixels "
                            f"({recipe_stat['matched_percent']:.2f}%)."
                        )
                    else:
                        st.warning(f"{recipe_label} found no exact match.")
                        st.caption(recipe_stat["nearest_message"])

                if target_result["status"] == "no_match":
                    st.warning(target_result["warning"])
                else:
                    preview_col, gray_col = st.columns(2)
                    with preview_col:
                        preview_title = (
                            "**Combined selected color separation**"
                            if len(target_result["target_recipes"]) > 1
                            else "**Selected color separation**"
                        )
                        st.markdown(preview_title)
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
                    stats_col1.metric(
                        "Combined matched pixels",
                        f"{target_result['matched_pixels']:,}",
                    )
                    stats_col2.metric(
                        "Combined matched area",
                        f"{target_result['matched_percent']:.2f}%",
                    )

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

                    st.download_button(
                        label="Download selected-color outputs as ZIP",
                        data=target_result["zip_bytes"],
                        file_name=f"{target_result['file_prefix']}_outputs.zip",
                        mime="application/zip",
                    )

                    st.success("Selected color extraction completed.")
                    if target_result["saved_to_disk"]:
                        st.info(f"Files also saved to: {target_result['output_folder']}")

            target_remove_result = st.session_state["target_remove_result"]
            if target_remove_result and target_remove_result["upload_signature"] == upload_signature:
                st.markdown("**Per-recipe removal match info**")
                for recipe_stat in target_remove_result["recipe_stats"]:
                    recipe = recipe_stat["target_cmyk"]
                    recipe_label = (
                        f"Separation {recipe_stat['recipe_index']}: "
                        f"C {recipe[0]} | M {recipe[1]} | Y {recipe[2]} | K {recipe[3]}"
                    )
                    if recipe_stat["matched_pixels"] > 0:
                        st.info(
                            f"{recipe_label} matched {recipe_stat['matched_pixels']:,} pixels "
                            f"({recipe_stat['matched_percent']:.2f}%)."
                        )
                    else:
                        st.warning(f"{recipe_label} found no exact match.")
                        st.caption(recipe_stat["nearest_message"])

                if target_remove_result["status"] == "no_match":
                    st.warning(target_remove_result["warning"])
                else:
                    preview_col, gray_col = st.columns(2)
                    with preview_col:
                        preview_title = (
                            "**TIFF after removing selected colors to white**"
                            if len(target_remove_result["target_recipes"]) > 1
                            else "**TIFF after removing selected color to white**"
                        )
                        st.markdown(preview_title)
                        st.image(
                            pil_from_rgb(target_remove_result["color_preview"]),
                            use_container_width=True,
                        )
                    with gray_col:
                        st.markdown("**Grayscale output after removal**")
                        st.image(
                            pil_from_gray(target_remove_result["grayscale_preview"]),
                            use_container_width=True,
                        )

                    stats_col1, stats_col2 = st.columns(2)
                    stats_col1.metric(
                        "Removed pixels",
                        f"{target_remove_result['matched_pixels']:,}",
                    )
                    stats_col2.metric(
                        "Removed area",
                        f"{target_remove_result['matched_percent']:.2f}%",
                    )

                    d1, d2 = st.columns(2)
                    with d1:
                        st.download_button(
                            label="Download removed-color TIFF",
                            data=target_remove_result["color_tiff_bytes"],
                            file_name=f"{target_remove_result['file_prefix']}.tif",
                            mime="image/tiff",
                            key="download_removed_target_color",
                        )
                    with d2:
                        st.download_button(
                            label="Download removed-color grayscale TIFF",
                            data=target_remove_result["grayscale_tiff_bytes"],
                            file_name=f"{target_remove_result['file_prefix']}_grayscale.tif",
                            mime="image/tiff",
                            key="download_removed_target_gray",
                        )

                    st.download_button(
                        label="Download removed-color outputs as ZIP",
                        data=target_remove_result["zip_bytes"],
                        file_name=f"{target_remove_result['file_prefix']}_outputs.zip",
                        mime="application/zip",
                    )

                    st.success("Selected color removal completed.")
                    if target_remove_result["saved_to_disk"]:
                        st.info(f"Files also saved to: {target_remove_result['output_folder']}")

    except Exception as exc:
        st.error(f"Failed to process TIFF: {exc}")
