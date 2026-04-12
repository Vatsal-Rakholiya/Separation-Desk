import hashlib
import io
import os
import zipfile
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import streamlit as st
import tifffile
from PIL import Image


st.set_page_config(page_title="Fast CMYK TIFF Separation Dashboard", layout="wide")

CHANNELS = ["cyan", "magenta", "yellow", "black"]
CHANNEL_INDEX = {name: idx for idx, name in enumerate(CHANNELS)}
DEFAULT_TARGET_CMYK = (100, 0, 100, 0)


def normalize_to_uint8(arr):
    if arr.dtype == np.uint8:
        return arr
    if arr.dtype == np.uint16:
        return ((arr.astype(np.uint32) + 128) // 257).astype(np.uint8)
    raise ValueError(f"Unsupported dtype: {arr.dtype}")


def extract_cmyk(arr):
    arr = normalize_to_uint8(arr)

    if arr.ndim != 3:
        raise ValueError(f"Unsupported TIFF shape: {arr.shape}")

    if arr.shape[2] in (4, 5):
        return arr[:, :, :4]

    if arr.shape[0] in (4, 5):
        return np.transpose(arr[:4, :, :], (1, 2, 0))

    raise ValueError(f"Cannot interpret TIFF as CMYK. Shape: {arr.shape}")


def cmyk_to_percent_array(cmyk_array):
    return ((cmyk_array.astype(np.uint16) * 100 + 127) // 255).astype(np.uint8)


def percent_to_uint8(value):
    value = int(np.clip(value, 0, 100))
    return (value * 255 + 50) // 100


def cmyk_percent_to_rgb(c, m, y, k):
    c = np.clip(c, 0, 100) / 100.0
    m = np.clip(m, 0, 100) / 100.0
    y = np.clip(y, 0, 100) / 100.0
    k = np.clip(k, 0, 100) / 100.0
    r = int(round(255 * (1 - c) * (1 - k)))
    g = int(round(255 * (1 - m) * (1 - k)))
    b = int(round(255 * (1 - y) * (1 - k)))
    return r, g, b


def cmyk_percent_to_hex(c, m, y, k):
    return "#{:02X}{:02X}{:02X}".format(*cmyk_percent_to_rgb(c, m, y, k))


def cmyk_to_rgb_preview(cmyk_array):
    cmyk_16 = cmyk_array.astype(np.uint16)
    c = cmyk_16[:, :, 0]
    m = cmyk_16[:, :, 1]
    y = cmyk_16[:, :, 2]
    k = cmyk_16[:, :, 3]

    rgb = np.stack(
        [
            ((255 - c) * (255 - k) + 127) // 255,
            ((255 - m) * (255 - k) + 127) // 255,
            ((255 - y) * (255 - k) + 127) // 255,
        ],
        axis=2,
    )
    return rgb.astype(np.uint8)


def pil_from_gray(gray_array):
    return Image.fromarray(gray_array.astype(np.uint8), mode="L")


def pil_from_rgb(rgb_array):
    return Image.fromarray(rgb_array.astype(np.uint8), mode="RGB")


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


def create_target_grayscale_preview(mask, target_uint8, white_threshold):
    ink_value = int(np.max(target_uint8))
    preview = np.full(mask.shape, 255, dtype=np.uint8)
    if ink_value > 0:
        preview[mask] = 255 - ink_value
    preview[preview >= white_threshold] = 255
    return preview


def build_zip(file_map):
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        for file_name, data in file_map.items():
            zf.writestr(file_name, data)
    zip_buffer.seek(0)
    return zip_buffer.getvalue()


def write_outputs_to_disk(output_folder, file_map):
    os.makedirs(output_folder, exist_ok=True)
    for file_name, data in file_map.items():
        with open(os.path.join(output_folder, file_name), "wb") as handle:
            handle.write(data)


def build_file_signature(file_bytes):
    return hashlib.sha1(file_bytes).hexdigest()


def get_preview_display_width(cmyk_array):
    width = int(cmyk_array.shape[1])
    return min(1280, max(760, width))


@st.cache_data(show_spinner=False)
def load_tiff_payload(file_bytes):
    with tifffile.TiffFile(io.BytesIO(file_bytes)) as tif:
        page = tif.pages[0]
        arr = page.asarray()
        info = {
            "shape": page.shape,
            "dtype": str(page.dtype),
            "photometric": str(page.photometric),
            "compression": str(page.compression),
        }

    cmyk = extract_cmyk(arr)
    cmyk_percent = cmyk_to_percent_array(cmyk)
    rgb_preview = cmyk_to_rgb_preview(cmyk)

    return {
        "cmyk": cmyk,
        "cmyk_percent": cmyk_percent,
        "rgb_preview": rgb_preview,
        "info": info,
    }


@st.cache_data(show_spinner=False)
def get_unique_cmyk_recipes(file_bytes):
    payload = load_tiff_payload(file_bytes)
    flat = payload["cmyk_percent"].reshape(-1, 4)
    unique_colors, counts = np.unique(flat, axis=0, return_counts=True)
    return unique_colors, counts


@st.cache_data(show_spinner=False)
def build_hover_preview_data(file_bytes, display_width):
    payload = load_tiff_payload(file_bytes)
    rgb_preview = payload["rgb_preview"]
    cmyk_percent = payload["cmyk_percent"]
    height, width = rgb_preview.shape[:2]

    render_width = min(width, max(display_width * 3, 2200))
    if width <= render_width:
        x_idx = np.arange(width, dtype=int)
        y_idx = np.arange(height, dtype=int)
    else:
        scaled_height = max(1, int(round(height * (render_width / width))))
        x_idx = np.linspace(0, width - 1, render_width).round().astype(int)
        y_idx = np.linspace(0, height - 1, scaled_height).round().astype(int)

    preview_percent = cmyk_percent[np.ix_(y_idx, x_idx)]
    preview_rgb = np.array(
        Image.fromarray(rgb_preview, mode="RGB").resize(
            (len(x_idx), len(y_idx)),
            Image.Resampling.LANCZOS,
        )
    )

    source_x = np.broadcast_to(x_idx.reshape(1, -1), (len(y_idx), len(x_idx)))
    source_y = np.broadcast_to(y_idx.reshape(-1, 1), (len(y_idx), len(x_idx)))
    customdata = np.dstack([preview_percent, source_x, source_y]).astype(np.int32)

    return preview_rgb, customdata


def create_hover_figure(preview_rgb, customdata, display_width):
    preview_height, preview_width = preview_rgb.shape[:2]
    figure_height = max(320, int(round(preview_height * (display_width / max(preview_width, 1)))))

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
        height=figure_height,
        margin=dict(l=0, r=0, t=0, b=0),
        hoverlabel=dict(
            bgcolor="white",
            bordercolor="#2e7d32",
            font=dict(color="black", size=18),
        ),
    )
    fig.update_xaxes(showticklabels=False, showgrid=False, zeroline=False)
    fig.update_yaxes(showticklabels=False, showgrid=False, zeroline=False, scaleanchor="x")
    return fig


@st.cache_data(show_spinner=False)
def build_standard_outputs(file_bytes, white_threshold):
    payload = load_tiff_payload(file_bytes)
    cmyk = payload["cmyk"]
    height, width, _ = cmyk.shape
    zeros = np.zeros((height, width), dtype=np.uint8)

    separation_bytes = {}
    preview_bytes = {}
    preview_images = {}
    output_files = {}

    for color in CHANNELS:
        idx = CHANNEL_INDEX[color]
        plates = [zeros, zeros, zeros, zeros]
        plates[idx] = cmyk[:, :, idx]
        separation = np.stack(plates, axis=2)
        separation_bytes[color] = save_cmyk_tiff_bytes(separation)

        preview_gray = create_plate_preview(cmyk[:, :, idx], white_threshold=white_threshold)
        preview_images[color] = preview_gray
        preview_bytes[color] = save_gray_tiff_bytes(preview_gray)

    return {
        "separation_bytes": separation_bytes,
        "preview_bytes": preview_bytes,
        "preview_images": preview_images,
    }


@st.cache_data(show_spinner=False)
def build_target_outputs(file_bytes, target_cmyk, white_threshold):
    payload = load_tiff_payload(file_bytes)
    cmyk = payload["cmyk"]
    cmyk_percent = payload["cmyk_percent"]
    target = np.array(target_cmyk, dtype=np.uint8)
    target_uint8 = np.array([percent_to_uint8(channel) for channel in target_cmyk], dtype=np.uint8)
    match_mask = np.all(cmyk_percent == target.reshape(1, 1, 4), axis=2)
    matched_pixels = int(match_mask.sum())
    matched_percent = (matched_pixels / match_mask.size) * 100 if match_mask.size else 0.0

    if matched_pixels == 0:
        unique_colors, counts = get_unique_cmyk_recipes(file_bytes)
        target_float = target.astype(np.float32)
        distances = np.sqrt(np.mean((unique_colors.astype(np.float32) - target_float) ** 2, axis=1))
        best_index = int(np.argmin(distances))
        return {
            "status": "no_match",
            "matched_pixels": 0,
            "matched_percent": 0.0,
            "nearest_cmyk": unique_colors[best_index].tolist(),
            "nearest_count": int(counts[best_index]),
            "nearest_distance": float(distances[best_index]),
        }

    picked_cmyk = np.zeros_like(cmyk, dtype=np.uint8)
    picked_cmyk[match_mask] = target_uint8

    target_rgb = np.array(cmyk_percent_to_rgb(*target_cmyk), dtype=np.uint8)
    color_preview = np.full((match_mask.shape[0], match_mask.shape[1], 3), 255, dtype=np.uint8)
    color_preview[match_mask] = target_rgb

    grayscale_preview = create_target_grayscale_preview(
        match_mask,
        target_uint8,
        white_threshold=white_threshold,
    )
    color_tiff_bytes = save_cmyk_tiff_bytes(picked_cmyk)
    grayscale_tiff_bytes = save_gray_tiff_bytes(grayscale_preview)

    return {
        "status": "matched",
        "matched_pixels": matched_pixels,
        "matched_percent": matched_percent,
        "color_preview": color_preview,
        "grayscale_preview": grayscale_preview,
        "color_tiff_bytes": color_tiff_bytes,
        "grayscale_tiff_bytes": grayscale_tiff_bytes,
    }


def initialize_state():
    st.session_state.setdefault("input_c", str(DEFAULT_TARGET_CMYK[0]))
    st.session_state.setdefault("input_m", str(DEFAULT_TARGET_CMYK[1]))
    st.session_state.setdefault("input_y", str(DEFAULT_TARGET_CMYK[2]))
    st.session_state.setdefault("input_k", str(DEFAULT_TARGET_CMYK[3]))
    st.session_state.setdefault("active_target_cmyk", DEFAULT_TARGET_CMYK)
    st.session_state.setdefault("show_hover_inspector_fast", False)
    st.session_state.setdefault("current_upload_signature_fast", "")
    st.session_state.setdefault("standard_result_fast", None)
    st.session_state.setdefault("target_result_fast", None)


def reset_state_for_upload(upload_signature):
    if st.session_state["current_upload_signature_fast"] != upload_signature:
        st.session_state["current_upload_signature_fast"] = upload_signature
        st.session_state["standard_result_fast"] = None
        st.session_state["target_result_fast"] = None
        st.session_state["show_hover_inspector_fast"] = False


def parse_form_cmyk_inputs():
    try:
        values = (
            int(str(st.session_state["input_c"]).strip()),
            int(str(st.session_state["input_m"]).strip()),
            int(str(st.session_state["input_y"]).strip()),
            int(str(st.session_state["input_k"]).strip()),
        )
    except ValueError:
        return None, "CMYK values must be whole numbers from 0 to 100."

    if any(value < 0 or value > 100 for value in values):
        return None, "CMYK values must stay between 0 and 100."

    return values, ""


def render_standard_outputs(result):
    st.subheader("Plate preview")
    preview_cols = st.columns(4)
    for idx, color in enumerate(CHANNELS):
        with preview_cols[idx]:
            st.markdown(f"**{color.title()}**")
            st.image(pil_from_gray(result["preview_images"][color]), use_container_width=True)

    st.subheader("Downloads")
    for color in CHANNELS:
        c1, c2 = st.columns(2)
        with c1:
            st.download_button(
                label=f"Download {color.title()} separation TIFF",
                data=result["output_files"][f"{result['original_stem']}_{color}.tif"],
                file_name=f"{result['original_stem']}_{color}.tif",
                mime="image/tiff",
                key=f"fast_sep_{color}",
            )
        with c2:
            st.download_button(
                label=f"Download {color.title()} plate preview TIFF",
                data=result["output_files"][f"{result['original_stem']}_{color}_plate_preview.tif"],
                file_name=f"{result['original_stem']}_{color}_plate_preview.tif",
                mime="image/tiff",
                key=f"fast_prev_{color}",
            )

    st.download_button(
        label="Download all outputs as ZIP",
        data=result["zip_bytes"],
        file_name=f"{result['original_stem']}_cmyk_outputs.zip",
        mime="application/zip",
        key="fast_standard_zip",
    )

    st.success("Standard CMYK pipeline completed.")
    if result["saved_to_disk"]:
        st.info(f"Files also saved to: {result['output_folder']}")


def render_target_outputs(result):
    if result["status"] == "no_match":
        st.warning("No exact CMYK match was found in this TIFF for the selected recipe.")
        st.info(
            f"Nearest CMYK recipe in the TIFF: "
            f"C {result['nearest_cmyk'][0]} | M {result['nearest_cmyk'][1]} | "
            f"Y {result['nearest_cmyk'][2]} | K {result['nearest_cmyk'][3]} "
            f"({result['nearest_count']:,} pixels, distance {result['nearest_distance']:.2f})."
        )
        return

    preview_col, gray_col = st.columns(2)
    with preview_col:
        st.markdown("**Selected color separation**")
        st.image(pil_from_rgb(result["color_preview"]), use_container_width=True)
    with gray_col:
        st.markdown("**Grayscale output**")
        st.image(pil_from_gray(result["grayscale_preview"]), use_container_width=True)

    stats_col1, stats_col2 = st.columns(2)
    stats_col1.metric("Matched pixels", f"{result['matched_pixels']:,}")
    stats_col2.metric("Matched area", f"{result['matched_percent']:.2f}%")

    d1, d2 = st.columns(2)
    with d1:
        st.download_button(
            label="Download selected color TIFF",
            data=result["output_files"][f"{result['file_prefix']}.tif"],
            file_name=f"{result['file_prefix']}.tif",
            mime="image/tiff",
            key="fast_target_color",
        )
    with d2:
        st.download_button(
            label="Download selected grayscale TIFF",
            data=result["output_files"][f"{result['file_prefix']}_grayscale.tif"],
            file_name=f"{result['file_prefix']}_grayscale.tif",
            mime="image/tiff",
            key="fast_target_gray",
        )

    st.download_button(
        label="Download selected-color outputs as ZIP",
        data=result["zip_bytes"],
        file_name=f"{result['file_prefix']}_outputs.zip",
        mime="application/zip",
        key="fast_target_zip",
    )

    st.success("Selected color extraction completed.")
    if result["saved_to_disk"]:
        st.info(f"Files also saved to: {result['output_folder']}")


initialize_state()


st.title("Fast CMYK TIFF Separation Dashboard")
st.caption(
    "This version is optimized with caching so TIFF loading, preview generation, and exact CMYK extraction rerun much faster."
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

uploaded_file = st.file_uploader("Upload CMYK TIFF", type=["tif", "tiff"], key="fast_upload")

if uploaded_file is None:
    st.info("Upload a CMYK TIFF to begin.")
else:
    original_stem = Path(uploaded_file.name).stem
    file_bytes = uploaded_file.getvalue()
    upload_signature = build_file_signature(file_bytes)
    reset_state_for_upload(upload_signature)

    try:
        with st.spinner("Loading TIFF..."):
            payload = load_tiff_payload(file_bytes)

        cmyk = payload["cmyk"]
        info = payload["info"]
        preview_rgb = payload["rgb_preview"]
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

            if st.button("Run full CMYK pipeline", type="primary", key="fast_run_standard"):
                with st.spinner("Building CMYK pipeline outputs..."):
                    computed = build_standard_outputs(file_bytes, white_threshold)

                output_files = {}
                for color in CHANNELS:
                    output_files[f"{original_stem}_{color}.tif"] = computed["separation_bytes"][color]
                    output_files[f"{original_stem}_{color}_plate_preview.tif"] = computed["preview_bytes"][color]

                if save_to_disk:
                    write_outputs_to_disk(output_folder, output_files)

                st.session_state["standard_result_fast"] = {
                    "upload_signature": upload_signature,
                    "original_stem": original_stem,
                    "preview_images": computed["preview_images"],
                    "output_files": output_files,
                    "zip_bytes": build_zip(output_files),
                    "saved_to_disk": save_to_disk,
                    "output_folder": os.path.abspath(output_folder),
                }

            standard_result = st.session_state["standard_result_fast"]
            if standard_result and standard_result["upload_signature"] == upload_signature:
                render_standard_outputs(standard_result)

        with target_tab:
            st.markdown(
                "Pick one target CMYK color. The app keeps only pixels whose CMYK channels match that exact CMYK recipe and removes all other colors."
            )

            st.markdown("**TIFF Preview**")
            control_col, help_col = st.columns([0.24, 0.76])
            with control_col:
                toggle_label = "[+]" if not st.session_state["show_hover_inspector_fast"] else "[x]"
                if st.button(toggle_label, key="fast_toggle_hover"):
                    st.session_state["show_hover_inspector_fast"] = not st.session_state["show_hover_inspector_fast"]
            with help_col:
                if st.session_state["show_hover_inspector_fast"]:
                    st.caption("Hover on the TIFF preview to see the CMYK percentages at that area.")
                else:
                    st.caption("Press the button to turn the TIFF preview into a CMYK hover inspector.")

            if st.session_state["show_hover_inspector_fast"]:
                with st.spinner("Preparing sharp hover preview..."):
                    hover_rgb, hover_customdata = build_hover_preview_data(file_bytes, preview_display_width)
                st.plotly_chart(
                    create_hover_figure(hover_rgb, hover_customdata, preview_display_width),
                    use_container_width=True,
                    config={"displayModeBar": False},
                )
            else:
                st.image(pil_from_rgb(preview_rgb), width=preview_display_width)

            with st.form("fast_target_cmyk_form", enter_to_submit=False):
                st.markdown("**CMYK Input**")
                input_col1, input_col2, input_col3, input_col4 = st.columns(4)
                with input_col1:
                    st.text_input("C", key="input_c")
                with input_col2:
                    st.text_input("M", key="input_m")
                with input_col3:
                    st.text_input("Y", key="input_y")
                with input_col4:
                    st.text_input("K", key="input_k")

                extract_selected_color = st.form_submit_button(
                    "Extract selected CMYK color",
                    type="primary",
                )

            current_target_cmyk = st.session_state["active_target_cmyk"]
            preview_hex = cmyk_percent_to_hex(
                current_target_cmyk[0],
                current_target_cmyk[1],
                current_target_cmyk[2],
                current_target_cmyk[3],
            )
            st.markdown(
                f"""
                <div style="display:flex; align-items:center; gap:10px; margin:8px 0 12px 0;">
                    <div style="width:34px; height:34px; border-radius:6px; border:1px solid #d7d7d7; background:{preview_hex};"></div>
                    <div style="font-size:14px;">Applied color preview</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.info(
                f"Applied CMYK: C {current_target_cmyk[0]} | "
                f"M {current_target_cmyk[1]} | "
                f"Y {current_target_cmyk[2]} | "
                f"K {current_target_cmyk[3]}"
            )
            st.caption("The boxes do not rerun the app while typing. Values are applied only when you click the button.")

            if extract_selected_color:
                target_cmyk, cmyk_input_error = parse_form_cmyk_inputs()
                if cmyk_input_error:
                    st.warning(cmyk_input_error)
                else:
                    st.session_state["active_target_cmyk"] = target_cmyk
                    with st.spinner("Extracting selected CMYK color..."):
                        computed = build_target_outputs(file_bytes, target_cmyk, white_threshold)

                    if computed["status"] == "matched":
                        file_prefix = (
                            f"{original_stem}_picked_"
                            f"c{target_cmyk[0]}_m{target_cmyk[1]}_y{target_cmyk[2]}_k{target_cmyk[3]}"
                        )
                        output_files = {
                            f"{file_prefix}.tif": computed["color_tiff_bytes"],
                            f"{file_prefix}_grayscale.tif": computed["grayscale_tiff_bytes"],
                        }
                        if save_to_disk:
                            write_outputs_to_disk(output_folder, output_files)

                        computed["file_prefix"] = file_prefix
                        computed["output_files"] = output_files
                        computed["zip_bytes"] = build_zip(output_files)
                        computed["saved_to_disk"] = save_to_disk
                        computed["output_folder"] = os.path.abspath(output_folder)
                    else:
                        computed["saved_to_disk"] = False

                    computed["upload_signature"] = upload_signature
                    st.session_state["target_result_fast"] = computed

            target_result = st.session_state["target_result_fast"]
            if target_result and target_result["upload_signature"] == upload_signature:
                render_target_outputs(target_result)

    except Exception as exc:
        st.error(f"Failed to process TIFF: {exc}")
