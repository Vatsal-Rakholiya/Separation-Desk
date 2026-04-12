import csv
import io
import os
import zipfile
from pathlib import Path

import numpy as np
import streamlit as st
import tifffile
from PIL import Image, ImageDraw


st.set_page_config(page_title="CMYK Detail Recovery Tool", layout="wide")


def normalize_to_uint8(arr):
    if arr.dtype == np.uint8:
        return arr
    if arr.dtype == np.uint16:
        return (arr / 257.0).round().astype(np.uint8)
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


def save_csv_report_bytes(rows):
    buf = io.StringIO()
    writer = csv.DictWriter(
        buf,
        fieldnames=[
            "mode",
            "c",
            "m",
            "y",
            "k",
            "pixels",
        ],
    )
    writer.writeheader()
    writer.writerows(rows)
    return buf.getvalue().encode("utf-8")


def build_zip(file_map):
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        for file_name, data in file_map.items():
            zf.writestr(file_name, data)
    zip_buffer.seek(0)
    return zip_buffer.getvalue()


def pil_from_gray(gray_array):
    return Image.fromarray(gray_array.astype(np.uint8), mode="L")


def pil_from_rgb(rgb_array):
    return Image.fromarray(rgb_array.astype(np.uint8), mode="RGB")


def percent_to_uint8(value):
    return int(round(np.clip(value, 0, 100) * 255 / 100))


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


def create_target_grayscale_preview(picked_cmyk, white_threshold):
    ink_strength = np.max(picked_cmyk, axis=2).astype(np.uint8)
    preview = 255 - ink_strength
    preview[preview >= white_threshold] = 255
    return preview


def resize_preview_image(rgb_array, max_width=1100):
    image = pil_from_rgb(rgb_array)
    scale = min(1.0, max_width / image.width)
    if scale == 1.0:
        return image, scale
    resized = image.resize(
        (int(round(image.width * scale)), int(round(image.height * scale))),
        Image.Resampling.LANCZOS,
    )
    return resized, scale


def draw_sampler_marker(preview_image, sample_x, sample_y, scale):
    marked = preview_image.copy()
    draw = ImageDraw.Draw(marked)
    px = int(round(sample_x * scale))
    py = int(round(sample_y * scale))
    marker_size = max(8, int(round(18 * scale)))
    draw.line((px - marker_size, py, px + marker_size, py), fill="#FF2D55", width=2)
    draw.line((px, py - marker_size, px, py + marker_size), fill="#FF2D55", width=2)
    draw.ellipse(
        (px - marker_size, py - marker_size, px + marker_size, py + marker_size),
        outline="#FFFFFF",
        width=2,
    )
    return marked


def create_zoom_crop(rgb_array, sample_x, sample_y, radius=90, scale=3):
    h, w = rgb_array.shape[:2]
    x0 = max(0, sample_x - radius)
    y0 = max(0, sample_y - radius)
    x1 = min(w, sample_x + radius + 1)
    y1 = min(h, sample_y + radius + 1)
    crop = pil_from_rgb(rgb_array[y0:y1, x0:x1])
    enlarged = crop.resize((crop.width * scale, crop.height * scale), Image.Resampling.NEAREST)
    draw = ImageDraw.Draw(enlarged)
    cx = (sample_x - x0) * scale
    cy = (sample_y - y0) * scale
    draw.line((cx - 16, cy, cx + 16, cy), fill="#FF2D55", width=2)
    draw.line((cx, cy - 16, cx, cy + 16), fill="#FF2D55", width=2)
    return enlarged


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


def initialize_picker_state():
    if "picker_c" not in st.session_state:
        set_picker_cmyk(89, 0, 50, 59)
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


def sync_from_cmyk():
    set_picker_cmyk(
        st.session_state["picker_c"],
        st.session_state["picker_m"],
        st.session_state["picker_y"],
        st.session_state["picker_k"],
    )


def sync_from_hex():
    try:
        c, m, y, k = hex_to_cmyk_percent(st.session_state["picker_hex"])
        set_picker_cmyk(c, m, y, k)
    except ValueError as exc:
        st.session_state["picker_hex_error"] = str(exc)


def sync_from_visual_picker():
    st.session_state["picker_hex"] = st.session_state["picker_visual"]
    sync_from_hex()


def sample_pixel_recipe(cmyk, sample_x, sample_y):
    pixel = cmyk[sample_y, sample_x]
    sampled = tuple(int(round(value * 100 / 255)) for value in pixel)
    set_picker_cmyk(*sampled)
    return sampled


def build_exact_masks(cmyk, target_cmyk_percent):
    cmyk_percent = np.round(cmyk.astype(np.float32) * 100.0 / 255.0).astype(np.uint8)
    target = np.array(target_cmyk_percent, dtype=np.uint8)

    exact_mask = np.all(cmyk_percent == target.reshape(1, 1, 4), axis=2)
    related_mask = np.zeros_like(exact_mask)

    final_mask = exact_mask | related_mask
    picked = np.zeros_like(cmyk, dtype=np.uint8)
    target_u8 = np.array([percent_to_uint8(v) for v in target_cmyk_percent], dtype=np.uint8)
    picked[final_mask] = target_u8

    return picked, exact_mask, related_mask, final_mask, cmyk_percent


def build_recipe_report_rows(cmyk_percent, exact_mask, related_mask):
    rows = []
    for mode_name, mask in [("exact", exact_mask), ("detail_recovery", related_mask)]:
        if not np.any(mask):
            continue
        flat = cmyk_percent[mask]
        unique_colors, counts = np.unique(flat, axis=0, return_counts=True)
        order = np.argsort(-counts)
        for index in order:
            rows.append(
                {
                    "mode": mode_name,
                    "c": int(unique_colors[index, 0]),
                    "m": int(unique_colors[index, 1]),
                    "y": int(unique_colors[index, 2]),
                    "k": int(unique_colors[index, 3]),
                    "pixels": int(counts[index]),
                }
            )
    return rows


def write_outputs_to_disk(output_folder, file_map):
    os.makedirs(output_folder, exist_ok=True)
    for file_name, data in file_map.items():
        with open(os.path.join(output_folder, file_name), "wb") as handle:
            handle.write(data)


initialize_picker_state()

st.title("CMYK Detail Recovery Tool")
st.caption(
    "Companion app for exact CMYK separation, TIFF pixel sampling, and recipe reporting."
)

with st.sidebar:
    st.header("Settings")
    white_threshold = st.slider(
        "White threshold for grayscale preview",
        min_value=0,
        max_value=255,
        value=245,
    )
    save_to_disk = st.checkbox("Also save files to a local output folder", value=False)
    output_folder = st.text_input("Local output folder", value="output")

uploaded_file = st.file_uploader("Upload CMYK TIFF", type=["tif", "tiff"])

if uploaded_file is None:
    st.info("Upload a CMYK TIFF to begin.")
else:
    original_stem = Path(uploaded_file.name).stem

    try:
        arr, info = read_uploaded_tiff(uploaded_file)
        cmyk = extract_cmyk(arr)
        rgb_preview = cmyk_to_rgb_preview(cmyk)
        preview_image, preview_scale = resize_preview_image(rgb_preview)

        st.success("TIFF loaded successfully.")

        info_col1, info_col2, info_col3, info_col4 = st.columns(4)
        info_col1.metric("Shape", str(info["shape"]))
        info_col2.metric("Dtype", info["dtype"])
        info_col3.metric("Photometric", info["photometric"])
        info_col4.metric("Compression", info["compression"])

        h, w = cmyk.shape[:2]
        st.subheader("TIFF Pixel Sampler")
        st.caption(
            "This file is separate from your main app. Use the X/Y sampler to pull the real CMYK recipe from the TIFF, then run exact-only separation."
        )

        sample_col1, sample_col2 = st.columns([1.5, 1.1])
        with sample_col1:
            xy_col1, xy_col2 = st.columns(2)
            with xy_col1:
                sample_x = st.number_input("Sample X", min_value=0, max_value=w - 1, value=min(100, w - 1))
            with xy_col2:
                sample_y = st.number_input("Sample Y", min_value=0, max_value=h - 1, value=min(100, h - 1))

            if st.button("Sample TIFF pixel into CMYK picker"):
                sampled = sample_pixel_recipe(cmyk, int(sample_x), int(sample_y))
                st.success(
                    f"Sampled TIFF pixel at ({int(sample_x)}, {int(sample_y)}): "
                    f"C {sampled[0]} | M {sampled[1]} | Y {sampled[2]} | K {sampled[3]}"
                )
                st.rerun()

            marked_preview = draw_sampler_marker(preview_image, int(sample_x), int(sample_y), preview_scale)
            st.image(marked_preview, use_container_width=True)

        with sample_col2:
            zoom_image = create_zoom_crop(rgb_preview, int(sample_x), int(sample_y))
            st.image(zoom_image, caption="Zoomed TIFF preview", use_container_width=True)

            sampled_pixel = cmyk[int(sample_y), int(sample_x)]
            sampled_cmyk = tuple(int(round(v * 100 / 255)) for v in sampled_pixel)
            sampled_hex = cmyk_percent_to_hex(*sampled_cmyk)
            st.info(
                f"Pixel ({int(sample_x)}, {int(sample_y)})\n\n"
                f"CMYK: C {sampled_cmyk[0]} | M {sampled_cmyk[1]} | Y {sampled_cmyk[2]} | K {sampled_cmyk[3]}\n\n"
                f"HEX: {sampled_hex}"
            )

        st.subheader("Target CMYK")
        picker_col, values_col = st.columns([1.05, 1.4])
        with picker_col:
            st.color_picker(
                "Visual picker",
                key="picker_visual",
                on_change=sync_from_visual_picker,
                help="This updates HEX and CMYK inputs for convenience.",
            )
            st.text_input(
                "HEX",
                key="picker_hex",
                on_change=sync_from_hex,
            )
            if st.session_state["picker_hex_error"]:
                st.warning(st.session_state["picker_hex_error"])

            swatch_hex = cmyk_percent_to_hex(
                st.session_state["picker_c"],
                st.session_state["picker_m"],
                st.session_state["picker_y"],
                st.session_state["picker_k"],
            )
            st.markdown(
                f"""
                <div style="height: 120px; border-radius: 12px; border: 1px solid #d7d7d7; background: {swatch_hex};"></div>
                <div style="margin-top: 8px; font-size: 14px;">Selected color preview: <strong>{swatch_hex}</strong></div>
                """,
                unsafe_allow_html=True,
            )

        with values_col:
            st.slider("Cyan (%)", 0, 100, key="picker_c", on_change=sync_from_cmyk)
            st.slider("Magenta (%)", 0, 100, key="picker_m", on_change=sync_from_cmyk)
            st.slider("Yellow (%)", 0, 100, key="picker_y", on_change=sync_from_cmyk)
            st.slider("Black (%)", 0, 100, key="picker_k", on_change=sync_from_cmyk)

            st.info(
                f"Target CMYK: C {st.session_state['picker_c']} | "
                f"M {st.session_state['picker_m']} | "
                f"Y {st.session_state['picker_y']} | "
                f"K {st.session_state['picker_k']} | "
                f"HEX {st.session_state['picker_hex']}"
            )
            st.caption(
                "Exact mode keeps only pixels with the same rounded CMYK percentages."
            )

        if st.button("Run exact separation", type="primary"):
            target_cmyk = (
                st.session_state["picker_c"],
                st.session_state["picker_m"],
                st.session_state["picker_y"],
                st.session_state["picker_k"],
            )
            picked_cmyk, exact_mask, related_mask, final_mask, cmyk_percent = build_exact_masks(
                cmyk,
                target_cmyk,
            )

            exact_pixels = int(exact_mask.sum())
            total_pixels = int(final_mask.sum())
            matched_percent = (total_pixels / final_mask.size) * 100 if final_mask.size else 0.0

            file_prefix = (
                f"{original_stem}_detail_"
                f"c{target_cmyk[0]}_m{target_cmyk[1]}_y{target_cmyk[2]}_k{target_cmyk[3]}"
            )

            if total_pixels == 0:
                st.warning("No matching pixels were found for the selected CMYK recipe.")
            else:
                grayscale_preview = create_target_grayscale_preview(
                    picked_cmyk,
                    white_threshold=white_threshold,
                )
                color_tiff_bytes = save_cmyk_tiff_bytes(picked_cmyk)
                grayscale_tiff_bytes = save_gray_tiff_bytes(grayscale_preview)
                report_rows = build_recipe_report_rows(cmyk_percent, exact_mask, related_mask)
                report_bytes = save_csv_report_bytes(report_rows)

                output_files = {
                    f"{file_prefix}.tif": color_tiff_bytes,
                    f"{file_prefix}_grayscale.tif": grayscale_tiff_bytes,
                    f"{file_prefix}_report.csv": report_bytes,
                }

                if save_to_disk:
                    write_outputs_to_disk(output_folder, output_files)

                preview_col, gray_col = st.columns(2)
                with preview_col:
                    st.markdown("**Selected color separation**")
                    st.image(pil_from_rgb(cmyk_to_rgb_preview(picked_cmyk)), use_container_width=True)
                with gray_col:
                    st.markdown("**Grayscale output**")
                    st.image(pil_from_gray(grayscale_preview), use_container_width=True)

                metric_col1, metric_col2, metric_col3 = st.columns(3)
                metric_col1.metric("Exact pixels", f"{exact_pixels:,}")
                metric_col2.metric("Total kept pixels", f"{total_pixels:,}")
                metric_col3.metric("Matched area", f"{matched_percent:.2f}%")

                st.subheader("Downloads")
                down_col1, down_col2, down_col3 = st.columns(3)
                with down_col1:
                    st.download_button(
                        label="Download selected color TIFF",
                        data=color_tiff_bytes,
                        file_name=f"{file_prefix}.tif",
                        mime="image/tiff",
                    )
                with down_col2:
                    st.download_button(
                        label="Download grayscale TIFF",
                        data=grayscale_tiff_bytes,
                        file_name=f"{file_prefix}_grayscale.tif",
                        mime="image/tiff",
                    )
                with down_col3:
                    st.download_button(
                        label="Download CMYK report CSV",
                        data=report_bytes,
                        file_name=f"{file_prefix}_report.csv",
                        mime="text/csv",
                    )

                st.download_button(
                    label="Download all outputs as ZIP",
                    data=build_zip(output_files),
                    file_name=f"{file_prefix}_outputs.zip",
                    mime="application/zip",
                )

                if report_rows:
                    st.subheader("Matched recipe report")
                    st.dataframe(report_rows, use_container_width=True, hide_index=True)

                st.success("Exact separation completed.")
                if save_to_disk:
                    st.info(f"Files also saved to: {os.path.abspath(output_folder)}")

    except Exception as exc:
        st.error(f"Failed to process TIFF: {exc}")
