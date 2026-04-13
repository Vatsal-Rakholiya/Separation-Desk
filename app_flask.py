import base64
import hashlib
import io
import json
import os
import time
import uuid
import zipfile
from datetime import datetime
from functools import lru_cache
from pathlib import Path

import numpy as np
import tifffile
from flask import (
    Flask,
    abort,
    jsonify,
    render_template,
    request,
    send_file,
    send_from_directory,
    session,
)
from PIL import Image, ImageFilter
from werkzeug.exceptions import HTTPException
from werkzeug.utils import secure_filename


BASE_DIR = Path(__file__).resolve().parent


def resolve_data_dir():
    override = os.environ.get("CMYK_STUDIO_DATA_DIR", "").strip()
    if override:
        return Path(override).expanduser()

    local_appdata = os.environ.get("LOCALAPPDATA", "").strip()
    if local_appdata:
        return Path(local_appdata) / "CMYKStudio" / "flask_studio_data"

    return BASE_DIR / "flask_studio_data"


DATA_DIR = resolve_data_dir()
SESSIONS_DIR = DATA_DIR / "sessions"
FRONTEND_DIST_DIR = BASE_DIR / "frontend" / "dist"
DATA_DIR.mkdir(parents=True, exist_ok=True)
SESSIONS_DIR.mkdir(parents=True, exist_ok=True)

APP = Flask(__name__)
APP.secret_key = os.environ.get("FLASK_SECRET_KEY", "cmyk-production-studio-dev-secret")
APP.config["MAX_CONTENT_LENGTH"] = 1024 * 1024 * 512

CHANNELS = ["cyan", "magenta", "yellow", "black"]
CHANNEL_INDEX = {name: idx for idx, name in enumerate(CHANNELS)}
RESULT_PREVIEW_MAX_WIDTH = 1100
PRESET_RECIPES = [
    {"name": "Print Green", "cmyk": [89, 33, 100, 27]},
    {"name": "Deep Green", "cmyk": [89, 0, 50, 59]},
    {"name": "Rich Red", "cmyk": [0, 100, 100, 0]},
    {"name": "Warm Yellow", "cmyk": [0, 8, 100, 0]},
    {"name": "Corporate Blue", "cmyk": [100, 70, 0, 10]},
]

HISTORY_LIMIT = 120
DEFAULT_SETTINGS = {
    "match_mode": "exact",
    "selection_mode": "keep",
    "spread": 6,
    "invert_grayscale": False,
    "gray_gamma": 1.0,
    "compression": "lzw",
    "preview_quality": "proxy",
    "preview_sharpen": False,
    "operator": "",
    "client": "",
    "revision": "1",
    "notes": "",
}


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
    return (
        int(round(255 * (1 - c) * (1 - k))),
        int(round(255 * (1 - m) * (1 - k))),
        int(round(255 * (1 - y) * (1 - k))),
    )


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


def pil_from_rgb(rgb_array):
    return Image.fromarray(rgb_array.astype(np.uint8), mode="RGB")


def pil_from_gray(gray_array):
    return Image.fromarray(gray_array.astype(np.uint8), mode="L")


def save_tiff_bytes(array, photometric, compression="lzw", resolution=None):
    buf = io.BytesIO()
    kwargs = {
        "data": array.astype(np.uint8),
        "compression": compression,
        "metadata": None,
    }
    if photometric == "separated":
        kwargs["photometric"] = "separated"
        kwargs["planarconfig"] = "contig"
    else:
        kwargs["photometric"] = photometric
    if resolution is not None:
        kwargs["resolution"] = resolution
    tifffile.imwrite(buf, **kwargs)
    buf.seek(0)
    return buf.getvalue()


def png_bytes_from_rgb(rgb_array):
    buf = io.BytesIO()
    pil_from_rgb(rgb_array).save(buf, format="PNG")
    buf.seek(0)
    return buf.getvalue()


def png_bytes_from_gray(gray_array):
    buf = io.BytesIO()
    pil_from_gray(gray_array).save(buf, format="PNG")
    buf.seek(0)
    return buf.getvalue()


def build_zip(file_map):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, data in file_map.items():
            archive.writestr(name, data)
    buf.seek(0)
    return buf.getvalue()


def sha1_bytes(file_bytes):
    return hashlib.sha1(file_bytes).hexdigest()


def parse_cmyk_recipe(payload):
    try:
        recipe = [int(payload["c"]), int(payload["m"]), int(payload["y"]), int(payload["k"])]
    except Exception as exc:
        raise ValueError("CMYK values must be whole numbers from 0 to 100.") from exc
    if any(value < 0 or value > 100 for value in recipe):
        raise ValueError("CMYK values must stay between 0 and 100.")
    return recipe


def session_id():
    if "studio_session_id" not in session:
        session["studio_session_id"] = uuid.uuid4().hex
    return session["studio_session_id"]


def session_dir():
    path = SESSIONS_DIR / session_id()
    path.mkdir(parents=True, exist_ok=True)
    (path / "uploads").mkdir(exist_ok=True)
    (path / "results").mkdir(exist_ok=True)
    return path


def metadata_path():
    return session_dir() / "metadata.json"


def default_metadata():
    return {
        "jobs": [],
        "active_job_id": None,
        "recent_colors": [],
        "favorites": [],
        "profiles": {},
        "history": [],
        "last_job": None,
        "settings": dict(DEFAULT_SETTINGS),
    }


def result_dir_for(job_id, result_id):
    return session_dir() / "results" / job_id / result_id


def append_history_entry(data, entry):
    entry["id"] = uuid.uuid4().hex
    data["history"].append(entry)
    data["history"] = data["history"][-HISTORY_LIMIT:]


def selected_history_entry(job, page_index, summary):
    return {
        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "job_name": job["job_name"],
        "file_name": job["file_name"],
        "page": page_index + 1,
        "matched_pixels": summary["stats"]["matched_pixels"],
        "matched_percent": round(summary["stats"]["matched_percent"], 2),
        "processing_seconds": round(summary["stats"]["processing_seconds"], 3),
        "kind": "selected",
        "summary": summary,
    }


def standard_history_entry(job, page_index, summary):
    return {
        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "job_name": job["job_name"],
        "file_name": job["file_name"],
        "page": page_index + 1,
        "matched_pixels": 0,
        "matched_percent": 0,
        "processing_seconds": round(summary["stats"]["processing_seconds"], 3),
        "kind": "standard",
        "summary": summary,
    }


def load_metadata():
    path = metadata_path()
    if not path.exists():
        return default_metadata()
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default_metadata()


def save_metadata(data):
    metadata_path().write_text(json.dumps(data, indent=2), encoding="utf-8")


def public_job(job):
    result = job.get("last_result")
    return {
        "id": job["id"],
        "file_name": job["file_name"],
        "job_name": job["job_name"],
        "signature": job["signature"],
        "pages": job["pages"],
        "current_page": job.get("current_page", 0),
        "selected_colors": job.get("selected_colors", []),
        "last_result": result,
        "last_standard_result": job.get("last_standard_result"),
    }


def api_request():
    return request.path.startswith("/api/")


def api_error_payload(message, status):
    return jsonify({"error": message, "status": status}), status


def current_state():
    data = load_metadata()
    history_changed = False
    for item in data.get("history", []):
        if "id" not in item:
            item["id"] = uuid.uuid4().hex
            history_changed = True
    if history_changed:
        save_metadata(data)
    return {
        "jobs": [public_job(job) for job in data["jobs"]],
        "active_job_id": data.get("active_job_id"),
        "recent_colors": data.get("recent_colors", []),
        "favorites": data.get("favorites", []),
        "profiles": sorted(data.get("profiles", {}).keys()),
        "history": data.get("history", [])[-24:],
        "settings": data.get("settings", dict(DEFAULT_SETTINGS)),
        "presets": PRESET_RECIPES,
        "click_pick_available": True,
    }


def get_job(data, job_id):
    for job in data["jobs"]:
        if job["id"] == job_id:
            return job
    abort(404, description="Job not found.")


def ensure_recent_color(data, recipe, name="Recent"):
    colors = data["recent_colors"]
    colors = [item for item in colors if item["cmyk"] != list(recipe)]
    colors.insert(0, {"name": name, "cmyk": list(recipe)})
    data["recent_colors"] = colors[:12]


def ensure_favorite(data, recipe, name="Favorite"):
    if not any(item["cmyk"] == list(recipe) for item in data["favorites"]):
        data["favorites"].append({"name": name, "cmyk": list(recipe)})


@lru_cache(maxsize=24)
def read_tiff_overview_cached(file_path, mtime_ns, size):
    pages = []
    with tifffile.TiffFile(file_path) as tif:
        for index, page in enumerate(tif.pages):
            dpi_x = 300.0
            dpi_y = 300.0
            try:
                x_res = page.tags.get("XResolution")
                y_res = page.tags.get("YResolution")
                if x_res:
                    value = x_res.value
                    dpi_x = float(value[0] / value[1] if isinstance(value, tuple) else value)
                if y_res:
                    value = y_res.value
                    dpi_y = float(value[0] / value[1] if isinstance(value, tuple) else value)
            except Exception:
                pass
            shape = page.shape
            pages.append(
                {
                    "index": index,
                    "label": f"Page {index + 1} - {shape}",
                    "shape": list(shape),
                    "dpi_x": dpi_x,
                    "dpi_y": dpi_y,
                    "photometric": str(page.photometric),
                    "compression": str(page.compression),
                    "dtype": str(page.dtype),
                }
            )
    return pages


@lru_cache(maxsize=32)
def load_page_cached(file_path, mtime_ns, size, page_index):
    with tifffile.TiffFile(file_path) as tif:
        arr = tif.pages[page_index].asarray()
    cmyk = extract_cmyk(arr)
    return {
        "cmyk": cmyk,
        "cmyk_percent": cmyk_to_percent_array(cmyk),
        "rgb_preview": cmyk_to_rgb_preview(cmyk),
    }


def file_cache_key(file_path):
    stat = Path(file_path).stat()
    return str(file_path), stat.st_mtime_ns, stat.st_size


def tiff_overview(file_path):
    return read_tiff_overview_cached(*file_cache_key(file_path))


def load_page_payload(file_path, page_index):
    return load_page_cached(*file_cache_key(file_path), page_index)


def sampled_preview(rgb_array, max_width, sharpen=False):
    width = rgb_array.shape[1]
    if width <= max_width:
        image = pil_from_rgb(rgb_array)
    else:
        height = max(1, int(round(rgb_array.shape[0] * (max_width / width))))
        image = pil_from_rgb(rgb_array).resize((max_width, height), Image.Resampling.LANCZOS)
    if sharpen:
        image = image.filter(ImageFilter.UnsharpMask(radius=1.2, percent=140, threshold=2))
    return np.array(image)


def sampled_gray_preview(gray_array, max_width):
    width = gray_array.shape[1]
    if width <= max_width:
        image = pil_from_gray(gray_array)
    else:
        height = max(1, int(round(gray_array.shape[0] * (max_width / width))))
        image = pil_from_gray(gray_array).resize((max_width, height), Image.Resampling.LANCZOS)
    return np.array(image)


def preview_dimensions(height, width, max_width):
    if width <= max_width:
        return width, height
    scaled_height = max(1, int(round(height * (max_width / width))))
    return max_width, scaled_height


@lru_cache(maxsize=32)
def inspector_preview_cached(file_path, mtime_ns, size, page_index, max_width):
    payload = load_page_cached(file_path, mtime_ns, size, page_index)
    source = payload["cmyk_percent"]
    source_height, source_width = source.shape[:2]
    preview_width, preview_height = preview_dimensions(source_height, source_width, max_width)

    if preview_width == source_width and preview_height == source_height:
        inspector = source
    else:
        x_idx = np.minimum(
            (np.arange(preview_width, dtype=np.int64) * source_width) // preview_width,
            source_width - 1,
        )
        y_idx = np.minimum(
            (np.arange(preview_height, dtype=np.int64) * source_height) // preview_height,
            source_height - 1,
        )
        inspector = source[y_idx[:, None], x_idx[None, :]]

    return {
        "width": int(preview_width),
        "height": int(preview_height),
        "source_width": int(source_width),
        "source_height": int(source_height),
        "data": base64.b64encode(inspector.tobytes()).decode("ascii"),
    }


def inspector_preview_payload(file_path, page_index, quality):
    max_width = 900 if quality == "proxy" else 1600
    return inspector_preview_cached(*file_cache_key(file_path), page_index, max_width)


def nearest_recipes(cmyk_percent, recipe, limit=5):
    flat = cmyk_percent.reshape(-1, 4)
    unique_colors, counts = np.unique(flat, axis=0, return_counts=True)
    distances = np.sqrt(np.mean((unique_colors.astype(np.float32) - np.array(recipe, dtype=np.float32)) ** 2, axis=1))
    order = np.argsort(distances)[:limit]
    return [
        {
            "cmyk": unique_colors[idx].tolist(),
            "pixels": int(counts[idx]),
            "distance": float(distances[idx]),
        }
        for idx in order
    ]


def build_mask(cmyk_percent, selected_colors, match_mode, spread):
    masks = []
    matched_recipes = []
    for item in selected_colors:
        recipe = np.array(item["cmyk"], dtype=np.uint8)
        if match_mode == "spread":
            diff = np.abs(cmyk_percent.astype(np.int16) - recipe.reshape(1, 1, 4).astype(np.int16))
            mask = np.max(diff, axis=2) <= int(spread)
            matched = recipe.tolist()
        elif match_mode == "nearest":
            nearest = nearest_recipes(cmyk_percent, recipe.tolist(), limit=1)[0]
            matched = nearest["cmyk"]
            mask = np.all(cmyk_percent == np.array(matched, dtype=np.uint8).reshape(1, 1, 4), axis=2)
        else:
            mask = np.all(cmyk_percent == recipe.reshape(1, 1, 4), axis=2)
            matched = recipe.tolist()
        masks.append(mask)
        matched_recipes.append({"name": item["name"], "requested": recipe.tolist(), "matched": matched})

    combined = np.logical_or.reduce(masks) if masks else np.zeros(cmyk_percent.shape[:2], dtype=bool)
    return combined, matched_recipes


def grayscale_from_mask(mask, selected_cmyk, invert, gamma):
    ink = np.max(selected_cmyk, axis=2).astype(np.uint8)
    gray = np.full(mask.shape, 255, dtype=np.uint8)
    gray[mask] = 255 - ink[mask]
    gamma = max(float(gamma), 0.1)
    gray = ((gray.astype(np.float32) / 255.0) ** gamma * 255.0).clip(0, 255).astype(np.uint8)
    if invert:
        gray = 255 - gray
    return gray


def grayscale_from_cmyk(cmyk, invert, gamma):
    ink = np.max(cmyk, axis=2).astype(np.uint8)
    gray = 255 - ink
    gamma = max(float(gamma), 0.1)
    gray = ((gray.astype(np.float32) / 255.0) ** gamma * 255.0).clip(0, 255).astype(np.uint8)
    if invert:
        gray = 255 - gray
    return gray


def build_selected_variant(payload, mask, settings, operation):
    if operation == "remove":
        processed_cmyk = payload["cmyk"].copy()
        processed_cmyk[mask] = 0
        processed_preview = cmyk_to_rgb_preview(processed_cmyk)
        grayscale = grayscale_from_cmyk(
            processed_cmyk,
            settings["invert_grayscale"],
            settings["gray_gamma"],
        )
    else:
        processed_cmyk = np.zeros_like(payload["cmyk"], dtype=np.uint8)
        processed_cmyk[mask] = payload["cmyk"][mask]
        processed_preview = composite_preview(payload["rgb_preview"], mask)
        grayscale = grayscale_from_mask(
            mask,
            processed_cmyk,
            settings["invert_grayscale"],
            settings["gray_gamma"],
        )

    return {
        "operation": operation,
        "selected_cmyk": processed_cmyk,
        "original_preview": payload["rgb_preview"],
        "selected_preview": processed_preview,
        "grayscale": grayscale,
        "coverage_mask": np.where(mask, 0, 255).astype(np.uint8),
        "matched_recipes": [],
        "mask": mask,
        "source_cmyk": payload["cmyk"],
        "source_percent": payload["cmyk_percent"],
    }


def checkerboard(shape, block=18):
    h, w = shape[:2]
    y_idx, x_idx = np.indices((h, w))
    pattern = ((x_idx // block) + (y_idx // block)) % 2
    base = np.where(pattern[..., None] == 0, 238, 220).astype(np.uint8)
    return np.repeat(base, 3, axis=2)


def composite_preview(rgb_preview, mask):
    board = checkerboard(rgb_preview.shape)
    result = board.copy()
    result[mask] = rgb_preview[mask]
    return result


def highlight_preview(rgb_preview, mask):
    highlighted = rgb_preview.astype(np.float32).copy()
    accent = np.array([255, 92, 47], dtype=np.float32)
    highlighted[mask] = highlighted[mask] * 0.68 + accent * 0.32
    return np.clip(highlighted, 0, 255).astype(np.uint8)


def write_result_files(job, settings, result_payload, file_base):
    result_id = uuid.uuid4().hex
    result_dir = result_dir_for(job["id"], result_id)
    result_dir.mkdir(parents=True, exist_ok=True)

    page = job["pages"][job.get("current_page", 0)]
    resolution = (page["dpi_x"], page["dpi_y"])
    compression = settings["compression"]
    output_files = {}

    processed_tiff = save_tiff_bytes(
        result_payload["selected_cmyk"],
        "separated",
        compression=compression,
        resolution=resolution,
    )
    grayscale_tiff = save_tiff_bytes(
        result_payload["grayscale"],
        "minisblack",
        compression=compression,
        resolution=resolution,
    )
    output_suffix = "removed" if result_payload["operation"] == "remove" else "selected"
    output_files[f"{file_base}_{output_suffix}.tif"] = processed_tiff
    output_files[f"{file_base}_grayscale.tif"] = grayscale_tiff

    zip_bytes = build_zip(output_files)
    (result_dir / "downloads.zip").write_bytes(zip_bytes)
    for name, data in output_files.items():
        (result_dir / name).write_bytes(data)

    selected_preview = sampled_preview(result_payload["selected_preview"], RESULT_PREVIEW_MAX_WIDTH)
    grayscale_preview = sampled_gray_preview(result_payload["grayscale"], RESULT_PREVIEW_MAX_WIDTH)
    original_preview = sampled_preview(result_payload["original_preview"], RESULT_PREVIEW_MAX_WIDTH)
    (result_dir / "selected_preview.png").write_bytes(png_bytes_from_rgb(selected_preview))
    (result_dir / "grayscale_preview.png").write_bytes(png_bytes_from_gray(grayscale_preview))
    (result_dir / "original_preview.png").write_bytes(png_bytes_from_rgb(original_preview))

    return result_id, result_dir, output_files


def result_summary(job, result_id, result_payload, file_base, processing_seconds):
    matched_pixels = int(result_payload["mask"].sum())
    matched_percent = (matched_pixels / result_payload["mask"].size) * 100 if result_payload["mask"].size else 0.0
    stats = {
        "matched_pixels": matched_pixels,
        "matched_percent": matched_percent,
        "processing_seconds": processing_seconds,
        "ink_coverage": round(
            float(np.mean(result_payload["selected_cmyk"])) * 100.0 / 255.0 * 4.0,
            2,
        ),
        "warnings": [],
    }
    if matched_pixels == 0:
        stats["warnings"].append("No exact output area was found with the current setup.")
    elif matched_pixels < 200:
        stats["warnings"].append("Matched area is very small. Please double-check the selected colors.")

    return {
        "result_id": result_id,
        "operation": result_payload["operation"],
        "status": "matched" if matched_pixels > 0 else "no_match",
        "file_base": file_base,
        "stats": stats,
        "previews": {
            "selected": f"/preview/{job['id']}/{result_id}/selected",
            "grayscale": f"/preview/{job['id']}/{result_id}/grayscale",
            "original": f"/preview/{job['id']}/{result_id}/original",
        },
        "downloads": {
            "zip": f"/download/{job['id']}/{result_id}/downloads.zip",
        },
    }


def write_selected_run_bundle(job, selected_files, removed_files):
    bundle_id = f"selected_run_{uuid.uuid4().hex}"
    bundle_dir = result_dir_for(job["id"], bundle_id)
    bundle_dir.mkdir(parents=True, exist_ok=True)

    combined_files = {**selected_files, **removed_files}
    bundle_zip = build_zip(combined_files)
    (bundle_dir / "downloads.zip").write_bytes(bundle_zip)
    return bundle_id


def selected_run_summary(job, bundle_id, selected_summary, removed_summary, processing_seconds):
    selected_stats = selected_summary["stats"]
    return {
        "result_id": bundle_id,
        "status": "matched" if selected_summary["status"] == "matched" else "no_match",
        "stats": {
            "matched_pixels": selected_stats["matched_pixels"],
            "matched_percent": selected_stats["matched_percent"],
            "processing_seconds": processing_seconds,
            "warnings": selected_stats.get("warnings", []),
        },
        "previews": {
            "original": selected_summary["previews"]["original"],
        },
        "downloads": {
            "zip": f"/download/{job['id']}/{bundle_id}/downloads.zip",
        },
        "selected_result": selected_summary,
        "removed_result": removed_summary,
    }


def write_standard_result(job, page_index, compression):
    payload = load_page_payload(job["file_path"], page_index)
    page = job["pages"][page_index]
    resolution = (page["dpi_x"], page["dpi_y"])
    cmyk = payload["cmyk"]
    zeros = np.zeros(cmyk.shape[:2], dtype=np.uint8)
    result_id = uuid.uuid4().hex
    result_dir = result_dir_for(job["id"], f"standard_{result_id}")
    result_dir.mkdir(parents=True, exist_ok=True)

    output_files = {}
    preview_files = {}
    for color in CHANNELS:
        idx = CHANNEL_INDEX[color]
        planes = [zeros, zeros, zeros, zeros]
        planes[idx] = cmyk[:, :, idx]
        separation = np.stack(planes, axis=2)
        output_files[f"{job['job_name']}_p{page_index + 1}_{color}.tif"] = save_tiff_bytes(
            separation,
            "separated",
            compression=compression,
            resolution=resolution,
        )
        grayscale_plate = 255 - cmyk[:, :, idx]
        output_files[f"{job['job_name']}_p{page_index + 1}_{color}_grayscale.tif"] = save_tiff_bytes(
            grayscale_plate,
            "minisblack",
            compression=compression,
            resolution=resolution,
        )
        color_preview = sampled_preview(cmyk_to_rgb_preview(separation), RESULT_PREVIEW_MAX_WIDTH)
        gray_preview = sampled_gray_preview(grayscale_plate, RESULT_PREVIEW_MAX_WIDTH)
        preview_files[f"{color}_color"] = png_bytes_from_rgb(color_preview)
        preview_files[f"{color}_gray"] = png_bytes_from_gray(gray_preview)
        (result_dir / f"{color}_color_preview.png").write_bytes(preview_files[f"{color}_color"])
        (result_dir / f"{color}_gray_preview.png").write_bytes(preview_files[f"{color}_gray"])

    original_preview = sampled_preview(payload["rgb_preview"], RESULT_PREVIEW_MAX_WIDTH)
    (result_dir / "original_preview.png").write_bytes(png_bytes_from_rgb(original_preview))

    zip_bytes = build_zip(output_files)
    (result_dir / "standard_outputs.zip").write_bytes(zip_bytes)
    for name, data in output_files.items():
        (result_dir / name).write_bytes(data)

    return {
        "result_id": f"standard_{result_id}",
        "downloads": {"zip": f"/download/{job['id']}/standard_{result_id}/standard_outputs.zip"},
        "previews": {
            "original": f"/preview/{job['id']}/standard_{result_id}/original",
            **{f"{color}_color": f"/preview/{job['id']}/standard_{result_id}/{color}_color" for color in CHANNELS},
            **{f"{color}_gray": f"/preview/{job['id']}/standard_{result_id}/{color}_gray" for color in CHANNELS},
        },
    }


@APP.get("/")
def index():
    react_index = FRONTEND_DIST_DIR / "index.html"
    if react_index.exists():
        return send_file(react_index)
    return render_template("index.html")


@APP.get("/assets/<path:filename>")
def frontend_assets(filename):
    asset_dir = FRONTEND_DIST_DIR / "assets"
    if asset_dir.exists():
        return send_from_directory(asset_dir, filename)
    abort(404)


@APP.errorhandler(Exception)
def handle_exception(error):
    if not api_request():
        if isinstance(error, HTTPException):
            return error
        APP.logger.exception("Unhandled page error", exc_info=error)
        return "Internal server error.", 500
    if isinstance(error, HTTPException):
        message = error.description or error.name or "Request failed."
        return api_error_payload(message, error.code or 500)
    APP.logger.exception("Unhandled API error", exc_info=error)
    return api_error_payload(str(error) or "Internal server error.", 500)


@APP.get("/api/state")
def api_state():
    return jsonify(current_state())


@APP.post("/api/upload")
def api_upload():
    data = load_metadata()
    files = request.files.getlist("files")
    if not files:
        return jsonify({"error": "No files uploaded."}), 400

    uploads_dir = session_dir() / "uploads"
    for file in files:
        file_bytes = file.read()
        if not file_bytes:
            continue
        file_id = uuid.uuid4().hex
        file_name = secure_filename(file.filename) or f"{file_id}.tif"
        file_path = uploads_dir / f"{file_id}_{file_name}"
        file_path.write_bytes(file_bytes)
        pages = tiff_overview(file_path)
        job = {
            "id": file_id,
            "file_name": file.filename,
            "job_name": Path(file.filename).stem,
            "signature": sha1_bytes(file_bytes),
            "file_path": str(file_path),
            "pages": pages,
            "current_page": 0,
            "selected_colors": [],
            "last_result": None,
            "last_standard_result": None,
            "created_at": datetime.now().isoformat(),
        }
        data["jobs"].append(job)
        if not data.get("active_job_id"):
            data["active_job_id"] = file_id

    save_metadata(data)
    return jsonify(current_state())


@APP.post("/api/reorder")
def api_reorder():
    data = load_metadata()
    ordered_ids = request.json.get("ordered_ids", [])
    order_map = {job_id: index for index, job_id in enumerate(ordered_ids)}
    data["jobs"].sort(key=lambda job: order_map.get(job["id"], len(order_map)))
    save_metadata(data)
    return jsonify(current_state())


@APP.post("/api/job/<job_id>/activate")
def api_activate(job_id):
    data = load_metadata()
    get_job(data, job_id)
    data["active_job_id"] = job_id
    save_metadata(data)
    return jsonify(current_state())


@APP.delete("/api/job/<job_id>")
def api_delete_job(job_id):
    data = load_metadata()
    existing_ids = [job["id"] for job in data["jobs"]]
    if job_id not in existing_ids:
        return jsonify({"error": "Job not found."}), 404
    data["jobs"] = [job for job in data["jobs"] if job["id"] != job_id]
    if data.get("active_job_id") == job_id:
        data["active_job_id"] = data["jobs"][0]["id"] if data["jobs"] else None
    save_metadata(data)
    return jsonify(current_state())


@APP.post("/api/job/<job_id>/rename")
def api_rename(job_id):
    data = load_metadata()
    job = get_job(data, job_id)
    name = (request.json.get("job_name") or "").strip()
    if not name:
        return jsonify({"error": "Job name is required."}), 400
    job["job_name"] = name
    save_metadata(data)
    return jsonify(current_state())


@APP.post("/api/job/<job_id>/page")
def api_set_page(job_id):
    data = load_metadata()
    job = get_job(data, job_id)
    page_index = int(request.json.get("page_index", 0))
    if page_index < 0 or page_index >= len(job["pages"]):
        return jsonify({"error": "Invalid page index."}), 400
    job["current_page"] = page_index
    save_metadata(data)
    return jsonify(current_state())


@APP.post("/api/job/<job_id>/settings")
def api_save_settings(job_id):
    data = load_metadata()
    get_job(data, job_id)
    settings = data["settings"]
    for key in DEFAULT_SETTINGS:
        if key in request.json:
            settings[key] = request.json[key]
    save_metadata(data)
    return jsonify(current_state())


@APP.post("/api/profile/save")
def api_profile_save():
    data = load_metadata()
    name = (request.json.get("name") or "").strip()
    if not name:
        return jsonify({"error": "Profile name is required."}), 400
    data["profiles"][name] = request.json.get("payload", {})
    save_metadata(data)
    return jsonify(current_state())


@APP.post("/api/profile/load")
def api_profile_load():
    data = load_metadata()
    name = request.json.get("name", "")
    payload = data["profiles"].get(name)
    if payload is None:
        return jsonify({"error": "Profile not found."}), 404
    return jsonify({"profile": payload})


@APP.post("/api/favorites/add")
def api_add_favorite():
    data = load_metadata()
    recipe = parse_cmyk_recipe(request.json)
    ensure_favorite(data, recipe, name=request.json.get("name", "Favorite"))
    save_metadata(data)
    return jsonify(current_state())


@APP.post("/api/job/<job_id>/colors")
def api_add_color(job_id):
    data = load_metadata()
    job = get_job(data, job_id)
    recipe = parse_cmyk_recipe(request.json)
    job["selected_colors"].append(
        {
            "name": (request.json.get("name") or f"Color {len(job['selected_colors']) + 1}").strip(),
            "cmyk": recipe,
        }
    )
    ensure_recent_color(data, recipe)
    save_metadata(data)
    return jsonify(current_state())


@APP.delete("/api/job/<job_id>/colors/<int:index>")
def api_remove_color(job_id, index):
    data = load_metadata()
    job = get_job(data, job_id)
    if 0 <= index < len(job["selected_colors"]):
        job["selected_colors"].pop(index)
        save_metadata(data)
    return jsonify(current_state())


@APP.post("/api/job/<job_id>/clear")
def api_clear_job(job_id):
    data = load_metadata()
    job = get_job(data, job_id)
    job["selected_colors"] = []
    save_metadata(data)
    return jsonify(current_state())


@APP.post("/api/job/<job_id>/duplicate-last")
def api_duplicate_last(job_id):
    data = load_metadata()
    job = get_job(data, job_id)
    last_job = data.get("last_job")
    if last_job:
        job["selected_colors"] = last_job.get("selected_colors", [])
    save_metadata(data)
    return jsonify(current_state())


@APP.delete("/api/history/<history_id>")
def api_delete_history(history_id):
    data = load_metadata()
    data["history"] = [item for item in data.get("history", []) if item.get("id") != history_id]
    save_metadata(data)
    return jsonify(current_state())


@APP.post("/api/history/clear")
def api_clear_history():
    data = load_metadata()
    data["history"] = []
    save_metadata(data)
    return jsonify(current_state())


@APP.get("/api/job/<job_id>/sample")
def api_sample(job_id):
    data = load_metadata()
    job = get_job(data, job_id)
    page_index = int(request.args.get("page", job.get("current_page", 0)))
    x = int(request.args.get("x", 0))
    y = int(request.args.get("y", 0))
    payload = load_page_payload(job["file_path"], page_index)
    h, w = payload["cmyk_percent"].shape[:2]
    x = min(max(x, 0), w - 1)
    y = min(max(y, 0), h - 1)
    recipe = payload["cmyk_percent"][y, x].tolist()
    return jsonify(
        {
            "x": x,
            "y": y,
            "cmyk": recipe,
            "hex": cmyk_percent_to_hex(*recipe),
            "nearest": nearest_recipes(payload["cmyk_percent"], recipe, limit=4),
        }
    )


@APP.get("/api/job/<job_id>/inspector")
def api_inspector(job_id):
    data = load_metadata()
    job = get_job(data, job_id)
    page_index = int(request.args.get("page", job.get("current_page", 0)))
    quality = request.args.get("quality", "proxy")
    if quality not in {"proxy", "full"}:
        quality = "proxy"
    return jsonify(inspector_preview_payload(job["file_path"], page_index, quality))


@APP.post("/api/job/<job_id>/process")
def api_process(job_id):
    data = load_metadata()
    job = get_job(data, job_id)
    request_payload = request.json or {}
    settings = dict(data.get("settings", dict(DEFAULT_SETTINGS)))
    settings.update({key: request_payload.get(key, settings[key]) for key in DEFAULT_SETTINGS})

    selected_colors = request_payload.get("selected_colors") or job.get("selected_colors") or []
    manual_recipe = request_payload.get("manual_recipe")
    if manual_recipe and not selected_colors:
        selected_colors = [{"name": "Current", "cmyk": parse_cmyk_recipe(manual_recipe)}]
    if not selected_colors:
        return jsonify({"error": "Add at least one color before processing."}), 400

    page_index = int(request_payload.get("page_index", job.get("current_page", 0)))
    if page_index < 0 or page_index >= len(job["pages"]):
        return jsonify({"error": "Invalid page index."}), 400
    job["current_page"] = page_index
    start_time = time.perf_counter()
    payload = load_page_payload(job["file_path"], page_index)
    mask, matched_recipes = build_mask(
        payload["cmyk_percent"],
        selected_colors,
        settings["match_mode"],
        settings["spread"],
    )

    for item in matched_recipes:
        matched = np.array(item["matched"], dtype=np.uint8)
        item["pixels"] = int(
            np.all(payload["cmyk_percent"] == matched.reshape(1, 1, 4), axis=2).sum()
        )

    selected_file_base = (
        f"{secure_filename(job['job_name']) or 'job'}"
        f"_p{page_index + 1}_keep_{settings['match_mode']}"
    )
    removed_file_base = (
        f"{secure_filename(job['job_name']) or 'job'}"
        f"_p{page_index + 1}_remove_{settings['match_mode']}"
    )

    selected_payload = build_selected_variant(payload, mask, settings, "keep")
    selected_payload["matched_recipes"] = matched_recipes
    removed_payload = build_selected_variant(payload, mask, settings, "remove")
    removed_payload["matched_recipes"] = matched_recipes

    selected_result_id, _selected_result_dir, selected_files = write_result_files(
        job,
        settings,
        selected_payload,
        selected_file_base,
    )
    removed_result_id, _removed_result_dir, removed_files = write_result_files(
        job,
        settings,
        removed_payload,
        removed_file_base,
    )
    bundle_id = write_selected_run_bundle(job, selected_files, removed_files)
    processing_seconds = time.perf_counter() - start_time
    selected_variant = result_summary(
        job,
        selected_result_id,
        selected_payload,
        selected_file_base,
        processing_seconds,
    )
    removed_variant = result_summary(
        job,
        removed_result_id,
        removed_payload,
        removed_file_base,
        processing_seconds,
    )
    summary = selected_run_summary(
        job,
        bundle_id,
        selected_variant,
        removed_variant,
        processing_seconds,
    )

    job["selected_colors"] = selected_colors
    job["last_result"] = summary
    data["settings"] = settings
    ensure_recent_color(data, selected_colors[0]["cmyk"], name=selected_colors[0]["name"])
    data["last_job"] = {"selected_colors": selected_colors}
    append_history_entry(data, selected_history_entry(job, page_index, summary))
    save_metadata(data)
    return jsonify(current_state())


@APP.post("/api/job/<job_id>/standard")
def api_standard(job_id):
    data = load_metadata()
    job = get_job(data, job_id)
    request_payload = request.json or {}
    settings = dict(data.get("settings", dict(DEFAULT_SETTINGS)))
    settings.update({key: request_payload.get(key, settings[key]) for key in DEFAULT_SETTINGS})
    page_index = int(request_payload.get("page_index", job.get("current_page", 0)))
    if page_index < 0 or page_index >= len(job["pages"]):
        return jsonify({"error": "Invalid page index."}), 400
    job["current_page"] = page_index

    start_time = time.perf_counter()
    summary = write_standard_result(job, page_index, settings["compression"])

    payload = load_page_payload(job["file_path"], page_index)
    page = job["pages"][page_index]
    processing_seconds = time.perf_counter() - start_time
    summary["stats"] = {
        "processing_seconds": processing_seconds,
        "width": int(payload["cmyk"].shape[1]),
        "height": int(payload["cmyk"].shape[0]),
        "dpi_x": float(page["dpi_x"]),
        "dpi_y": float(page["dpi_y"]),
    }

    job["last_standard_result"] = summary
    data["settings"] = settings
    append_history_entry(data, standard_history_entry(job, page_index, summary))
    save_metadata(data)
    return jsonify(current_state())


@APP.get("/preview/<job_id>/original")
def preview_original(job_id):
    data = load_metadata()
    job = get_job(data, job_id)
    page_index = int(request.args.get("page", job.get("current_page", 0)))
    sharpen = request.args.get("sharpen", "0") == "1"
    quality = request.args.get("quality", "proxy")
    payload = load_page_payload(job["file_path"], page_index)
    preview = sampled_preview(
        payload["rgb_preview"],
        900 if quality == "proxy" else 1600,
        sharpen=sharpen,
    )
    return send_file(io.BytesIO(png_bytes_from_rgb(preview)), mimetype="image/png")


@APP.get("/preview/<job_id>/<result_id>/<kind>")
def preview_result(job_id, result_id, kind):
    standard_kinds = {f"{color}_color" for color in CHANNELS} | {f"{color}_gray" for color in CHANNELS}
    if kind not in {"selected", "grayscale", "original", *standard_kinds}:
        abort(404)
    result_dir = result_dir_for(job_id, result_id)
    file_map = {
        "selected": result_dir / "selected_preview.png",
        "grayscale": result_dir / "grayscale_preview.png",
        "original": result_dir / "original_preview.png",
        "cyan_color": result_dir / "cyan_color_preview.png",
        "magenta_color": result_dir / "magenta_color_preview.png",
        "yellow_color": result_dir / "yellow_color_preview.png",
        "black_color": result_dir / "black_color_preview.png",
        "cyan_gray": result_dir / "cyan_gray_preview.png",
        "magenta_gray": result_dir / "magenta_gray_preview.png",
        "yellow_gray": result_dir / "yellow_gray_preview.png",
        "black_gray": result_dir / "black_gray_preview.png",
    }
    file_path = file_map[kind]
    if not file_path.exists():
        abort(404)
    return send_file(file_path, mimetype="image/png")


@APP.get("/download/<job_id>/<result_id>/<path:file_name>")
def download_result(job_id, result_id, file_name):
    file_path = session_dir() / "results" / job_id / result_id / file_name
    if not file_path.exists():
        abort(404)
    return send_file(file_path, as_attachment=True, download_name=file_name)


if __name__ == "__main__":
    APP.run(debug=False, port=5000, threaded=True, use_reloader=False)
