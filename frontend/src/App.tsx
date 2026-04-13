import { startTransition, useDeferredValue, useEffect, useRef, useState } from "react";
import "./App.css";
import { api, ApiError, uploadFiles } from "./api";
import type {
  AppState,
  HistoryEntry,
  InspectorPreviewPayload,
  ManualRecipe,
  Recipe,
  SampleResponse,
  SelectedRunSummary,
  Settings,
  ResultSummary,
} from "./types";

const FALLBACK_RECIPE: ManualRecipe = { c: 89, m: 33, y: 100, k: 27 };

const EMPTY_SETTINGS: Settings = {
  match_mode: "exact",
  selection_mode: "keep",
  spread: 6,
  invert_grayscale: false,
  gray_gamma: 1,
  compression: "lzw",
  preview_quality: "proxy",
  preview_sharpen: false,
  operator: "",
  client: "",
  revision: "1",
  notes: "",
};

function cmykToHex(recipe: ManualRecipe | Recipe["cmyk"]) {
  const values = Array.isArray(recipe)
    ? { c: recipe[0], m: recipe[1], y: recipe[2], k: recipe[3] }
    : recipe;
  const c = Math.max(0, Math.min(100, values.c)) / 100;
  const m = Math.max(0, Math.min(100, values.m)) / 100;
  const y = Math.max(0, Math.min(100, values.y)) / 100;
  const k = Math.max(0, Math.min(100, values.k)) / 100;
  const r = Math.round(255 * (1 - c) * (1 - k));
  const g = Math.round(255 * (1 - m) * (1 - k));
  const b = Math.round(255 * (1 - y) * (1 - k));
  return `#${[r, g, b].map((value) => value.toString(16).padStart(2, "0")).join("").toUpperCase()}`;
}

function activeJobFrom(state: AppState | null) {
  if (!state || !state.active_job_id) return null;
  return state.jobs.find((job) => job.id === state.active_job_id) ?? null;
}

function historyImageEntries(item: HistoryEntry) {
  if (item.kind === "selected") {
    if ("selected_result" in item.summary) {
      return [
        { label: "Selected", src: item.summary.selected_result.previews.selected },
        { label: "Selected Gray", src: item.summary.selected_result.previews.grayscale },
        { label: "Removed", src: item.summary.removed_result.previews.selected },
        { label: "Removed Gray", src: item.summary.removed_result.previews.grayscale },
      ];
    }
    const summary = item.summary as ResultSummary;
    return [
      { label: "Selected", src: summary.previews.selected },
      { label: "Gray", src: summary.previews.grayscale },
    ];
  }

  const summary = item.summary as ResultSummary;
  return [
    { label: "Cyan", src: summary.previews.cyan_color },
    { label: "Magenta", src: summary.previews.magenta_color },
    { label: "Yellow", src: summary.previews.yellow_color },
    { label: "Black", src: summary.previews.black_color },
  ];
}

function isSelectedRunSummary(
  value: ResultSummary | SelectedRunSummary | null | undefined,
): value is SelectedRunSummary {
  return !!value && "selected_result" in value && "removed_result" in value;
}

function decodeBase64Bytes(value: string) {
  const binary = window.atob(value);
  const bytes = new Uint8Array(binary.length);
  for (let index = 0; index < binary.length; index += 1) {
    bytes[index] = binary.charCodeAt(index);
  }
  return bytes;
}

function manualRecipeFromSample(sample: SampleResponse): ManualRecipe {
  return {
    c: sample.cmyk[0],
    m: sample.cmyk[1],
    y: sample.cmyk[2],
    k: sample.cmyk[3],
  };
}

function recipeTuple(recipe: ManualRecipe): [number, number, number, number] {
  return [recipe.c, recipe.m, recipe.y, recipe.k];
}

export default function App() {
  const [state, setState] = useState<AppState | null>(null);
  const [settingsDraft, setSettingsDraft] = useState<Settings>(EMPTY_SETTINGS);
  const [manualRecipe, setManualRecipe] = useState<ManualRecipe>(FALLBACK_RECIPE);
  const [profileName, setProfileName] = useState("");
  const [selectedHistoryId, setSelectedHistoryId] = useState<string | null>(null);
  const [selectedRunView, setSelectedRunView] = useState<"keep" | "remove">("keep");
  const [standardRunView, setStandardRunView] = useState<"color" | "gray">("color");
  const [error, setError] = useState("");
  const [uploading, setUploading] = useState(false);
  const [processingSelected, setProcessingSelected] = useState(false);
  const [processingStandard, setProcessingStandard] = useState(false);
  const [loadingProfile, setLoadingProfile] = useState(false);
  const [savingProfile, setSavingProfile] = useState(false);
  const [selectedProfile, setSelectedProfile] = useState("");
  const [resultsView, setResultsView] = useState<"selected" | "standard">("selected");
  const [pickerEnabled, setPickerEnabled] = useState(false);
  const [sample, setSample] = useState<SampleResponse | null>(null);
  const [sampling, setSampling] = useState(false);
  const [inspectorLoading, setInspectorLoading] = useState(false);
  const [inspectorMeta, setInspectorMeta] = useState<Omit<InspectorPreviewPayload, "data"> | null>(
    null,
  );
  const uploadRef = useRef<HTMLInputElement | null>(null);
  const sampleAbortRef = useRef<AbortController | null>(null);
  const inspectorAbortRef = useRef<AbortController | null>(null);
  const inspectorBytesRef = useRef<Uint8Array | null>(null);
  const hoverFrameRef = useRef<number | null>(null);
  const hoverPointRef = useRef<{
    clientX: number;
    clientY: number;
    imageElement: HTMLImageElement;
  } | null>(null);

  const deferredState = useDeferredValue(state);
  const activeJob = activeJobFrom(deferredState);
  const selectedHistory =
    deferredState?.history.find((item) => item.id === selectedHistoryId) ?? null;
  const selectedRun = isSelectedRunSummary(activeJob?.last_result) ? activeJob.last_result : null;

  useEffect(() => {
    void fetchState();
  }, []);

  useEffect(() => {
    if (!state) return;
    setSettingsDraft(state.settings);
  }, [state?.settings]);

  useEffect(() => {
    if (!deferredState?.history.length) {
      setSelectedHistoryId(null);
      return;
    }
    if (selectedHistoryId && deferredState.history.some((item) => item.id === selectedHistoryId)) {
      return;
    }
    setSelectedHistoryId(null);
  }, [deferredState?.history, selectedHistoryId]);

  useEffect(() => {
    if (activeJob?.last_result && !activeJob?.last_standard_result) {
      setResultsView("selected");
      return;
    }
    if (activeJob?.last_standard_result && !activeJob?.last_result) {
      setResultsView("standard");
    }
  }, [activeJob?.id, activeJob?.last_result, activeJob?.last_standard_result]);

  useEffect(() => {
    sampleAbortRef.current?.abort();
    inspectorAbortRef.current?.abort();
    inspectorBytesRef.current = null;
    setSample(null);
    setSampling(false);
    setInspectorMeta(null);

    if (hoverFrameRef.current !== null) {
      window.cancelAnimationFrame(hoverFrameRef.current);
      hoverFrameRef.current = null;
    }

    if (!activeJob) {
      setInspectorLoading(false);
      return;
    }

    const controller = new AbortController();
    inspectorAbortRef.current = controller;
    setInspectorLoading(true);

    void fetch(
      `/api/job/${activeJob.id}/inspector?page=${activeJob.current_page}&quality=${encodeURIComponent(
        settingsDraft.preview_quality,
      )}`,
      { signal: controller.signal },
    )
      .then(async (response) => {
        const payload = (await response.json()) as InspectorPreviewPayload | { error?: string };
        if (!response.ok) {
          throw new ApiError(
            "error" in payload && payload.error ? payload.error : "Failed to prepare inspector.",
            response.status,
          );
        }
        return payload as InspectorPreviewPayload;
      })
      .then((payload) => {
        inspectorBytesRef.current = decodeBase64Bytes(payload.data);
        setInspectorMeta({
          width: payload.width,
          height: payload.height,
          source_width: payload.source_width,
          source_height: payload.source_height,
        });
      })
      .catch((err) => {
        if (controller.signal.aborted) return;
        setError(err instanceof ApiError ? err.message : "Failed to prepare live inspector.");
      })
      .finally(() => {
        if (!controller.signal.aborted) {
          setInspectorLoading(false);
        }
      });

    return () => {
      controller.abort();
      if (hoverFrameRef.current !== null) {
        window.cancelAnimationFrame(hoverFrameRef.current);
        hoverFrameRef.current = null;
      }
    };
  }, [activeJob?.id, activeJob?.current_page, settingsDraft.preview_quality]);

  function applyState(nextState: AppState) {
    startTransition(() => {
      setState(nextState);
    });
  }

  function applyOptimisticColor(jobId: string, color: Recipe) {
    startTransition(() => {
      setState((current) => {
        if (!current) return current;
        return {
          ...current,
          jobs: current.jobs.map((job) =>
            job.id === jobId
              ? { ...job, selected_colors: [...job.selected_colors, color] }
              : job,
          ),
        };
      });
    });
  }

  async function fetchState() {
    try {
      const nextState = await api<AppState>("/api/state");
      applyState(nextState);
      setError("");
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Failed to load workspace.");
    }
  }

  async function run<T>(task: () => Promise<T>) {
    try {
      setError("");
      return await task();
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Request failed.");
      throw err;
    }
  }

  async function handleUpload(files: FileList | null) {
    if (!files?.length) return;
    setUploading(true);
    try {
      const nextState = await run(() => uploadFiles<AppState>(files));
      applyState(nextState);
      if (uploadRef.current) uploadRef.current.value = "";
    } finally {
      setUploading(false);
    }
  }

  async function commitJobState(path: string, method: "POST" | "DELETE", json?: unknown) {
    const nextState = await run(() => api<AppState>(path, { method, json }));
    applyState(nextState);
  }

  async function saveSettings(nextSettings: Settings) {
    setSettingsDraft(nextSettings);
    if (!activeJob) return;
    await commitJobState(`/api/job/${activeJob.id}/settings`, "POST", nextSettings);
  }

  async function updateSetting<K extends keyof Settings>(key: K, value: Settings[K]) {
    const nextSettings = { ...settingsDraft, [key]: value };
    setSettingsDraft(nextSettings);
    if (!activeJob) return;
    await commitJobState(`/api/job/${activeJob.id}/settings`, "POST", nextSettings);
  }

  async function addColorFromManualRecipe() {
    if (!activeJob) return;
    const colorName = `Color ${activeJob.selected_colors.length + 1}`;
    const nextColor: Recipe = {
      name: colorName,
      cmyk: recipeTuple(manualRecipe),
    };
    applyOptimisticColor(activeJob.id, nextColor);
    try {
      await commitJobState(`/api/job/${activeJob.id}/colors`, "POST", {
        ...manualRecipe,
        name: colorName,
      });
    } catch (error) {
      void fetchState();
      throw error;
    }
  }

  async function processSelectedSeparation() {
    if (!activeJob) return;
    setProcessingSelected(true);
    setResultsView("selected");
    setSelectedRunView("keep");
    try {
      const currentRecipe: Recipe = {
        name: "Current",
        cmyk: recipeTuple(manualRecipe),
      };
      const hasSelectedColors = activeJob.selected_colors.length > 0;
      const selectedColorsForRun = hasSelectedColors
        ? activeJob.selected_colors.map((item) =>
            item.name === "Current" ? currentRecipe : item,
          )
        : [currentRecipe];
      const nextState = await run(() =>
        api<AppState>(`/api/job/${activeJob.id}/process`, {
          method: "POST",
          json: {
            ...settingsDraft,
            page_index: activeJob.current_page,
            selected_colors: selectedColorsForRun,
            manual_recipe: manualRecipe,
          },
        }),
      );
      applyState(nextState);
    } finally {
      setProcessingSelected(false);
    }
  }

  async function processStandardSeparation() {
    if (!activeJob) return;
    setProcessingStandard(true);
    setResultsView("standard");
    setStandardRunView("color");
    try {
      const nextState = await run(() =>
        api<AppState>(`/api/job/${activeJob.id}/standard`, {
          method: "POST",
          json: {
            ...settingsDraft,
            page_index: activeJob.current_page,
          },
        }),
      );
      applyState(nextState);
    } finally {
      setProcessingStandard(false);
    }
  }

  async function saveProfile() {
    if (!profileName.trim()) return;
    setSavingProfile(true);
    try {
      const nextState = await run(() =>
        api<AppState>("/api/profile/save", {
          method: "POST",
          json: {
            name: profileName.trim(),
            payload: {
              settings: settingsDraft,
              selected_colors: activeJob?.selected_colors ?? [],
              manual_recipe: manualRecipe,
            },
          },
        }),
      );
      applyState(nextState);
      setSelectedProfile(profileName.trim());
    } finally {
      setSavingProfile(false);
    }
  }

  async function loadProfile() {
    if (!selectedProfile) return;
    setLoadingProfile(true);
    try {
      const payload = await run(() =>
        api<{ profile?: { settings?: Settings; manual_recipe?: ManualRecipe } }>(
          "/api/profile/load",
          {
            method: "POST",
            json: { name: selectedProfile },
          },
        ),
      );
      const profile = payload.profile;
      if (!profile) return;
      const nextSettings = profile.settings ?? settingsDraft;
      setSettingsDraft(nextSettings);
      if (profile.manual_recipe) setManualRecipe(profile.manual_recipe);
      if (activeJob) {
        const nextState = await run(() =>
          api<AppState>(`/api/job/${activeJob.id}/settings`, {
            method: "POST",
            json: nextSettings,
          }),
        );
        applyState(nextState);
      }
    } finally {
      setLoadingProfile(false);
    }
  }

  function readInspectorSample(clientX: number, clientY: number, imageElement: HTMLImageElement) {
    if (!inspectorMeta || !inspectorBytesRef.current) return null;

    const rect = imageElement.getBoundingClientRect();
    if (!rect.width || !rect.height) return null;

    const previewX = Math.min(
      inspectorMeta.width - 1,
      Math.max(0, Math.floor(((clientX - rect.left) / rect.width) * inspectorMeta.width)),
    );
    const previewY = Math.min(
      inspectorMeta.height - 1,
      Math.max(0, Math.floor(((clientY - rect.top) / rect.height) * inspectorMeta.height)),
    );
    const offset = (previewY * inspectorMeta.width + previewX) * 4;
    const recipe: [number, number, number, number] = [
      inspectorBytesRef.current[offset] ?? 0,
      inspectorBytesRef.current[offset + 1] ?? 0,
      inspectorBytesRef.current[offset + 2] ?? 0,
      inspectorBytesRef.current[offset + 3] ?? 0,
    ];

    return {
      x: Math.min(
        inspectorMeta.source_width - 1,
        Math.max(0, Math.floor((previewX * inspectorMeta.source_width) / inspectorMeta.width)),
      ),
      y: Math.min(
        inspectorMeta.source_height - 1,
        Math.max(0, Math.floor((previewY * inspectorMeta.source_height) / inspectorMeta.height)),
      ),
      cmyk: recipe,
      hex: cmykToHex(recipe),
      nearest: [],
    } satisfies SampleResponse;
  }

  function queueInspectorSample(
    clientX: number,
    clientY: number,
    imageElement: HTMLImageElement,
  ) {
    hoverPointRef.current = { clientX, clientY, imageElement };

    if (hoverFrameRef.current !== null) return;

    hoverFrameRef.current = window.requestAnimationFrame(() => {
      hoverFrameRef.current = null;
      if (!hoverPointRef.current) return;
      const nextSample = readInspectorSample(
        hoverPointRef.current.clientX,
        hoverPointRef.current.clientY,
        hoverPointRef.current.imageElement,
      );
      if (nextSample) setSample(nextSample);
    });
  }

  async function inspectPixel(
    clientX: number,
    clientY: number,
    imageElement: HTMLImageElement,
  ) {
    if (!activeJob) return;

    const rect = imageElement.getBoundingClientRect();
    if (!rect.width || !rect.height) return;

    const pageShape = activeJob.pages[activeJob.current_page]?.shape ?? [];
    const sourceHeight = pageShape[0] ?? 0;
    const sourceWidth = pageShape[1] ?? 0;
    if (!sourceWidth || !sourceHeight) return;

    const x = Math.floor(((clientX - rect.left) / rect.width) * sourceWidth);
    const y = Math.floor(((clientY - rect.top) / rect.height) * sourceHeight);

    sampleAbortRef.current?.abort();
    const controller = new AbortController();
    sampleAbortRef.current = controller;
    setSampling(true);

    try {
      const response = await fetch(
        `/api/job/${activeJob.id}/sample?page=${activeJob.current_page}&x=${x}&y=${y}`,
        { signal: controller.signal },
      );
      if (!response.ok) return;
      const data = (await response.json()) as SampleResponse;
      setSample(data);
      setManualRecipe(manualRecipeFromSample(data));
    } catch {
      // Ignore aborted sampling requests.
    } finally {
      setSampling(false);
    }
  }

  const previewUrl = activeJob
    ? `/preview/${activeJob.id}/original?page=${activeJob.current_page}&quality=${encodeURIComponent(
        settingsDraft.preview_quality,
      )}&sharpen=${settingsDraft.preview_sharpen ? "1" : "0"}`
    : "";

  function renderSelectedOutput(summary: ResultSummary, mode: "keep" | "remove") {
    const isRemove = mode === "remove";

    return (
      <div className="result-block">
        <div className="result-head">
          <div>
            <h3>{isRemove ? "Removed Color Output" : "Selected Color Output"}</h3>
            <p>Processed in {summary.stats.processing_seconds.toFixed(3)}s</p>
          </div>
          <a className="download-link" href={summary.downloads.zip}>
            {isRemove ? "Download Removed ZIP" : "Download Selected ZIP"}
          </a>
        </div>
        {!!summary.stats.warnings?.length && (
          <div className="warning-stack">
            {summary.stats.warnings.map((warning) => (
              <div key={warning} className="warning-item">
                {warning}
              </div>
            ))}
          </div>
        )}
        <div className="metric-row">
          <div className="metric-card">
            <span>{isRemove ? "Removed Pixels" : "Matched Pixels"}</span>
            <strong>{summary.stats.matched_pixels?.toLocaleString() ?? 0}</strong>
          </div>
          <div className="metric-card">
            <span>{isRemove ? "Removed Area" : "Matched Area"}</span>
            <strong>{summary.stats.matched_percent?.toFixed(2) ?? "0.00"}%</strong>
          </div>
          <div className="metric-card">
            <span>Ink Coverage</span>
            <strong>{summary.stats.ink_coverage?.toFixed(2) ?? "0.00"}%</strong>
          </div>
        </div>
        <div className="result-grid two-up">
          <article className="result-panel">
            <h4>{isRemove ? "Original TIFF After Removal" : "Selected Separation"}</h4>
            <img
              src={summary.previews.selected}
              alt={isRemove ? "Removed preview" : "Selected preview"}
              loading="lazy"
              decoding="async"
            />
          </article>
          <article className="result-panel">
            <h4>{isRemove ? "Grayscale After Removal" : "Grayscale Output"}</h4>
            <img
              src={summary.previews.grayscale}
              alt={isRemove ? "Removed grayscale preview" : "Grayscale preview"}
              loading="lazy"
              decoding="async"
            />
          </article>
        </div>
      </div>
    );
  }

  function handleActionClick(
    event: React.MouseEvent<HTMLButtonElement>,
    action: () => void | Promise<void>,
  ) {
    event.preventDefault();
    event.stopPropagation();
    void action();
  }

  return (
    <div className="app-shell">
      <div className="atmosphere" />
      <header className="hero">
        <div className="hero-copy">
          <span className="eyebrow">React + TypeScript Production Surface</span>
          <h1>CMYK Studio</h1>
          <p>
            Faster queue handling, cleaner preview flow, stronger production controls, and the
            same TIFF-processing backend.
          </p>
        </div>
        <div className="hero-actions">
          <label className="upload-pill">
            <input
              ref={uploadRef}
              type="file"
              accept=".tif,.tiff"
              multiple
              onChange={(event) => void handleUpload(event.target.files)}
            />
            <span>{uploading ? "Uploading..." : "Upload TIFF Files"}</span>
          </label>
          <button className="secondary-pill" onClick={() => void fetchState()}>
            Refresh
          </button>
        </div>
      </header>

      {error ? <div className="error-banner">{error}</div> : null}

      <div className="workspace-grid">
        <aside className="panel stack controls-panel">
          <section className="card">
            <div className="card-head">
              <div>
                <span className="micro-label">Processing</span>
                <h2>Settings</h2>
              </div>
            </div>
            <div className="form-grid">
              <label>
                <span>Match</span>
                <select
                  value={settingsDraft.match_mode}
                  onChange={(event) =>
                    void updateSetting("match_mode", event.target.value as Settings["match_mode"])
                  }
                >
                  <option value="exact">Exact</option>
                  <option value="nearest">Nearest</option>
                  <option value="spread">Spread</option>
                </select>
              </label>
              <label>
                <span>Spread</span>
                <input
                  type="number"
                  min={0}
                  max={40}
                  value={settingsDraft.spread}
                  onChange={(event) =>
                    setSettingsDraft((current) => ({
                      ...current,
                      spread: Number(event.target.value || 0),
                    }))
                  }
                  onBlur={() => void saveSettings(settingsDraft)}
                />
              </label>
              <label>
                <span>Compression</span>
                <select
                  value={settingsDraft.compression}
                  onChange={(event) =>
                    void updateSetting(
                      "compression",
                      event.target.value as Settings["compression"],
                    )
                  }
                >
                  <option value="lzw">LZW</option>
                  <option value="deflate">Deflate</option>
                  <option value="none">None</option>
                </select>
              </label>
              <label>
                <span>Preview</span>
                <select
                  value={settingsDraft.preview_quality}
                  onChange={(event) =>
                    void updateSetting(
                      "preview_quality",
                      event.target.value as Settings["preview_quality"],
                    )
                  }
                >
                  <option value="proxy">Fast Proxy</option>
                  <option value="full">Sharp Preview</option>
                </select>
              </label>
              <label>
                <span>Gray Gamma</span>
                <input
                  type="number"
                  min={0.2}
                  max={3}
                  step={0.1}
                  value={settingsDraft.gray_gamma}
                  onChange={(event) =>
                    setSettingsDraft((current) => ({
                      ...current,
                      gray_gamma: Number(event.target.value || 1),
                    }))
                  }
                  onBlur={() => void saveSettings(settingsDraft)}
                />
              </label>
              <label className="toggle-row">
                <input
                  type="checkbox"
                  checked={settingsDraft.invert_grayscale}
                  onChange={(event) =>
                    void updateSetting("invert_grayscale", event.target.checked)
                  }
                />
                <span>Invert grayscale</span>
              </label>
              <label className="toggle-row">
                <input
                  type="checkbox"
                  checked={settingsDraft.preview_sharpen}
                  onChange={(event) => void updateSetting("preview_sharpen", event.target.checked)}
                />
                <span>Sharpen preview</span>
              </label>
            </div>
          </section>

          <section className="card">
            <div className="card-head">
              <div>
                <span className="micro-label">Color Lab</span>
                <h2>Recipe Builder</h2>
              </div>
              <div className="swatch" style={{ background: cmykToHex(manualRecipe) }} />
            </div>
            <div className="recipe-grid">
              {(["c", "m", "y", "k"] as const).map((channel) => (
                <label key={channel}>
                  <span>{channel.toUpperCase()}</span>
                  <input
                    type="number"
                    min={0}
                    max={100}
                    value={manualRecipe[channel]}
                    onChange={(event) =>
                      setManualRecipe((current) => ({
                        ...current,
                        [channel]: Number(event.target.value || 0),
                      }))
                    }
                  />
                </label>
              ))}
            </div>
            <div className="button-row">
              <button type="button" className="primary-button" onClick={() => void addColorFromManualRecipe()}>
                Add Color
              </button>
              <button
                type="button"
                className="accent-button"
                disabled={!activeJob || processingSelected}
                onClick={() => void processSelectedSeparation()}
              >
                {processingSelected ? "Running..." : "Run Color"}
              </button>
            </div>
            <div className="inline-note">Run Color now generates both selected and removed outputs.</div>
            <button
              type="button"
              className="secondary-wide"
              disabled={!activeJob || processingStandard}
              onClick={() => void processStandardSeparation()}
            >
              {processingStandard ? "Building 4 Channels..." : "Run 4 Channel Separation"}
            </button>
            <div className="subsection">
              <div className="subsection-head">
                <span className="micro-label">Selected</span>
                <button
                  type="button"
                  className="ghost-button"
                  disabled={!activeJob}
                  onClick={() =>
                    activeJob && void commitJobState(`/api/job/${activeJob.id}/clear`, "POST", {})
                  }
                >
                  Clear
                </button>
              </div>
              <div className="selected-list">
                {activeJob?.selected_colors.length ? (
                  activeJob.selected_colors.map((item, index) => (
                    <div key={`${item.name}-${index}`} className="selected-item">
                      <span
                        className="chip-swatch"
                        style={{ background: cmykToHex(item.cmyk) }}
                      />
                      <div className="selected-copy">
                        <strong>{item.name}</strong>
                        <span>
                          C {item.cmyk[0]} M {item.cmyk[1]} Y {item.cmyk[2]} K {item.cmyk[3]}
                        </span>
                      </div>
                      <button
                        type="button"
                        className="mini-button"
                        onClick={() =>
                          void commitJobState(
                            `/api/job/${activeJob.id}/colors/${index}`,
                            "DELETE",
                          )
                        }
                      >
                        Remove
                      </button>
                    </div>
                  ))
                ) : (
                  <div className="empty-note">No selected colors yet.</div>
                )}
              </div>
            </div>

            <div className="subsection">
              <span className="micro-label">Recent</span>
              <div className="chip-cloud">
                {deferredState?.recent_colors.map((item) => (
                  <button
                    type="button"
                    key={`${item.name}-${item.cmyk.join("-")}`}
                    className="recipe-chip"
                    onClick={() =>
                      setManualRecipe({
                        c: item.cmyk[0],
                        m: item.cmyk[1],
                        y: item.cmyk[2],
                        k: item.cmyk[3],
                      })
                    }
                  >
                    <span className="chip-swatch" style={{ background: cmykToHex(item.cmyk) }} />
                    <span>{item.name}</span>
                  </button>
                ))}
              </div>
            </div>
          </section>

          <section className="card">
            <div className="card-head">
              <div>
                <span className="micro-label">Profiles</span>
                <h2>Saved Setups</h2>
              </div>
            </div>
            <div className="form-grid profile-grid">
              <label>
                <span>Name</span>
                <input
                  type="text"
                  value={profileName}
                  onChange={(event) => setProfileName(event.target.value)}
                  placeholder="Press profile"
                />
              </label>
              <label>
                <span>Load</span>
                <select
                  value={selectedProfile}
                  onChange={(event) => setSelectedProfile(event.target.value)}
                >
                  <option value="">Select profile</option>
                  {deferredState?.profiles.map((profile) => (
                    <option key={profile} value={profile}>
                      {profile}
                    </option>
                  ))}
                </select>
              </label>
            </div>
            <div className="button-row">
              <button
                type="button"
                className="secondary-pill"
                disabled={savingProfile}
                onClick={() => void saveProfile()}
              >
                {savingProfile ? "Saving..." : "Save Profile"}
              </button>
              <button
                type="button"
                className="secondary-pill"
                disabled={loadingProfile}
                onClick={() => void loadProfile()}
              >
                {loadingProfile ? "Loading..." : "Load Profile"}
              </button>
            </div>
          </section>
        </aside>

        <main className="panel stack stage-panel">
          <section className="card preview-card">
            <div className="card-head">
              <div>
                <span className="micro-label">Preview Workspace</span>
                <h2>{activeJob?.job_name ?? "No Active Job"}</h2>
                <p className="meta-line">
                  {activeJob
                    ? `${activeJob.file_name} · ${activeJob.pages[activeJob.current_page]?.shape.join(" × ")} · ${activeJob.pages[activeJob.current_page]?.dpi_x.toFixed(
                        0,
                      )} DPI`
                    : "Upload a TIFF file to begin."}
                </p>
                <div className="preview-source-note">Original TIFF preview from CMYK source</div>
              </div>
              <div className="preview-tools">
                <button
                  type="button"
                  className={`picker-button ${pickerEnabled ? "active" : ""}`}
                  disabled={!activeJob}
                  onClick={() => {
                    setPickerEnabled((current) => !current);
                    setSample(null);
                  }}
                  title="CMYK Inspector"
                >
                  <img src="/color-picker.png" alt="Color picker" />
                </button>
                <select
                  value={activeJob?.current_page ?? 0}
                  disabled={!activeJob}
                  onChange={(event) =>
                    activeJob &&
                    void commitJobState(`/api/job/${activeJob.id}/page`, "POST", {
                      page_index: Number(event.target.value),
                    })
                  }
                >
                  {activeJob?.pages.map((page) => (
                    <option key={page.index} value={page.index}>
                      {page.label}
                    </option>
                  ))}
                </select>
                <button
                  type="button"
                  className="ghost-button"
                  disabled={!activeJob}
                  onClick={() =>
                    activeJob &&
                    void commitJobState(`/api/job/${activeJob.id}/duplicate-last`, "POST", {})
                  }
                >
                  Duplicate Last
                </button>
              </div>
            </div>
            <div className={`preview-frame ${pickerEnabled ? "pick-mode" : ""}`}>
              {activeJob ? (
                <>
                  <img
                    src={previewUrl}
                    alt="TIFF preview"
                    onMouseMove={(event) => {
                      if (!pickerEnabled || inspectorLoading) return;
                      queueInspectorSample(event.clientX, event.clientY, event.currentTarget);
                    }}
                    onMouseLeave={() => {
                      if (hoverFrameRef.current !== null) {
                        window.cancelAnimationFrame(hoverFrameRef.current);
                        hoverFrameRef.current = null;
                      }
                      setSample(null);
                    }}
                    onClick={(event) => {
                      if (!pickerEnabled) return;
                      if (event.button !== 0) return;
                      const hoveredSample = readInspectorSample(
                        event.clientX,
                        event.clientY,
                        event.currentTarget,
                      );
                      if (hoveredSample) {
                        setSample(hoveredSample);
                        setManualRecipe(manualRecipeFromSample(hoveredSample));
                      }
                      void inspectPixel(
                        event.clientX,
                        event.clientY,
                        event.currentTarget,
                      );
                    }}
                  />
                  {pickerEnabled ? (
                    <div className="picker-hint">
                      {inspectorLoading
                        ? "Preparing live CMYK inspector..."
                        : sampling
                          ? "Locking exact sample..."
                          : "Inspector on. Move for live CMYK, click to lock exact values."}
                    </div>
                  ) : null}
                  {sample ? (
                    <div className="sample-card">
                      <div className="sample-card-head">
                        <span
                          className="sample-swatch"
                          style={{ background: sample.hex }}
                        />
                        <div>
                          <strong>
                            C {sample.cmyk[0]} M {sample.cmyk[1]} Y {sample.cmyk[2]} K {sample.cmyk[3]}
                          </strong>
                          <span>
                            X {sample.x} · Y {sample.y}
                          </span>
                        </div>
                      </div>
                      <div className="sample-mode">
                        {sample.nearest.length
                          ? "Exact pixel from original TIFF"
                          : "Live inspector feed from original TIFF preview"}
                      </div>
                      {sample.nearest.length ? (
                        <div className="sample-nearest">
                          <span>Nearest</span>
                          <div className="nearest-list">
                            {sample.nearest.slice(0, 3).map((entry, index) => (
                              <div key={`${entry.cmyk.join("-")}-${index}`} className="nearest-item">
                                <span
                                  className="chip-swatch"
                                  style={{
                                    background: cmykToHex(entry.cmyk),
                                  }}
                                />
                                <span>
                                  C {entry.cmyk[0]} M {entry.cmyk[1]} Y {entry.cmyk[2]} K {entry.cmyk[3]}
                                </span>
                              </div>
                            ))}
                          </div>
                        </div>
                      ) : null}
                    </div>
                  ) : null}
                </>
              ) : (
                <div className="empty-view">Preview will appear here.</div>
              )}
            </div>
            {activeJob ? (
              <div className="info-grid">
                <div className="info-pill">
                  <span>Photometric</span>
                  <strong>{activeJob.pages[activeJob.current_page]?.photometric}</strong>
                </div>
                <div className="info-pill">
                  <span>Compression</span>
                  <strong>{activeJob.pages[activeJob.current_page]?.compression}</strong>
                </div>
                <div className="info-pill">
                  <span>Dtype</span>
                  <strong>{activeJob.pages[activeJob.current_page]?.dtype}</strong>
                </div>
                <div className="info-pill">
                  <span>Pages</span>
                  <strong>{activeJob.pages.length}</strong>
                </div>
              </div>
            ) : null}
          </section>

          <section className="card results-card">
            <div className="card-head">
              <div>
                <span className="micro-label">Outputs</span>
                <h2>Results</h2>
              </div>
              <div className="view-switch">
                <button
                  type="button"
                  className={`view-chip ${resultsView === "selected" ? "active" : ""}`}
                  disabled={!activeJob?.last_result}
                  onClick={() => setResultsView("selected")}
                >
                  Run Color
                </button>
                <button
                  type="button"
                  className={`view-chip ${resultsView === "standard" ? "active" : ""}`}
                  disabled={!activeJob?.last_standard_result}
                  onClick={() => setResultsView("standard")}
                >
                  4 Channel
                </button>
              </div>
            </div>

            {resultsView === "selected" && selectedRun ? (
              <>
                <div className="result-head split-head">
                  <div>
                    <h3>Selected Color Run</h3>
                    <p>One click generated both selected and removed outputs.</p>
                  </div>
                  <a className="download-link" href={selectedRun.downloads.zip}>
                    Download Combined ZIP
                  </a>
                </div>
                <div className="view-switch sub-switch">
                  <button
                    type="button"
                    className={`view-chip ${selectedRunView === "keep" ? "active" : ""}`}
                    onClick={() => setSelectedRunView("keep")}
                  >
                    Selected Preview
                  </button>
                  <button
                    type="button"
                    className={`view-chip ${selectedRunView === "remove" ? "active" : ""}`}
                    onClick={() => setSelectedRunView("remove")}
                  >
                    Removed Preview
                  </button>
                </div>
                {selectedRunView === "keep"
                  ? renderSelectedOutput(selectedRun.selected_result, "keep")
                  : renderSelectedOutput(selectedRun.removed_result, "remove")}
              </>
            ) : resultsView === "selected" && activeJob?.last_result ? (
              renderSelectedOutput(
                activeJob.last_result as ResultSummary,
                (activeJob.last_result as ResultSummary).operation === "remove" ? "remove" : "keep",
              )
            ) : resultsView === "selected" ? (
              <div className="empty-note large">
                Run selected color to generate both selected and removed outputs.
              </div>
            ) : null}

            {resultsView === "standard" && activeJob?.last_standard_result ? (
              <div className="result-block">
                <div className="result-head">
                  <div>
                    <h3>4 Channel Separation</h3>
                    <p>
                      Ready in {activeJob.last_standard_result.stats.processing_seconds.toFixed(3)}s
                    </p>
                  </div>
                  <a className="download-link" href={activeJob.last_standard_result.downloads.zip}>
                    Download 4 Channel ZIP
                  </a>
                </div>
                <div className="metric-row">
                  <div className="metric-card">
                    <span>Width</span>
                    <strong>
                      {activeJob.last_standard_result.stats.width?.toLocaleString() ?? 0}
                    </strong>
                  </div>
                  <div className="metric-card">
                    <span>Height</span>
                    <strong>
                      {activeJob.last_standard_result.stats.height?.toLocaleString() ?? 0}
                    </strong>
                  </div>
                  <div className="metric-card">
                    <span>DPI</span>
                    <strong>{activeJob.last_standard_result.stats.dpi_x?.toFixed(0) ?? 0}</strong>
                  </div>
                </div>
                <div className="view-switch sub-switch">
                  <button
                    type="button"
                    className={`view-chip ${standardRunView === "color" ? "active" : ""}`}
                    onClick={() => setStandardRunView("color")}
                  >
                    Color Plates
                  </button>
                  <button
                    type="button"
                    className={`view-chip ${standardRunView === "gray" ? "active" : ""}`}
                    onClick={() => setStandardRunView("gray")}
                  >
                    Gray Plates
                  </button>
                </div>
                <div className="result-grid four-up">
                  {(["cyan", "magenta", "yellow", "black"] as const).map((color) => (
                    <article key={`${color}-color`} className="result-panel">
                      <h4>
                        {color[0].toUpperCase() + color.slice(1)} {standardRunView === "color" ? "Plate" : "Gray"}
                      </h4>
                      <img
                        src={
                          activeJob.last_standard_result?.previews[
                            `${color}_${standardRunView === "color" ? "color" : "gray"}`
                          ]
                        }
                        alt={`${color} ${standardRunView} plate`}
                        loading="lazy"
                        decoding="async"
                      />
                    </article>
                  ))}
                </div>
              </div>
            ) : resultsView === "standard" ? (
              <div className="empty-note large">
                Run 4 Channel Separation to generate CMYK plate outputs.
              </div>
            ) : null}
          </section>
        </main>

        <aside className="panel stack queue-panel">
          <section className="card">
            <div className="card-head">
              <div>
                <span className="micro-label">Batch Queue</span>
                <h2>Jobs</h2>
              </div>
            </div>
            <div className="queue-list">
              {deferredState?.jobs.length ? (
                deferredState.jobs.map((job) => (
                  <article
                    key={job.id}
                    className={`queue-card ${job.id === deferredState.active_job_id ? "active" : ""}`}
                  >
                    <button
                      type="button"
                      className="queue-main"
                      onClick={() => void commitJobState(`/api/job/${job.id}/activate`, "POST", {})}
                    >
                      <strong>{job.job_name}</strong>
                      <span>{job.file_name}</span>
                      <span>{job.pages.length} page(s)</span>
                    </button>
                    <button
                      type="button"
                      className="mini-button"
                      onClick={(event) =>
                        handleActionClick(event, () => commitJobState(`/api/job/${job.id}`, "DELETE"))
                      }
                    >
                      Delete
                    </button>
                  </article>
                ))
              ) : (
                <div className="empty-note">No TIFF files loaded yet.</div>
              )}
            </div>
          </section>

          <section className="card">
            <div className="card-head">
              <div>
                <span className="micro-label">History</span>
                <h2>Recent Runs</h2>
              </div>
              <button
                type="button"
                className="ghost-button"
                onClick={(event) =>
                  handleActionClick(event, () => commitJobState("/api/history/clear", "POST", {}))
                }
              >
                Clear
              </button>
            </div>
            <div className="history-list">
              {deferredState?.history.length ? (
                [...deferredState.history].reverse().map((item) => (
                  <article
                    key={item.id}
                    className={`history-card ${item.id === selectedHistoryId ? "active" : ""}`}
                  >
                    <button type="button" className="history-main" onClick={() => setSelectedHistoryId(item.id)}>
                      <strong>{item.job_name}</strong>
                      <span>
                        {item.file_name} · page {item.page}
                      </span>
                      <span>
                        {item.kind === "selected"
                          ? `${item.matched_percent.toFixed(2)}% matched`
                          : `${item.processing_seconds.toFixed(3)}s`}
                      </span>
                    </button>
                    <button
                      type="button"
                      className="mini-button"
                      onClick={(event) =>
                        handleActionClick(event, () => commitJobState(`/api/history/${item.id}`, "DELETE"))
                      }
                    >
                      Delete
                    </button>
                  </article>
                ))
              ) : (
                <div className="empty-note">No history yet.</div>
              )}
            </div>

            {selectedHistory ? (
              <div className="history-viewer">
                <div className="history-head">
                  <div>
                    <strong>{selectedHistory.job_name}</strong>
                    <span>{selectedHistory.time}</span>
                  </div>
                  <a className="download-link" href={selectedHistory.summary.downloads.zip}>
                    ZIP
                  </a>
                </div>
                <div className="history-grid">
                  <article className="history-thumb">
                    <h4>Original</h4>
                    <img
                      src={selectedHistory.summary.previews.original}
                      alt="Original preview"
                      loading="lazy"
                      decoding="async"
                    />
                  </article>
                  {historyImageEntries(selectedHistory).map((entry) => (
                    <article key={entry.label} className="history-thumb">
                      <h4>{entry.label}</h4>
                      <img src={entry.src} alt={entry.label} loading="lazy" decoding="async" />
                    </article>
                  ))}
                </div>
              </div>
            ) : deferredState?.history.length ? (
              <div className="empty-note large">Select a history row to load its preview images.</div>
            ) : null}
          </section>
        </aside>
      </div>
    </div>
  );
}
