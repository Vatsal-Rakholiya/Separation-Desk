const ui = {
  uploadInput: document.getElementById("uploadInput"),
  refreshButton: document.getElementById("refreshButton"),
  queueList: document.getElementById("queueList"),
  queueEmpty: document.getElementById("queueEmpty"),
  pageSelect: document.getElementById("pageSelect"),
  previewImage: document.getElementById("previewImage"),
  previewScroll: document.getElementById("previewScroll"),
  fileInfo: document.getElementById("fileInfo"),
  activeJobTitle: document.getElementById("activeJobTitle"),
  activeJobMeta: document.getElementById("activeJobMeta"),
  cInput: document.getElementById("cInput"),
  mInput: document.getElementById("mInput"),
  yInput: document.getElementById("yInput"),
  kInput: document.getElementById("kInput"),
  currentSwatch: document.getElementById("currentSwatch"),
  currentRecipeText: document.getElementById("currentRecipeText"),
  addColorButton: document.getElementById("addColorButton"),
  processButton: document.getElementById("processButton"),
  standardProcessButton: document.getElementById("standardProcessButton"),
  duplicateButton: document.getElementById("duplicateButton"),
  clearButton: document.getElementById("clearButton"),
  selectedColors: document.getElementById("selectedColors"),
  recentColors: document.getElementById("recentColors"),
  matchModeSelect: document.getElementById("matchModeSelect"),
  selectionModeSelect: document.getElementById("selectionModeSelect"),
  spreadInput: document.getElementById("spreadInput"),
  compressionSelect: document.getElementById("compressionSelect"),
  previewQualitySelect: document.getElementById("previewQualitySelect"),
  invertGrayInput: document.getElementById("invertGrayInput"),
  sharpenPreviewInput: document.getElementById("sharpenPreviewInput"),
  gammaInput: document.getElementById("gammaInput"),
  saveProfileButton: document.getElementById("saveProfileButton"),
  profileNameInput: document.getElementById("profileNameInput"),
  profileSelect: document.getElementById("profileSelect"),
  loadProfileButton: document.getElementById("loadProfileButton"),
  resultsPanel: document.getElementById("resultsPanel"),
  selectedResultSection: document.getElementById("selectedResultSection"),
  standardResultSection: document.getElementById("standardResultSection"),
  historySection: document.getElementById("historySection"),
  warningStack: document.getElementById("warningStack"),
  statGrid: document.getElementById("statGrid"),
  selectedPreview: document.getElementById("selectedPreview"),
  grayscalePreview: document.getElementById("grayscalePreview"),
  selectedEmpty: document.getElementById("selectedEmpty"),
  grayscaleEmpty: document.getElementById("grayscaleEmpty"),
  resultMeta: document.getElementById("resultMeta"),
  zipDownload: document.getElementById("zipDownload"),
  standardZipDownload: document.getElementById("standardZipDownload"),
  standardMeta: document.getElementById("standardMeta"),
  cyanPreview: document.getElementById("cyanPreview"),
  magentaPreview: document.getElementById("magentaPreview"),
  yellowPreview: document.getElementById("yellowPreview"),
  blackPreview: document.getElementById("blackPreview"),
  cyanGrayPreview: document.getElementById("cyanGrayPreview"),
  magentaGrayPreview: document.getElementById("magentaGrayPreview"),
  yellowGrayPreview: document.getElementById("yellowGrayPreview"),
  blackGrayPreview: document.getElementById("blackGrayPreview"),
  cyanEmpty: document.getElementById("cyanEmpty"),
  magentaEmpty: document.getElementById("magentaEmpty"),
  yellowEmpty: document.getElementById("yellowEmpty"),
  blackEmpty: document.getElementById("blackEmpty"),
  cyanGrayEmpty: document.getElementById("cyanGrayEmpty"),
  magentaGrayEmpty: document.getElementById("magentaGrayEmpty"),
  yellowGrayEmpty: document.getElementById("yellowGrayEmpty"),
  blackGrayEmpty: document.getElementById("blackGrayEmpty"),
  historyList: document.getElementById("historyList"),
  historyViewer: document.getElementById("historyViewer"),
  clearHistoryButton: document.getElementById("clearHistoryButton"),
  queueItemTemplate: document.getElementById("queueItemTemplate"),
};

const appState = {
  state: null,
  draggingJobId: null,
  selectedHistoryId: null,
};

async function api(path, options = {}) {
  let response;
  try {
    response = await fetch(path, {
      headers: { "Content-Type": "application/json", ...(options.headers || {}) },
      ...options,
    });
  } catch (_error) {
    throw new Error(`Request failed for ${path}. Check that the Flask server is still running.`);
  }
  const contentType = response.headers.get("content-type") || "";
  const payload = contentType.includes("application/json") ? await response.json() : await response.text();
  if (!response.ok) {
    throw new Error(payload.error || payload || "Request failed.");
  }
  return payload;
}

function activeJob() {
  if (!appState.state) return null;
  return appState.state.jobs.find((job) => job.id === appState.state.active_job_id) || null;
}

function currentSettings() {
  return {
    match_mode: ui.matchModeSelect.value,
    selection_mode: ui.selectionModeSelect.value,
    spread: Number(ui.spreadInput.value || 0),
    compression: ui.compressionSelect.value,
    preview_quality: ui.previewQualitySelect.value,
    invert_grayscale: ui.invertGrayInput.checked,
    preview_sharpen: ui.sharpenPreviewInput.checked,
    gray_gamma: Number(ui.gammaInput.value || 1),
  };
}

function manualRecipe() {
  return {
    c: Number(ui.cInput.value || 0),
    m: Number(ui.mInput.value || 0),
    y: Number(ui.yInput.value || 0),
    k: Number(ui.kInput.value || 0),
  };
}

function recipeArray() {
  const recipe = manualRecipe();
  return [recipe.c, recipe.m, recipe.y, recipe.k];
}

function cmykToHex(c, m, y, k) {
  const cn = Math.max(0, Math.min(100, c)) / 100;
  const mn = Math.max(0, Math.min(100, m)) / 100;
  const yn = Math.max(0, Math.min(100, y)) / 100;
  const kn = Math.max(0, Math.min(100, k)) / 100;
  const r = Math.round(255 * (1 - cn) * (1 - kn));
  const g = Math.round(255 * (1 - mn) * (1 - kn));
  const b = Math.round(255 * (1 - yn) * (1 - kn));
  return `#${[r, g, b].map((value) => value.toString(16).padStart(2, "0")).join("").toUpperCase()}`;
}

function updateCurrentRecipeDisplay() {
  const [c, m, y, k] = recipeArray();
  ui.currentRecipeText.textContent = `C ${c} M ${m} Y ${y} K ${k}`;
  ui.currentSwatch.style.background = cmykToHex(c, m, y, k);
}

function renderQueue() {
  ui.queueList.innerHTML = "";
  const jobs = appState.state?.jobs || [];
  ui.queueEmpty.classList.toggle("hidden", jobs.length > 0);
  jobs.forEach((job) => {
    const node = ui.queueItemTemplate.content.firstElementChild.cloneNode(true);
    node.dataset.jobId = job.id;
    node.classList.toggle("active", job.id === appState.state.active_job_id);
    node.querySelector(".queue-title").textContent = job.job_name;
    node.querySelector(".queue-meta").textContent = `${job.file_name} - ${job.pages.length} page(s)`;
    node.querySelector(".queue-main").addEventListener("click", () => activateJob(job.id));
    node.querySelector(".queue-delete").addEventListener("click", async (event) => {
      event.stopPropagation();
      appState.state = await api(`/api/job/${job.id}`, { method: "DELETE" });
      render();
    });
    node.addEventListener("dragstart", () => {
      appState.draggingJobId = job.id;
      node.classList.add("dragging");
    });
    node.addEventListener("dragend", () => {
      appState.draggingJobId = null;
      node.classList.remove("dragging");
    });
    node.addEventListener("dragover", (event) => {
      event.preventDefault();
    });
    node.addEventListener("drop", async (event) => {
      event.preventDefault();
      if (!appState.draggingJobId || appState.draggingJobId === job.id) return;
      const ids = [...appState.state.jobs.map((item) => item.id)];
      const from = ids.indexOf(appState.draggingJobId);
      const to = ids.indexOf(job.id);
      const [moved] = ids.splice(from, 1);
      ids.splice(to, 0, moved);
      appState.state = await api("/api/reorder", {
        method: "POST",
        body: JSON.stringify({ ordered_ids: ids }),
      });
      render();
    });
    ui.queueList.appendChild(node);
  });
}

function renderPageOptions(job) {
  ui.pageSelect.innerHTML = "";
  if (!job) return;
  job.pages.forEach((page) => {
    const option = document.createElement("option");
    option.value = String(page.index);
    option.textContent = page.label;
    option.selected = page.index === job.current_page;
    ui.pageSelect.appendChild(option);
  });
}

function renderLists() {
  const job = activeJob();
  ui.selectedColors.innerHTML = "";
  (job?.selected_colors || []).forEach((item, index) => {
    const row = document.createElement("div");
    row.className = "selected-item";
    row.innerHTML = `<div><strong>${item.name}</strong><div>C ${item.cmyk[0]} M ${item.cmyk[1]} Y ${item.cmyk[2]} K ${item.cmyk[3]}</div></div>`;
    const remove = document.createElement("button");
    remove.className = "ghost-button remove-chip";
    remove.type = "button";
    remove.textContent = "Remove";
    remove.addEventListener("click", () => removeColor(index));
    row.appendChild(remove);
    ui.selectedColors.appendChild(row);
  });

  renderChipList(ui.recentColors, appState.state?.recent_colors || [], true);
  renderHistory(appState.state?.history || []);
}

function renderChipList(container, items, applyOnClick) {
  container.innerHTML = "";
  items.forEach((item) => {
    const button = document.createElement("button");
    button.className = "chip";
    button.innerHTML = `<strong>${item.name}</strong><span>C ${item.cmyk[0]} M ${item.cmyk[1]} Y ${item.cmyk[2]} K ${item.cmyk[3]}</span>`;
    if (applyOnClick) {
      button.addEventListener("click", () => applyRecipe(item.cmyk));
    }
    container.appendChild(button);
  });
}

function renderHistory(items) {
  ui.historyList.innerHTML = "";
  if (!items.length) {
    ui.historyViewer.classList.add("hidden");
    ui.historyViewer.innerHTML = "";
  }
  items.slice().reverse().forEach((item) => {
    const row = document.createElement("div");
    row.className = "history-item";
    row.classList.toggle("active", item.id === appState.selectedHistoryId);
    const trailing = item.kind === "standard" ? "4 ch" : `${item.matched_percent ?? 0}%`;
    row.innerHTML = `<button class="history-open" type="button"><div><strong>${item.job_name}</strong><div>${item.file_name} - page ${item.page}</div></div><div>${trailing}</div></button>`;
    const remove = document.createElement("button");
    remove.className = "queue-delete ghost-button";
    remove.type = "button";
    remove.textContent = "Delete";
    remove.addEventListener("click", async (event) => {
      event.stopPropagation();
      appState.state = await api(`/api/history/${item.id}`, { method: "DELETE" });
      if (appState.selectedHistoryId === item.id) {
        appState.selectedHistoryId = null;
      }
      render();
    });
    row.querySelector(".history-open").addEventListener("click", () => {
      appState.selectedHistoryId = item.id;
      renderHistoryViewer(item);
      renderHistory(appState.state?.history || []);
    });
    row.appendChild(remove);
    ui.historyList.appendChild(row);
  });
}

function historyPreviewCards(title, entries) {
  return `
    <div class="history-card-block">
      <h4>${title}</h4>
      <div class="history-card-grid">
        ${entries.map((entry) => `
          <article class="result-card">
            <h3>${entry.label}</h3>
            <img src="${entry.src}" alt="${entry.label}">
          </article>
        `).join("")}
      </div>
    </div>
  `;
}

function renderHistoryViewer(item) {
  if (!item?.summary) {
    ui.historyViewer.classList.add("hidden");
    ui.historyViewer.innerHTML = "";
    return;
  }
  const summary = item.summary;
  const originalBlock = historyPreviewCards("Original", [{ label: "Original Preview", src: `${summary.previews.original}?t=${Date.now()}` }]);
  let outputBlock = "";
  if (item.kind === "selected") {
    outputBlock = historyPreviewCards("Selected Separation", [
      { label: "Selected Separation", src: `${summary.previews.selected}?t=${Date.now()}` },
      { label: "Grayscale Output", src: `${summary.previews.grayscale}?t=${Date.now()}` },
    ]);
  } else {
    outputBlock = historyPreviewCards("4 Channel Color Plates", [
      { label: "Cyan Plate", src: `${summary.previews.cyan_color}?t=${Date.now()}` },
      { label: "Magenta Plate", src: `${summary.previews.magenta_color}?t=${Date.now()}` },
      { label: "Yellow Plate", src: `${summary.previews.yellow_color}?t=${Date.now()}` },
      { label: "Black Plate", src: `${summary.previews.black_color}?t=${Date.now()}` },
    ]) + historyPreviewCards("4 Channel Grayscale Plates", [
      { label: "Cyan Grayscale", src: `${summary.previews.cyan_gray}?t=${Date.now()}` },
      { label: "Magenta Grayscale", src: `${summary.previews.magenta_gray}?t=${Date.now()}` },
      { label: "Yellow Grayscale", src: `${summary.previews.yellow_gray}?t=${Date.now()}` },
      { label: "Black Grayscale", src: `${summary.previews.black_gray}?t=${Date.now()}` },
    ]);
  }
  ui.historyViewer.classList.remove("hidden");
  ui.historyViewer.innerHTML = `
    <div class="history-viewer-head">
      <div>
        <p class="muted-label">History Preview</p>
        <strong>${item.job_name} - ${item.time}</strong>
      </div>
      <a class="download-link" href="${summary.downloads.zip}" target="_blank" rel="noopener">Download ZIP</a>
    </div>
    ${originalBlock}
    ${outputBlock}
  `;
}

function renderProfiles() {
  const current = ui.profileSelect.value;
  ui.profileSelect.innerHTML = `<option value="">Select profile</option>`;
  (appState.state?.profiles || []).forEach((name) => {
    const option = document.createElement("option");
    option.value = name;
    option.textContent = name;
    ui.profileSelect.appendChild(option);
  });
  ui.profileSelect.value = current;
}

function renderMeta(job) {
  if (!job) {
    ui.activeJobTitle.textContent = "Preview Workspace";
    ui.activeJobMeta.textContent = "Upload TIFF files to begin.";
    ui.fileInfo.innerHTML = "";
    return;
  }
  const page = job.pages[job.current_page];
  ui.activeJobTitle.textContent = job.job_name;
  ui.activeJobMeta.textContent = `${job.file_name} - ${page.shape.join(" x ")} - ${page.dpi_x.toFixed(0)} DPI`;
  ui.fileInfo.innerHTML = [
    `<div class="recipe-item"><span>Photometric</span><strong>${page.photometric}</strong></div>`,
    `<div class="recipe-item"><span>Compression</span><strong>${page.compression}</strong></div>`,
    `<div class="recipe-item"><span>Dtype</span><strong>${page.dtype}</strong></div>`,
    `<div class="recipe-item"><span>Pages</span><strong>${job.pages.length}</strong></div>`,
  ].join("");
}

function renderPreview() {
  const job = activeJob();
  if (!job) {
    ui.previewImage.removeAttribute("src");
    return;
  }
  const settings = currentSettings();
  const previewUrl = `/preview/${job.id}/original?page=${job.current_page}&quality=${encodeURIComponent(settings.preview_quality)}&sharpen=${settings.preview_sharpen ? "1" : "0"}&t=${Date.now()}`;
  ui.previewImage.src = previewUrl;
}

function setDownloadLink(link, href, label) {
  if (href) {
    link.href = href;
    link.textContent = label;
    link.classList.remove("disabled-link");
    return;
  }
  link.removeAttribute("href");
  link.textContent = label;
  link.classList.add("disabled-link");
}

function renderWarnings(result) {
  ui.warningStack.innerHTML = "";
  (result?.stats?.warnings || []).forEach((warning) => {
    const note = document.createElement("div");
    note.className = "warning";
    note.textContent = warning;
    ui.warningStack.appendChild(note);
  });
}

function renderStats(result) {
  ui.statGrid.innerHTML = "";
  if (!result) {
    ui.resultMeta.textContent = "Run a separation to generate outputs.";
    return;
  }
  ui.resultMeta.textContent = `Processed in ${result.stats.processing_seconds.toFixed(3)}s`;
  const stats = [
    ["Matched pixels", result.stats.matched_pixels.toLocaleString()],
    ["Matched area", `${result.stats.matched_percent.toFixed(2)}%`],
    ["Ink coverage", `${result.stats.ink_coverage.toFixed(2)}%`],
    ["Status", result.status],
  ];
  stats.forEach(([label, value]) => {
    const card = document.createElement("div");
    card.className = "stat-card";
    card.innerHTML = `<span class="muted-label">${label}</span><strong>${value}</strong>`;
    ui.statGrid.appendChild(card);
  });
}

function renderResults() {
  const job = activeJob();
  const result = job?.last_result || null;
  const standard = job?.last_standard_result || null;
  const hasJobs = (appState.state?.jobs || []).length > 0;
  const hasHistory = (appState.state?.history || []).length > 0;
  ui.resultsPanel.classList.toggle("hidden", !hasJobs && !hasHistory);
  ui.selectedResultSection.classList.toggle("hidden", !result);
  ui.standardResultSection.classList.toggle("hidden", !standard);
  ui.historySection.classList.toggle("hidden", !hasHistory);
  renderWarnings(result);
  renderStats(result);
  ui.warningStack.classList.toggle("hidden", !result || !(result.stats?.warnings || []).length);
  ui.statGrid.classList.toggle("hidden", !result);
  if (!result && standard) {
    ui.resultMeta.textContent = `4 channel separation ready in ${standard.stats.processing_seconds.toFixed(3)}s`;
  }
  if (!result) {
    [ui.selectedPreview, ui.grayscalePreview].forEach((img) => {
      img.removeAttribute("src");
      img.classList.add("hidden");
    });
    [ui.selectedEmpty, ui.grayscaleEmpty].forEach((node) => node.classList.remove("hidden"));
    setDownloadLink(ui.zipDownload, "", "Download Selected ZIP");
  } else {
    [ui.selectedPreview, ui.grayscalePreview].forEach((img) => img.classList.remove("hidden"));
    [ui.selectedEmpty, ui.grayscaleEmpty].forEach((node) => node.classList.add("hidden"));
    ui.selectedPreview.src = `${result.previews.selected}?t=${Date.now()}`;
    ui.grayscalePreview.src = `${result.previews.grayscale}?t=${Date.now()}`;
    setDownloadLink(ui.zipDownload, result.downloads.zip, "Download Selected ZIP");
  }

  const standardImages = [
    [ui.cyanPreview, ui.cyanEmpty, "cyan_color"],
    [ui.magentaPreview, ui.magentaEmpty, "magenta_color"],
    [ui.yellowPreview, ui.yellowEmpty, "yellow_color"],
    [ui.blackPreview, ui.blackEmpty, "black_color"],
    [ui.cyanGrayPreview, ui.cyanGrayEmpty, "cyan_gray"],
    [ui.magentaGrayPreview, ui.magentaGrayEmpty, "magenta_gray"],
    [ui.yellowGrayPreview, ui.yellowGrayEmpty, "yellow_gray"],
    [ui.blackGrayPreview, ui.blackGrayEmpty, "black_gray"],
  ];
  if (!standard) {
    standardImages.forEach(([img, empty]) => {
      img.removeAttribute("src");
      img.classList.add("hidden");
      empty.classList.remove("hidden");
    });
    setDownloadLink(ui.standardZipDownload, "", "Download 4 Channel ZIP");
    ui.standardMeta.textContent = "Run 4 channel separation to generate CMYK plates.";
    return;
  }

  standardImages.forEach(([img, empty, key]) => {
    img.classList.remove("hidden");
    empty.classList.add("hidden");
    img.src = `${standard.previews[key]}?t=${Date.now()}`;
  });
  setDownloadLink(ui.standardZipDownload, standard.downloads.zip, "Download 4 Channel ZIP");
  ui.standardMeta.textContent = `4 channel separation ready in ${standard.stats.processing_seconds.toFixed(3)}s at ${standard.stats.width.toLocaleString()} x ${standard.stats.height.toLocaleString()} pixels.`;
}

function renderSettings() {
  const settings = appState.state?.settings || {};
  ui.matchModeSelect.value = settings.match_mode || "exact";
  ui.selectionModeSelect.value = settings.selection_mode || "keep";
  ui.spreadInput.value = settings.spread ?? 6;
  ui.compressionSelect.value = settings.compression || "lzw";
  ui.previewQualitySelect.value = settings.preview_quality || "proxy";
  ui.invertGrayInput.checked = !!settings.invert_grayscale;
  ui.sharpenPreviewInput.checked = !!settings.preview_sharpen;
  ui.gammaInput.value = settings.gray_gamma ?? 1.0;
}

function render() {
  renderQueue();
  renderProfiles();
  renderSettings();
  updateCurrentRecipeDisplay();
  const job = activeJob();
  renderPageOptions(job);
  renderMeta(job);
  renderPreview();
  renderLists();
  renderResults();
  const selectedHistory = (appState.state?.history || []).find((item) => item.id === appState.selectedHistoryId) || null;
  renderHistoryViewer(selectedHistory);
}

async function fetchState() {
  appState.state = await api("/api/state");
  render();
}

async function uploadFiles(files) {
  const form = new FormData();
  [...files].forEach((file) => form.append("files", file));
  const response = await fetch("/api/upload", { method: "POST", body: form });
  if (!response.ok) {
    const text = await response.text();
    throw new Error(text || "Upload failed.");
  }
  appState.state = await response.json();
  render();
}

async function activateJob(jobId) {
  appState.state = await api(`/api/job/${jobId}/activate`, { method: "POST", body: JSON.stringify({}) });
  render();
}

async function saveSettings() {
  const job = activeJob();
  if (!job) return;
  appState.state = await api(`/api/job/${job.id}/settings`, {
    method: "POST",
    body: JSON.stringify(currentSettings()),
  });
  render();
}

function applyRecipe(recipe) {
  [ui.cInput.value, ui.mInput.value, ui.yInput.value, ui.kInput.value] = recipe;
  updateCurrentRecipeDisplay();
}

async function addColor() {
  const job = activeJob();
  if (!job) return;
  const [c, m, y, k] = recipeArray();
  appState.state = await api(`/api/job/${job.id}/colors`, {
    method: "POST",
    body: JSON.stringify({ c, m, y, k, name: `Color ${job.selected_colors.length + 1}` }),
  });
  render();
}

async function removeColor(index) {
  const job = activeJob();
  if (!job) return;
  appState.state = await api(`/api/job/${job.id}/colors/${index}`, { method: "DELETE" });
  render();
}

async function processJob() {
  const job = activeJob();
  if (!job) return;
  ui.processButton.disabled = true;
  ui.processButton.textContent = "Running...";
  try {
    appState.state = await api(`/api/job/${job.id}/process`, {
      method: "POST",
      body: JSON.stringify({
        ...currentSettings(),
        page_index: Number(ui.pageSelect.value || 0),
        selected_colors: job.selected_colors,
        manual_recipe: manualRecipe(),
      }),
    });
    render();
  } finally {
    ui.processButton.disabled = false;
    ui.processButton.textContent = "Run Selected Separation";
  }
}

async function processStandardJob() {
  const job = activeJob();
  if (!job) return;
  ui.standardProcessButton.disabled = true;
  ui.standardProcessButton.textContent = "Building 4 Channels...";
  try {
    appState.state = await api(`/api/job/${job.id}/standard`, {
      method: "POST",
      body: JSON.stringify({
        ...currentSettings(),
        page_index: Number(ui.pageSelect.value || 0),
      }),
    });
    render();
  } finally {
    ui.standardProcessButton.disabled = false;
    ui.standardProcessButton.textContent = "Run 4 Channel Separation";
  }
}

async function saveProfile() {
  const name = ui.profileNameInput.value.trim();
  if (!name) return;
  const job = activeJob();
  const payload = {
    settings: currentSettings(),
    selected_colors: job?.selected_colors || [],
    manual_recipe: manualRecipe(),
  };
  appState.state = await api("/api/profile/save", {
    method: "POST",
    body: JSON.stringify({ name, payload }),
  });
  render();
}

async function loadProfile() {
  if (!ui.profileSelect.value) return;
  const result = await api("/api/profile/load", {
    method: "POST",
    body: JSON.stringify({ name: ui.profileSelect.value }),
  });
  const profile = result.profile || {};
  ui.matchModeSelect.value = profile.settings?.match_mode || "exact";
  ui.selectionModeSelect.value = profile.settings?.selection_mode || "keep";
  ui.spreadInput.value = profile.settings?.spread ?? 6;
  ui.compressionSelect.value = profile.settings?.compression || "lzw";
  ui.previewQualitySelect.value = profile.settings?.preview_quality || "proxy";
  ui.invertGrayInput.checked = !!profile.settings?.invert_grayscale;
  ui.sharpenPreviewInput.checked = !!profile.settings?.preview_sharpen;
  ui.gammaInput.value = profile.settings?.gray_gamma ?? 1.0;
  if (profile.manual_recipe) {
    applyRecipe([profile.manual_recipe.c, profile.manual_recipe.m, profile.manual_recipe.y, profile.manual_recipe.k]);
  }
  await saveSettings();
}

function bindEvents() {
  ui.uploadInput.addEventListener("change", (event) => uploadFiles(event.target.files).catch(alert));
  ui.refreshButton.addEventListener("click", () => fetchState().catch(alert));
  [ui.cInput, ui.mInput, ui.yInput, ui.kInput].forEach((input) => input.addEventListener("input", updateCurrentRecipeDisplay));
  ui.addColorButton.addEventListener("click", () => addColor().catch(alert));
  ui.processButton.addEventListener("click", () => processJob().catch(alert));
  ui.standardProcessButton.addEventListener("click", () => processStandardJob().catch(alert));
  ui.clearButton.addEventListener("click", async () => {
    const job = activeJob();
    if (!job) return;
    appState.state = await api(`/api/job/${job.id}/clear`, { method: "POST", body: JSON.stringify({}) });
    render();
  });
  ui.duplicateButton.addEventListener("click", async () => {
    const job = activeJob();
    if (!job) return;
    appState.state = await api(`/api/job/${job.id}/duplicate-last`, { method: "POST", body: JSON.stringify({}) });
    render();
  });
  ui.pageSelect.addEventListener("change", async () => {
    const job = activeJob();
    if (!job) return;
    appState.state = await api(`/api/job/${job.id}/page`, {
      method: "POST",
      body: JSON.stringify({ page_index: Number(ui.pageSelect.value) }),
    });
    render();
  });
  [
    ui.matchModeSelect,
    ui.selectionModeSelect,
    ui.spreadInput,
    ui.compressionSelect,
    ui.previewQualitySelect,
    ui.invertGrayInput,
    ui.sharpenPreviewInput,
    ui.gammaInput,
  ].forEach((element) => element.addEventListener("change", () => saveSettings().catch(console.error)));
  ui.saveProfileButton.addEventListener("click", () => saveProfile().catch(alert));
  ui.loadProfileButton.addEventListener("click", () => loadProfile().catch(alert));
  ui.clearHistoryButton.addEventListener("click", async () => {
    appState.state = await api("/api/history/clear", { method: "POST", body: JSON.stringify({}) });
    appState.selectedHistoryId = null;
    render();
  });
}

bindEvents();
fetchState().catch(console.error);
