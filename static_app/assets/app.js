"use strict";

const pageEl = document.getElementById("page");
const navLinks = Array.from(document.querySelectorAll(".nav a"));

const featureLabels = {
  acc: "Accuracy (%)",
  z_acc: "Accuracy (z-score)",
  consistency: "WPM consistency (%)",
  is_pb: "Personal best",
  raw_wpm: "Raw WPM",
  test_duration: "Test duration (s)",
  time_of_day_sec: "Time of day (s)",
  trial_type_num: "Trial type completed",
  wpm: "Words per minute",
  z_wpm: "WPM (z-score)",
  log_norm_wpm: "WPM (Log-fit norm.)",
  datetime: "Date",
  trial_type_id: "Trial type",
  trial_num: "Trial number",
};

const scatterColumns = [
  "acc",
  "z_acc",
  "consistency",
  "is_pb",
  "raw_wpm",
  "test_duration",
  "time_of_day_sec",
  "trial_type_num",
  "wpm",
  "z_wpm",
];

const state = {
  cache: new Map(),
  manifest: null,
  uploadRows: null,
  useUploaded: false,
  uploadMessage: "",
  home: {
    timeFeature: "wpm",
    timeTrialType: "",
    xFeature: "acc",
    yFeature: "wpm",
    pairTrialType: "",
  },
  simIndex: 0,
  table: {
    search: "",
    sortKey: "datetime",
    sortDir: "desc",
    page: 0,
    pageSize: 50,
  },
};

async function fetchJson(path) {
  if (state.cache.has(path)) {
    return state.cache.get(path);
  }
  const response = await fetch(path);
  if (!response.ok) {
    throw new Error(`Could not load ${path}`);
  }
  const payload = await response.json();
  state.cache.set(path, payload);
  return payload;
}

function setActiveNav(page) {
  navLinks.forEach((link) => {
    link.classList.toggle("active", link.dataset.page === page);
  });
}

function getPageFromHash() {
  const hash = window.location.hash.replace("#", "");
  return ["home", "trial", "simulation", "model"].includes(hash) ? hash : "home";
}

async function render() {
  const page = getPageFromHash();
  setActiveNav(page);
  try {
    if (!state.manifest) {
      state.manifest = await fetchJson("data/manifest.json");
    }
    if (page === "trial") {
      await renderTrial();
    } else if (page === "simulation") {
      await renderSimulation();
    } else if (page === "model") {
      await renderModel();
    } else {
      await renderHome();
    }
  } catch (error) {
    pageEl.innerHTML = `
      <section class="notice error">
        Static app data could not be loaded. Serve the generated dist directory with a local web server rather than opening index.html directly.
        <br><br><code>${escapeHtml(error.message)}</code>
      </section>`;
  }
}

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function fmt(value, digits = 1) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) {
    return "n/a";
  }
  return Number(value).toLocaleString(undefined, {
    maximumFractionDigits: digits,
  });
}

function formatDate(value) {
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return "n/a";
  return date.toLocaleDateString(undefined, { year: "numeric", month: "short", day: "numeric" });
}

function panel(title, body, intro = "") {
  return `
    <section class="panel">
      <div>
        <h2>${title}</h2>
        ${intro ? `<p>${intro}</p>` : ""}
      </div>
      ${body}
    </section>`;
}

function warningNotice(text) {
  return `<section class="notice warning">${escapeHtml(text)}</section>`;
}

function currentRows(defaultRows) {
  if (state.useUploaded && state.uploadRows) {
    return state.uploadRows;
  }
  return defaultRows || [];
}

function featureOptionsFromRows(rows, fallbackOptions = []) {
  if (fallbackOptions.length > 0) return fallbackOptions;
  if (!rows.length) return [];
  return scatterColumns
    .filter((column) => rows.some((row) => row[column] !== null && row[column] !== undefined))
    .map((column) => ({ column, label: featureLabels[column] || column }));
}

function trialOptionsFromRows(rows, fallbackOptions = []) {
  if (fallbackOptions.length > 0) return fallbackOptions;
  const values = Array.from(new Set(rows.map((row) => row.trial_type_id).filter(Boolean)));
  values.sort((a, b) => Number(a) - Number(b));
  return values.map((value) => ({ value, label: `Trial type ${value}` }));
}

function selectHtml(id, label, options, value, includeAll = false) {
  const optionHtml = [
    includeAll ? `<option value="">All trial types</option>` : "",
    ...options.map((option) => {
      const optionValue = option.value ?? option.column;
      const selected = String(optionValue) === String(value) ? "selected" : "";
      return `<option value="${escapeHtml(optionValue)}" ${selected}>${escapeHtml(option.label)}</option>`;
    }),
  ].join("");
  return `
    <div class="control">
      <label for="${id}">${label}</label>
      <select id="${id}" class="select">${optionHtml}</select>
    </div>`;
}

async function renderHome() {
  const data = await fetchJson("data/home.json");
  const rows = currentRows(data.rows);
  const featureOptions = featureOptionsFromRows(rows, data.feature_options || []);
  const trialOptions = trialOptionsFromRows(rows, data.trial_type_options || []);
  const summary = summarizeRows(rows, data.summary);
  const showDataWarning = !data.available && !state.uploadRows;

  pageEl.innerHTML = `
    <section class="hero">
      <h1>monkeytype.com Data Analysis</h1>
      <p class="lead">
        Analyze Monkeytype typing history with the same pipeline as the Streamlit app,
        delivered here as static files. Expensive processing is done at build time;
        lightweight chart controls run in the browser.
      </p>
    </section>

    ${showDataWarning ? warningNotice(data.warning || "Default typing data is unavailable in this build.") : ""}

    <section class="stats-grid">
      ${statCard("Trials", fmt(summary.trial_count, 0))}
      ${statCard("Average WPM", fmt(summary.avg_wpm, 1))}
      ${statCard("Average accuracy", fmt(summary.avg_acc, 2))}
      ${statCard("Trial types", fmt(summary.trial_type_count, 0))}
    </section>

    ${uploadPanel()}

    ${panel(
      "Feature correlations over time",
      `
        <div class="control-grid">
          ${selectHtml("timeFeature", "Select a feature to plot", featureOptions, state.home.timeFeature)}
          ${selectHtml("timeTrialType", "Select a trial-type subset", trialOptions, state.home.timeTrialType, true)}
          ${useUploadedControl()}
        </div>
        <canvas id="timeChart" class="chart" aria-label="Feature over time scatter plot"></canvas>
      `,
      "Select a feature and optionally filter to one trial type."
    )}

    ${panel(
      "Cross-feature correlations",
      `
        <div class="control-grid">
          ${selectHtml("xFeature", "X-axis feature", featureOptions, state.home.xFeature)}
          ${selectHtml("yFeature", "Y-axis feature", featureOptions, state.home.yFeature)}
          ${selectHtml("pairTrialType", "Trial-type subset", trialOptions, state.home.pairTrialType, true)}
        </div>
        <canvas id="pairChart" class="chart" aria-label="Cross feature scatter plot"></canvas>
      `,
      "Compare two selected features and view the linear trend."
    )}

    ${tablePanel(rows)}

    <section class="panel">
      <h2>Notebook reference</h2>
      <p>Additional exploratory analyses, including day-of-week, autocorrelation, and trial-feature plots, remain in the source notebooks.</p>
      <div class="link-list">
        <a href="https://github.com/jbreffle/monkeytype-analysis/blob/main/notebooks/1_explore.ipynb">notebooks/1_explore.ipynb</a>
      </div>
    </section>
  `;

  attachHomeHandlers(data);
  if (rows.length) {
    drawScatter(
      document.getElementById("timeChart"),
      filterTrialType(rows, state.home.timeTrialType),
      "datetime",
      state.home.timeFeature,
      {
        xLabel: featureLabels.datetime,
        yLabel: featureLabels[state.home.timeFeature] || state.home.timeFeature,
        regression: true,
        colorByTrial: false,
      }
    );
    drawScatter(
      document.getElementById("pairChart"),
      filterTrialType(rows, state.home.pairTrialType),
      state.home.xFeature,
      state.home.yFeature,
      {
        xLabel: featureLabels[state.home.xFeature] || state.home.xFeature,
        yLabel: featureLabels[state.home.yFeature] || state.home.yFeature,
        regression: true,
        colorByTrial: false,
      }
    );
  }
}

function statCard(label, value) {
  return `<div class="stat"><strong>${escapeHtml(value)}</strong><span>${escapeHtml(label)}</span></div>`;
}

function uploadPanel() {
  return panel(
    "User file upload",
    `
      <p>Upload a Monkeytype <code>results.csv</code> export to process it entirely in the browser. The file stays local to this page.</p>
      <div class="control-grid">
        <div class="control">
          <label for="uploadFile">Monkeytype export</label>
          <input id="uploadFile" class="input" type="file" accept=".csv,.psv,text/csv,text/plain" />
        </div>
        ${useUploadedControl()}
        <div class="control">
          <label>Upload status</label>
          <div class="notice ${state.uploadRows ? "" : "warning"}">${escapeHtml(state.uploadMessage || "No uploaded file processed yet.")}</div>
        </div>
      </div>
    `,
    "The Streamlit app uses pandas for uploads; this static replica implements the same core column processing in JavaScript for interactive charts."
  );
}

function useUploadedControl() {
  const disabled = state.uploadRows ? "" : "disabled";
  const checked = state.useUploaded && state.uploadRows ? "checked" : "";
  return `
    <label class="checkbox-row">
      <input id="useUploaded" type="checkbox" ${checked} ${disabled} />
      Use uploaded data
    </label>`;
}

function attachHomeHandlers(data) {
  const controls = [
    ["timeFeature", "timeFeature"],
    ["timeTrialType", "timeTrialType"],
    ["xFeature", "xFeature"],
    ["yFeature", "yFeature"],
    ["pairTrialType", "pairTrialType"],
  ];
  controls.forEach(([id, key]) => {
    const element = document.getElementById(id);
    if (element) {
      element.addEventListener("change", () => {
        state.home[key] = element.value;
        renderHome();
      });
    }
  });

  document.querySelectorAll("#useUploaded").forEach((element) => {
    element.addEventListener("change", () => {
      state.useUploaded = element.checked;
      state.table.page = 0;
      renderHome();
    });
  });

  const upload = document.getElementById("uploadFile");
  if (upload) {
    upload.addEventListener("change", async () => {
      const file = upload.files && upload.files[0];
      if (!file) return;
      try {
        const text = await file.text();
        state.uploadRows = processUploadedText(text);
        state.useUploaded = true;
        state.uploadMessage = `${state.uploadRows.length.toLocaleString()} rows processed from ${file.name}.`;
        state.table.page = 0;
      } catch (error) {
        state.uploadRows = null;
        state.useUploaded = false;
        state.uploadMessage = error.message;
      }
      await renderHome(data);
    });
  }

  attachTableHandlers();
}

async function renderTrial() {
  const data = await fetchJson("data/trial_difficulty.json");
  const rows = currentRows(data.rows);
  const usingUpload = state.useUploaded && state.uploadRows;
  const hist = usingUpload ? histogramFromRows(rows) : data.histogram || [];
  const logCurves = usingUpload
    ? { top_one: logCurvesFromRows(rows, 1), top_four: logCurvesFromRows(rows, 4) }
    : data.log_curves || { top_one: [], top_four: [] };

  pageEl.innerHTML = `
    <section class="hero">
      <h1>Trial difficulty</h1>
      <p class="lead">
        Monkeytype trials vary by duration, language, vocabulary, punctuation,
        numbers, special characters, and mode. This page compares performance
        across trial types.
      </p>
    </section>

    ${!data.available && !state.uploadRows ? warningNotice(data.warning || "Default typing data is unavailable in this build.") : ""}

    <section class="image-row">
      <figure class="image-card">
        <img src="assets/images/english_600x150.png" alt="Example English Monkeytype trial" />
        <figcaption class="caption">Example of an English trial from monkeytype.com</figcaption>
      </figure>
      <figure class="image-card">
        <img src="assets/images/ascii_600x150.png" alt="Example ASCII Monkeytype trial" />
        <figcaption class="caption">Example of an ASCII trial from monkeytype.com</figcaption>
      </figure>
    </section>

    ${panel(
      "Comparing trials of different difficulty",
      `
        <div class="control-grid">${useUploadedControl()}</div>
        <canvas id="trialTimeRaw" class="chart" aria-label="WPM over time"></canvas>
        <canvas id="trialHist" class="chart short" aria-label="Trial type histogram"></canvas>
        <canvas id="trialTimeColor" class="chart" aria-label="WPM over time by trial type"></canvas>
      `,
      "WPM jumps between average values when a user completes trials of different difficulty."
    )}

    ${panel(
      "Z-scoring by trial type",
      `<canvas id="zScoreChart" class="chart" aria-label="Z-scored WPM over time"></canvas>`,
      "Z-scoring shifts each trial-type distribution to mean 0 and scales by standard deviation 1."
    )}

    ${panel(
      "Logarithmic learning curves",
      `
        <div class="two-col">
          <canvas id="logOne" class="chart" aria-label="Log fit for top trial type"></canvas>
          <canvas id="logFour" class="chart" aria-label="Log fit for top four trial types"></canvas>
        </div>
        <canvas id="logNormByTrial" class="chart" aria-label="Log normalized WPM by trial type completed"></canvas>
        <canvas id="logNormByTime" class="chart" aria-label="Log normalized WPM over time"></canvas>
      `,
      "The fitted curve captures trial-specific learning while preserving an interpretable residual-like normalized score."
    )}

    <section class="panel">
      <h2>Notebook reference</h2>
      <div class="link-list">
        <a href="https://github.com/jbreffle/monkeytype-analysis/blob/main/notebooks/2a_z_scoring.ipynb">notebooks/2a_z_scoring.ipynb</a>
        <a href="https://github.com/jbreffle/monkeytype-analysis/blob/main/notebooks/2b_learning_curve.ipynb">notebooks/2b_learning_curve.ipynb</a>
      </div>
    </section>
  `;

  document.querySelectorAll("#useUploaded").forEach((element) => {
    element.addEventListener("change", () => {
      state.useUploaded = element.checked;
      renderTrial();
    });
  });

  if (!rows.length) return;
  drawScatter(document.getElementById("trialTimeRaw"), rows, "datetime", "wpm", {
    xLabel: "Date",
    yLabel: "Words per minute",
    colorByTrial: false,
  });
  drawHistogram(document.getElementById("trialHist"), hist, {
    xLabel: "Trial type ID (sorted)",
    yLabel: "Trials completed",
  });
  drawScatter(document.getElementById("trialTimeColor"), rows, "datetime", "wpm", {
    xLabel: "Date",
    yLabel: "Words per minute",
    colorByTrial: true,
    colorCount: 5,
  });
  drawScatter(document.getElementById("zScoreChart"), rows, "datetime", "z_wpm", {
    xLabel: "Date",
    yLabel: "WPM (z-score)",
    colorByTrial: true,
    colorCount: 5,
  });
  drawScatterWithCurves(document.getElementById("logOne"), rows, logCurves.top_one, {
    xKey: "trial_type_num",
    yKey: "wpm",
    xLabel: "Trial type completed",
    yLabel: "Words per minute",
    trialLimit: 1,
  });
  drawScatterWithCurves(document.getElementById("logFour"), rows, logCurves.top_four, {
    xKey: "trial_type_num",
    yKey: "wpm",
    xLabel: "Trial type completed",
    yLabel: "Words per minute",
    trialLimit: 4,
  });
  drawScatter(document.getElementById("logNormByTrial"), rows, "trial_type_num", "log_norm_wpm", {
    xLabel: "Trial type completed",
    yLabel: "WPM (Log-fit norm.)",
    colorByTrial: true,
    colorCount: 5,
  });
  drawScatter(document.getElementById("logNormByTime"), rows, "datetime", "log_norm_wpm", {
    xLabel: "Date",
    yLabel: "WPM (Log-fit norm.)",
    colorByTrial: true,
    colorCount: 5,
  });
}

async function renderSimulation() {
  const data = await fetchJson("data/simulations.json");
  const run = data.runs[state.simIndex % data.runs.length];
  pageEl.innerHTML = `
    <section class="hero">
      <h1>Simulated typing</h1>
      <p class="lead">
        Simulations test whether random mistakes and fixed correction time can explain
        the observed relationship between accuracy and WPM.
      </p>
    </section>

    ${panel(
      "Simulation controls",
      `
        <p>Showing precomputed seed ${run.seed}. The Streamlit rerun button is represented by cycling through seeded build-time variants.</p>
        <button id="rerunSim" class="button primary" type="button">Re-run simulations</button>
      `
    )}

    <section class="two-col">
      ${panel(
        "Simulated typing: random mistake draws",
        `<canvas id="simpleSim" class="chart" aria-label="Simple simulation scatter plot"></canvas>`,
        "Each trial draws random mistakes and random correction durations."
      )}
      ${panel(
        "Simulated typing: Poisson process",
        `<canvas id="poissonSim" class="chart" aria-label="Poisson simulation scatter plot"></canvas>`,
        "Typing and mistakes are simulated over time as a Poisson process."
      )}
    </section>

    <section class="panel">
      <h2>Notebook reference</h2>
      <div class="link-list">
        <a href="https://github.com/jbreffle/monkeytype-analysis/blob/main/notebooks/3a_sim_simple.ipynb">notebooks/3a_sim_simple.ipynb</a>
        <a href="https://github.com/jbreffle/monkeytype-analysis/blob/main/notebooks/3b_sim_poisson.ipynb">notebooks/3b_sim_poisson.ipynb</a>
      </div>
    </section>
  `;

  document.getElementById("rerunSim").addEventListener("click", () => {
    state.simIndex = (state.simIndex + 1) % data.runs.length;
    renderSimulation();
  });
  drawSimulation(document.getElementById("simpleSim"), run.simple, data.avg_wpm, data.avg_acc);
  drawSimulation(document.getElementById("poissonSim"), run.poisson, data.avg_wpm, data.avg_acc);
}

async function renderModel() {
  const data = await fetchJson("data/model.json");
  pageEl.innerHTML = `
    <section class="hero">
      <h1>Predicting performance</h1>
      <p class="lead">
        A feedforward neural network predicts typing speed from trial type, accuracy,
        total experience, and trial-type-specific experience.
      </p>
    </section>

    ${!data.available ? warningNotice(data.warning || "Model data is unavailable in this build.") : ""}

    ${panel(
      "Neural network model",
      `
        <canvas id="lossChart" class="chart" aria-label="Train and test loss over epochs"></canvas>
        <canvas id="predChart" class="chart" aria-label="Predicted WPM versus actual WPM"></canvas>
        <canvas id="featureChart" class="chart" aria-label="Actual and predicted values by feature"></canvas>
      `,
      "The static build runs Torch inference once and exports the prediction payload for browser rendering."
    )}

    <section class="panel">
      <h2>Notebook reference</h2>
      <div class="link-list">
        <a href="https://github.com/jbreffle/monkeytype-analysis/blob/main/notebooks/4_nn_predict.ipynb">notebooks/4_nn_predict.ipynb</a>
        <a href="https://github.com/jbreffle/monkeytype-analysis/blob/main/notebooks/5_nn_hyperopt.ipynb">notebooks/5_nn_hyperopt.ipynb</a>
      </div>
    </section>
  `;

  if (!data.available) return;
  drawLoss(document.getElementById("lossChart"), data.train_loss, data.test_loss);
  drawPredictionScatter(document.getElementById("predChart"), data.predictions);
  drawModelFeatureScatter(document.getElementById("featureChart"), data.predictions);
}

function summarizeRows(rows, fallback) {
  if (!rows.length && fallback) return fallback;
  if (!rows.length) {
    return {
      trial_count: 0,
      avg_wpm: null,
      avg_acc: null,
      trial_type_count: 0,
    };
  }
  const avg = (key) => mean(rows.map((row) => Number(row[key])).filter(Number.isFinite));
  return {
    trial_count: rows.length,
    avg_wpm: avg("wpm"),
    avg_acc: avg("acc"),
    trial_type_count: new Set(rows.map((row) => row.trial_type_id)).size,
  };
}

function filterTrialType(rows, trialType) {
  if (!trialType) return rows;
  return rows.filter((row) => String(row.trial_type_id) === String(trialType));
}

function tablePanel(rows) {
  if (!rows.length) {
    return panel("Processed data table", `<div class="empty">No processed rows available.</div>`);
  }
  const columns = [
    "datetime",
    "wpm",
    "acc",
    "raw_wpm",
    "consistency",
    "test_duration",
    "trial_type_id",
    "trial_type_num",
    "is_pb",
  ];
  const search = state.table.search.trim().toLowerCase();
  let filtered = rows.filter((row) => {
    if (!search) return true;
    return columns.some((column) => String(row[column] ?? "").toLowerCase().includes(search));
  });
  filtered = filtered.slice().sort((a, b) => compareValues(a[state.table.sortKey], b[state.table.sortKey]));
  if (state.table.sortDir === "desc") filtered.reverse();
  const pageCount = Math.max(1, Math.ceil(filtered.length / state.table.pageSize));
  state.table.page = Math.min(state.table.page, pageCount - 1);
  const start = state.table.page * state.table.pageSize;
  const pageRows = filtered.slice(start, start + state.table.pageSize);
  const headers = columns
    .map((column) => `<th data-sort="${column}">${escapeHtml(featureLabels[column] || column)}</th>`)
    .join("");
  const body = pageRows
    .map(
      (row) => `
        <tr>
          ${columns
            .map((column) => `<td>${escapeHtml(formatTableValue(row[column], column))}</td>`)
            .join("")}
        </tr>`
    )
    .join("");
  return panel(
    "Processed data table",
    `
      <div class="table-tools">
        <input id="tableSearch" class="input" type="search" placeholder="Search table" value="${escapeHtml(state.table.search)}" />
        <div>
          <button id="prevPage" class="button" type="button">Prev</button>
          <button id="nextPage" class="button" type="button">Next</button>
        </div>
        <p>${filtered.length.toLocaleString()} matching rows, page ${state.table.page + 1} of ${pageCount}</p>
      </div>
      <div class="table-wrap">
        <table>
          <thead><tr>${headers}</tr></thead>
          <tbody>${body}</tbody>
        </table>
      </div>
    `
  );
}

function attachTableHandlers() {
  const search = document.getElementById("tableSearch");
  if (search) {
    search.addEventListener("input", () => {
      state.table.search = search.value;
      state.table.page = 0;
      renderHome();
    });
  }
  document.querySelectorAll("th[data-sort]").forEach((header) => {
    header.addEventListener("click", () => {
      const key = header.dataset.sort;
      if (state.table.sortKey === key) {
        state.table.sortDir = state.table.sortDir === "asc" ? "desc" : "asc";
      } else {
        state.table.sortKey = key;
        state.table.sortDir = "asc";
      }
      renderHome();
    });
  });
  const prev = document.getElementById("prevPage");
  const next = document.getElementById("nextPage");
  if (prev) {
    prev.addEventListener("click", () => {
      state.table.page = Math.max(0, state.table.page - 1);
      renderHome();
    });
  }
  if (next) {
    next.addEventListener("click", () => {
      state.table.page += 1;
      renderHome();
    });
  }
}

function compareValues(a, b) {
  const aNum = Number(a);
  const bNum = Number(b);
  if (Number.isFinite(aNum) && Number.isFinite(bNum)) return aNum - bNum;
  return String(a ?? "").localeCompare(String(b ?? ""));
}

function formatTableValue(value, column) {
  if (column === "datetime") return formatDate(value);
  if (value === null || value === undefined) return "";
  if (typeof value === "number") return fmt(value, column === "trial_type_id" ? 0 : 2);
  return value;
}

function processUploadedText(text) {
  const delimiter = detectDelimiter(text);
  const lines = text.split(/\r?\n/).filter((line) => line.trim().length > 0);
  if (lines.length < 2) throw new Error("Upload does not contain data rows.");
  const headers = lines[0].split(delimiter).map((value) => value.trim());
  const rawRows = lines.slice(1).map((line) => {
    const values = line.split(delimiter);
    const row = {};
    headers.forEach((header, index) => {
      row[header] = values[index] ?? "";
    });
    return row;
  });
  const rows = rawRows.map(renameAndCoerceRow).filter((row) => Number.isFinite(row.timestamp));
  if (!rows.length) throw new Error("No rows with valid Monkeytype timestamps were found.");
  rows.sort((a, b) => a.timestamp - b.timestamp);
  addProcessedColumns(rows);
  return rows;
}

function detectDelimiter(text) {
  const firstLine = text.split(/\r?\n/, 1)[0] || "";
  return (firstLine.match(/\|/g) || []).length >= (firstLine.match(/,/g) || []).length ? "|" : ",";
}

function renameAndCoerceRow(row) {
  const rename = {
    _id: "id",
    isPb: "is_pb",
    rawWpm: "raw_wpm",
    testDuration: "test_duration",
    lazyMode: "lazy_mode",
    blindMode: "blind_mode",
    bailedOut: "bailed_out",
    charStats: "char_stats",
    quoteLength: "quote_length",
    restartCount: "restart_count",
    afkDuration: "afk_duration",
    incompleteTestSeconds: "incomplete_test_seconds",
  };
  const normalized = {};
  Object.entries(row).forEach(([key, value]) => {
    normalized[rename[key] || key] = value;
  });
  [
    "timestamp",
    "wpm",
    "acc",
    "raw_wpm",
    "consistency",
    "test_duration",
    "punctuation",
    "numbers",
  ].forEach((key) => {
    normalized[key] = toNumber(normalized[key]);
  });
  normalized.test_duration = Math.round(normalized.test_duration || 0);
  normalized.is_pb = normalized.is_pb === 1 || normalized.is_pb === "1" ? 1 : 0;
  return normalized;
}

function addProcessedColumns(rows) {
  const groupIdMap = new Map();
  rows.forEach((row, index) => {
    const date = new Date(row.timestamp);
    row.datetime = date.toISOString().slice(0, 19);
    row.time_of_day_sec = date.getUTCHours() * 3600 + date.getUTCMinutes() * 60 + date.getUTCSeconds();
    row.trial_num = index + 1;
    row.time_diff_sec = index === 0 ? null : (row.timestamp - rows[index - 1].timestamp) / 1000;
    row.new_sesh_ind = row.time_diff_sec === null || row.time_diff_sec >= 600 ? 1 : 0;
    row.combined_key = [
      row.mode,
      row.mode2,
      row.punctuation,
      row.numbers,
      row.language,
      row.funbox,
      row.difficulty,
      row.lazy_mode,
      row.blind_mode,
    ].join("|");
    if (!groupIdMap.has(row.combined_key)) {
      groupIdMap.set(row.combined_key, groupIdMap.size);
    }
    row.combined_id = groupIdMap.get(row.combined_key);
  });
  const groups = groupRows(rows, "combined_id");
  const ranks = Array.from(groups.entries())
    .map(([combinedId, group]) => ({ combinedId, count: group.length }))
    .sort((a, b) => b.count - a.count)
    .reduce((acc, item, index) => {
      acc.set(item.combinedId, index + 1);
      return acc;
    }, new Map());
  groups.forEach((group, combinedId) => {
    const wpmStats = stats(group.map((row) => row.wpm));
    const accStats = stats(group.map((row) => row.acc));
    group.forEach((row, index) => {
      row.trial_type_id = ranks.get(combinedId);
      row.trial_type_num = index + 1;
      row.z_wpm = wpmStats.std ? (row.wpm - wpmStats.mean) / wpmStats.std : 0;
      row.z_acc = accStats.std ? (row.acc - accStats.mean) / accStats.std : 0;
    });
  });
  addLogNormRows(rows);
}

function addLogNormRows(rows) {
  const groups = groupRows(rows, "combined_id");
  groups.forEach((group) => {
    if (group.length <= 10) {
      group.forEach((row) => {
        row.log_norm_wpm = null;
      });
      return;
    }
    let best = { sse: Infinity, y0: 0, alpha: 0.5, residualStd: 1 };
    for (let alpha = 0.05; alpha <= 1.5; alpha += 0.025) {
      const offsets = group.map((row, index) => row.wpm - Math.pow(index + 1, alpha));
      const y0 = mean(offsets);
      const residuals = group.map((row, index) => row.wpm - (y0 + Math.pow(index + 1, alpha)));
      const sse = residuals.reduce((sum, value) => sum + value * value, 0);
      if (sse < best.sse) {
        best = { sse, y0, alpha, residualStd: stats(residuals).std || 1 };
      }
    }
    group.forEach((row) => {
      row.log_norm_wpm = (row.wpm - best.y0) / best.residualStd;
    });
  });
}

function histogramFromRows(rows) {
  return Array.from(groupRows(rows, "trial_type_id").entries())
    .map(([trialTypeId, group]) => ({ trial_type_id: Number(trialTypeId), count: group.length }))
    .sort((a, b) => a.trial_type_id - b.trial_type_id);
}

function logCurvesFromRows(rows, limit) {
  const groups = groupRows(rows, "trial_type_id");
  const curves = [];
  for (let trialTypeId = 1; trialTypeId <= limit; trialTypeId += 1) {
    const group = groups.get(trialTypeId);
    if (!group || group.length <= 10) continue;
    let best = { sse: Infinity, y0: 0, alpha: 0.5 };
    for (let alpha = 0.05; alpha <= 1.5; alpha += 0.025) {
      const offsets = group.map((row, index) => row.wpm - Math.pow(index + 1, alpha));
      const y0 = mean(offsets);
      const sse = group.reduce((sum, row, index) => {
        const residual = row.wpm - (y0 + Math.pow(index + 1, alpha));
        return sum + residual * residual;
      }, 0);
      if (sse < best.sse) best = { sse, y0, alpha };
    }
    curves.push({
      trial_type_id: trialTypeId,
      points: group.map((row, index) => ({
        x: index + 1,
        y: best.y0 + Math.pow(index + 1, best.alpha),
      })),
    });
  }
  return curves;
}

function toNumber(value) {
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : null;
}

function groupRows(rows, key) {
  const groups = new Map();
  rows.forEach((row) => {
    const value = row[key];
    if (!groups.has(value)) groups.set(value, []);
    groups.get(value).push(row);
  });
  return groups;
}

function mean(values) {
  const valid = values.filter(Number.isFinite);
  if (!valid.length) return null;
  return valid.reduce((sum, value) => sum + value, 0) / valid.length;
}

function stats(values) {
  const valid = values.filter(Number.isFinite);
  const avg = mean(valid);
  if (avg === null) return { mean: null, std: null };
  const variance = valid.reduce((sum, value) => sum + Math.pow(value - avg, 2), 0) / Math.max(1, valid.length - 1);
  return { mean: avg, std: Math.sqrt(variance) };
}

function setupCanvas(canvas) {
  const dpr = window.devicePixelRatio || 1;
  const rect = canvas.getBoundingClientRect();
  canvas.width = Math.max(320, Math.floor(rect.width * dpr));
  canvas.height = Math.max(240, Math.floor(rect.height * dpr));
  const ctx = canvas.getContext("2d");
  ctx.scale(dpr, dpr);
  return { ctx, width: canvas.width / dpr, height: canvas.height / dpr };
}

function chartBounds(width, height) {
  return { left: 58, right: width - 24, top: 24, bottom: height - 48 };
}

function getValue(row, key) {
  if (key === "datetime") {
    return Date.parse(row.datetime);
  }
  return Number(row[key]);
}

function extent(values) {
  const valid = values.filter(Number.isFinite);
  if (!valid.length) return [0, 1];
  let min = Math.min(...valid);
  let max = Math.max(...valid);
  if (min === max) {
    min -= 1;
    max += 1;
  }
  const pad = (max - min) * 0.05;
  return [min - pad, max + pad];
}

function scale(value, domainMin, domainMax, rangeMin, rangeMax) {
  return rangeMin + ((value - domainMin) / (domainMax - domainMin)) * (rangeMax - rangeMin);
}

function drawBase(ctx, width, height, xLabel, yLabel, xDomain, yDomain, xIsDate = false) {
  const bounds = chartBounds(width, height);
  ctx.clearRect(0, 0, width, height);
  ctx.fillStyle = "#12151b";
  ctx.fillRect(0, 0, width, height);
  ctx.strokeStyle = "#303643";
  ctx.lineWidth = 1;
  ctx.strokeRect(bounds.left, bounds.top, bounds.right - bounds.left, bounds.bottom - bounds.top);
  ctx.font = "12px Inter, system-ui, sans-serif";
  ctx.fillStyle = "#a8afbd";
  ctx.textAlign = "center";
  ctx.fillText(xLabel, (bounds.left + bounds.right) / 2, height - 14);
  ctx.save();
  ctx.translate(15, (bounds.top + bounds.bottom) / 2);
  ctx.rotate(-Math.PI / 2);
  ctx.fillText(yLabel, 0, 0);
  ctx.restore();
  ctx.strokeStyle = "rgba(168, 175, 189, 0.16)";
  ctx.fillStyle = "#767f90";
  ctx.textAlign = "center";
  for (let i = 0; i <= 4; i += 1) {
    const x = scale(i, 0, 4, bounds.left, bounds.right);
    const y = scale(i, 0, 4, bounds.bottom, bounds.top);
    ctx.beginPath();
    ctx.moveTo(x, bounds.top);
    ctx.lineTo(x, bounds.bottom);
    ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(bounds.left, y);
    ctx.lineTo(bounds.right, y);
    ctx.stroke();
    const xValue = scale(i, 0, 4, xDomain[0], xDomain[1]);
    const yValue = scale(i, 0, 4, yDomain[0], yDomain[1]);
    ctx.fillText(xIsDate ? formatDate(xValue) : shortNumber(xValue), x, bounds.bottom + 18);
    ctx.textAlign = "right";
    ctx.fillText(shortNumber(yValue), bounds.left - 8, y + 4);
    ctx.textAlign = "center";
  }
  return bounds;
}

function shortNumber(value) {
  const abs = Math.abs(value);
  if (abs >= 1000000) return `${(value / 1000000).toFixed(1)}m`;
  if (abs >= 1000) return `${(value / 1000).toFixed(1)}k`;
  if (abs >= 100) return value.toFixed(0);
  if (abs >= 10) return value.toFixed(1);
  return value.toFixed(2);
}

function pointColor(row, options = {}) {
  if (!options.colorByTrial) return "#7dd3fc";
  const palette = ["#7dd3fc", "#6ee7b7", "#fbbf24", "#f472b6", "#a78bfa"];
  const trialType = Number(row.trial_type_id);
  if (trialType >= 1 && trialType < (options.colorCount || 5)) {
    return palette[(trialType - 1) % palette.length];
  }
  return "#6b7280";
}

function drawScatter(canvas, rows, xKey, yKey, options = {}) {
  if (!canvas) return;
  const { ctx, width, height } = setupCanvas(canvas);
  const points = rows
    .map((row) => ({ row, x: getValue(row, xKey), y: getValue(row, yKey) }))
    .filter((point) => Number.isFinite(point.x) && Number.isFinite(point.y));
  const xDomain = extent(points.map((point) => point.x));
  const yDomain = extent(points.map((point) => point.y));
  const bounds = drawBase(
    ctx,
    width,
    height,
    options.xLabel || xKey,
    options.yLabel || yKey,
    xDomain,
    yDomain,
    xKey === "datetime"
  );
  points.forEach((point) => {
    const x = scale(point.x, xDomain[0], xDomain[1], bounds.left, bounds.right);
    const y = scale(point.y, yDomain[0], yDomain[1], bounds.bottom, bounds.top);
    ctx.fillStyle = pointColor(point.row, options);
    ctx.globalAlpha = 0.68;
    ctx.beginPath();
    ctx.arc(x, y, 2.4, 0, Math.PI * 2);
    ctx.fill();
  });
  ctx.globalAlpha = 1;
  if (options.regression && points.length > 2) {
    const regression = linearRegression(points.map((point) => point.x), points.map((point) => point.y));
    const x1 = xDomain[0];
    const x2 = xDomain[1];
    const y1 = regression.slope * x1 + regression.intercept;
    const y2 = regression.slope * x2 + regression.intercept;
    ctx.strokeStyle = "#fb7185";
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(scale(x1, xDomain[0], xDomain[1], bounds.left, bounds.right), scale(y1, yDomain[0], yDomain[1], bounds.bottom, bounds.top));
    ctx.lineTo(scale(x2, xDomain[0], xDomain[1], bounds.left, bounds.right), scale(y2, yDomain[0], yDomain[1], bounds.bottom, bounds.top));
    ctx.stroke();
    ctx.fillStyle = "#fda4af";
    ctx.textAlign = "left";
    ctx.fillText(`R2=${regression.r2.toFixed(4)}`, bounds.left + 10, bounds.top + 18);
  }
}

function drawHistogram(canvas, histogram, options = {}) {
  const { ctx, width, height } = setupCanvas(canvas);
  const bounds = chartBounds(width, height);
  const maxCount = Math.max(1, ...histogram.map((item) => item.count));
  drawBase(ctx, width, height, options.xLabel || "Value", options.yLabel || "Count", [1, histogram.length || 1], [0, maxCount], false);
  const barWidth = (bounds.right - bounds.left) / Math.max(1, histogram.length);
  histogram.forEach((item, index) => {
    const barHeight = scale(item.count, 0, maxCount, 0, bounds.bottom - bounds.top);
    ctx.fillStyle = index < 5 ? "#7dd3fc" : "#6b7280";
    ctx.fillRect(bounds.left + index * barWidth + 1, bounds.bottom - barHeight, Math.max(1, barWidth - 2), barHeight);
  });
}

function drawScatterWithCurves(canvas, rows, curves, options) {
  drawScatter(canvas, rows.filter((row) => Number(row.trial_type_id) <= options.trialLimit), options.xKey, options.yKey, {
    xLabel: options.xLabel,
    yLabel: options.yLabel,
    colorByTrial: true,
    colorCount: options.trialLimit + 1,
  });
  const { ctx, width, height } = setupCanvas(canvas);
  const filtered = rows.filter((row) => Number(row.trial_type_id) <= options.trialLimit);
  const xDomain = extent(filtered.map((row) => getValue(row, options.xKey)));
  const yDomain = extent(filtered.map((row) => getValue(row, options.yKey)));
  const bounds = drawBase(ctx, width, height, options.xLabel, options.yLabel, xDomain, yDomain, false);
  drawPoints(ctx, filtered, options.xKey, options.yKey, bounds, xDomain, yDomain, { colorByTrial: true, colorCount: options.trialLimit + 1 });
  curves.forEach((curve, index) => {
    ctx.strokeStyle = ["#7dd3fc", "#6ee7b7", "#fbbf24", "#f472b6"][index % 4];
    ctx.lineWidth = 2.5;
    ctx.beginPath();
    curve.points.forEach((point, pointIndex) => {
      const x = scale(point.x, xDomain[0], xDomain[1], bounds.left, bounds.right);
      const y = scale(point.y, yDomain[0], yDomain[1], bounds.bottom, bounds.top);
      if (pointIndex === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    ctx.stroke();
  });
}

function drawPoints(ctx, rows, xKey, yKey, bounds, xDomain, yDomain, options) {
  rows.forEach((row) => {
    const xValue = getValue(row, xKey);
    const yValue = getValue(row, yKey);
    if (!Number.isFinite(xValue) || !Number.isFinite(yValue)) return;
    const x = scale(xValue, xDomain[0], xDomain[1], bounds.left, bounds.right);
    const y = scale(yValue, yDomain[0], yDomain[1], bounds.bottom, bounds.top);
    ctx.fillStyle = pointColor(row, options);
    ctx.globalAlpha = 0.58;
    ctx.beginPath();
    ctx.arc(x, y, 2.2, 0, Math.PI * 2);
    ctx.fill();
  });
  ctx.globalAlpha = 1;
}

function drawSimulation(canvas, points, avgWpm, avgAcc) {
  drawScatter(canvas, points, "wpm", "acc", {
    xLabel: "WPM",
    yLabel: "Accuracy",
    regression: true,
  });
  const { ctx, width, height } = setupCanvas(canvas);
  const xDomain = extent(points.map((point) => Number(point.wpm)));
  const yDomain = extent(points.map((point) => Number(point.acc)));
  const bounds = drawBase(ctx, width, height, "WPM", "Accuracy", xDomain, yDomain, false);
  drawPoints(ctx, points, "wpm", "acc", bounds, xDomain, yDomain, {});
  ctx.strokeStyle = "rgba(243, 245, 248, 0.55)";
  ctx.setLineDash([5, 5]);
  const targetX = scale(avgWpm, xDomain[0], xDomain[1], bounds.left, bounds.right);
  const targetY = scale(avgAcc, yDomain[0], yDomain[1], bounds.bottom, bounds.top);
  ctx.beginPath();
  ctx.moveTo(targetX, bounds.top);
  ctx.lineTo(targetX, bounds.bottom);
  ctx.moveTo(bounds.left, targetY);
  ctx.lineTo(bounds.right, targetY);
  ctx.stroke();
  ctx.setLineDash([]);
  const meanWpm = mean(points.map((point) => point.wpm));
  const meanAcc = mean(points.map((point) => point.acc));
  ctx.fillStyle = "#fb7185";
  ctx.beginPath();
  ctx.arc(scale(meanWpm, xDomain[0], xDomain[1], bounds.left, bounds.right), scale(meanAcc, yDomain[0], yDomain[1], bounds.bottom, bounds.top), 5, 0, Math.PI * 2);
  ctx.fill();
}

function drawLoss(canvas, trainLoss, testLoss) {
  const { ctx, width, height } = setupCanvas(canvas);
  const rows = trainLoss.map((value, index) => ({ epoch: index + 1, train: value, test: testLoss[index] }));
  const xDomain = [1, rows.length];
  const yValues = rows.flatMap((row) => [Math.log10(row.train), Math.log10(row.test)]).filter(Number.isFinite);
  const yDomain = extent(yValues);
  const bounds = drawBase(ctx, width, height, "Epoch", "Log10 loss", xDomain, yDomain, false);
  drawLine(ctx, rows, "train", "#7dd3fc", bounds, xDomain, yDomain, true);
  drawLine(ctx, rows, "test", "#6ee7b7", bounds, xDomain, yDomain, true);
  drawLegend(ctx, bounds, [
    ["Train loss", "#7dd3fc"],
    ["Test loss", "#6ee7b7"],
  ]);
}

function drawLine(ctx, rows, key, color, bounds, xDomain, yDomain, logY = false) {
  ctx.strokeStyle = color;
  ctx.lineWidth = 2;
  ctx.beginPath();
  rows.forEach((row, index) => {
    const x = scale(row.epoch, xDomain[0], xDomain[1], bounds.left, bounds.right);
    const rawY = Number(row[key]);
    const yValue = logY ? Math.log10(rawY) : rawY;
    const y = scale(yValue, yDomain[0], yDomain[1], bounds.bottom, bounds.top);
    if (index === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  });
  ctx.stroke();
}

function drawPredictionScatter(canvas, predictions) {
  drawScatter(canvas, predictions, "actual", "predicted", {
    xLabel: "Actual WPM",
    yLabel: "Predicted WPM",
    regression: false,
  });
  const { ctx, width, height } = setupCanvas(canvas);
  const values = predictions.flatMap((row) => [row.actual, row.predicted]);
  const domain = extent(values);
  const bounds = drawBase(ctx, width, height, "Actual WPM", "Predicted WPM", domain, domain, false);
  drawPoints(ctx, predictions, "actual", "predicted", bounds, domain, domain, {});
  ctx.strokeStyle = "#f3f5f8";
  ctx.setLineDash([5, 5]);
  ctx.beginPath();
  ctx.moveTo(bounds.left, bounds.bottom);
  ctx.lineTo(bounds.right, bounds.top);
  ctx.stroke();
  ctx.setLineDash([]);
}

function drawModelFeatureScatter(canvas, predictions) {
  const { ctx, width, height } = setupCanvas(canvas);
  ctx.clearRect(0, 0, width, height);
  const half = width / 2;
  drawFeaturePane(ctx, predictions, "trial_num", "Trial number", 0, half, height);
  drawFeaturePane(ctx, predictions, "trial_type_num", "Trial type completed", half, half, height);
}

function drawFeaturePane(ctx, predictions, xKey, label, xOffset, paneWidth, height) {
  ctx.save();
  ctx.translate(xOffset, 0);
  const bounds = chartBounds(paneWidth, height);
  const xDomain = extent(predictions.map((row) => Number(row[xKey])));
  const yDomain = extent(predictions.flatMap((row) => [row.actual, row.predicted]));
  drawBase(ctx, paneWidth, height, label, "Words per minute", xDomain, yDomain, false);
  predictions.forEach((row) => {
    const x = scale(row[xKey], xDomain[0], xDomain[1], bounds.left, bounds.right);
    const actualY = scale(row.actual, yDomain[0], yDomain[1], bounds.bottom, bounds.top);
    const predictedY = scale(row.predicted, yDomain[0], yDomain[1], bounds.bottom, bounds.top);
    ctx.fillStyle = "#7dd3fc";
    ctx.globalAlpha = 0.54;
    ctx.beginPath();
    ctx.arc(x, actualY, 2.2, 0, Math.PI * 2);
    ctx.fill();
    ctx.fillStyle = "#fbbf24";
    ctx.beginPath();
    ctx.arc(x, predictedY, 2.2, 0, Math.PI * 2);
    ctx.fill();
  });
  ctx.globalAlpha = 1;
  drawLegend(ctx, bounds, [
    ["Actual", "#7dd3fc"],
    ["Predicted", "#fbbf24"],
  ]);
  ctx.restore();
}

function drawLegend(ctx, bounds, items) {
  ctx.font = "12px Inter, system-ui, sans-serif";
  ctx.textAlign = "left";
  items.forEach(([label, color], index) => {
    const x = bounds.left + 10 + index * 110;
    const y = bounds.top + 16;
    ctx.fillStyle = color;
    ctx.fillRect(x, y - 8, 10, 10);
    ctx.fillStyle = "#a8afbd";
    ctx.fillText(label, x + 15, y);
  });
}

function linearRegression(xValues, yValues) {
  const n = xValues.length;
  const xMean = mean(xValues);
  const yMean = mean(yValues);
  let numerator = 0;
  let xDenominator = 0;
  let yDenominator = 0;
  for (let i = 0; i < n; i += 1) {
    const xDiff = xValues[i] - xMean;
    const yDiff = yValues[i] - yMean;
    numerator += xDiff * yDiff;
    xDenominator += xDiff * xDiff;
    yDenominator += yDiff * yDiff;
  }
  const slope = numerator / xDenominator;
  const intercept = yMean - slope * xMean;
  const r = numerator / Math.sqrt(xDenominator * yDenominator);
  return { slope, intercept, r2: Number.isFinite(r) ? r * r : 0 };
}

window.addEventListener("hashchange", render);
let resizeTimer = null;
window.addEventListener("resize", () => {
  window.clearTimeout(resizeTimer);
  resizeTimer = window.setTimeout(render, 120);
});

render();
