// Trading System Monitor — vanilla JS, no build step, no external dependencies.

const fmt = {
  num: (v, d = 2) => (v === null || v === undefined || Number.isNaN(v) ? "—" : Number(v).toFixed(d)),
  pct: (v, d = 2) => (v === null || v === undefined || Number.isNaN(v) ? "—" : `${(Number(v) * 100).toFixed(d)}%`),
  money: (v) => (v === null || v === undefined || Number.isNaN(v) ? "—" : `$${Number(v).toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`),
  time: (v) => (v ? new Date(v).toLocaleString() : "—"),
};

async function fetchJSON(url, opts) {
  const res = await fetch(url, opts);
  if (!res.ok) throw new Error(`${url} -> ${res.status}`);
  return res.json();
}

// ---------- SVG line chart (no dependencies) ----------
function renderLineChart(container, points, { color = "#5b8cff", height = 160, zeroLine = false } = {}) {
  container.innerHTML = "";
  if (!points || points.length < 2) {
    container.innerHTML = '<div class="empty-state">Not enough data yet.</div>';
    return;
  }
  const width = 600;
  const pad = 8;
  const values = points.map((p) => p.y);
  const minV = Math.min(...values, zeroLine ? 0 : Infinity);
  const maxV = Math.max(...values, zeroLine ? 0 : -Infinity);
  const range = maxV - minV || 1;

  const xStep = (width - pad * 2) / (points.length - 1);
  const coords = points.map((p, i) => {
    const x = pad + i * xStep;
    const y = pad + (height - pad * 2) * (1 - (p.y - minV) / range);
    return [x, y];
  });
  const path = coords.map(([x, y], i) => `${i === 0 ? "M" : "L"}${x.toFixed(1)},${y.toFixed(1)}`).join(" ");

  let zeroY = null;
  if (zeroLine) {
    zeroY = pad + (height - pad * 2) * (1 - (0 - minV) / range);
  }

  const svg = `
    <svg viewBox="0 0 ${width} ${height}" preserveAspectRatio="none">
      ${zeroLine ? `<line x1="${pad}" y1="${zeroY}" x2="${width - pad}" y2="${zeroY}" stroke="#8b92a6" stroke-dasharray="3,3" stroke-width="1" />` : ""}
      <path d="${path}" fill="none" stroke="${color}" stroke-width="2" />
    </svg>
    <div class="muted" style="display:flex;justify-content:space-between;font-size:11px;margin-top:4px;">
      <span>${fmt.time(points[0].label)}</span>
      <span>${fmt.time(points[points.length - 1].label)}</span>
    </div>`;
  container.innerHTML = svg;
}

// ---------- Positions ----------

// Legacy fallback for decisions logged before the 7-phase reasoning model —
// a flat list of {feature_name, value, contribution} SHAP contributions.
function legacyReasoningBarsHTML(reasoning) {
  const maxAbs = Math.max(...reasoning.map((r) => Math.abs(r.contribution))) || 1;
  return reasoning
    .map((r) => {
      const pct = (Math.abs(r.contribution) / maxAbs) * 50;
      const cls = r.contribution >= 0 ? "pos" : "neg";
      return `
        <div class="reasoning-row">
          <div class="reasoning-feature" title="${r.feature_name}">${r.feature_name}</div>
          <div class="reasoning-bar-track"><div class="reasoning-bar ${cls}" style="width:${pct}%"></div></div>
          <div class="reasoning-value">${fmt.num(r.contribution, 4)}</div>
        </div>`;
    })
    .join("");
}

// Current format: a list of phase dicts {phase, title, summary, lines[]},
// one per stage of the decision (see monitoring/reasoning.py). Rendered as
// plain-English cards in phase order, not a features/contribution chart.
function reasoningPhasesHTML(reasoning) {
  if (!reasoning || reasoning.length === 0) {
    return '<div class="no-decision">No reasoning captured for this decision (logged before reasoning tracking was added).</div>';
  }
  if (reasoning[0].phase === undefined) {
    return legacyReasoningBarsHTML(reasoning);
  }
  return reasoning
    .map(
      (p) => `
      <div class="phase-card">
        <div class="phase-head">Phase ${p.phase} — ${p.title}</div>
        <div class="phase-summary">${p.summary}</div>
        <ul class="phase-lines">${p.lines.map((line) => `<li>${line}</li>`).join("")}</ul>
      </div>`
    )
    .join("");
}

function positionCardHTML(p, idx) {
  const plClass = p.unrealized_pl >= 0 ? "pl-pos" : "pl-neg";
  const d = p.decision;
  const decisionBlock = d
    ? `
      <div class="position-grid-stats">
        <div><div class="label">Forecast</div>${fmt.pct(d.forecast, 2)}</div>
        <div><div class="label">Regime</div>${d.regime || "—"}</div>
        <div><div class="label">Target %</div>${fmt.pct(d.target_position, 1)}</div>
        <div><div class="label">Decided</div>${fmt.time(d.ts)}</div>
      </div>
      <div class="reasoning-toggle" data-idx="${idx}">▸ Why this trade — all 7 phases</div>
      <div class="reasoning-body" id="reasoning-${idx}">${reasoningPhasesHTML(d.reasoning)}</div>
    `
    : '<div class="no-decision">No matching decision found for this position.</div>';

  return `
    <div class="position-card">
      <div class="position-card-head">
        <span class="position-symbol">${p.symbol}</span>
        <span class="side-badge ${p.side}">${p.side}</span>
      </div>
      <div class="position-grid-stats">
        <div><div class="label">Qty</div>${fmt.num(p.qty, 4)}</div>
        <div><div class="label">Market value</div>${fmt.money(p.market_value)}</div>
        <div><div class="label">Entry price</div>${fmt.money(p.avg_entry_price)}</div>
        <div><div class="label">Current price</div>${fmt.money(p.current_price)}</div>
        <div><div class="label">Unrealized P/L</div><span class="${plClass}">${fmt.money(p.unrealized_pl)} (${fmt.pct(p.unrealized_plpc)})</span></div>
      </div>
      ${decisionBlock}
    </div>`;
}

async function loadPositions() {
  const positions = await fetchJSON("/api/positions");
  document.getElementById("positions-count").textContent = `(${positions.length})`;

  const totalValue = positions.reduce((s, p) => s + p.market_value, 0);
  const totalPL = positions.reduce((s, p) => s + p.unrealized_pl, 0);
  document.getElementById("positions-summary").innerHTML = `
    <div class="stat-card"><div class="value">${positions.length}</div><div class="label">Open positions</div></div>
    <div class="stat-card"><div class="value">${fmt.money(totalValue)}</div><div class="label">Total market value</div></div>
    <div class="stat-card ${totalPL >= 0 ? "good" : "bad"}"><div class="value">${fmt.money(totalPL)}</div><div class="label">Unrealized P/L</div></div>
  `;

  const list = document.getElementById("positions-list");
  if (positions.length === 0) {
    list.innerHTML = '<div class="empty-state">No open positions.</div>';
    return;
  }
  list.innerHTML = positions.map(positionCardHTML).join("");
  list.querySelectorAll(".reasoning-toggle").forEach((el) => {
    el.addEventListener("click", () => {
      const body = document.getElementById(`reasoning-${el.dataset.idx}`);
      body.classList.toggle("open");
      el.textContent = (body.classList.contains("open") ? "▾ " : "▸ ") + "Why this trade";
    });
  });
}

// ---------- Equity & drawdown ----------
async function loadEquity() {
  const rows = await fetchJSON("/api/equity_curve?mode=paper");
  if (rows.length === 0) {
    document.getElementById("equity-summary").innerHTML = '<div class="empty-state">No equity snapshots recorded yet.</div>';
    renderLineChart(document.getElementById("equity-chart"), []);
    renderLineChart(document.getElementById("drawdown-chart"), []);
    return;
  }
  const values = rows.map((r) => r.equity_value);
  const latest = values[values.length - 1];
  const peak = Math.max(...values);
  const drawdown = peak > 0 ? latest / peak - 1 : 0;

  document.getElementById("equity-summary").innerHTML = `
    <div class="stat-card"><div class="value">${fmt.money(latest)}</div><div class="label">Current equity</div></div>
    <div class="stat-card"><div class="value">${fmt.money(peak)}</div><div class="label">Peak equity</div></div>
    <div class="stat-card ${drawdown >= 0 ? "good" : "bad"}"><div class="value">${fmt.pct(drawdown)}</div><div class="label">Current drawdown</div></div>
  `;

  renderLineChart(
    document.getElementById("equity-chart"),
    rows.map((r) => ({ y: r.equity_value, label: r.ts })),
    { color: "#5b8cff" }
  );

  let runningPeak = -Infinity;
  const ddPoints = rows.map((r) => {
    runningPeak = Math.max(runningPeak, r.equity_value);
    return { y: runningPeak > 0 ? r.equity_value / runningPeak - 1 : 0, label: r.ts };
  });
  renderLineChart(document.getElementById("drawdown-chart"), ddPoints, { color: "#e5484d", zeroLine: true });
}

// ---------- Circuit breakers ----------
async function loadBreakers() {
  const rows = await fetchJSON("/api/circuit_breakers");
  const list = document.getElementById("breakers-list");
  if (rows.length === 0) {
    list.innerHTML = '<div class="empty-state">No breaker checks recorded yet.</div>';
    return;
  }
  const latestByName = {};
  for (const r of rows) {
    if (!latestByName[r.breaker_name] || new Date(r.ts) > new Date(latestByName[r.breaker_name].ts)) {
      latestByName[r.breaker_name] = r;
    }
  }
  list.innerHTML = Object.values(latestByName)
    .sort((a, b) => a.breaker_name.localeCompare(b.breaker_name))
    .map(
      (r) => `
      <div class="breaker-card ${r.triggered ? "triggered" : "ok"}">
        <div class="breaker-name">${r.triggered ? "🚨" : "✅"} ${r.breaker_name}</div>
        <div class="breaker-meta">${r.triggered ? r.reason : "OK"} — ${fmt.time(r.ts)}</div>
      </div>`
    )
    .join("");
}

// ---------- Model analysis ----------
async function loadAnalysis() {
  const rows = await fetchJSON("/api/analysis/runs");
  const tbody = document.querySelector("#analysis-table tbody");
  if (rows.length === 0) {
    tbody.innerHTML = '<tr><td colspan="8" class="empty-state">No walk-forward runs found in MLflow.</td></tr>';
    renderLineChart(document.getElementById("analysis-chart"), []);
    return;
  }
  tbody.innerHTML = rows
    .map(
      (r) => `
      <tr>
        <td>${r.fold_id ?? "—"}</td>
        <td>${r.feature_set_id ?? "—"}</td>
        <td>${(r.test_start || "").slice(0, 10)} → ${(r.test_end || "").slice(0, 10)}</td>
        <td>${fmt.num(r.mae, 4)}</td>
        <td>${fmt.num(r.rmse, 4)}</td>
        <td>${fmt.pct(r.directional_accuracy, 1)}</td>
        <td>${fmt.pct(r.directional_accuracy_when_confident, 1)}</td>
        <td>${fmt.pct(r.pct_rows_confident, 1)}</td>
      </tr>`
    )
    .join("");

  renderLineChart(
    document.getElementById("analysis-chart"),
    rows.filter((r) => r.directional_accuracy !== null).map((r) => ({ y: r.directional_accuracy, label: r.start_time })),
    { color: "#26a65b", zeroLine: false }
  );
}

// ---------- Live forecast accuracy ----------
async function loadLiveAccuracy() {
  const result = await fetchJSON("/api/analysis/live_accuracy");
  const box = document.getElementById("live-accuracy-summary");
  if (result.hit_rate === null) {
    box.innerHTML = '<div class="empty-state">No matured decisions yet to score (needs a price bar after the decision).</div>';
    return;
  }
  const cls = result.hit_rate >= 0.5 ? "good" : "bad";
  box.innerHTML = `
    <div class="stat-card ${cls}"><div class="value">${fmt.pct(result.hit_rate, 1)}</div><div class="label">Live directional hit rate</div></div>
    <div class="stat-card"><div class="value">${result.n_matured}</div><div class="label">Matured decisions scored</div></div>
  `;
}

// ---------- Decision history ----------
async function loadDecisions(symbol) {
  const url = symbol ? `/api/decisions?symbol=${encodeURIComponent(symbol)}` : "/api/decisions";
  const rows = await fetchJSON(url);
  const tbody = document.querySelector("#decisions-table tbody");
  if (rows.length === 0) {
    tbody.innerHTML = '<tr><td colspan="8" class="empty-state">No decisions logged yet.</td></tr>';
    return;
  }
  tbody.innerHTML = rows
    .map((r, i) => {
      const reasoningCell = r.reasoning && r.reasoning.length
        ? `<span class="reasoning-link" data-decision-idx="${i}">${r.reasoning.length} phase(s)</span>`
        : '<span class="muted">—</span>';
      return `
      <tr>
        <td>${fmt.time(r.ts)}</td>
        <td><strong>${r.symbol}</strong></td>
        <td>${fmt.pct(r.forecast, 2)}</td>
        <td>${r.regime || "—"}</td>
        <td>${fmt.pct(r.target_position, 1)}</td>
        <td>${r.executed_position === null ? "—" : fmt.num(r.executed_position, 2)}</td>
        <td>${r.mode}</td>
        <td>${reasoningCell}</td>
      </tr>`;
    })
    .join("");

  tbody.querySelectorAll(".reasoning-link").forEach((el) => {
    el.addEventListener("click", () => {
      const r = rows[Number(el.dataset.decisionIdx)];
      const text = r.reasoning[0].phase !== undefined
        ? r.reasoning.map((p) => `Phase ${p.phase} — ${p.title}\n${p.lines.map((l) => `  • ${l}`).join("\n")}`).join("\n\n")
        : r.reasoning.map((f) => `${f.feature_name}: ${fmt.num(f.contribution, 4)} (value=${fmt.num(f.value, 3)})`).join("\n");
      alert(text);
    });
  });
}

// ---------- Tests ----------
function renderTestStatus(result) {
  const box = document.getElementById("test-status");
  const output = document.getElementById("test-output");
  if (!result) {
    box.textContent = "Never run";
    box.className = "stat-card";
    output.textContent = "";
    return;
  }
  box.innerHTML = `<div class="value">${result.passed ? "✅ Passing" : "❌ Failing"}</div><div class="label">${result.summary} — ${fmt.time(result.ts)}</div>`;
  box.className = `stat-card ${result.passed ? "good" : "bad"}`;
  output.textContent = result.output;
}

async function loadLastTestRun() {
  const result = await fetchJSON("/api/tests/last");
  renderTestStatus(result);
}

async function runTestsNow() {
  const btn = document.getElementById("run-tests-btn");
  btn.disabled = true;
  btn.textContent = "Running… (can take a minute)";
  try {
    const result = await fetchJSON("/api/tests/run", { method: "POST" });
    renderTestStatus(result);
  } finally {
    btn.disabled = false;
    btn.textContent = "Run tests now";
  }
}

// ---------- Mode badge ----------
async function loadModeBadge() {
  try {
    const positions = await fetchJSON("/api/positions");
    const mode = positions[0]?.decision?.mode || "paper";
    const badge = document.getElementById("mode-badge");
    badge.textContent = mode.toUpperCase();
    badge.className = `badge ${mode}`;
  } catch {
    document.getElementById("mode-badge").textContent = "unknown";
  }
}

// ---------- Orchestration ----------
async function loadAll() {
  await Promise.allSettled([
    loadModeBadge(),
    loadPositions(),
    loadEquity(),
    loadBreakers(),
    loadAnalysis(),
    loadLiveAccuracy(),
    loadDecisions(),
    loadLastTestRun(),
  ]);
}

document.getElementById("refresh-btn").addEventListener("click", loadAll);
document.getElementById("run-tests-btn").addEventListener("click", runTestsNow);
document.getElementById("decision-filter-btn").addEventListener("click", () => {
  loadDecisions(document.getElementById("decision-symbol-filter").value.trim().toUpperCase());
});
document.getElementById("decision-symbol-filter").addEventListener("keydown", (e) => {
  if (e.key === "Enter") document.getElementById("decision-filter-btn").click();
});

loadAll();
