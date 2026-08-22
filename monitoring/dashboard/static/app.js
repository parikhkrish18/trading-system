// Trading System Monitor — vanilla JS, no build step, no external dependencies.

const fmt = {
  num: (v, d = 2) => (v === null || v === undefined || Number.isNaN(v) ? "—" : Number(v).toFixed(d)),
  pct: (v, d = 2) => (v === null || v === undefined || Number.isNaN(v) ? "—" : `${(Number(v) * 100).toFixed(d)}%`),
  money: (v) => (v === null || v === undefined || Number.isNaN(v) ? "—" : `$${Number(v).toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`),
  time: (v) => (v ? new Date(v).toLocaleString() : "—"),
};

// ---------- Operator token ----------
// The mutating endpoints (POST /api/tests/run, /api/jobs/*/run) are gated by
// DASHBOARD_API_TOKEN, which is mandatory once the dashboard is bound to a
// public interface. Without this the hosted dashboard's own buttons would be
// permanently 403 and the only way to run a job would be curl. The token
// lives in this browser's localStorage and is never sent anywhere except as
// a bearer header back to this same origin.
const TOKEN_STORAGE_KEY = "dashboardApiToken";

function getApiToken() {
  try {
    return localStorage.getItem(TOKEN_STORAGE_KEY) || "";
  } catch {
    return ""; // private browsing / storage disabled — the field just won't persist
  }
}

function setApiToken(value) {
  try {
    if (value) localStorage.setItem(TOKEN_STORAGE_KEY, value);
    else localStorage.removeItem(TOKEN_STORAGE_KEY);
  } catch {
    /* not persisting is survivable; the in-page value still works this session */
  }
}

async function fetchJSON(url, opts) {
  const options = { ...opts };
  const token = getApiToken();
  if (token) options.headers = { ...(options.headers || {}), Authorization: `Bearer ${token}` };
  const res = await fetch(url, options);
  if (res.status === 403) {
    throw new Error(
      token
        ? "403 — the operator token was rejected. Check it matches DASHBOARD_API_TOKEN on the server."
        : "403 — this action needs the operator token. Paste DASHBOARD_API_TOKEN into the box at the top right."
    );
  }
  if (!res.ok) throw new Error(`${url} -> ${res.status}`);
  return res.json();
}

// ---------- SVG line chart (no dependencies) ----------
// `regimeAt(label)` is an optional function mapping a point's label (ts) to
// 'chop'|'trend'|null -- when given, shades the background behind each
// point-to-point segment so regime is visible directly on the chart instead
// of only in a separate table.
function renderLineChart(container, points, { color = "#5b8cff", height = 160, zeroLine = false, regimeAt = null } = {}) {
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

  let bandsSvg = "";
  if (regimeAt) {
    bandsSvg = points
      .slice(0, -1)
      .map((p, i) => {
        const regime = regimeAt(p.label);
        if (regime !== "chop") return "";
        const x0 = coords[i][0];
        const x1 = coords[i + 1][0];
        return `<rect x="${x0.toFixed(1)}" y="0" width="${(x1 - x0).toFixed(1)}" height="${height}" fill="#e5484d" opacity="0.08" />`;
      })
      .join("");
  }

  const svg = `
    <svg viewBox="0 0 ${width} ${height}" preserveAspectRatio="none">
      ${bandsSvg}
      ${zeroLine ? `<line x1="${pad}" y1="${zeroY}" x2="${width - pad}" y2="${zeroY}" stroke="#8b92a6" stroke-dasharray="3,3" stroke-width="1" />` : ""}
      <path d="${path}" fill="none" stroke="${color}" stroke-width="2" />
    </svg>
    <div class="muted" style="display:flex;justify-content:space-between;font-size:11px;margin-top:4px;">
      <span>${fmt.time(points[0].label)}</span>
      <span>${fmt.time(points[points.length - 1].label)}</span>
    </div>`;
  container.innerHTML = svg;
}

// Grouped bar chart for the report card — same hand-rolled SVG approach as
// renderLineChart. `rows`: [{fold_label, series, accuracy}], long form.
function renderGroupedBarChart(container, rows, { height = 180, colors = { [0]: "#5b8cff", [1]: "#26a65b" } } = {}) {
  container.innerHTML = "";
  if (!rows || rows.length === 0) {
    container.innerHTML = '<div class="empty-state">No fold metrics recorded yet.</div>';
    return;
  }
  const width = 600;
  const pad = 8;
  const labelBand = 16;
  const folds = [...new Set(rows.map((r) => r.fold_label))];
  const seriesNames = [...new Set(rows.map((r) => r.series))];
  const groupWidth = (width - pad * 2) / folds.length;
  const barWidth = Math.min(24, (groupWidth - 6) / seriesNames.length);
  const plotHeight = height - pad * 2 - labelBand;

  const bars = rows
    .map((r) => {
      const gi = folds.indexOf(r.fold_label);
      const si = seriesNames.indexOf(r.series);
      const h = Math.max(1, plotHeight * Math.min(Math.max(r.accuracy, 0), 1));
      const x = pad + gi * groupWidth + (groupWidth - barWidth * seriesNames.length) / 2 + si * barWidth;
      const y = pad + plotHeight - h;
      return `<rect x="${x.toFixed(1)}" y="${y.toFixed(1)}" width="${(barWidth - 2).toFixed(1)}" height="${h.toFixed(1)}" fill="${colors[si] || "#5b8cff"}"><title>${r.fold_label} — ${r.series}: ${(r.accuracy * 100).toFixed(1)}%</title></rect>`;
    })
    .join("");

  const fiftyY = pad + plotHeight * 0.5;
  const labels = folds
    .map((f, gi) => {
      const x = pad + gi * groupWidth + groupWidth / 2;
      return `<text x="${x.toFixed(1)}" y="${height - 4}" text-anchor="middle" font-size="10" fill="#8b92a6">${f}</text>`;
    })
    .join("");

  const legend = seriesNames
    .map((name, si) => `<span class="legend-item"><span class="legend-swatch" style="background:${colors[si] || "#5b8cff"}"></span>${name}</span>`)
    .join(" ");

  container.innerHTML = `
    <svg viewBox="0 0 ${width} ${height}">
      <line x1="${pad}" y1="${fiftyY}" x2="${width - pad}" y2="${fiftyY}" stroke="#8b92a6" stroke-dasharray="3,3" stroke-width="1" />
      <text x="${width - pad}" y="${fiftyY - 3}" text-anchor="end" font-size="9" fill="#8b92a6">coin flip (50%)</text>
      ${bars}
      ${labels}
    </svg>
    <div class="muted" style="font-size:11px;margin-top:4px;">${legend}</div>`;
}

// ---------- Model report card ----------
async function loadReportCard() {
  const result = await fetchJSON("/api/analysis/report_card");
  const summary = document.getElementById("report-card-summary");
  const callouts = document.getElementById("report-card-callouts");
  if (!result.available) {
    summary.innerHTML = '<div class="empty-state">No training runs found in MLflow yet — run models/train.py to grade the model.</div>';
    renderGroupedBarChart(document.getElementById("report-card-chart"), []);
    callouts.innerHTML = "";
    return;
  }
  const h = result.headline;
  summary.innerHTML = `
    <div class="stat-card"><div class="value">${h.n_folds}</div><div class="label">Folds graded</div></div>
    <div class="stat-card"><div class="value">${fmt.pct(h.directional_accuracy, 1)}</div><div class="label">Direction right (all predictions)</div></div>
    <div class="stat-card"><div class="value">${fmt.pct(h.directional_accuracy_when_confident, 1)}</div><div class="label">Direction right (models agreed)</div></div>
    <div class="stat-card"><div class="value">${fmt.pct(h.pct_rows_confident, 1)}</div><div class="label">Share clearing the agreement bar</div></div>
  `;
  renderGroupedBarChart(document.getElementById("report-card-chart"), result.chart);
  callouts.innerHTML = result.callouts.map((c) => `<div class="callout">${c}</div>`).join("");
}

// ---------- What-if thresholds ----------
async function loadWhatif() {
  const minAgreement = document.getElementById("whatif-agreement").value;
  const minMove = document.getElementById("whatif-move").value;
  document.getElementById("whatif-agreement-value").textContent = `${Math.round(minAgreement * 100)}%`;
  document.getElementById("whatif-move-value").textContent = `${(minMove * 100).toFixed(1)}%`;

  const result = await fetchJSON(`/api/whatif?min_agreement=${minAgreement}&min_abs_move=${minMove}`);
  const summary = document.getElementById("whatif-summary");
  const tbody = document.querySelector("#whatif-table tbody");
  if (!result.available) {
    summary.innerHTML = `<div class="empty-state">${result.message}</div>`;
    tbody.innerHTML = "";
    return;
  }
  summary.innerHTML = `<div class="stat-card"><div class="value">${result.n_after}/${result.n_before}</div><div class="label">${result.summary}</div></div>`;
  if (result.rows.length === 0) {
    tbody.innerHTML = '<tr><td colspan="7" class="empty-state">Nothing clears these thresholds.</td></tr>';
    return;
  }
  tbody.innerHTML = result.rows
    .map(
      (r) => `
      <tr>
        <td><strong>${r["Symbol"]}</strong></td>
        <td>${r["Direction"]}</td>
        <td>${r["Market regime"]}</td>
        <td>${fmt.pct(r["Predicted move"], 2)}</td>
        <td>${fmt.pct(r["Target size"], 1)}</td>
        <td>${fmt.pct(r["Model agreement"], 0)}</td>
        <td>${r["Placed?"]}</td>
      </tr>`
    )
    .join("");
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

function newsFeedHTML(items) {
  if (!items || items.length === 0) {
    return '<div class="no-decision">No recent news found for this symbol.</div>';
  }
  return items
    .map((n) => {
      const sentClass = n.sentiment === null || n.sentiment === undefined ? "" : n.sentiment >= 0 ? "pl-pos" : "pl-neg";
      const sentLabel = n.sentiment === null || n.sentiment === undefined ? "unscored" : fmt.num(n.sentiment, 2);
      return `
        <div class="news-row">
          <div class="news-headline">${n.headline}</div>
          <div class="news-meta"><span class="${sentClass}">${sentLabel}</span> · ${fmt.time(n.ts)} · ${n.source || ""}</div>
        </div>`;
    })
    .join("");
}

function positionCardHTML(p, idx, newsBySymbol) {
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

  const news = (newsBySymbol && newsBySymbol[p.symbol]) || [];
  const newsBlock = `
      <div class="reasoning-toggle" data-news-idx="${idx}">▸ Recent news (${news.length})</div>
      <div class="reasoning-body" id="news-${idx}">${newsFeedHTML(news)}</div>
    `;

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
      ${newsBlock}
    </div>`;
}

async function loadPositions() {
  const [positions, newsBySymbol] = await Promise.all([
    fetchJSON("/api/positions"),
    fetchJSON("/api/positions/news").catch(() => ({})),
  ]);
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
  list.innerHTML = positions.map((p, i) => positionCardHTML(p, i, newsBySymbol)).join("");
  list.querySelectorAll(".reasoning-toggle[data-idx]").forEach((el) => {
    el.addEventListener("click", () => {
      const body = document.getElementById(`reasoning-${el.dataset.idx}`);
      body.classList.toggle("open");
      el.textContent = (body.classList.contains("open") ? "▾ " : "▸ ") + "Why this trade";
    });
  });
  list.querySelectorAll(".reasoning-toggle[data-news-idx]").forEach((el) => {
    el.addEventListener("click", () => {
      const body = document.getElementById(`news-${el.dataset.newsIdx}`);
      body.classList.toggle("open");
      el.textContent = (body.classList.contains("open") ? "▾ " : "▸ ") + el.textContent.replace(/^[▾▸]\s*/, "");
    });
  });
}

// ---------- Equity & drawdown ----------
async function loadEquity() {
  const [rows, regimeRows] = await Promise.all([
    fetchJSON("/api/equity_curve?mode=paper"),
    fetchJSON("/api/regime_history").catch(() => []),
  ]);
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

  // regime_history is dense (daily); equity snapshots are sparser -- for each
  // equity point, find the regime on or before that date rather than requiring
  // an exact timestamp match.
  const sortedRegimes = [...regimeRows].sort((a, b) => new Date(a.ts) - new Date(b.ts));
  const regimeAt = (label) => {
    const t = new Date(label).getTime();
    let match = null;
    for (const r of sortedRegimes) {
      if (new Date(r.ts).getTime() > t) break;
      match = r.regime;
    }
    return match;
  };

  renderLineChart(
    document.getElementById("equity-chart"),
    rows.map((r) => ({ y: r.equity_value, label: r.ts })),
    { color: "#5b8cff", regimeAt: sortedRegimes.length ? regimeAt : null }
  );

  let runningPeak = -Infinity;
  const ddPoints = rows.map((r) => {
    runningPeak = Math.max(runningPeak, r.equity_value);
    return { y: runningPeak > 0 ? r.equity_value / runningPeak - 1 : 0, label: r.ts };
  });
  renderLineChart(document.getElementById("drawdown-chart"), ddPoints, { color: "#e5484d", zeroLine: true });
}

// ---------- Closed trades ----------
async function loadClosedTrades() {
  const rows = await fetchJSON("/api/trades/closed");
  const tbody = document.querySelector("#closed-trades-table tbody");
  if (rows.length === 0) {
    tbody.innerHTML = '<tr><td colspan="8" class="empty-state">No closed trades yet.</td></tr>';
    document.getElementById("closed-trades-summary").innerHTML = "";
    return;
  }

  const totalPnl = rows.reduce((s, t) => s + t.realized_pnl, 0);
  const wins = rows.filter((t) => t.realized_pnl > 0).length;
  const winRate = wins / rows.length;
  document.getElementById("closed-trades-summary").innerHTML = `
    <div class="stat-card ${totalPnl >= 0 ? "good" : "bad"}"><div class="value">${fmt.money(totalPnl)}</div><div class="label">Total realized P/L</div></div>
    <div class="stat-card"><div class="value">${rows.length}</div><div class="label">Closed trades</div></div>
    <div class="stat-card ${winRate >= 0.5 ? "good" : "bad"}"><div class="value">${fmt.pct(winRate, 0)}</div><div class="label">Win rate</div></div>
  `;

  tbody.innerHTML = rows
    .map((t) => {
      const plClass = t.realized_pnl >= 0 ? "pl-pos" : "pl-neg";
      return `
      <tr>
        <td><strong>${t.symbol}</strong></td>
        <td><span class="side-badge ${t.side}">${t.side}</span></td>
        <td>${fmt.time(t.entry_ts)}</td>
        <td>${fmt.time(t.exit_ts)}</td>
        <td>${fmt.num(t.shares, 2)}</td>
        <td>${fmt.money(t.entry_price)}</td>
        <td>${fmt.money(t.exit_price)}</td>
        <td class="${plClass}">${fmt.money(t.realized_pnl)} (${fmt.pct(t.realized_pnl_pct)})</td>
      </tr>`;
    })
    .join("");
}

// ---------- Feature importance over time ----------
async function loadFeatureImportance() {
  const rows = await fetchJSON("/api/analysis/feature_frequency");
  const box = document.getElementById("feature-importance-list");
  if (rows.length === 0) {
    box.innerHTML = '<div class="empty-state">No structured feature data yet (only decisions logged after the 7-phase reasoning model count).</div>';
    return;
  }
  const maxCount = Math.max(...rows.map((r) => r.times_in_top5));
  box.innerHTML = rows
    .map((r) => {
      const pct = (r.times_in_top5 / maxCount) * 100;
      return `
      <div class="feature-freq-row">
        <div class="feature-freq-name" title="${r.feature_name}">${r.feature_name}</div>
        <div class="feature-freq-bar-track"><div class="feature-freq-bar" style="width:${pct}%"></div></div>
        <div class="feature-freq-count">${r.times_in_top5}×</div>
      </div>`;
    })
    .join("");
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
  const backfill = result.backfill || { hit_rate: null, n_matured: 0 };
  let html = "";
  if (result.hit_rate === null) {
    html += '<div class="empty-state">No matured live decisions yet to score (needs a price bar after the decision).</div>';
  } else {
    const cls = result.hit_rate >= 0.5 ? "good" : "bad";
    html += `
    <div class="stat-card ${cls}"><div class="value">${fmt.pct(result.hit_rate, 1)}</div><div class="label">Live directional hit rate — real decisions only</div></div>
    <div class="stat-card"><div class="value">${result.n_matured}</div><div class="label">Matured live decisions scored</div></div>`;
  }
  if (backfill.hit_rate !== null) {
    html += `
    <div class="stat-card"><div class="value">${fmt.pct(backfill.hit_rate, 1)}</div><div class="label">Backfilled replay hit rate — historical simulation, NOT live results</div></div>
    <div class="stat-card"><div class="value">${backfill.n_matured}</div><div class="label">Backfilled rows (excluded from the live number)</div></div>`;
  }
  box.innerHTML = html;
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

// ---------- Manual triggers (ingest / trading cycle) ----------
// Same shape as the tests panel: POST to start, then poll /api/jobs while
// anything is running. The cycle button only STARTS a cycle — trades still
// wait on the Telegram approval gate.

let jobsPollTimer = null;

function renderJobStatus(name, job) {
  const box = document.getElementById(`job-${name}-status`);
  const btn = document.getElementById(name === "ingest" ? "run-ingest-btn" : "run-cycle-btn");
  if (!box) return;
  const labels = { never_run: "Never run", running: "⏳ Running…", finished: "✅ Finished", failed: "❌ Failed" };
  const when = job.finished_at || job.started_at;
  box.innerHTML = `<div class="value">${labels[job.status] || job.status}</div><div class="label">${job.label}${when ? ` — ${fmt.time(when)}` : ""}</div>`;
  box.className = `stat-card ${job.status === "finished" ? "good" : job.status === "failed" ? "bad" : ""}`;
  if (btn) {
    btn.disabled = job.status === "running";
    btn.textContent = job.status === "running"
      ? "Running… (runs in background)"
      : name === "ingest" ? "Run data ingestion" : "Run trading cycle";
  }
}

async function loadJobs() {
  const jobs = await fetchJSON("/api/jobs");
  let tail = "";
  let anyRunning = false;
  for (const [name, job] of Object.entries(jobs)) {
    renderJobStatus(name, job);
    anyRunning = anyRunning || job.status === "running";
    if (job.output_tail) tail += `=== ${name} ===\n${job.output_tail}\n\n`;
  }
  document.getElementById("job-output").textContent = tail;
  // Keep polling while something is running; stop when everything settles.
  if (anyRunning && !jobsPollTimer) {
    jobsPollTimer = setInterval(loadJobs, 5000);
  } else if (!anyRunning && jobsPollTimer) {
    clearInterval(jobsPollTimer);
    jobsPollTimer = null;
  }
}

async function startJob(name) {
  try {
    await fetchJSON(`/api/jobs/${name}/run`, { method: "POST" });
    await loadJobs();
  } catch (e) {
    // 409 = already running, 403 = missing/rejected token. Show it: a button
    // that silently does nothing is the worst way to report a missing token.
    await loadJobs();
    document.getElementById("job-output").textContent =
      `Could not start ${name}: ${e.message}\n\n` + document.getElementById("job-output").textContent;
    document.getElementById("job-output-wrap").open = true;
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
    loadClosedTrades(),
    loadEquity(),
    loadBreakers(),
    loadAnalysis(),
    loadReportCard(),
    loadWhatif(),
    loadFeatureImportance(),
    loadLiveAccuracy(),
    loadDecisions(),
    loadLastTestRun(),
    loadJobs(),
  ]);
}

const tokenInput = document.getElementById("api-token-input");
tokenInput.value = getApiToken();
tokenInput.addEventListener("change", () => setApiToken(tokenInput.value.trim()));

document.getElementById("refresh-btn").addEventListener("click", loadAll);
document.getElementById("run-tests-btn").addEventListener("click", runTestsNow);
document.getElementById("run-ingest-btn").addEventListener("click", () => startJob("ingest"));
document.getElementById("run-cycle-btn").addEventListener("click", () => startJob("cycle"));
document.getElementById("decision-filter-btn").addEventListener("click", () => {
  loadDecisions(document.getElementById("decision-symbol-filter").value.trim().toUpperCase());
});
document.getElementById("decision-symbol-filter").addEventListener("keydown", (e) => {
  if (e.key === "Enter") document.getElementById("decision-filter-btn").click();
});
// The sliders re-hit the endpoint on every input tick — the endpoint is a
// read-only re-filter of one already-logged batch, so that's cheap.
document.getElementById("whatif-agreement").addEventListener("input", loadWhatif);
document.getElementById("whatif-move").addEventListener("input", loadWhatif);

loadAll();
