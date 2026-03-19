'use strict';

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------

const state = {
  config:        {},
  uploads:       [],
  examples:      [],
  jobs:          [],
  batchSelected: new Set(),
  pollTimer:     null,
};

// ---------------------------------------------------------------------------
// Active-operations tracker
// Drives the global progress rail + header chip.
// ---------------------------------------------------------------------------

const _ops = { uploads: 0, processing: 0, batch: 0, pull: 0 };

function setActive(type, value) {
  _ops[type] = typeof value === 'boolean' ? (value ? 1 : 0) : Math.max(0, value);
  _updateGlobalProgress();
}

function _updateGlobalProgress() {
  const total = _ops.uploads + _ops.processing + _ops.batch + _ops.pull;
  const rail  = document.getElementById('global-rail');
  const chip  = document.getElementById('active-ops-chip');
  const text  = document.getElementById('active-ops-text');

  if (rail) rail.classList.toggle('active', total > 0);

  if (chip) {
    if (total > 0) {
      chip.style.display = 'flex';
      const parts = [];
      if (_ops.uploads > 0)    parts.push(`${_ops.uploads} upload${_ops.uploads > 1 ? 's' : ''}`);
      if (_ops.processing > 0) parts.push('analyzing');
      if (_ops.batch > 0)      parts.push(`${_ops.batch} job${_ops.batch > 1 ? 's' : ''}`);
      if (_ops.pull > 0)       parts.push('pulling model');
      if (text) text.textContent = parts.join(' · ');
    } else {
      chip.style.display = 'none';
    }
  }
}

// ---------------------------------------------------------------------------
// Navigation
// ---------------------------------------------------------------------------

const PAGE_TITLES = {
  analysis: 'Image Analysis',
  batch:    'Batch Process',
  results:  'Results',
  images:   'Manage Images',
  config:   'Configuration',
};

const USE_CASE_HINTS = {
  'Transcribing Typed Text':       'Extract printed or typed text from scanned documents, PDFs, or screenshots.',
  'Transcribing Handwritten Text': 'Convert handwritten notes into searchable, editable digital text.',
  'Transcribing Forms':            'Extract all field labels and values from structured forms.',
  'Complicated Document QA':       'Ask a specific question and receive an answer grounded in the document content.',
  'Unstructured Information → JSON':'Convert the document content into machine-readable structured JSON.',
};

function navigate(section) {
  document.querySelectorAll('.section').forEach(s => s.classList.remove('active'));
  document.querySelectorAll('.nav-item').forEach(n => n.classList.remove('active'));
  document.getElementById(`section-${section}`).classList.add('active');
  document.querySelector(`[data-section="${section}"]`).classList.add('active');
  document.getElementById('page-title').textContent = PAGE_TITLES[section];

  if (section === 'analysis') refreshSingleImageSelect();
  if (section === 'batch')    { refreshBatchGrid(); refreshJobList(); }
  if (section === 'results')  refreshResultsList();
  if (section === 'images')   refreshImagesList();
  if (section === 'config')   loadConfigForm();
}

// ---------------------------------------------------------------------------
// Toast
// ---------------------------------------------------------------------------

function toast(msg, type = 'info', ms = 3500) {
  const el = document.createElement('div');
  el.className = `toast ${type}`;
  el.textContent = msg;
  document.getElementById('toast-container').appendChild(el);
  setTimeout(() => el.remove(), ms);
}

// ---------------------------------------------------------------------------
// API helpers
// ---------------------------------------------------------------------------

async function api(method, path, body) {
  const opts = { method, headers: {} };
  if (body && !(body instanceof FormData)) {
    opts.headers['Content-Type'] = 'application/json';
    opts.body = JSON.stringify(body);
  } else if (body instanceof FormData) {
    opts.body = body;
  }
  const r = await fetch(path, opts);
  if (!r.ok) {
    const err = await r.json().catch(() => ({ detail: r.statusText }));
    throw new Error(err.detail || r.statusText);
  }
  return r.json();
}

// ---------------------------------------------------------------------------
// App init
// ---------------------------------------------------------------------------

async function init() {
  try {
    state.config = await api('GET', '/api/config');
    updateSidebarPills();
  } catch (e) {
    console.error('Init error', e);
  }

  await refreshImages();
  onSingleUseCaseChange();
  refreshSingleImageSelect();

  // Always-on: poll job queue + Ollama status
  state.pollTimer   = setInterval(pollJobs,        2000);
  state.ollamaTimer = setInterval(_globalOllamaCheck, 5000);
  _globalOllamaCheck();   // immediate first check

  navigate('analysis');
}

// ---------------------------------------------------------------------------
// Ollama header chip
// ---------------------------------------------------------------------------

async function _globalOllamaCheck() {
  try {
    const data    = await api('GET', '/api/ollama/status');
    const running = data.running;
    const pulling = !!(data.pull && data.pull.running);

    setActive('pull', pulling);

    // ── Ollama header chip ─────────────────────────────────────────────
    setDot('dot-ollama', running ? 'ok' : 'error');
    const lbl = document.getElementById('ollama-chip-label');
    if (lbl) lbl.textContent = running
      ? `Ollama · ${state.config.local_model || ''}`
      : 'Ollama offline';

    // ── GPU header chip ────────────────────────────────────────────────
    const gpuChip = document.getElementById('gpu-chip');
    const gpuDot  = document.getElementById('dot-gpu');
    const gpuLbl  = document.getElementById('gpu-chip-label');
    const gpu     = data.gpu || {};

    if (gpu.available && gpu.gpus && gpu.gpus.length > 0) {
      const g = gpu.gpus[0];
      if (gpuChip) gpuChip.style.display = 'flex';
      if (data.gpu_in_use) {
        // Model actively loaded in VRAM
        setDot('dot-gpu', 'ok');
        const vramPct = Math.round(g.memory_used_mb / g.memory_total_mb * 100);
        if (gpuLbl) gpuLbl.textContent = `GPU · ${vramPct}% VRAM`;
      } else {
        // GPU present & allocated to this session but no model in VRAM yet
        setDot('dot-gpu', 'warn');
        if (gpuLbl) gpuLbl.textContent = `GPU · ready`;
      }
      const chip = document.getElementById('gpu-chip');
      if (chip) chip.title = `${g.name} — ${g.memory_used_mb} / ${g.memory_total_mb} MB · ${g.utilization_pct}% util`;
    } else {
      if (gpuChip) gpuChip.style.display = 'none';
    }

    // ── Sidebar pills ──────────────────────────────────────────────────
    const statusPill = document.getElementById('sidebar-ollama-status');
    if (statusPill) {
      statusPill.style.color = running ? 'var(--cldr-teal)' : 'var(--red)';
      const gpuAvail = gpu.available && gpu.gpus && gpu.gpus.length > 0;
      statusPill.textContent = running
        ? (data.gpu_in_use ? '● Ollama · GPU active' : (gpuAvail ? '● Ollama · GPU ready' : '● Ollama · CPU'))
        : '○ Ollama';
    }

    // ── Config-dot nav warning ─────────────────────────────────────────
    const configDot = document.getElementById('config-dot');
    if (configDot) configDot.className = `nav-dot ${running ? 'success' : 'warning'}`;
  } catch (_) {}
}

function setDot(id, status) {
  const el = document.getElementById(id);
  if (!el) return;
  el.className = 'status-dot';
  if (status) el.classList.add(status);
}

function updateSidebarPills() {
  const namePill = document.getElementById('sidebar-model-name');
  if (namePill) namePill.textContent = state.config.local_model || 'llama3.2-vision:11b';
}

// ---------------------------------------------------------------------------
// Images
// ---------------------------------------------------------------------------

async function refreshImages() {
  try {
    const data     = await api('GET', '/api/images');
    state.uploads  = data.uploads  || [];
    state.examples = data.examples || [];
  } catch (e) {
    console.error('refreshImages', e);
  }
}

function formatBytes(b) {
  if (b < 1024)    return b + ' B';
  if (b < 1048576) return (b / 1024).toFixed(1) + ' KB';
  return (b / 1048576).toFixed(1) + ' MB';
}

// ---------------------------------------------------------------------------
// Single analysis
// ---------------------------------------------------------------------------

function onSingleUseCaseChange() {
  const uc = document.getElementById('single-use-case').value;
  document.getElementById('single-use-case-hint').textContent = USE_CASE_HINTS[uc] || '';
  document.getElementById('single-question-group').style.display =
    uc === 'Complicated Document QA' ? 'block' : 'none';
}

function onSingleSrcChange() { refreshSingleImageSelect(); }

function refreshSingleImageSelect() {
  const src    = document.querySelector('input[name="single-src"]:checked')?.value || 'uploads';
  const images = src === 'examples' ? state.examples : state.uploads;
  const sel    = document.getElementById('single-image-select');
  sel.innerHTML = images.length
    ? images.map(i => `<option value="${i.name}">${i.name}</option>`).join('')
    : '<option value="">No images available</option>';
  updateSinglePreview();
}

function updateSinglePreview() {
  const src  = document.querySelector('input[name="single-src"]:checked')?.value || 'uploads';
  const name = document.getElementById('single-image-select').value;
  const wrap = document.getElementById('single-preview-wrap');
  const img  = document.getElementById('single-preview');
  if (!name) { wrap.style.display = 'none'; return; }
  wrap.style.display = 'block';
  img.src = src === 'examples' ? `/examples/${name}` : `/images/${name}`;
}

document.addEventListener('DOMContentLoaded', () => {
  const sel = document.getElementById('single-image-select');
  if (sel) sel.addEventListener('change', updateSinglePreview);
  const batchUC = document.getElementById('batch-use-case');
  if (batchUC) batchUC.addEventListener('change', () => {
    document.getElementById('batch-question-group').style.display =
      batchUC.value === 'Complicated Document QA' ? 'block' : 'none';
  });
});

// Pipeline stage HTML builder
function _buildPipelineHTML(stages, activeIdx, doneSet = new Set()) {
  const checkIcon = `<svg viewBox="0 0 20 20" fill="currentColor"><path fill-rule="evenodd" d="M16.704 4.153a.75.75 0 01.143 1.052l-8 10.5a.75.75 0 01-1.127.075l-4.5-4.5a.75.75 0 011.06-1.06l3.894 3.893 7.48-9.817a.75.75 0 011.05-.143z"/></svg>`;
  const clockIcon = `<svg viewBox="0 0 20 20" fill="currentColor"><path fill-rule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm.75-13a.75.75 0 00-1.5 0v5c0 .414.336.75.75.75h4a.75.75 0 000-1.5h-3.25V5z"/></svg>`;

  const items = stages.map((s, i) => {
    const done   = doneSet.has(i);
    const active = i === activeIdx && !done;
    const status = done ? 'done' : active ? 'active' : 'pending';
    const icon   = done   ? checkIcon
                 : active ? `<span class="spinner" style="width:15px;height:15px;border-width:2px"></span>`
                 : clockIcon;
    return `<div class="pipeline-stage ${status}">
      <div class="stage-icon">${icon}</div>
      <div class="stage-body">
        <div class="stage-label">${escHtml(s.name)}</div>
        <div class="stage-sublabel">${escHtml(s.desc)}</div>
      </div>
    </div>`;
  });

  const withArrows = [];
  items.forEach((item, i) => {
    withArrows.push(item);
    if (i < items.length - 1) withArrows.push(`<div class="pipeline-arrow">→</div>`);
  });
  return `<div class="pipeline-stages">${withArrows.join('')}</div>
    <p class="pipeline-note">Running in background — you can switch sections freely</p>`;
}

async function runSingleProcess() {
  const uc       = document.getElementById('single-use-case').value;
  const question = document.getElementById('single-question')?.value || '';
  const src      = document.querySelector('input[name="single-src"]:checked')?.value || 'uploads';
  const filename = document.getElementById('single-image-select').value;

  if (!filename) { toast('Please select an image.', 'error'); return; }

  const btn  = document.getElementById('single-process-btn');
  const body = document.getElementById('single-result-body');

  btn.disabled = true;
  btn.innerHTML = '<span class="spinner"></span> Processing…';

  const model  = state.config.local_model || 'llava';
  const stages = [{ name: 'Vision LLM', desc: model }];
  body.innerHTML = _buildPipelineHTML(stages, 0);
  setActive('processing', true);

  try {
    const result = await api('POST', '/api/process', { filename, use_case: uc, question, source: src });
    renderSingleResult(result);
    document.getElementById('single-copy-btn').style.display = 'inline-flex';
    document.getElementById('single-copy-btn').dataset.text = result.result;
    toast('Analysis complete', 'success');
  } catch (e) {
    body.innerHTML = `<div class="empty-state"><p style="color:var(--red)">Error: ${escHtml(e.message)}</p></div>`;
    toast(e.message, 'error');
  } finally {
    setActive('processing', false);
    btn.disabled = false;
    btn.innerHTML = `<svg viewBox="0 0 20 20" fill="currentColor"><path fill-rule="evenodd" d="M2 10a8 8 0 1116 0 8 8 0 01-16 0zm6.39-2.908a.75.75 0 01.766.027l3.5 2.25a.75.75 0 010 1.262l-3.5 2.25A.75.75 0 018 12.25v-4.5a.75.75 0 01.39-.658z"/></svg> Process Image`;
  }
}

function renderSingleResult(result) {
  const body = document.getElementById('single-result-body');
  body.innerHTML = `
    <div class="result-section-label">Analysis Result</div>
    <pre class="result-pre">${escHtml(result.result)}</pre>`;
}

function copyResult() {
  const text = document.getElementById('single-copy-btn').dataset.text || '';
  navigator.clipboard.writeText(text).then(() => toast('Copied to clipboard', 'success'));
}

// ---------------------------------------------------------------------------
// File upload (XHR with % progress)
// ---------------------------------------------------------------------------

function uploadFiles(files, progressId = 'batch') {
  if (!files.length) return;

  const fd = new FormData();
  for (const f of files) fd.append('files', f);

  const wrap  = document.getElementById(`${progressId}-upload-progress`);
  const fill  = document.getElementById(`${progressId}-upload-fill`);
  const pct   = document.getElementById(`${progressId}-upload-pct`);
  const label = document.getElementById(`${progressId}-upload-label`);

  if (wrap)  wrap.style.display  = 'flex';
  if (fill)  fill.style.width    = '0%';
  if (pct)   pct.textContent     = '0%';
  if (label) label.textContent   = `Uploading ${files.length} file${files.length > 1 ? 's' : ''}…`;

  setActive('uploads', true);

  const xhr = new XMLHttpRequest();
  xhr.upload.addEventListener('progress', (e) => {
    if (e.lengthComputable) {
      const p = Math.round(e.loaded / e.total * 100);
      if (fill) fill.style.width = p + '%';
      if (pct)  pct.textContent  = p + '%';
    }
  });
  xhr.addEventListener('load', async () => {
    setActive('uploads', false);
    if (wrap) wrap.style.display = 'none';
    if (xhr.status < 400) {
      await refreshImages();
      refreshBatchGrid();
      refreshImagesList();
      toast(`Uploaded ${files.length} file(s)`, 'success');
    } else {
      toast('Upload failed', 'error');
    }
  });
  xhr.addEventListener('error', () => {
    setActive('uploads', false);
    if (wrap) wrap.style.display = 'none';
    toast('Upload failed (network error)', 'error');
  });
  xhr.open('POST', '/api/images/upload');
  xhr.send(fd);
}

function onBatchDrop(e)      { e.preventDefault(); uploadFiles([...e.dataTransfer.files], 'batch'); }
function onBatchFileInput(e) { uploadFiles([...e.target.files], 'batch');  e.target.value = ''; }
function onImagesDrop(e)     { e.preventDefault(); uploadFiles([...e.dataTransfer.files], 'images'); }
function onImagesFileInput(e){ uploadFiles([...e.target.files], 'images'); e.target.value = ''; }

// ---------------------------------------------------------------------------
// Batch image grid
// ---------------------------------------------------------------------------

function refreshBatchGrid() {
  const grid = document.getElementById('batch-image-grid');
  if (!state.uploads.length) {
    grid.innerHTML = `<p style="color:var(--tx-3);font-size:13px;grid-column:1/-1">No uploaded images yet. Use the upload zone above.</p>`;
    return;
  }
  grid.innerHTML = state.uploads.map(img => `
    <div class="image-thumb ${state.batchSelected.has(img.name) ? 'selected' : ''}"
         onclick="toggleBatchSelect('${escAttr(img.name)}')" title="${escAttr(img.name)}">
      <img src="/images/${escAttr(img.name)}" loading="lazy" />
      <span class="thumb-name">${escHtml(img.name)}</span>
      <span class="thumb-check">
        <svg viewBox="0 0 20 20" fill="currentColor"><path fill-rule="evenodd" d="M16.704 4.153a.75.75 0 01.143 1.052l-8 10.5a.75.75 0 01-1.127.075l-4.5-4.5a.75.75 0 011.06-1.06l3.894 3.893 7.48-9.817a.75.75 0 011.05-.143z"/></svg>
      </span>
    </div>
  `).join('');
  updateBatchCount();
}

function toggleBatchSelect(name) {
  state.batchSelected.has(name) ? state.batchSelected.delete(name) : state.batchSelected.add(name);
  refreshBatchGrid();
}

function selectAllBatch()    { state.uploads.forEach(i => state.batchSelected.add(i.name)); refreshBatchGrid(); }
function clearBatchSelection(){ state.batchSelected.clear(); refreshBatchGrid(); }

function updateBatchCount() {
  const n     = state.batchSelected.size;
  document.getElementById('batch-selected-count').textContent = n;
  document.getElementById('batch-process-btn').disabled = n === 0;
  const badge = document.getElementById('nav-badge-batch');
  if (n > 0) { badge.textContent = n; badge.style.display = 'inline-block'; }
  else badge.style.display = 'none';
}

// ---------------------------------------------------------------------------
// Batch — run
// ---------------------------------------------------------------------------

async function runBatch() {
  const filenames = [...state.batchSelected];
  if (!filenames.length) { toast('Select at least one image.', 'error'); return; }

  try {
    await api('POST', '/api/batch', {
      filenames,
      use_case: document.getElementById('batch-use-case').value,
      question: document.getElementById('batch-question')?.value || '',
    });
    state.batchSelected.clear();
    refreshBatchGrid();
    toast(`${filenames.length} job(s) queued`, 'success');
    refreshJobList();
    updateResultsBadge();
  } catch (e) {
    toast('Batch error: ' + e.message, 'error');
  }
}

// ---------------------------------------------------------------------------
// Job polling (always-on)
// ---------------------------------------------------------------------------

let _notifiedJobIds = new Set();

async function pollJobs() {
  try {
    const data    = await api('GET', '/api/jobs');
    const newJobs = data.jobs || [];

    // Toast on newly completed jobs regardless of current section
    const nowComplete = new Set(newJobs.filter(j => j.status === 'complete').map(j => j.id));
    if (_notifiedJobIds.size > 0) {
      for (const id of nowComplete) {
        if (!_notifiedJobIds.has(id)) {
          const job = newJobs.find(j => j.id === id);
          if (job) toast(`✓ ${job.filename} — complete`, 'success', 4000);
        }
      }
    }
    _notifiedJobIds = nowComplete;

    const running = newJobs.filter(j => j.status === 'processing' || j.status === 'queued').length;
    setActive('batch', running);

    state.jobs = newJobs;
    refreshJobList();
    refreshResultsList();
    updateResultsBadge();
  } catch (_) {}
}

function updateResultsBadge() {
  const complete = state.jobs.filter(j => j.status === 'complete').length;
  const badge    = document.getElementById('nav-badge-results');
  if (complete > 0) { badge.textContent = complete; badge.style.display = 'inline-block'; }
  else badge.style.display = 'none';
}

// ---------------------------------------------------------------------------
// Job list
// ---------------------------------------------------------------------------

function statusIcon(status) {
  const icons = {
    queued:     `<svg viewBox="0 0 20 20" fill="currentColor"><path fill-rule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm.75-13a.75.75 0 00-1.5 0v5c0 .414.336.75.75.75h4a.75.75 0 000-1.5h-3.25V5z"/></svg>`,
    processing: `<span class="spinner"></span>`,
    complete:   `<svg viewBox="0 0 20 20" fill="currentColor"><path fill-rule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.857-9.809a.75.75 0 00-1.214-.882l-3.483 4.79-1.88-1.88a.75.75 0 10-1.06 1.061l2.5 2.5a.75.75 0 001.137-.089l4-5.5z"/></svg>`,
    error:      `<svg viewBox="0 0 20 20" fill="currentColor"><path fill-rule="evenodd" d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-8-5a.75.75 0 01.75.75v4.5a.75.75 0 01-1.5 0v-4.5A.75.75 0 0110 5zm0 10a1 1 0 100-2 1 1 0 000 2z"/></svg>`,
  };
  return icons[status] || '';
}

function relTime(iso) {
  if (!iso) return '';
  const s = Math.floor((Date.now() - new Date(iso + 'Z').getTime()) / 1000);
  if (s < 60)   return `${s}s ago`;
  if (s < 3600) return `${Math.floor(s/60)}m ago`;
  return `${Math.floor(s/3600)}h ago`;
}

function refreshJobList() {
  const el = document.getElementById('batch-job-list');
  if (!el) return;
  if (!state.jobs.length) {
    el.innerHTML = `<div class="empty-state"><p>No jobs yet. Select images above and click <strong>Process Selected</strong>.</p></div>`;
    return;
  }
  const sorted = [...state.jobs].sort((a,b) => b.created_at.localeCompare(a.created_at));
  el.innerHTML = sorted.map(job => `
    <div class="job-item">
      <div class="job-status-icon ${job.status}">${statusIcon(job.status)}</div>
      <div class="job-info">
        <div class="job-filename">${escHtml(job.filename)}</div>
        <div class="job-meta">${escHtml(job.use_case)} · ${relTime(job.created_at)}</div>
      </div>
      <span class="job-tag ${job.status}">${job.status}</span>
      ${job.status === 'complete' ? `<button class="btn btn-sm btn-ghost" onclick="openResultModal('${job.id}')">View</button>` : ''}
      ${job.status === 'error' ? `<span title="${escAttr(job.error)}" style="color:var(--red);font-size:11px;max-width:120px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap">${escHtml(job.error||'')}</span>` : ''}
    </div>
  `).join('');
}

async function clearCompletedJobs() {
  try {
    const data = await api('DELETE', '/api/jobs');
    toast(`Cleared ${data.cleared} completed job(s)`, 'success');
    await pollJobs();
  } catch (e) {
    toast(e.message, 'error');
  }
}

// ---------------------------------------------------------------------------
// Results tab
// ---------------------------------------------------------------------------

function refreshResultsList() {
  const el = document.getElementById('results-list');
  if (!el) return;
  const done = state.jobs.filter(j => j.status === 'complete' || j.status === 'error');
  if (!done.length) {
    el.innerHTML = `<div class="empty-state"><p>Results from processed images will appear here.</p></div>`;
    return;
  }
  const sorted = [...done].sort((a,b) => (b.completed_at||'').localeCompare(a.completed_at||''));
  el.innerHTML = sorted.map(job => `
    <div class="result-card" id="rc-${job.id}">
      <div class="result-card-header" onclick="toggleResultCard('${job.id}')">
        <span class="job-tag ${job.status}" style="margin-right:8px">${job.status}</span>
        <strong style="font-size:13px;flex:1">${escHtml(job.filename)}</strong>
        <span style="font-size:11px;color:var(--tx-3);margin:0 12px">${escHtml(job.use_case)}</span>
        <span style="font-size:11px;color:var(--tx-3)">${relTime(job.completed_at)}</span>
        <svg viewBox="0 0 20 20" fill="currentColor" style="width:14px;margin-left:10px;color:var(--tx-3)"><path fill-rule="evenodd" d="M5.23 7.21a.75.75 0 011.06.02L10 11.168l3.71-3.938a.75.75 0 111.08 1.04l-4.25 4.5a.75.75 0 01-1.08 0l-4.25-4.5a.75.75 0 01.02-1.06z"/></svg>
      </div>
      <div class="result-card-body">
        ${job.status === 'error'
          ? `<p style="color:var(--red);font-size:13px">${escHtml(job.error||'')}</p>`
          : `<div class="result-section-label">Analysis Result</div>
             <pre class="result-pre">${escHtml(job.result||'')}</pre>
             <button class="btn btn-sm btn-ghost" onclick="navigator.clipboard.writeText(${JSON.stringify(job.result||'')}).then(()=>toast('Copied','success'))">Copy Result</button>`
        }
      </div>
    </div>
  `).join('');
}

function toggleResultCard(id) {
  document.getElementById(`rc-${id}`)?.classList.toggle('open');
}

// ---------------------------------------------------------------------------
// Result modal
// ---------------------------------------------------------------------------

function openResultModal(jobId) {
  const job = state.jobs.find(j => j.id === jobId);
  if (!job) return;

  document.getElementById('modal-title').textContent    = job.filename;
  document.getElementById('modal-subtitle').textContent = `${job.use_case} · ${relTime(job.completed_at)}`;

  const imgEl = document.getElementById('modal-image');
  imgEl.src = `/images/${job.filename}`;
  imgEl.onerror = () => { imgEl.style.display = 'none'; };
  imgEl.style.display = 'block';

  document.getElementById('modal-result').textContent = job.result || '';
  document.getElementById('result-modal').classList.add('open');
}

function closeModal(e) {
  if (e.target.id === 'result-modal')
    document.getElementById('result-modal').classList.remove('open');
}

function copyModalResult() {
  navigator.clipboard.writeText(document.getElementById('modal-result').textContent)
    .then(() => toast('Copied to clipboard', 'success'));
}

// ---------------------------------------------------------------------------
// Images tab
// ---------------------------------------------------------------------------

async function refreshImagesList() {
  await refreshImages();
  const el = document.getElementById('images-upload-list');
  if (!el) return;
  if (!state.uploads.length) {
    el.innerHTML = `<div class="empty-state"><p>No images uploaded yet. Use the upload zone above.</p></div>`;
    return;
  }
  el.innerHTML = state.uploads.map(img => `
    <div class="img-list-row">
      <img class="img-list-thumb" src="/images/${escAttr(img.name)}" onclick="previewImage('/images/${escAttr(img.name)}')" loading="lazy" />
      <span class="img-list-name" title="${escAttr(img.name)}">${escHtml(img.name)}</span>
      <span class="img-list-size">${formatBytes(img.size)}</span>
      <button class="btn btn-sm btn-ghost" onclick="previewImage('/images/${escAttr(img.name)}')">View</button>
      <button class="btn btn-sm btn-danger" onclick="deleteImage('${escAttr(img.name)}')">Delete</button>
    </div>
  `).join('');
}

function previewImage(src) {
  document.getElementById('images-preview-area').innerHTML =
    `<img src="${src}" style="max-width:100%;max-height:400px;object-fit:contain;display:block;border-radius:var(--radius);border:1px solid var(--border)" />`;
}

async function deleteImage(name) {
  if (!confirm(`Delete "${name}"?`)) return;
  try {
    await api('DELETE', `/api/images/${encodeURIComponent(name)}`);
    await refreshImages();
    refreshImagesList();
    refreshBatchGrid();
    refreshSingleImageSelect();
    toast(`Deleted ${name}`, 'success');
  } catch (e) {
    toast('Delete failed: ' + e.message, 'error');
  }
}

// ---------------------------------------------------------------------------
// Configuration tab
// ---------------------------------------------------------------------------

async function loadConfigForm() {
  try {
    state.config = await api('GET', '/api/config');

    // Populate model dropdown
    const catalog  = state.config.local_model_catalog || {};
    const modelSel = document.getElementById('cfg-local-model');
    modelSel.innerHTML = Object.entries(catalog).map(([label, id]) =>
      `<option value="${escAttr(id)}" ${id === state.config.local_model ? 'selected' : ''}>${escHtml(label)}</option>`
    ).join('');

    updateSidebarPills();
    refreshOllamaStatus();
    if (!_ollamaStatusTimer) _ollamaStatusTimer = setInterval(refreshOllamaStatus, 4000);
  } catch (e) {
    toast('Failed to load config', 'error');
  }
}

// Config-section detailed Ollama timer (runs only while config is open)
let _ollamaStatusTimer = null;

// Override navigate to stop the config-section timer when leaving
const _origNavigate = navigate;
// (navigate is already defined above; we patch it after DOMContentLoaded)

async function saveConfig() {
  const model = document.getElementById('cfg-local-model')?.value || 'llava:7b';
  try {
    await api('POST', '/api/config', { local_model: model, max_tokens: 4096 });
    state.config = await api('GET', '/api/config');
    updateSidebarPills();
    toast('Configuration saved', 'success');
  } catch (e) {
    toast('Save failed: ' + e.message, 'error');
  }
}

// ---------------------------------------------------------------------------
// Ollama management (config section detail view)
// ---------------------------------------------------------------------------

let _wasPulling = false;

async function refreshOllamaStatus() {
  try {
    const data = await api('GET', '/api/ollama/status');

    setDot('ollama-dot-installed', data.installed ? 'ok' : 'error');
    document.getElementById('ollama-label-installed').textContent =
      data.installed ? 'Ollama installed' : 'Ollama not installed';
    document.getElementById('ollama-install-hint').style.display = data.installed ? 'none' : 'block';

    setDot('ollama-dot-running', data.running ? 'ok' : 'error');
    document.getElementById('ollama-label-running').textContent =
      data.running ? 'Server running' : 'Server not running';
    document.getElementById('ollama-start-btn').style.display =
      data.installed && !data.running ? 'inline-flex' : 'none';

    // Pull progress
    const pull       = data.pull || {};
    const pullSec    = document.getElementById('pull-progress');
    const pullFill   = document.getElementById('pull-fill');
    const pullLabel  = document.getElementById('pull-label');
    const pullBtn    = document.getElementById('pull-model-btn');

    setActive('pull', pull.running);

    if (pull.running) {
      pullSec.style.display = 'block';
      pullFill.classList.add('indeterminate');
      pullLabel.textContent = `Pulling ${pull.model} — this may take several minutes…`;
      if (pullBtn) pullBtn.disabled = true;
      _wasPulling = true;
      clearInterval(_ollamaStatusTimer);
      _ollamaStatusTimer = setInterval(refreshOllamaStatus, 1500);
    } else if (pull.error) {
      pullSec.style.display = 'block';
      pullFill.classList.remove('indeterminate');
      pullFill.style.width = '0';
      pullLabel.style.color = 'var(--red)';
      pullLabel.textContent = `Error: ${pull.error}`;
      if (pullBtn) pullBtn.disabled = false;
      _wasPulling = false;
      clearInterval(_ollamaStatusTimer);
      _ollamaStatusTimer = setInterval(refreshOllamaStatus, 4000);
    } else {
      if (_wasPulling) {
        pullSec.style.display = 'block';
        pullFill.classList.remove('indeterminate');
        pullFill.style.width = '100%';
        pullLabel.style.color = 'var(--green)';
        pullLabel.textContent = '✓ Model pulled successfully';
        setTimeout(() => { pullSec.style.display = 'none'; pullFill.style.width = '0'; pullLabel.style.color = ''; }, 4000);
        toast('Model pulled successfully', 'success');
      } else {
        pullSec.style.display = 'none';
      }
      if (pullBtn) pullBtn.disabled = false;
      _wasPulling = false;
      clearInterval(_ollamaStatusTimer);
      _ollamaStatusTimer = setInterval(refreshOllamaStatus, 4000);
    }

    renderPulledModels(data.models || []);
    renderGpuStatus(data);

    // Model availability badge
    const localModel = document.getElementById('cfg-local-model')?.value || '';
    const avail = (data.models || []).some(
      m => m === localModel || m.startsWith(localModel.split(':')[0])
    );
    const badge = document.getElementById('local-model-status');
    if (data.running && avail) {
      badge.textContent = '✓ Model ready';
      badge.className = 'model-avail-badge ready';
    } else if (data.running && !avail) {
      badge.textContent = '⬇ Not pulled yet';
      badge.className = 'model-avail-badge missing';
    } else {
      badge.textContent = '';
      badge.className = 'model-avail-badge';
    }
  } catch (e) {
    console.warn('Ollama status error', e);
  }
}

function renderGpuStatus(data) {
  const section = document.getElementById('gpu-status-section');
  const badge   = document.getElementById('gpu-in-use-badge');
  const cards   = document.getElementById('gpu-cards');
  if (!section) return;

  const gpu = data.gpu || {};

  if (!gpu.available || !gpu.gpus || gpu.gpus.length === 0) {
    section.style.display = 'none';
    return;
  }

  section.style.display = 'block';

  // Three-state GPU badge: active (model in VRAM) / ready (GPU present, no model) / CPU
  if (badge) {
    badge.style.display = 'inline-block';
    if (data.gpu_in_use) {
      badge.textContent = '● GPU active — model in VRAM';
      badge.className = 'gpu-in-use-badge active';
    } else {
      badge.textContent = '● GPU ready — will load on first inference';
      badge.className = 'gpu-in-use-badge ready';
    }
  }

  // One card per GPU (most setups have 1)
  if (cards) {
    const loaded = data.loaded || [];
    cards.innerHTML = gpu.gpus.map((g, i) => {
      const vramPct    = Math.round(g.memory_used_mb / g.memory_total_mb * 100);
      const activeClass = data.gpu_in_use ? 'active' : 'ready';

      // Find if any model is loaded on this GPU and show its VRAM allocation
      const modelVram = loaded
        .filter(m => m.size_vram > 0)
        .map(m => `${escHtml(m.name)} (${(m.size_vram / 1073741824).toFixed(1)} GB)`)
        .join(', ');

      return `<div class="gpu-card ${activeClass}">
        <div class="gpu-card-icon">GPU${i}</div>
        <div class="gpu-card-body">
          <div class="gpu-card-name">${escHtml(g.name)}</div>
          <div class="gpu-card-meta">
            Driver ${escHtml(g.driver_version)} ·
            ${g.utilization_pct}% util
            ${modelVram
              ? `· <strong style="color:var(--cldr-teal)">${modelVram} in VRAM</strong>`
              : `· <span style="color:var(--tx-3);font-style:italic">Model loads into VRAM on first inference</span>`}
          </div>
        </div>
        <div class="gpu-vram-bar-wrap">
          <div class="gpu-vram-bar">
            <div class="gpu-vram-fill" style="width:${vramPct}%"></div>
          </div>
          <span class="gpu-vram-label">${g.memory_used_mb.toLocaleString()} / ${g.memory_total_mb.toLocaleString()} MB</span>
        </div>
      </div>`;
    }).join('');
  }
}

function renderPulledModels(models) {
  const el = document.getElementById('pulled-models-list');
  if (!models.length) {
    el.innerHTML = `<p class="form-hint">No models pulled yet.</p>`;
    return;
  }
  el.innerHTML = `<div class="result-section-label" style="margin-bottom:8px">Pulled Models</div>` +
    models.map(m => `<div class="pulled-model-row"><span class="model-pill" style="font-size:12px">${escHtml(m)}</span></div>`).join('');
}

async function startOllama() {
  const btn = document.getElementById('ollama-start-btn');
  btn.disabled = true; btn.textContent = 'Starting…';
  try {
    const r = await api('POST', '/api/ollama/start');
    toast(r.message, 'success');
    setTimeout(refreshOllamaStatus, 2500);
  } catch (e) {
    toast(e.message, 'error');
  } finally {
    btn.disabled = false; btn.textContent = 'Start Ollama';
  }
}

async function pullModel() {
  const model = document.getElementById('cfg-local-model')?.value;
  if (!model) { toast('Select a model first.', 'error'); return; }

  const btn      = document.getElementById('pull-model-btn');
  const pullSec  = document.getElementById('pull-progress');
  const pullFill = document.getElementById('pull-fill');
  const pullLabel= document.getElementById('pull-label');

  btn.disabled = true;
  pullSec.style.display = 'block';
  pullFill.classList.add('indeterminate');
  pullLabel.style.color = '';
  pullLabel.textContent = `Starting pull for ${model}…`;
  _wasPulling = true;

  try {
    const r = await api('POST', '/api/ollama/pull', { model });
    if (!r.ok) {
      toast(r.message, 'error');
      pullSec.style.display = 'none';
      pullFill.classList.remove('indeterminate');
      btn.disabled = false;
      _wasPulling = false;
      return;
    }
    toast(`Pulling ${model} — see progress below`, 'success');
    clearInterval(_ollamaStatusTimer);
    _ollamaStatusTimer = setInterval(refreshOllamaStatus, 1500);
  } catch (e) {
    toast('Pull failed: ' + e.message, 'error');
    pullSec.style.display = 'none';
    pullFill.classList.remove('indeterminate');
    btn.disabled = false;
    _wasPulling = false;
  }
}

// Stop config-section timer when navigating away
document.addEventListener('DOMContentLoaded', () => {
  // Patch navigate to manage the config-section timer
  const _origNav = navigate;
  window.navigate = function(section) {
    if (section !== 'config') {
      clearInterval(_ollamaStatusTimer);
      _ollamaStatusTimer = null;
    }
    _origNav(section);
  };
});

// ---------------------------------------------------------------------------
// Utilities
// ---------------------------------------------------------------------------

function escHtml(str) {
  return String(str ?? '')
    .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;').replace(/'/g, '&#039;');
}

function escAttr(str) {
  return String(str ?? '').replace(/"/g, '&quot;').replace(/'/g, '&#039;');
}

// ---------------------------------------------------------------------------
// Boot
// ---------------------------------------------------------------------------

window.addEventListener('DOMContentLoaded', init);
