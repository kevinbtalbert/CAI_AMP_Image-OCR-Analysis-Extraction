'use strict';

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------

const state = {
  config: {},
  uploads: [],
  examples: [],
  jobs: [],
  batchSelected: new Set(),
  pollTimer: null,
};

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

// ---------------------------------------------------------------------------
// Navigation
// ---------------------------------------------------------------------------

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
    updateStatusChips();
    updateConfigDot();
  } catch (e) {
    console.error('Init error', e);
  }

  await refreshImages();
  onSingleUseCaseChange();
  refreshSingleImageSelect();

  // Poll jobs every 2 s
  state.pollTimer = setInterval(pollJobs, 2000);

  navigate('analysis');
}

// ---------------------------------------------------------------------------
// Status chips
// ---------------------------------------------------------------------------

function updateStatusChips() {
  const cfg = state.config;
  const hasToken = cfg.has_token;
  const hasOcr   = !!cfg.ocr_endpoint_url;
  const hasLlm   = !!cfg.llm_endpoint_url;

  setDot('dot-ocr', hasToken && hasOcr ? 'ok' : 'error');
  setDot('dot-llm', hasToken && hasLlm ? 'ok' : 'error');
}

function setDot(id, status) {
  const el = document.getElementById(id);
  el.className = 'status-dot';
  if (status) el.classList.add(status);
}

function updateConfigDot() {
  const cfg = state.config;
  const ok = cfg.has_token && cfg.ocr_endpoint_url && cfg.llm_endpoint_url;
  const dot = document.getElementById('config-dot');
  dot.className = `nav-dot ${ok ? 'success' : 'warning'}`;
}

// ---------------------------------------------------------------------------
// Images
// ---------------------------------------------------------------------------

async function refreshImages() {
  try {
    const data = await api('GET', '/api/images');
    state.uploads  = data.uploads  || [];
    state.examples = data.examples || [];
  } catch (e) {
    console.error('refreshImages', e);
  }
}

function formatBytes(b) {
  if (b < 1024)       return b + ' B';
  if (b < 1048576)    return (b / 1024).toFixed(1) + ' KB';
  return (b / 1048576).toFixed(1) + ' MB';
}

// ---------------------------------------------------------------------------
// Single analysis
// ---------------------------------------------------------------------------

function onSingleUseCaseChange() {
  const uc = document.getElementById('single-use-case').value;
  document.getElementById('single-use-case-hint').textContent = USE_CASE_HINTS[uc] || '';
  const isQA = uc === 'Complicated Document QA';
  document.getElementById('single-question-group').style.display = isQA ? 'block' : 'none';
}

function onSingleSrcChange() {
  refreshSingleImageSelect();
}

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
    const isQA = batchUC.value === 'Complicated Document QA';
    document.getElementById('batch-question-group').style.display = isQA ? 'block' : 'none';
  });
});

async function runSingleProcess() {
  const uc       = document.getElementById('single-use-case').value;
  const question = document.getElementById('single-question')?.value || '';
  const src      = document.querySelector('input[name="single-src"]:checked')?.value || 'uploads';
  const filename = document.getElementById('single-image-select').value;

  if (!filename) { toast('Please select an image.', 'error'); return; }

  const btn = document.getElementById('single-process-btn');
  const body = document.getElementById('single-result-body');

  btn.disabled = true;
  btn.innerHTML = '<span class="spinner"></span> Processing…';
  body.innerHTML = `<div class="empty-state"><span class="spinner" style="width:32px;height:32px;border-width:3px"></span><p style="margin-top:12px">Running OCR → LLM pipeline…</p></div>`;

  try {
    const result = await api('POST', '/api/process', { filename, use_case: uc, question, source: src });
    renderSingleResult(result);
    document.getElementById('single-copy-btn').style.display = 'inline-flex';
    document.getElementById('single-copy-btn').dataset.text = result.result;
  } catch (e) {
    body.innerHTML = `<div class="empty-state"><p style="color:var(--red)">Error: ${escHtml(e.message)}</p></div>`;
    toast(e.message, 'error');
  } finally {
    btn.disabled = false;
    btn.innerHTML = `<svg viewBox="0 0 20 20" fill="currentColor"><path fill-rule="evenodd" d="M2 10a8 8 0 1116 0 8 8 0 01-16 0zm6.39-2.908a.75.75 0 01.766.027l3.5 2.25a.75.75 0 010 1.262l-3.5 2.25A.75.75 0 018 12.25v-4.5a.75.75 0 01.39-.658z"/></svg> Process Image`;
  }
}

function renderSingleResult(result) {
  const body = document.getElementById('single-result-body');
  let html = '';

  if (result.ocr_text) {
    html += `<details style="margin-bottom:12px">
      <summary class="result-section-label" style="cursor:pointer;user-select:none">📝 OCR Extracted Text (Stage 1)</summary>
      <pre class="result-pre" style="margin-top:6px">${escHtml(result.ocr_text)}</pre>
    </details>`;
  }
  html += `<div class="result-section-label">🤖 LLM Analysis</div>`;
  html += `<pre class="result-pre">${escHtml(result.result)}</pre>`;
  body.innerHTML = html;
}

function copyResult() {
  const btn  = document.getElementById('single-copy-btn');
  const text = btn.dataset.text || '';
  navigator.clipboard.writeText(text).then(() => toast('Copied to clipboard', 'success'));
}

// ---------------------------------------------------------------------------
// Batch — file upload
// ---------------------------------------------------------------------------

async function uploadFiles(files) {
  if (!files.length) return;
  const fd = new FormData();
  for (const f of files) fd.append('files', f);
  try {
    await api('POST', '/api/images/upload', fd);
    await refreshImages();
    refreshBatchGrid();
    refreshImagesList();
    toast(`Uploaded ${files.length} file(s)`, 'success');
  } catch (e) {
    toast('Upload failed: ' + e.message, 'error');
  }
}

function onBatchDrop(e) {
  e.preventDefault();
  uploadFiles([...e.dataTransfer.files]);
}

function onBatchFileInput(e) {
  uploadFiles([...e.target.files]);
  e.target.value = '';
}

function onImagesDrop(e) {
  e.preventDefault();
  uploadFiles([...e.dataTransfer.files]);
}

function onImagesFileInput(e) {
  uploadFiles([...e.target.files]);
  e.target.value = '';
}

// ---------------------------------------------------------------------------
// Batch — image grid
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
  if (state.batchSelected.has(name)) {
    state.batchSelected.delete(name);
  } else {
    state.batchSelected.add(name);
  }
  refreshBatchGrid();
}

function selectAllBatch() {
  state.uploads.forEach(i => state.batchSelected.add(i.name));
  refreshBatchGrid();
}

function clearBatchSelection() {
  state.batchSelected.clear();
  refreshBatchGrid();
}

function updateBatchCount() {
  const n = state.batchSelected.size;
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

  const uc       = document.getElementById('batch-use-case').value;
  const question = document.getElementById('batch-question')?.value || '';

  try {
    await api('POST', '/api/batch', { filenames, use_case: uc, question });
    state.batchSelected.clear();
    refreshBatchGrid();
    toast(`${filenames.length} job(s) queued`, 'success');
    refreshJobList();
    // Show badge on results nav
    updateResultsBadge();
  } catch (e) {
    toast('Batch error: ' + e.message, 'error');
  }
}

// ---------------------------------------------------------------------------
// Job polling
// ---------------------------------------------------------------------------

async function pollJobs() {
  try {
    const data = await api('GET', '/api/jobs');
    state.jobs = data.jobs || [];
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
// Job list (batch tab)
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
  const diff = Date.now() - new Date(iso + 'Z').getTime();
  const s = Math.floor(diff / 1000);
  if (s < 60)  return `${s}s ago`;
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
          : `
            ${job.ocr_text ? `<div class="result-section-label">OCR Extracted Text</div><pre class="result-pre">${escHtml(job.ocr_text)}</pre>` : ''}
            <div class="result-section-label">LLM Analysis</div>
            <pre class="result-pre">${escHtml(job.result||'')}</pre>
            <button class="btn btn-sm btn-ghost" onclick="navigator.clipboard.writeText(${JSON.stringify(job.result||'')}).then(()=>toast('Copied','success'))">Copy Result</button>
          `
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

  document.getElementById('modal-title').textContent = job.filename;
  document.getElementById('modal-subtitle').textContent = `${job.use_case} · ${relTime(job.completed_at)}`;

  const imgEl = document.getElementById('modal-image');
  imgEl.src = `/images/${job.filename}`;
  imgEl.onerror = () => { imgEl.style.display = 'none'; };
  imgEl.style.display = 'block';

  const ocrSection = document.getElementById('modal-ocr-section');
  if (job.ocr_text) {
    ocrSection.style.display = 'block';
    document.getElementById('modal-ocr').textContent = job.ocr_text;
  } else {
    ocrSection.style.display = 'none';
  }

  document.getElementById('modal-result').textContent = job.result || '';
  document.getElementById('result-modal').classList.add('open');
}

function closeModal(e) {
  if (e.target.id === 'result-modal') {
    document.getElementById('result-modal').classList.remove('open');
  }
}

function copyModalResult() {
  const text = document.getElementById('modal-result').textContent;
  navigator.clipboard.writeText(text).then(() => toast('Copied to clipboard', 'success'));
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
  const area = document.getElementById('images-preview-area');
  area.innerHTML = `<img src="${src}" style="max-width:100%;max-height:400px;object-fit:contain;display:block;border-radius:var(--radius);border:1px solid var(--border)" />`;
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
    // Don't pre-fill the token (it's write-only from UI)
    document.getElementById('cfg-ocr-url').value = state.config.ocr_endpoint_url || '';
    document.getElementById('cfg-llm-url').value = state.config.llm_endpoint_url || '';
    updateStatusChips();
    updateConfigDot();
  } catch (e) {
    toast('Failed to load config', 'error');
  }
}

async function saveConfig() {
  const token  = document.getElementById('cfg-token').value.trim();
  const ocrUrl = document.getElementById('cfg-ocr-url').value.trim();
  const llmUrl = document.getElementById('cfg-llm-url').value.trim();

  try {
    await api('POST', '/api/config', {
      inference_token: token,
      ocr_endpoint_url: ocrUrl,
      llm_endpoint_url: llmUrl,
      max_tokens: 4096,
    });
    state.config = await api('GET', '/api/config');
    updateStatusChips();
    updateConfigDot();
    toast('Configuration saved', 'success');
    // Clear token field after saving (security)
    document.getElementById('cfg-token').value = '';
    document.getElementById('cfg-token').placeholder = state.config.has_token ? '● Saved — paste new token to update' : 'Paste your CDP JWT token';
  } catch (e) {
    toast('Save failed: ' + e.message, 'error');
  }
}

async function testConnection(service) {
  const token  = document.getElementById('cfg-token').value.trim();
  const ocrUrl = document.getElementById('cfg-ocr-url').value.trim();
  const llmUrl = document.getElementById('cfg-llm-url').value.trim();

  const btn    = document.getElementById(`test-${service}-btn`);
  const status = document.getElementById(`conn-status-${service}`);

  btn.disabled = true;
  btn.textContent = '…';
  status.textContent = '';
  status.className = 'conn-status';

  try {
    const result = await api('POST', '/api/test-connection', {
      service,
      inference_token: token || undefined,
      ocr_endpoint_url: ocrUrl,
      llm_endpoint_url: llmUrl,
    });
    if (result.ok) {
      status.textContent = '✓ Connected';
      status.className = 'conn-status ok';
    } else {
      status.textContent = '✗ ' + result.message.slice(0, 60);
      status.className = 'conn-status error';
    }
  } catch (e) {
    status.textContent = '✗ ' + e.message.slice(0, 60);
    status.className = 'conn-status error';
  } finally {
    btn.disabled = false;
    btn.textContent = 'Test';
  }
}

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
