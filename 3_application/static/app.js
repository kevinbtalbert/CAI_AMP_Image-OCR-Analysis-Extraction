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
  results:  'Results',
  search:   'Search',
  files:    'Files',
  config:   'Configuration',
};

const USE_CASE_HINTS = {
  'Transcribing Typed Text':        'Verbatim OCR — outputs the exact printed characters, line breaks, and punctuation. Nothing added, nothing omitted.',
  'Transcribing Handwritten Text':  'Verbatim OCR of handwritten content — preserves original spelling, capitalisation, and line breaks as closely as possible.',
  'Transcribing Forms':             'Outputs every field label and its value verbatim, one per line (e.g. "First Name: John"). Blank fields are preserved.',
  'Complicated Document QA':        'Ask a specific question and receive an answer grounded in the document content.',
  'Unstructured Information → JSON':'Converts document content into structured, machine-readable JSON.',
  'Summarize Image':                'Describes and summarises the image — what it shows, its purpose, and key takeaways. Not a verbatim transcription.',
  'tpl:invoice':      '📄 Extracts vendor, invoice number, date, line items, subtotal, tax, and total amount.',
  'tpl:receipt':      '🧾 Extracts merchant name, date, itemised purchases, subtotals, tax, and payment method.',
  'tpl:business_card':'👤 Extracts name, title, company, email, phone, address, and website.',
  'tpl:purchase_order':'📦 Extracts PO number, vendor, buyer, ordered items, quantities, unit prices, and totals.',
  'tpl:medical_form': '🏥 Extracts patient name, DOB, provider, visit date, chief complaint, and notes.',
  'tpl:id_document':  '🪪 Extracts full name, date of birth, document number, issue date, and expiry date.',
};

// Template → use case mapping for backend
const TEMPLATE_USE_CASE = {
  'tpl:invoice':       'Unstructured Information → JSON',
  'tpl:receipt':       'Unstructured Information → JSON',
  'tpl:business_card': 'Unstructured Information → JSON',
  'tpl:purchase_order':'Unstructured Information → JSON',
  'tpl:medical_form':  'Transcribing Forms',
  'tpl:id_document':   'Transcribing Forms',
};

function navigate(section) {
  document.querySelectorAll('.section').forEach(s => s.classList.remove('active'));
  document.querySelectorAll('.nav-item').forEach(n => n.classList.remove('active'));
  document.getElementById(`section-${section}`).classList.add('active');
  document.querySelector(`[data-section="${section}"]`)?.classList.add('active');
  document.getElementById('page-title').textContent = PAGE_TITLES[section] || section;

  if (section === 'analysis') refreshSingleImageSelect();
  if (section === 'results')  refreshResultsList();
  if (section === 'search')   document.getElementById('search-input')?.focus();
  if (section === 'files')    loadFilesSection();
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

async function api(method, path, body, signal) {
  const opts = { method, headers: {} };
  if (signal) opts.signal = signal;
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

  loadUserInfo();       // personalise header (fire-and-forget)
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
    const warming = !!data.model_warming;

    setActive('pull', pulling);

    // ── Model warm-up banner on Analysis page ──────────────────────────
    _updateWarmingBanner(warming, data.gpu_in_use);

    // ── Ollama header chip ─────────────────────────────────────────────
    setDot('dot-ollama', running ? 'ok' : 'error');
    const lbl = document.getElementById('ollama-chip-label');
    if (lbl) lbl.textContent = running
      ? (warming ? `Ollama · loading model…` : `Ollama · ${state.config.local_model || ''}`)
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
        setDot('dot-gpu', 'ok');
        // Always use Ollama ps size_vram as the authoritative VRAM figure
        const modelVramMb    = (data.loaded || []).filter(m => m.size_vram > 0)
          .reduce((s, m) => s + m.size_vram, 0) / 1048576;
        const nvTotalReliable = g.memory_total_mb > 0 && g.memory_total_mb < 90000;
        let gpuLabel = 'GPU · active';
        if (modelVramMb > 0 && nvTotalReliable) {
          const vramPct = Math.min(100, Math.round(modelVramMb / g.memory_total_mb * 100));
          gpuLabel = `GPU · ${vramPct}% VRAM`;
        } else if (modelVramMb > 0) {
          gpuLabel = `GPU · ${Math.round(modelVramMb).toLocaleString()} MB`;
        }
        if (gpuLbl) gpuLbl.textContent = gpuLabel;
      } else if (warming) {
        setDot('dot-gpu', 'warn');
        if (gpuLbl) gpuLbl.textContent = 'GPU · loading…';
      } else {
        setDot('dot-gpu', 'warn');
        if (gpuLbl) gpuLbl.textContent = 'GPU · ready';
      }
      const chip = document.getElementById('gpu-chip');
      const memInfo = g.memory_total_mb < 90000
        ? `${g.memory_used_mb.toLocaleString()} / ${g.memory_total_mb.toLocaleString()} MB`
        : 'VRAM via nvidia-smi unreliable in this environment';
      if (chip) chip.title = `${g.name} — ${memInfo} · ${g.utilization_pct}% util`;
    } else {
      if (gpuChip) gpuChip.style.display = 'none';
    }

    // ── Sidebar pills ──────────────────────────────────────────────────
    const statusPill = document.getElementById('sidebar-ollama-status');
    if (statusPill) {
      statusPill.style.color = '#ffffff';
      const gpuAvail = gpu.available && gpu.gpus && gpu.gpus.length > 0;
      statusPill.textContent = running
        ? (warming ? '↻ Ollama · loading model' : (data.gpu_in_use ? '● Ollama · GPU active' : (gpuAvail ? '● Ollama · GPU ready' : '● Ollama · CPU')))
        : '○ Ollama';
    }

    // ── Config-dot nav warning ─────────────────────────────────────────
    const configDot = document.getElementById('config-dot');
    if (configDot) configDot.className = `nav-dot ${running ? 'success' : 'warning'}`;
  } catch (_) {}
}

function _updateWarmingBanner(warming, gpuInUse) {
  const banner  = document.getElementById('model-warming-banner');
  const procBtn = document.getElementById('single-process-btn');
  if (!banner) return;

  if (warming) {
    banner.style.display = 'flex';
    if (procBtn && !procBtn.dataset.userDisabled) {
      procBtn.disabled = true;
      procBtn.title    = 'Model is loading into GPU — please wait';
    }
  } else {
    banner.style.display = 'none';
    if (procBtn && !procBtn.dataset.userDisabled) {
      procBtn.disabled = false;
      procBtn.title    = '';
    }
    if (!warming && gpuInUse && !_warmingToastShown) {
      _warmingToastShown = true;
      toast('Model loaded into GPU — ready to process', 'success');
    }
  }
}
let _warmingToastShown = false;

function setDot(id, status) {
  const el = document.getElementById(id);
  if (!el) return;
  el.className = 'status-dot';
  if (status) el.classList.add(status);
}

function updateSidebarPills() {
  const namePill = document.getElementById('sidebar-model-name');
  if (namePill) namePill.textContent = state.config.local_model || 'qwen2.5vl:7b';
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
  const isQA = uc === 'Complicated Document QA';
  document.getElementById('single-question-group').style.display = isQA ? 'flex' : 'none';
}

function onSingleSrcChange() { refreshSingleImageSelect(); }
function onSingleImageChange() { updateSinglePreview(); }

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
  const src     = document.querySelector('input[name="single-src"]:checked')?.value || 'uploads';
  const name    = document.getElementById('single-image-select').value;
  const imgEl   = document.getElementById('single-preview');
  const empty   = document.getElementById('analysis-img-empty');
  const metaEl  = document.getElementById('analysis-img-meta');
  if (!name) {
    if (imgEl)  { imgEl.style.display = 'none'; }
    if (empty)  { empty.style.display = 'flex'; }
    if (metaEl) { metaEl.textContent = ''; }
    return;
  }
  const imgSrc = src === 'examples' ? `/examples/${encodeURIComponent(name)}` : `/images/${encodeURIComponent(name)}`;
  if (imgEl) { imgEl.src = imgSrc; imgEl.style.display = 'block'; }
  if (empty)  { empty.style.display = 'none'; }
  if (metaEl) { metaEl.textContent = name; }
}

document.addEventListener('DOMContentLoaded', () => {
  const sel = document.getElementById('single-image-select');
  if (sel) sel.addEventListener('change', updateSinglePreview);

  // ── Clipboard paste ─────────────────────────────────────────────────────
  document.addEventListener('paste', async (e) => {
    const items = Array.from(e.clipboardData?.items || []);
    const imgItem = items.find(it => it.type.startsWith('image/'));
    if (!imgItem) return;
    e.preventDefault();
    const file = imgItem.getAsFile();
    if (!file) return;
    const ext  = file.type.split('/')[1] || 'png';
    const name = `paste_${Date.now()}.${ext}`;
    const renamed = new File([file], name, { type: file.type });
    toast('Image pasted — uploading…', 'info', 2000);
    const fd = new FormData();
    fd.append('files', renamed);
    try {
      await fetch('/api/images/upload', { method: 'POST', body: fd });
      await refreshImages();
      // Switch single-src to uploads and select the pasted image
      const uploadRadio = document.querySelector('input[name="single-src"][value="uploads"]');
      if (uploadRadio) { uploadRadio.checked = true; }
      await refreshSingleImageSelect();
      const sel = document.getElementById('single-image-select');
      if (sel) {
        const opt = Array.from(sel.options).find(o => o.value === name);
        if (opt) { sel.value = name; updateSinglePreview(); }
      }
      toast('Pasted image ready', 'success');
    } catch (err) {
      toast('Paste upload failed: ' + err.message, 'error');
    }
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

let _singleAbort = null;

function _setSingleProcessing(active) {
  const btn     = document.getElementById('single-process-btn');
  const stopBtn = document.getElementById('single-stop-btn');
  if (!btn) return;
  if (active) {
    btn.disabled = true;
    btn.classList.add('btn-processing');
    btn.innerHTML = '<span class="spinner"></span> Processing…';
    if (stopBtn) stopBtn.style.display = 'inline-flex';
  } else {
    btn.disabled = false;
    btn.classList.remove('btn-processing');
    btn.innerHTML = `<svg viewBox="0 0 20 20" fill="currentColor"><path fill-rule="evenodd" d="M2 10a8 8 0 1116 0 8 8 0 01-16 0zm6.39-2.908a.75.75 0 01.766.027l3.5 2.25a.75.75 0 010 1.262l-3.5 2.25A.75.75 0 018 12.25v-4.5a.75.75 0 01.39-.658z"/></svg> Process Image`;
    if (stopBtn) stopBtn.style.display = 'none';
    _singleAbort = null;
  }
}

// Current single-process state: filename and folder for result editing
let _singleResultFilename = '';
let _singleResultFolder   = '';

async function runSingleProcess() {
  const ucRaw    = document.getElementById('single-use-case').value;
  const uc       = TEMPLATE_USE_CASE[ucRaw] || ucRaw;  // resolve template → use case
  const question = document.getElementById('single-question')?.value || '';
  const src      = document.querySelector('input[name="single-src"]:checked')?.value || 'uploads';
  const filename = document.getElementById('single-image-select').value;

  if (!filename) { toast('Please select an image.', 'error'); return; }

  _singleResultFilename = filename;
  _singleResultFolder   = '';   // single-process always saves to root

  const resultPre   = document.getElementById('single-result');
  const resultEmpty = document.getElementById('single-result-empty');
  const streamInd   = document.getElementById('single-stream-indicator');
  const editBtn     = document.getElementById('single-edit-btn');
  const copyBtn     = document.getElementById('single-copy-btn');

  // Switch to streaming state
  if (resultEmpty)  resultEmpty.style.display = 'none';
  if (resultPre)    { resultPre.textContent = ''; resultPre.style.display = 'block'; }
  if (streamInd)    streamInd.style.display = 'inline-flex';
  if (editBtn)      editBtn.style.display = 'none';
  if (copyBtn)      copyBtn.style.display = 'none';
  cancelResultEdit();

  _singleAbort = new AbortController();
  _setSingleProcessing(true);
  setActive('processing', true);

  let fullText = '';

  try {
    const resp = await fetch('/api/process/stream', {
      method:  'POST',
      headers: { 'Content-Type': 'application/json' },
      body:    JSON.stringify({ filename, use_case: uc, question, source: src }),
      signal:  _singleAbort.signal,
    });

    if (!resp.ok) {
      const err = await resp.json().catch(() => ({ detail: resp.statusText }));
      throw new Error(err.detail || resp.statusText);
    }

    const reader  = resp.body.getReader();
    const decoder = new TextDecoder();
    let   buf     = '';

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buf += decoder.decode(value, { stream: true });
      const parts = buf.split('\n\n');
      buf = parts.pop();
      for (const part of parts) {
        for (const line of part.split('\n')) {
          if (!line.startsWith('data: ')) continue;
          const data = JSON.parse(line.slice(6));
          if (data.error) { throw new Error(data.error); }
          if (data.token) {
            fullText += data.token;
            if (resultPre) resultPre.textContent = fullText;
          }
        }
      }
    }

    if (streamInd) streamInd.style.display = 'none';
    if (fullText) {
      if (copyBtn) { copyBtn.style.display = 'inline-flex'; copyBtn.dataset.text = fullText; }
      if (editBtn) editBtn.style.display = 'inline-flex';
      toast('Analysis complete', 'success');
    }

  } catch (e) {
    if (streamInd) streamInd.style.display = 'none';
    if (e.name === 'AbortError') {
      toast('Processing stopped', 'info');
    } else {
      if (resultPre) resultPre.textContent = `Error: ${e.message}`;
      toast(e.message, 'error');
    }
  } finally {
    setActive('processing', false);
    _setSingleProcessing(false);
  }
}

function stopSingleProcess() {
  if (_singleAbort) _singleAbort.abort();
}

function copyResult() {
  const text = document.getElementById('single-copy-btn').dataset.text
            || document.getElementById('single-result')?.textContent || '';
  navigator.clipboard.writeText(text).then(() => toast('Copied to clipboard', 'success'));
}

// ---------------------------------------------------------------------------
// Result editing (Analysis + Files modal)
// ---------------------------------------------------------------------------

function toggleResultEdit() {
  const pre      = document.getElementById('single-result');
  const textarea = document.getElementById('single-result-edit');
  const actions  = document.getElementById('single-result-edit-actions');
  const editBtn  = document.getElementById('single-edit-btn');
  if (!pre || !textarea) return;
  const editing = textarea.style.display !== 'none';
  if (editing) {
    cancelResultEdit();
  } else {
    textarea.value = pre.textContent;
    pre.style.display = 'none';
    textarea.style.display = 'block';
    actions.style.display  = 'flex';
    editBtn.textContent = 'View';
    textarea.focus();
  }
}

async function saveResultEdit() {
  const textarea = document.getElementById('single-result-edit');
  const pre      = document.getElementById('single-result');
  if (!textarea || !_singleResultFilename) return;
  try {
    await api('PUT', '/api/results/save', {
      filename: _singleResultFilename,
      folder:   _singleResultFolder,
      content:  textarea.value,
    });
    if (pre) { pre.textContent = textarea.value; }
    document.getElementById('single-copy-btn').dataset.text = textarea.value;
    cancelResultEdit();
    toast('Result saved', 'success');
  } catch (e) {
    toast('Save failed: ' + e.message, 'error');
  }
}

function cancelResultEdit() {
  const pre      = document.getElementById('single-result');
  const textarea = document.getElementById('single-result-edit');
  const actions  = document.getElementById('single-result-edit-actions');
  const editBtn  = document.getElementById('single-edit-btn');
  if (textarea) textarea.style.display = 'none';
  if (actions)  actions.style.display  = 'none';
  if (pre && pre.textContent) pre.style.display = 'block';
  if (editBtn)  editBtn.textContent = 'Edit';
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// Search
// ---------------------------------------------------------------------------

let _searchDebounce = null;

function onSearchInput() {
  clearTimeout(_searchDebounce);
  const q = document.getElementById('search-input')?.value.trim();
  if (!q) { document.getElementById('search-results').innerHTML = ''; return; }
  _searchDebounce = setTimeout(runSearch, 400);
}

async function runSearch() {
  const q  = document.getElementById('search-input')?.value.trim();
  const el = document.getElementById('search-results');
  if (!q || !el) return;
  el.innerHTML = `<div class="empty-state"><span class="spinner"></span><p>Searching…</p></div>`;
  try {
    const data = await api('GET', `/api/results/search?q=${encodeURIComponent(q)}`);
    if (!data.matches.length) {
      el.innerHTML = `<div class="empty-state"><p>No results found for <strong>${escHtml(q)}</strong></p></div>`;
      return;
    }
    el.innerHTML = data.matches.map(m => {
      // Highlight the match within the snippet
      const before  = escHtml(m.snippet.slice(0, m.match_start));
      const matched = escHtml(m.snippet.slice(m.match_start, m.match_start + m.match_len));
      const after   = escHtml(m.snippet.slice(m.match_start + m.match_len));
      const folder  = m.folder ? `<span class="search-match-folder">${escHtml(m.folder)}</span>` : '';
      return `<div class="search-match-card" onclick="openSearchResult('${escAttr(m.img_filename)}','${escAttr(m.folder)}')">
        <div class="search-match-meta">
          <span class="search-match-file">${escHtml(m.img_filename)}</span>
          ${folder}
        </div>
        <div class="search-match-snippet">${before}<mark>${matched}</mark>${after}</div>
      </div>`;
    }).join('');
  } catch (e) {
    el.innerHTML = `<div class="empty-state"><p style="color:var(--red)">Search failed: ${escHtml(e.message)}</p></div>`;
  }
}

async function openSearchResult(filename, folder) {
  try {
    const q   = `?filename=${encodeURIComponent(filename)}${folder ? '&folder=' + encodeURIComponent(folder) : ''}`;
    const data = await api('GET', `/api/results/read${q}`);
    const lines   = (data.content || '').split('\n');
    const sepIdx  = lines.findIndex(l => l.startsWith('─'));
    const body    = sepIdx >= 0 ? lines.slice(sepIdx + 1).join('\n').trimStart() : data.content;
    document.getElementById('modal-title').textContent    = filename;
    document.getElementById('modal-subtitle').textContent = folder ? `Folder: ${folder}` : 'All Files';
    const imgEl = document.getElementById('modal-image');
    if (imgEl) {
      const src = folder
        ? `/images/${encodeURIComponent(folder)}/${encodeURIComponent(filename)}`
        : `/images/${encodeURIComponent(filename)}`;
      imgEl.src = src; imgEl.style.display = 'block';
    }
    document.getElementById('modal-result').textContent = body;
    document.getElementById('result-modal').classList.add('open');
  } catch (e) {
    toast('Could not open result: ' + e.message, 'error');
  }
}

// ---------------------------------------------------------------------------
// Keyboard shortcuts
// ---------------------------------------------------------------------------

document.addEventListener('keydown', e => {
  // Don't fire shortcuts when typing in inputs
  const tag = document.activeElement?.tagName;
  if (tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT') return;

  // ? — show shortcut cheat sheet toast
  if (e.key === '?') {
    toast('Shortcuts: Space=select · ↑↓←→=navigate · Enter=process · Ctrl+A=select all · Escape=clear', 'info', 6000);
    return;
  }

  // Only grid shortcuts apply when Files section is active
  const filesActive = document.getElementById('section-files')?.classList.contains('active');
  if (!filesActive) return;

  const tiles = Array.from(document.querySelectorAll('.file-tile'));
  if (!tiles.length) return;

  const focusedTile = document.querySelector('.file-tile.kb-focus');
  const focusedIdx  = focusedTile ? tiles.indexOf(focusedTile) : -1;

  const moveFocus = (newIdx) => {
    tiles.forEach(t => t.classList.remove('kb-focus'));
    const t = tiles[Math.max(0, Math.min(newIdx, tiles.length - 1))];
    t.classList.add('kb-focus');
    t.scrollIntoView({ block: 'nearest' });
  };

  if (e.key === 'ArrowRight') { e.preventDefault(); moveFocus(focusedIdx + 1); }
  if (e.key === 'ArrowLeft')  { e.preventDefault(); moveFocus(Math.max(0, focusedIdx - 1)); }
  if (e.key === 'ArrowDown')  { e.preventDefault(); moveFocus(focusedIdx + 4); } // approx grid cols
  if (e.key === 'ArrowUp')    { e.preventDefault(); moveFocus(Math.max(0, focusedIdx - 4)); }

  if (e.key === ' ' && focusedTile) {
    e.preventDefault();
    const name = focusedTile.id.replace('tile-', '');
    filesToggle(name);
  }

  if ((e.key === 'a' || e.key === 'A') && (e.ctrlKey || e.metaKey)) {
    e.preventDefault();
    filesSelectAll();
  }

  if (e.key === 'Enter') {
    e.preventDefault();
    queueFilesSelected();
  }
});

// ---------------------------------------------------------------------------
// File upload (XHR with % progress)
// ---------------------------------------------------------------------------

function uploadFiles(files, progressId = 'batch', folder = '') {
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
      if (document.getElementById('section-files')?.classList.contains('active')) {
        await _loadFilesTree();
        await _loadFilesBrowser();
      }
      // Show PDF page info if any PDFs were expanded
      try {
        const resp = JSON.parse(xhr.responseText);
        const pdfPages = resp.pdf_pages || {};
        const pdfNames = Object.keys(pdfPages);
        if (pdfNames.length) {
          const summary = pdfNames.map(n => `${n} → ${pdfPages[n].length} pages`).join(', ');
          toast(`PDFs expanded: ${summary}`, 'success', 5000);
        } else {
          toast(`Uploaded ${files.length} file(s)${folder ? ` to "${folder}"` : ''}`, 'success');
        }
      } catch {
        toast(`Uploaded ${files.length} file(s)${folder ? ` to "${folder}"` : ''}`, 'success');
      }
    } else {
      toast('Upload failed', 'error');
    }
  });
  xhr.addEventListener('error', () => {
    setActive('uploads', false);
    if (wrap) wrap.style.display = 'none';
    toast('Upload failed (network error)', 'error');
  });
  const url = folder ? `/api/images/upload?folder=${encodeURIComponent(folder)}` : '/api/images/upload';
  xhr.open('POST', url);
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

let _notifiedJobIds  = new Set();
let _prevWasRunning  = false;   // tracks transition: running → done, to reload browser once

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

    const running    = newJobs.filter(j => j.status === 'processing' || j.status === 'queued').length;
    const nowRunning = running > 0;
    setActive('batch', running);

    state.jobs = newJobs;
    refreshJobList();
    refreshResultsList();
    updateResultsBadge();

    // Update Files job panel if visible — but only rebuild the file BROWSER
    // on the one-time transition from "had running jobs" → "all done".
    // Calling _loadFilesBrowser() every poll causes the icon-flashing.
    if (document.getElementById('files-jobs-panel')?.style.display !== 'none') {
      _refreshFilesJobList();
    }
    if (_prevWasRunning && !nowRunning) {
      // Jobs just finished — refresh the file browser once to update OCR badges
      _loadFilesBrowser();
    }
    _prevWasRunning = nowRunning;
  } catch (_) {}
}

function updateResultsBadge() {
  const complete = state.jobs.filter(j => j.status === 'complete').length;
  const running  = state.jobs.filter(j => j.status === 'processing' || j.status === 'queued').length;

  const resBadge = document.getElementById('nav-badge-results');
  if (resBadge) {
    if (complete > 0) { resBadge.textContent = complete; resBadge.style.display = 'inline-block'; }
    else resBadge.style.display = 'none';
  }
  const filesBadge = document.getElementById('nav-badge-files');
  if (filesBadge) {
    if (running > 0) { filesBadge.textContent = running; filesBadge.style.display = 'inline-block'; }
    else filesBadge.style.display = 'none';
  }
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

// Safe store for job results — avoids embedding arbitrary text inside HTML attributes
const _jobResultStore = new Map();

function copyJobResult(id) {
  const text = _jobResultStore.get(id) || '';
  navigator.clipboard.writeText(text).then(() => toast('Copied', 'success'));
}

function refreshResultsList() {
  const el = document.getElementById('results-list');
  if (!el) return;
  const done = state.jobs.filter(j => j.status === 'complete' || j.status === 'error');
  if (!done.length) {
    el.innerHTML = `<div class="empty-state"><p>Results from processed images will appear here.</p></div>`;
    return;
  }
  const sorted = [...done].sort((a,b) => (b.completed_at||'').localeCompare(a.completed_at||''));

  // Populate the result store before rendering HTML so copy buttons always work
  for (const job of sorted) {
    if (job.result) _jobResultStore.set(job.id, job.result);
  }

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
             <button class="btn btn-sm btn-ghost" onclick="copyJobResult('${job.id}')">Copy Result</button>`
        }
      </div>
    </div>
  `).join('');

  // Restore expanded state — pollJobs re-renders every 2 s which would otherwise
  // collapse any card the user has open.
  for (const id of _openResultCards) {
    document.getElementById(`rc-${id}`)?.classList.add('open');
  }
}

// Track which result cards are expanded so polling re-renders don't collapse them
const _openResultCards = new Set();

function toggleResultCard(id) {
  const card = document.getElementById(`rc-${id}`);
  if (!card) return;
  card.classList.toggle('open');
  if (card.classList.contains('open')) {
    _openResultCards.add(id);
  } else {
    _openResultCards.delete(id);
  }
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
// Image lightbox — click any thumbnail to enlarge
// ---------------------------------------------------------------------------

function openLightbox(src, caption) {
  const lb  = document.getElementById('img-lightbox');
  const img = document.getElementById('img-lightbox-img');
  const cap = document.getElementById('img-lightbox-caption');
  img.src         = src;
  cap.textContent = caption || '';
  lb.classList.add('open');
}

function closeLightbox(e) {
  if (e.target.id === 'img-lightbox' || e.target.id === 'img-lightbox-img')
    document.getElementById('img-lightbox').classList.remove('open');
}

// Close lightbox on Escape key
document.addEventListener('keydown', e => {
  if (e.key === 'Escape') {
    document.getElementById('img-lightbox')?.classList.remove('open');
    document.getElementById('result-modal')?.classList.remove('open');
  }
});

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
    refreshSingleImageSelect();
    if (document.getElementById('section-files')?.classList.contains('active')) {
      await _loadFilesTree(); await _loadFilesBrowser();
    }
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
  const model = document.getElementById('cfg-local-model')?.value || 'qwen2.5vl:7b';
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
        pullLabel.textContent = '✓ Model pulled — saving as active model…';
        setTimeout(() => { pullSec.style.display = 'none'; pullFill.style.width = '0'; pullLabel.style.color = ''; }, 5000);
        // Auto-save the pulled model as the active model so it takes effect immediately
        (async () => {
          try {
            const model = document.getElementById('cfg-local-model')?.value || 'qwen2.5vl:7b';
            await api('POST', '/api/config', { local_model: model, max_tokens: 4096 });
            state.config = await api('GET', '/api/config');
            updateSidebarPills();
            toast(`Model "${model}" pulled and set as active`, 'success');
          } catch (e) {
            toast('Pulled OK but could not save config: ' + e.message, 'error');
          }
        })();
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
      const activeClass = data.gpu_in_use ? 'active' : 'ready';

      // Ollama ps is the authoritative source for VRAM actually in use by the model
      const modelEntries   = loaded.filter(m => m.size_vram > 0);
      const modelVramBytes = modelEntries.reduce((s, m) => s + m.size_vram, 0);
      const modelVramMb    = Math.round(modelVramBytes / 1048576);
      const modelVramLabel = modelEntries
        .map(m => `${escHtml(m.name)} (${(m.size_vram / 1073741824).toFixed(1)} GB)`)
        .join(', ');

      // nvidia-smi memory.total can report host RAM in some CML/Kubernetes environments
      // (e.g. an L4 showing 130 GB instead of its actual 24 GB).
      // Only use it for a capacity denominator when it looks plausible (< 90 GB).
      const nvTotalMb       = g.memory_total_mb;
      const nvTotalReliable = nvTotalMb > 0 && nvTotalMb < 90000;

      // Build bar and label — Ollama ps VRAM is always the "used" value
      let vramPct, barLabel;
      if (modelVramMb > 0 && nvTotalReliable) {
        vramPct  = Math.min(100, Math.round(modelVramMb / nvTotalMb * 100));
        barLabel = `${modelVramMb.toLocaleString()} / ${nvTotalMb.toLocaleString()} MB`;
      } else if (modelVramMb > 0) {
        vramPct  = null;
        barLabel = `${modelVramMb.toLocaleString()} MB in VRAM`;
      } else if (nvTotalReliable) {
        vramPct  = 0;
        barLabel = `0 / ${nvTotalMb.toLocaleString()} MB`;
      } else {
        vramPct  = 0;
        barLabel = 'Model not yet loaded';
      }

      const barHtml = vramPct !== null
        ? `<div class="gpu-vram-bar"><div class="gpu-vram-fill" style="width:${vramPct}%"></div></div>`
        : `<div class="gpu-vram-bar"><div class="gpu-vram-fill" style="width:100%;opacity:0.45;background:var(--cldr-teal)"></div></div>`;

      return `<div class="gpu-card ${activeClass}">
        <div class="gpu-card-icon">GPU${i}</div>
        <div class="gpu-card-body">
          <div class="gpu-card-name">${escHtml(g.name)}</div>
          <div class="gpu-card-meta">
            Driver ${escHtml(g.driver_version)} ·
            ${g.utilization_pct}% util
            ${modelVramLabel
              ? `· <strong style="color:var(--cldr-teal)">${modelVramLabel} in VRAM</strong>`
              : `· <span style="color:var(--tx-3);font-style:italic">Model loads into VRAM on first inference</span>`}
          </div>
        </div>
        <div class="gpu-vram-bar-wrap">
          ${barHtml}
          <span class="gpu-vram-label">${barLabel}</span>
        </div>
      </div>`;
    }).join('');
  }
}

function renderPulledModels(models) {
  const el = document.getElementById('pulled-models-list');
  if (!el) return;
  if (!models.length) {
    el.innerHTML = `<p class="form-hint" style="margin-top:4px">No models pulled yet.</p>`;
    return;
  }
  el.innerHTML =
    `<div class="result-section-label" style="margin-bottom:8px">PULLED MODELS</div>` +
    `<div style="display:flex;flex-wrap:wrap;gap:6px">` +
    models.map(m =>
      `<span style="display:inline-flex;align-items:center;gap:6px;font-size:12px;color:var(--tx);` +
      `background:var(--blue-bg);border:1px solid var(--border-md);border-radius:4px;padding:4px 10px;font-family:inherit">` +
      `<svg width="12" height="12" viewBox="0 0 20 20" fill="currentColor" style="color:var(--green);flex-shrink:0">` +
      `<path fill-rule="evenodd" d="M16.704 4.153a.75.75 0 01.143 1.052l-8 10.5a.75.75 0 01-1.127.075l-4.5-4.5a.75.75 0 011.06-1.06l3.894 3.893 7.48-9.817a.75.75 0 011.05-.143z"/>` +
      `</svg>${escHtml(m)}</span>`
    ).join('') +
    `</div>`;
}

async function startOllama() {
  const btn = document.getElementById('ollama-start-btn');
  btn.disabled = true; btn.textContent = 'Starting…';
  try {
    const r = await api('POST', '/api/ollama/start');
    toast(r.message, 'success');
    setTimeout(refreshOllamaStatus, 2500);
    // Refresh log after a short delay so GPU detection output is visible
    setTimeout(refreshOllamaLog, 4000);
  } catch (e) {
    toast(e.message, 'error');
  } finally {
    btn.disabled = false; btn.textContent = 'Start Ollama';
  }
}

let _ollamaLogVisible = false;

function toggleOllamaLog() {
  const panel  = document.getElementById('ollama-log-panel');
  const hint   = document.getElementById('ollama-log-toggle-hint');
  _ollamaLogVisible = !_ollamaLogVisible;
  panel.style.display = _ollamaLogVisible ? 'block' : 'none';
  hint.textContent    = _ollamaLogVisible ? 'Hide' : 'Show';
  if (_ollamaLogVisible) refreshOllamaLog();
}

async function refreshOllamaLog() {
  const pre = document.getElementById('ollama-log-pre');
  if (!pre) return;
  try {
    const data = await api('GET', '/api/ollama/log');
    pre.textContent = data.log || '(empty)';
    pre.scrollTop   = pre.scrollHeight;
  } catch (e) {
    pre.textContent = 'Error fetching log: ' + e.message;
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
// User personalisation
// ---------------------------------------------------------------------------

async function loadUserInfo() {
  try {
    const u = await api('GET', '/api/user-info');

    // Header avatar chip
    const avatar = document.getElementById('user-avatar');
    const name   = document.getElementById('user-name');
    const chip   = document.getElementById('user-chip');
    if (avatar) avatar.textContent = u.initials || '?';
    if (name)   name.textContent   = u.username || u.full_name || '';
    if (chip)   chip.title = `${u.full_name || u.username}  ·  ${u.project}${u.domain ? '  ·  ' + u.domain : ''}`;

    // Welcome banner (Workbench-style greeting)
    const banner    = document.getElementById('welcome-banner');
    const welcomeTx = document.getElementById('welcome-title');
    if (banner) banner.style.display = 'block';
    if (welcomeTx && u.full_name) {
      welcomeTx.textContent = `Welcome to Cloudera Image Analysis, ${u.full_name}.`;
    }

    // Sidebar footer user pill
    const pills = document.getElementById('sidebar-model-pills');
    if (pills && (u.full_name || u.username)) {
      const existing = document.getElementById('sidebar-user-pill');
      if (!existing) {
        const pill = document.createElement('span');
        pill.id = 'sidebar-user-pill';
        pill.className = 'model-pill';
        pill.style.cssText = 'color:#ffffff;border-color:var(--sb-border);display:flex;align-items:center;gap:5px';
        pill.innerHTML = `<svg viewBox="0 0 20 20" fill="currentColor" style="width:10px;height:10px"><path fill-rule="evenodd" d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-5.5-2.5a2.5 2.5 0 11-5 0 2.5 2.5 0 015 0zM10 12a5.99 5.99 0 00-4.793 2.39A6.483 6.483 0 0010 16.5a6.483 6.483 0 004.793-2.11A5.99 5.99 0 0010 12z"/></svg>${escHtml(u.full_name || u.username)}`;
        pills.prepend(pill);
      }
    }
  } catch (_) {}
}

// ---------------------------------------------------------------------------
// Folder management
// ---------------------------------------------------------------------------

let _activeFolder = null;

// ---------------------------------------------------------------------------
// Unified Files section
// ---------------------------------------------------------------------------

let _filesFolder   = null;   // null = root "All Files"; string = named folder
let _filesSelected = new Set();
let _filesResultSet = new Set(); // OCR_*.txt stems that have results
// Per-file use case overrides: { filename → use_case_string }
const _filesUseCaseOverrides = new Map();

const USE_CASE_SHORT = {
  'Transcribing Typed Text':        'Typed',
  'Transcribing Handwritten Text':  'Handwritten',
  'Transcribing Forms':             'Forms',
  'Complicated Document QA':        'QA',
  'Unstructured Information → JSON':'→ JSON',
  'Summarize Image':                'Summarize',
};
const USE_CASE_LIST = Object.keys(USE_CASE_SHORT);

async function loadFilesSection() {
  _filesSelected.clear();
  _updateFilesProcessBtn();
  await _loadFilesTree();
  await _loadFilesBrowser();
}

async function _loadFilesTree() {
  const tree = document.getElementById('files-tree-list');
  if (!tree) return;
  const { folders } = await api('GET', '/api/folders').catch(() => ({ folders: [] }));
  const rootImages  = await api('GET', '/api/images').catch(() => ({ uploads: [] }));
  const rootCount   = (rootImages.uploads || []).length;

  tree.innerHTML = [
    // "All Files" row
    `<div class="files-tree-item${_filesFolder === null ? ' active' : ''}" onclick="filesSelectFolder(null)">
      <svg viewBox="0 0 20 20" fill="currentColor"><path fill-rule="evenodd" d="M1 5.25A2.25 2.25 0 013.25 3h13.5A2.25 2.25 0 0119 5.25v9.5A2.25 2.25 0 0116.75 17H3.25A2.25 2.25 0 011 14.75v-9.5zm1.5 5.81v3.69c0 .414.336.75.75.75h13.5a.75.75 0 00.75-.75v-2.69l-2.22-2.219a.75.75 0 00-1.06 0l-1.91 1.909-.48-.484a.75.75 0 00-1.06 0L6.75 13.5l-1.5-1.5-2.75.06zm0-1.56L5.03 7.45a.75.75 0 011.06 0l1.5 1.5.48-.484a.75.75 0 011.06 0l.48.483 1.91-1.908a.75.75 0 011.06 0L15 9.28V5.25a.75.75 0 00-.75-.75H3.25a.75.75 0 00-.75.75v4.25zm5-3a1 1 0 100-2 1 1 0 000 2z"/></svg>
      All Files
      ${rootCount ? `<span class="files-tree-count">${rootCount}</span>` : ''}
    </div>`,
    // Named folder rows
    ...folders.map(f => `
      <div class="files-tree-item${_filesFolder === f.name ? ' active' : ''}" id="ftree-${escAttr(f.name)}" onclick="filesSelectFolder('${escAttr(f.name)}')">
        <svg viewBox="0 0 20 20" fill="currentColor"><path d="M3.75 3A1.75 1.75 0 002 4.75v10.5c0 .966.784 1.75 1.75 1.75h12.5A1.75 1.75 0 0018 15.25v-8.5A1.75 1.75 0 0016.25 5h-4.836a.25.25 0 01-.177-.073L9.823 3.513A1.75 1.75 0 008.586 3H3.75z"/></svg>
        <span class="files-tree-label" id="ftree-label-${escAttr(f.name)}"
              ondblclick="event.stopPropagation();startFolderRename('${escAttr(f.name)}')"
              title="Double-click to rename"
              style="flex:1;overflow:hidden;text-overflow:ellipsis;white-space:nowrap">${escHtml(f.name)}</span>
        ${f.result_count ? `<span class="files-tree-result-dot" title="${f.result_count} result${f.result_count > 1 ? 's' : ''}"></span>` : ''}
        <span class="files-tree-count">${f.count}</span>
        <button class="files-tree-rename" title="Rename folder" onclick="event.stopPropagation();startFolderRename('${escAttr(f.name)}')">✎</button>
        <button class="files-tree-delete" title="Delete folder" onclick="event.stopPropagation();filesDeleteFolder('${escAttr(f.name)}')">✕</button>
      </div>`)
  ].join('');
}

async function _loadFilesBrowser() {
  const grid  = document.getElementById('files-grid');
  const empty = document.getElementById('files-empty');
  const drop  = document.getElementById('files-drop-zone');
  if (!grid) return;

  // Update breadcrumb + topbar buttons
  const bc = document.getElementById('files-breadcrumb');
  if (bc) bc.textContent = _filesFolder ? `📁 ${_filesFolder}` : 'All Files';

  const dlBtn = document.getElementById('files-download-btn');

  grid.innerHTML = '<div style="padding:20px;color:var(--tx-3);font-size:13px">Loading…</div>';
  empty.style.display = 'none';
  drop.style.display  = 'none';

  let images = [];
  _filesResultSet.clear();

  try {
    if (_filesFolder) {
      const r = await api('GET', `/api/folders/${encodeURIComponent(_filesFolder)}/images`);
      images = r.images || [];
      // Load which images have results
      const rr = await api('GET', `/api/folders/${encodeURIComponent(_filesFolder)}/results`).catch(() => ({ results: [] }));
      (rr.results || []).forEach(fn => {
        // OCR_{stem}.txt → stem
        const stem = fn.replace(/^OCR_/, '').replace(/\.txt$/, '');
        _filesResultSet.add(stem);
      });
      const hasResults = (rr.results || []).length > 0;
      dlBtn.style.display = hasResults ? 'inline-flex' : 'none';
      const csvBtn = document.getElementById('files-csv-btn');
      if (csvBtn) csvBtn.style.display = hasResults ? 'inline-flex' : 'none';
    } else {
      const r = await api('GET', '/api/images');
      images = r.uploads || [];
      const rr = await api('GET', '/api/folders/root/results').catch(() => ({ results: [] }));
      (rr.results || []).forEach(fn => {
        const stem = fn.replace(/^OCR_/, '').replace(/\.txt$/, '');
        _filesResultSet.add(stem);
      });
      const hasResults = (rr.results || []).length > 0;
      dlBtn.style.display = hasResults ? 'inline-flex' : 'none';
      const csvBtn = document.getElementById('files-csv-btn');
      if (csvBtn) csvBtn.style.display = hasResults ? 'inline-flex' : 'none';
    }
  } catch (e) {
    grid.innerHTML = `<div style="padding:20px;color:var(--red);font-size:13px">${escHtml(e.message)}</div>`;
    return;
  }

  if (!images.length) {
    grid.innerHTML = '';
    empty.style.display = 'flex';
    drop.style.display  = 'flex';
    document.getElementById('files-select-all-btn').style.display = 'none';
    document.getElementById('files-clear-btn').style.display = 'none';
    return;
  }

  document.getElementById('files-select-all-btn').style.display = 'inline-flex';
  document.getElementById('files-clear-btn').style.display = 'inline-flex';

  const imgSrc = name => _filesFolder
    ? `/images/${encodeURIComponent(_filesFolder)}/${encodeURIComponent(name)}`
    : `/images/${encodeURIComponent(name)}`;

  grid.innerHTML = images.map(img => {
    const sel       = _filesSelected.has(img.name);
    const hasResult = _filesResultSet.has(Path_stem(img.name));
    const ucOverride = _filesUseCaseOverrides.get(img.name);
    const ucLabel    = ucOverride ? USE_CASE_SHORT[ucOverride] || ucOverride : null;

    const ocrBadge = hasResult
      ? `<button class="file-tile-result-badge" onclick="event.stopPropagation();viewFileResult('${escAttr(img.name)}')" title="View OCR result">OCR ↗</button>`
      : '';
    const ucBadge = ucLabel
      ? `<button class="file-tile-uc-badge" onclick="event.stopPropagation();cycleFileUseCase('${escAttr(img.name)}')" title="Use case override (click to change)">${escHtml(ucLabel)}</button>`
      : `<button class="file-tile-uc-badge file-tile-uc-default" onclick="event.stopPropagation();cycleFileUseCase('${escAttr(img.name)}')" title="Click to set a use case override for this file">+</button>`;

    return `<div class="file-tile${sel ? ' selected' : ''}" id="tile-${escAttr(img.name)}" onclick="filesToggle('${escAttr(img.name)}')">
      <div class="file-tile-cb"></div>
      ${ocrBadge}
      ${ucBadge}
      <img src="${imgSrc(img.name)}" alt="${escAttr(img.name)}" loading="lazy"
           onclick="event.stopPropagation();openLightbox('${escAttr(imgSrc(img.name))}','${escAttr(img.name)}')"
           title="Click to enlarge" />
      <div class="file-tile-name" title="${escAttr(img.name)}">${escHtml(img.name)}</div>
      <button class="file-tile-rename" onclick="event.stopPropagation();startImageRename('${escAttr(img.name)}')" title="Rename">✎</button>
      <button class="file-tile-del" onclick="event.stopPropagation();filesDeleteImage('${escAttr(img.name)}')" title="Delete">✕</button>
    </div>`;
  }).join('');
}

function Path_stem(filename) {
  return filename.replace(/\.[^.]+$/, '');
}

function filesToggle(name) {
  const tile = document.getElementById(`tile-${name}`);
  if (!tile) return;
  if (_filesSelected.has(name)) {
    _filesSelected.delete(name);
    tile.classList.remove('selected');
  } else {
    _filesSelected.add(name);
    tile.classList.add('selected');
  }
  _updateFilesProcessBtn();
}

function filesSelectAll() {
  document.querySelectorAll('.file-tile').forEach(t => {
    const name = t.id.replace('tile-', '');
    _filesSelected.add(name);
    t.classList.add('selected');
  });
  _updateFilesProcessBtn();
}

function filesClearSelection() {
  _filesSelected.clear();
  document.querySelectorAll('.file-tile').forEach(t => t.classList.remove('selected'));
  _updateFilesProcessBtn();
}

function _updateFilesProcessBtn() {
  const btn   = document.getElementById('files-process-btn');
  const count = document.getElementById('files-sel-count');
  if (!btn) return;
  const n = _filesSelected.size;
  btn.style.display = n > 0 ? 'inline-flex' : 'none';
  if (count) count.textContent = n;
}

async function filesSelectFolder(name) {
  _filesFolder = name;
  _filesSelected.clear();
  hideFilesProcessBar();
  await _loadFilesTree();
  await _loadFilesBrowser();
}

function showCreateFolderForm() {
  document.getElementById('create-folder-form').style.display = 'block';
  document.getElementById('new-folder-name').focus();
}
function hideCreateFolderForm() {
  document.getElementById('create-folder-form').style.display = 'none';
  document.getElementById('new-folder-name').value = '';
}

async function createFolder() {
  const inp  = document.getElementById('new-folder-name');
  const name = inp.value.trim();
  if (!name) { toast('Enter a folder name.', 'error'); return; }
  try {
    await api('POST', '/api/folders', { name });
    hideCreateFolderForm();
    toast(`Folder "${name}" created.`, 'success');
    await filesSelectFolder(name);
  } catch (e) {
    toast(e.message, 'error');
  }
}

async function filesDeleteFolder(name) {
  if (!confirm(`Delete folder "${name}" and all its images?`)) return;
  try {
    await api('DELETE', `/api/folders/${encodeURIComponent(name)}`);
    toast(`Folder "${name}" deleted.`, 'success');
    if (_filesFolder === name) await filesSelectFolder(null);
    else await _loadFilesTree();
  } catch (e) {
    toast(e.message, 'error');
  }
}

function startFolderRename(name) {
  const labelEl = document.getElementById(`ftree-label-${name}`);
  if (!labelEl) return;

  // Swap the label span for an inline input
  const input = document.createElement('input');
  input.type      = 'text';
  input.value     = name;
  input.className = 'files-tree-rename-input';
  input.onclick   = e => e.stopPropagation();   // don't navigate on click
  input.onkeydown = e => {
    e.stopPropagation();
    if (e.key === 'Enter')  { e.preventDefault(); commitFolderRename(name, input.value.trim()); }
    if (e.key === 'Escape') { cancelFolderRename(name, labelEl, input); }
  };
  input.onblur = () => commitFolderRename(name, input.value.trim());

  labelEl.replaceWith(input);
  input.select();
}

async function commitFolderRename(oldName, newName) {
  // Guard against blur firing after Enter already handled it
  const inputEl = document.querySelector(`.files-tree-rename-input`);
  if (!inputEl) return;   // already committed/cancelled

  if (!newName || newName === oldName) {
    cancelFolderRenameById(oldName);
    return;
  }
  // Remove the input immediately to prevent double-fire on blur
  inputEl.onblur = null;
  try {
    await api('PUT', `/api/folders/${encodeURIComponent(oldName)}/rename`, { new_name: newName });
    // If we were viewing the renamed folder, stay in it under the new name
    if (_filesFolder === oldName) _filesFolder = newName;
    toast(`Renamed to "${newName}"`, 'success');
    await _loadFilesTree();
    if (_filesFolder === newName) await _loadFilesBrowser();
  } catch (e) {
    toast(e.message, 'error');
    await _loadFilesTree();  // restore original state
  }
}

function cancelFolderRenameById(name) {
  const input = document.querySelector('.files-tree-rename-input');
  if (!input) return;
  input.onblur = null;
  _loadFilesTree();  // cheapest full reset
}

function cancelFolderRename(name, labelEl, input) {
  input.onblur = null;
  input.replaceWith(labelEl);
}

async function filesDeleteImage(name) {
  if (!confirm(`Delete "${name}"?`)) return;
  try {
    const q = _filesFolder ? `?folder=${encodeURIComponent(_filesFolder)}` : '';
    await api('DELETE', `/api/images/${encodeURIComponent(name)}${q}`);
    _filesSelected.delete(name);
    _updateFilesProcessBtn();
    await _loadFilesBrowser();
  } catch (e) {
    toast(e.message, 'error');
  }
}

function startImageRename(filename) {
  const tile    = document.getElementById(`tile-${filename}`);
  const nameEl  = tile?.querySelector('.file-tile-name');
  if (!tile || !nameEl) return;

  // Swap the name label for an inline input
  const input = document.createElement('input');
  input.type      = 'text';
  input.value     = filename;
  input.className = 'file-tile-rename-input';
  // Select just the stem (before extension) for convenience
  const dotIdx = filename.lastIndexOf('.');
  input.onclick   = e => e.stopPropagation();
  input.onkeydown = e => {
    e.stopPropagation();
    if (e.key === 'Enter')  { e.preventDefault(); commitImageRename(filename, input); }
    if (e.key === 'Escape') { input.onblur = null; nameEl.style.display = ''; input.remove(); }
  };
  input.onblur = () => commitImageRename(filename, input);

  nameEl.style.display = 'none';
  tile.appendChild(input);
  input.focus();
  if (dotIdx > 0) input.setSelectionRange(0, dotIdx);
  else            input.select();
}

async function commitImageRename(oldName, input) {
  if (!input.isConnected) return;  // already removed
  input.onblur = null;             // prevent double-fire

  const newName = input.value.trim();
  const tile    = document.getElementById(`tile-${oldName}`);
  const nameEl  = tile?.querySelector('.file-tile-name');

  // Restore display regardless of outcome
  const restore = () => { if (nameEl) nameEl.style.display = ''; input.remove(); };

  if (!newName || newName === oldName) { restore(); return; }

  try {
    const resp = await api('PUT', `/api/images/${encodeURIComponent(oldName)}/rename`, {
      new_name: newName,
      folder:   _filesFolder || '',
    });
    restore();
    // Update selection state to track the new name
    if (_filesSelected.has(oldName)) {
      _filesSelected.delete(oldName);
      _filesSelected.add(resp.new_name);
    }
    if (_filesUseCaseOverrides.has(oldName)) {
      _filesUseCaseOverrides.set(resp.new_name, _filesUseCaseOverrides.get(oldName));
      _filesUseCaseOverrides.delete(oldName);
    }
    const msg = resp.result_renamed
      ? `Renamed to "${resp.new_name}" — OCR result updated`
      : `Renamed to "${resp.new_name}"`;
    toast(msg, 'success');
    await _loadFilesBrowser();
    await _loadFilesTree();          // update result-dot counts
  } catch (e) {
    restore();
    toast(e.message, 'error');
  }
}

function showFilesProcessBar() {
  if (!_filesSelected.size) return;
  document.getElementById('files-process-bar').style.display = 'block';
  _onFilesUseCaseChange();
}
function hideFilesProcessBar() {
  const bar = document.getElementById('files-process-bar');
  if (bar) bar.style.display = 'none';
}
function _onFilesUseCaseChange() {
  const uc = document.getElementById('files-use-case')?.value || '';
  const qg = document.getElementById('files-question-group');
  if (qg) qg.style.display = uc === 'Complicated Document QA' ? 'block' : 'none';
}

function cycleFileUseCase(filename) {
  const current = _filesUseCaseOverrides.get(filename);
  const idx     = current ? USE_CASE_LIST.indexOf(current) : -1;
  if (idx === -1 || idx === USE_CASE_LIST.length - 1) {
    // First click: set first use case; last: clear override
    if (idx === USE_CASE_LIST.length - 1) {
      _filesUseCaseOverrides.delete(filename);
    } else {
      _filesUseCaseOverrides.set(filename, USE_CASE_LIST[0]);
    }
  } else {
    _filesUseCaseOverrides.set(filename, USE_CASE_LIST[idx + 1]);
  }
  // Re-render just this tile
  const tile     = document.getElementById(`tile-${filename}`);
  if (!tile) return;
  const ucOverride = _filesUseCaseOverrides.get(filename);
  const ucLabel    = ucOverride ? USE_CASE_SHORT[ucOverride] || ucOverride : null;
  const existing   = tile.querySelector('.file-tile-uc-badge');
  if (existing) {
    if (ucLabel) {
      existing.className = 'file-tile-uc-badge';
      existing.title     = 'Use case override (click to change)';
      existing.textContent = ucLabel;
    } else {
      existing.className = 'file-tile-uc-badge file-tile-uc-default';
      existing.title     = 'Click to set a use case override for this file';
      existing.textContent = '+';
    }
  }
}

async function queueFilesSelected() {
  const defaultUseCase = document.getElementById('files-use-case')?.value || 'Summarize Image';
  const question       = document.getElementById('files-question')?.value || '';
  const names          = [..._filesSelected];
  if (!names.length) return;
  try {
    // Group files by their effective use case so we can batch each group separately
    const groups = new Map();  // useCase → filenames[]
    for (const name of names) {
      const uc = _filesUseCaseOverrides.get(name) || defaultUseCase;
      if (!groups.has(uc)) groups.set(uc, []);
      groups.get(uc).push(name);
    }
    for (const [uc, filenames] of groups) {
      await api('POST', '/api/batch', {
        filenames,
        use_case:  uc,
        question,
        folder:    _filesFolder || '',
      });
    }
    toast(`Queued ${names.length} job${names.length > 1 ? 's' : ''} across ${groups.size} use case${groups.size > 1 ? 's' : ''}.`, 'success');
    hideFilesProcessBar();
    filesClearSelection();
    _showFilesJobsPanel();
  } catch (e) {
    toast(e.message, 'error');
  }
}

function _showFilesJobsPanel() {
  const panel = document.getElementById('files-jobs-panel');
  if (panel) panel.style.display = 'block';
  _refreshFilesJobList();
}

function _refreshFilesJobList() {
  const list = document.getElementById('files-job-list');
  if (!list) return;
  const jobs = (state.jobs || []).slice().reverse().slice(0, 30);
  if (!jobs.length) {
    list.innerHTML = '<div style="padding:12px 14px;font-size:13px;color:var(--tx-3)">No jobs yet.</div>';
    return;
  }
  const icons = {
    queued:     '<svg class="files-job-icon" style="color:var(--tx-3)" viewBox="0 0 20 20" fill="currentColor"><circle cx="10" cy="10" r="7" stroke="currentColor" stroke-width="1.5" fill="none"/></svg>',
    processing: '<svg class="files-job-icon" style="color:var(--blue)" viewBox="0 0 20 20" fill="currentColor"><path fill-rule="evenodd" d="M15.312 11.424a5.5 5.5 0 01-9.201 2.466l-.312-.311h2.433a.75.75 0 000-1.5H3.989a.75.75 0 00-.75.75v4.242a.75.75 0 001.5 0v-2.43l.31.31a7 7 0 0011.712-3.138.75.75 0 00-1.449-.39z"/></svg>',
    complete:   '<svg class="files-job-icon" style="color:var(--green)" viewBox="0 0 20 20" fill="currentColor"><path fill-rule="evenodd" d="M16.704 4.153a.75.75 0 01.143 1.052l-8 10.5a.75.75 0 01-1.127.075l-4.5-4.5a.75.75 0 011.06-1.06l3.894 3.893 7.48-9.817a.75.75 0 011.05-.143z"/></svg>',
    error:      '<svg class="files-job-icon" style="color:var(--red)" viewBox="0 0 20 20" fill="currentColor"><path fill-rule="evenodd" d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-8-5a.75.75 0 01.75.75v4.5a.75.75 0 01-1.5 0v-4.5A.75.75 0 0110 5zm0 10a1 1 0 100-2 1 1 0 000 2z"/></svg>',
  };
  list.innerHTML = jobs.map(j => `
    <div class="files-job-row">
      ${icons[j.status] || ''}
      <span class="files-job-file">${escHtml(j.filename)}</span>
      <span class="files-job-use-case">${escHtml(j.use_case)}</span>
      <span class="files-job-status ${j.status}">${j.status === 'complete' ? 'Done' : j.status === 'error' ? 'Error' : j.status === 'processing' ? 'Processing…' : 'Queued'}</span>
    </div>`).join('');
}

function onFilesUpload(e) {
  const files = [...e.target.files];
  if (files.length) uploadFiles(files, 'folder', _filesFolder);
  e.target.value = '';
}
function onFilesDrop(e) {
  e.preventDefault();
  const files = [...e.dataTransfer.files].filter(f => f.type.startsWith('image/'));
  if (files.length) uploadFiles(files, 'folder', _filesFolder);
}

function downloadFilesResults() {
  const q = _filesFolder ? `?folder=${encodeURIComponent(_filesFolder)}` : '';
  window.location.href = `/api/results/download${q}`;
}

async function viewFileResult(filename) {
  const folder = _filesFolder || '';
  try {
    const q    = `?filename=${encodeURIComponent(filename)}${folder ? '&folder=' + encodeURIComponent(folder) : ''}`;
    const data = await api('GET', `/api/results/read${q}`);

    // Strip the header lines (File/Folder/Use case/Date/─────) from the content
    // so only the actual OCR text is shown in the preview.
    const lines   = (data.content || '').split('\n');
    const sepIdx  = lines.findIndex(l => l.startsWith('─'));
    const body    = sepIdx >= 0 ? lines.slice(sepIdx + 1).join('\n').trimStart() : data.content;

    // Reuse the existing result modal
    document.getElementById('modal-title').textContent    = filename;
    document.getElementById('modal-subtitle').textContent = folder ? `Folder: ${folder}` : 'All Files';
    const imgEl = document.getElementById('modal-image');
    if (imgEl) {
      const src = folder
        ? `/images/${encodeURIComponent(folder)}/${encodeURIComponent(filename)}`
        : `/images/${encodeURIComponent(filename)}`;
      imgEl.src   = src;
      imgEl.style.display = 'block';
    }
    document.getElementById('modal-result').textContent = body;
    document.getElementById('result-modal').classList.add('open');
  } catch (e) {
    toast('Could not load result: ' + e.message, 'error');
  }
}

// ---------------------------------------------------------------------------
// Results download
// ---------------------------------------------------------------------------

function downloadResults() {
  window.location.href = '/api/results/download';
}

function downloadResultsCsv(folder = '') {
  const url = folder
    ? `/api/results/export-csv?folder=${encodeURIComponent(folder)}`
    : '/api/results/export-csv';
  window.location.href = url;
}

// Keep old name as alias so any remaining references still work
function downloadFolderResults() { downloadFilesResults(); }

// ---------------------------------------------------------------------------
// Utilities
// ---------------------------------------------------------------------------

function _fmtSize(bytes) {
  if (bytes < 1024)        return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

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
