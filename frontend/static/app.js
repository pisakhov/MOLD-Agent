const $ = id => document.getElementById(id);
const api = (url, options = {}) => fetch(url, { headers: { 'Content-Type': 'application/json' }, ...options }).then(async r => {
  const data = await r.json();
  if (!r.ok) throw new Error(data.detail || 'Request failed');
  return data;
});

function escapeHtml(value) {
  return String(value).replace(/[&<>"]/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
}

if (window.marked) window.marked.setOptions({ gfm: true, breaks: true });

function markdownHtml(value) {
  const text = String(value ?? '');
  if (!window.marked || !window.DOMPurify) return escapeHtml(text).replace(/\n/g, '<br>');
  return window.DOMPurify.sanitize(window.marked.parse(text));
}

function renderMarkdown(id, value, muted = false) {
  const el = $(id);
  el.classList.toggle('muted', muted);
  el.classList.add('markdown');
  el.innerHTML = markdownHtml(value);
}

async function loadMolds() {
  const { molds } = await api('/api/molds');
  $('molds').innerHTML = molds.map((m, i) => `
    <label class="mold-card">
      <input type="radio" name="mold" value="${m.name}" ${i === 0 ? 'checked' : ''}>
      <span><strong>${escapeHtml(m.title)}</strong><span>${escapeHtml(m.description)}</span></span>
    </label>
  `).join('');
}

async function loadStatus() {
  const [{ models }, { chain }] = await Promise.all([api('/api/models/'), api('/api/models/chain')]);
  const active = chain.length || models.some(m => m.enabled && m.is_default);
  $('model-status').textContent = active ? 'model ready' : 'configure model first';
}

function setRunning(on) {
  $('run').disabled = on;
  $('run').textContent = on ? 'Running A/B…' : 'Run A/B';
}

function showPending() {
  for (const id of ['simple-answer', 'mold-answer']) {
    $(id).classList.add('muted');
    $(id).textContent = 'Thinking…';
  }
  $('state').textContent = '{}';
  $('trace').textContent = '[]';
}

function showArm(id, arm) {
  const el = $(id);
  if (arm?.error) {
    el.classList.add('muted');
    el.textContent = arm.error;
    return;
  }
  renderMarkdown(id, arm?.answer || '(no final answer)');
}

async function run() {
  const mold = document.querySelector('input[name="mold"]:checked')?.value;
  setRunning(true);
  showPending();
  try {
    const result = await api('/api/compare', {
      method: 'POST',
      body: JSON.stringify({
        message: $('message').value,
        mold_names: [mold],
        system_prompt: $('system-prompt').value || null,
      }),
    });
    showArm('simple-answer', result.simple);
    showArm('mold-answer', result.mold);
    $('state').textContent = JSON.stringify(result.mold?.mold_state || {}, null, 2);
    $('trace').textContent = JSON.stringify(result.mold?.trace || [], null, 2);
  } catch (err) {
    $('simple-answer').textContent = err.message;
    $('mold-answer').textContent = err.message;
  } finally {
    setRunning(false);
  }
}

$('run').addEventListener('click', run);
loadMolds();
loadStatus();
