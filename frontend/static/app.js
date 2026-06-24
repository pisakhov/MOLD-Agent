const $ = id => document.getElementById(id);
const api = (url, options = {}) => fetch(url, { headers: { 'Content-Type': 'application/json' }, ...options }).then(async r => {
  const data = await r.json();
  if (!r.ok) throw new Error(data.detail || 'Request failed');
  return data;
});

function escapeHtml(value) {
  return String(value).replace(/[&<>"]/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
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

async function run() {
  const mold = document.querySelector('input[name="mold"]:checked')?.value;
  $('run').disabled = true;
  $('run').textContent = 'Running…';
  $('answer').classList.add('muted');
  $('answer').textContent = 'Thinking through the mold…';
  try {
    const result = await api('/api/run', {
      method: 'POST',
      body: JSON.stringify({
        message: $('message').value,
        mold_names: [mold],
        system_prompt: $('system-prompt').value || null,
      }),
    });
    $('answer').classList.remove('muted');
    $('answer').textContent = result.answer || '(no final answer)';
    $('state').textContent = JSON.stringify(result.mold_state, null, 2);
    $('trace').textContent = JSON.stringify(result.trace, null, 2);
  } catch (err) {
    $('answer').textContent = err.message;
  } finally {
    $('run').disabled = false;
    $('run').textContent = 'Run';
  }
}

$('run').addEventListener('click', run);
loadMolds();
loadStatus();
