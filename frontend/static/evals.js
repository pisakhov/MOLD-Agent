const $ = id => document.getElementById(id);
const state = { case: null, votes: {} };

async function api(url, options = {}) {
  const r = await fetch(url, { headers: { 'Content-Type': 'application/json' }, ...options });
  const data = await r.json();
  if (!r.ok) throw new Error(data.detail || 'Request failed');
  return data;
}

function esc(value) {
  return String(value ?? '').replace(/[&<>"]/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
}

function selectedMold() {
  return document.querySelector('input[name="mold"]:checked')?.value || 'refined_minimal_final_response';
}

async function loadMolds() {
  const { molds } = await api('/api/molds');
  $('molds').innerHTML = molds.map((m, i) => `
    <label class="mold-card">
      <input type="radio" name="mold" value="${esc(m.name)}" ${i === 0 ? 'checked' : ''}>
      <span><strong>${esc(m.title)}</strong><span>${esc(m.description)}</span></span>
    </label>
  `).join('');
}

async function loadStatsBadge() {
  const data = await api('/api/evals/stats');
  const pending = data.summary.pending_cases || 0;
  $('pending-count').textContent = `${pending} pending`;
}

function show(id, on = true) {
  $(id).classList.toggle('hidden', !on);
}

function resetVote() {
  state.votes = {};
  $('submit-vote').disabled = true;
}

async function loadNext() {
  resetVote();
  await loadStatsBadge();
  const { case: item } = await api('/api/evals/next');
  state.case = item;
  show('reveal-card', false);
  if (!item) {
    show('empty', true);
    show('case-card', false);
    show('rubric-card', false);
    return;
  }
  show('empty', false);
  show('case-card', true);
  show('rubric-card', true);
  $('case-id').textContent = item.id.slice(0, 8);
  $('system-prompt').textContent = item.system_prompt;
  $('user-message').textContent = item.user_message;
  $('option-a').textContent = item.option_a_answer;
  $('option-b').textContent = item.option_b_answer;
  renderRubric();
}

function renderRubric() {
  const criteria = state.case?.rubric || [];
  $('vote-score').textContent = `${Object.keys(state.votes).length} / ${criteria.length}`;
  $('rubric').innerHTML = criteria.map((text, i) => `
    <div class="criterion">
      <div class="criterion-text">${esc(text)}</div>
      <div class="criterion-actions">
        <button class="btn ${state.votes[i] === 'A' ? 'primary' : ''}" data-i="${i}" data-vote="A">A</button>
        <button class="btn ${state.votes[i] === 'B' ? 'primary' : ''}" data-i="${i}" data-vote="B">B</button>
      </div>
    </div>
  `).join('');
  $('rubric').querySelectorAll('button').forEach(btn => btn.addEventListener('click', () => {
    state.votes[btn.dataset.i] = btn.dataset.vote;
    $('submit-vote').disabled = Object.keys(state.votes).length !== criteria.length;
    renderRubric();
  }));
}

function kindLabel(kind) {
  return kind === 'mold' ? 'Mold agent' : 'Simple agent';
}

function renderReveal(item) {
  show('reveal-card', true);
  const winnerKind = item.human_winner === 'A' ? item.option_a_kind : item.option_b_kind;
  const judge = item.judge_suggestion || '—';
  const agreed = item.judge_agreed ? 'agreed' : 'missed';
  $('reveal').innerHTML = `
    <div class="item"><strong>Your winner:</strong> Option ${esc(item.human_winner)} · ${kindLabel(winnerKind)}</div>
    <div class="item"><strong>Option A:</strong> ${kindLabel(item.option_a_kind)}</div>
    <div class="item"><strong>Option B:</strong> ${kindLabel(item.option_b_kind)}</div>
    <div class="item"><strong>Hidden judge suggestion:</strong> Option ${esc(judge)} · ${agreed} · confidence ${Math.round((item.judge_confidence || 0) * 100)}%</div>
    <div class="item muted">${esc(item.judge_rationale || '')}</div>
  `;
  $('mold-state').textContent = JSON.stringify(item.mold_state || {}, null, 2);
  $('mold-trace').textContent = JSON.stringify(item.mold_trace || [], null, 2);
}

async function submitVote() {
  if (!state.case) return;
  $('submit-vote').disabled = true;
  try {
    const { case: item } = await api(`/api/evals/cases/${state.case.id}/vote`, {
      method: 'POST',
      body: JSON.stringify({ criteria_votes: state.votes }),
    });
    state.case = item;
    await loadStatsBadge();
    renderReveal(item);
  } catch (err) {
    alert(err.message);
    $('submit-vote').disabled = false;
  }
}

async function generateBatch() {
  $('generate').disabled = true;
  $('generate').textContent = 'Generating…';
  $('generate-status').textContent = 'Creating 5 random blind tests. This can take a few minutes.';
  try {
    await api('/api/evals/batch', {
      method: 'POST',
      body: JSON.stringify({ mold_name: selectedMold() }),
    });
    $('generate-status').textContent = 'Batch ready.';
    await loadNext();
  } catch (err) {
    $('generate-status').textContent = err.message;
  } finally {
    $('generate').disabled = false;
    $('generate').textContent = 'Generate 5 blind tests';
  }
}

$('generate').addEventListener('click', generateBatch);
$('submit-vote').addEventListener('click', submitVote);
$('next').addEventListener('click', loadNext);

await loadMolds();
await loadNext();
