const $ = id => document.getElementById(id);
const state = { case: null, votes: {}, batchPoll: null, activeBatchId: null };

async function api(url, options = {}) {
  const r = await fetch(url, { headers: { 'Content-Type': 'application/json' }, ...options });
  const data = await r.json();
  if (!r.ok) throw new Error(data.detail || 'Request failed');
  return data;
}

function esc(value) {
  return String(value ?? '').replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
}

if (window.marked) window.marked.setOptions({ gfm: true, breaks: true });

function markdownHtml(value) {
  const text = String(value ?? '');
  if (!window.marked || !window.DOMPurify) return esc(text).replace(/\n/g, '<br>');
  return window.DOMPurify.sanitize(window.marked.parse(text));
}

function show(id, on = true) {
  const el = $(id);
  if (el) el.classList.toggle('hidden', !on);
}

function feedbackItems() {
  return (state.case?.feedback_items || state.case?.rubric || []).map((item, i) => ({ ...item, id: item.id || `q${i + 1}` }));
}

function itemById(id) {
  return feedbackItems().find(item => item.id === id);
}

function pulse(el) {
  el.classList.add('pulse');
  setTimeout(() => el.classList.remove('pulse'), 900);
}

function focusFeedback(itemId) {
  const card = [...document.querySelectorAll('[data-feedback-item]')].find(el => el.dataset.feedbackItem === itemId);
  if (!card) return;
  card.scrollIntoView({ behavior: 'smooth', block: 'center' });
  pulse(card);
}

function focusAnchor(itemId) {
  const mark = [...document.querySelectorAll('.feedback-anchor')].find(el => el.dataset.itemId === itemId);
  if (!mark) return focusFeedback(itemId);
  mark.scrollIntoView({ behavior: 'smooth', block: 'center' });
  pulse(mark);
}

function insertAnchorSlot(root, mark, item) {
  const slot = document.createElement('div');
  slot.className = 'inline-feedback-slot';
  slot.dataset.slotItem = item.id;
  slot.dataset.option = item.anchor_option;
  const block = mark.closest('p, li, blockquote, pre, h1, h2, h3, h4, h5, h6') || mark.parentElement;
  if (block && block !== root && root.contains(block)) block.insertAdjacentElement('afterend', slot);
  else root.appendChild(slot);
}

function wrapFirstMatch(root, needle, item) {
  if (!needle || needle.length < 2) return false;
  const walker = document.createTreeWalker(root, NodeFilter.SHOW_TEXT);
  let node;
  while ((node = walker.nextNode())) {
    const index = node.nodeValue.indexOf(needle);
    if (index < 0) continue;
    const range = document.createRange();
    range.setStart(node, index);
    range.setEnd(node, index + needle.length);
    const mark = document.createElement('mark');
    mark.className = 'feedback-anchor';
    mark.dataset.itemId = item.id;
    mark.title = item.question;
    mark.addEventListener('click', () => focusFeedback(item.id));
    range.surroundContents(mark);

    const pin = document.createElement('button');
    pin.type = 'button';
    pin.className = 'anchor-pin';
    pin.textContent = 'feedback';
    pin.title = item.question;
    pin.addEventListener('click', () => focusFeedback(item.id));
    mark.insertAdjacentElement('afterend', pin);

    insertAnchorSlot(root, mark, item);
    return true;
  }
  return false;
}

function applyAnchors(id, option) {
  const root = $(id);
  feedbackItems()
    .filter(item => item.anchor_option === option && item.anchor_text)
    .forEach(item => wrapFirstMatch(root, item.anchor_text, item));
}

function renderMarkdown(id, value, option) {
  const el = $(id);
  el.classList.add('markdown');
  el.innerHTML = markdownHtml(value);
  applyAnchors(id, option);
}

function signal(choice, option) {
  return Number(choice?.signals?.[option] || 0);
}

function targetOptions(item) {
  const target = String(item.target || '').toUpperCase();
  if (target === 'A' || target === 'B') return [target];
  if (item.anchor_option === 'A' || item.anchor_option === 'B') return [item.anchor_option];
  return ['A', 'B'];
}

function choicesForOption(item, option) {
  const choices = item.choices || [];
  const targets = targetOptions(item);
  if (!targets.includes(option)) return [];
  if (targets.length === 1) return choices;
  const other = option === 'A' ? 'B' : 'A';
  const localChoices = choices.filter(choice => {
    const own = signal(choice, option);
    const otherSignal = signal(choice, other);
    if (own === 0 && otherSignal === 0) return true;
    if (own === otherSignal && own !== 0) return true;
    return own > otherSignal;
  });
  return localChoices.length ? localChoices : choices;
}

function feedbackCard(item, choices, option) {
  const answered = state.votes[item.id];
  return `
    <div class="inline-feedback-card ${answered ? 'answered' : ''}" data-feedback-item="${esc(item.id)}" data-option="${esc(option)}">
      <div class="inline-feedback-meta row">
        ${item.focus ? `<span class="badge">${esc(item.focus)}</span>` : ''}
        ${item.kind ? `<span class="badge">${esc(item.kind)}</span>` : ''}
        <span class="feedback-status">${answered ? 'answered' : 'needs feedback'}</span>
      </div>
      <div class="inline-feedback-question">${esc(item.question)}</div>
      <div class="inline-choice-grid">
        ${choices.map(choice => `
          <button class="btn feedback-choice ${state.votes[item.id] === choice.id ? 'primary' : ''}" data-item-id="${esc(item.id)}" data-choice-id="${esc(choice.id)}">
            ${esc(choice.label)}
          </button>
        `).join('')}
      </div>
      ${item.why_ask ? `<div class="inline-feedback-why">${esc(item.why_ask)}</div>` : ''}
    </div>
  `;
}

function findSlot(itemId, option) {
  return [...document.querySelectorAll('[data-slot-item]')].find(slot => slot.dataset.slotItem === itemId && slot.dataset.option === option);
}

function ensureFlowSlot(option, item) {
  const existing = findSlot(item.id, option);
  if (existing) return existing;
  const root = option === 'A' ? $('option-a') : $('option-b');
  const slot = document.createElement('div');
  slot.className = 'inline-feedback-slot flow-feedback-slot';
  slot.dataset.slotItem = item.id;
  slot.dataset.option = option;
  const blocks = [...root.children].filter(el => !el.classList.contains('inline-feedback-slot'));
  const slotCount = root.querySelectorAll('.flow-feedback-slot').length;
  const anchor = blocks[Math.min(slotCount, Math.max(0, blocks.length - 1))];
  if (anchor) anchor.insertAdjacentElement('afterend', slot);
  else root.appendChild(slot);
  return slot;
}

function renderInlineFeedback() {
  const items = feedbackItems();
  const rendered = new Set();
  document.querySelectorAll('.inline-feedback-slot').forEach(slot => { slot.innerHTML = ''; });

  document.querySelectorAll('[data-slot-item]').forEach(slot => {
    const item = itemById(slot.dataset.slotItem);
    if (!item) return;
    const option = slot.dataset.option;
    const choices = choicesForOption(item, option);
    if (!choices.length) return;
    slot.innerHTML = feedbackCard(item, choices, option);
    rendered.add(`${item.id}:${option}`);
  });

  items.forEach(item => {
    ['A', 'B'].forEach(option => {
      const choices = choicesForOption(item, option);
      if (!choices.length || rendered.has(`${item.id}:${option}`)) return;
      const slot = ensureFlowSlot(option, item);
      slot.innerHTML = feedbackCard(item, choices, option);
      rendered.add(`${item.id}:${option}`);
    });
  });

  document.querySelectorAll('[data-choice-id]').forEach(btn => btn.addEventListener('click', () => {
    state.votes[btn.dataset.itemId] = btn.dataset.choiceId;
    renderInlineFeedback();
  }));
  updateVoteProgress();
}

function updateVoteProgress() {
  const items = feedbackItems();
  const count = items.filter(item => state.votes[item.id]).length;
  $('vote-score').textContent = `${count} / ${items.length} answered`;
  $('submit-vote').disabled = Boolean(state.case?.voted) || !items.length || count !== items.length;
}

function resetVote() {
  state.votes = {};
  updateVoteProgress();
}

async function loadStatsBadge() {
  const data = await api('/api/evals/stats');
  const pending = data.summary.pending_cases || 0;
  $('pending-count').textContent = `${pending} pending`;
}

async function loadNext() {
  await loadStatsBadge();
  const { case: item } = await api('/api/evals/next');
  state.case = item;
  resetVote();
  show('reveal-card', false);
  if (!item) {
    show('empty', true);
    show('case-card', false);
    return;
  }
  show('empty', false);
  show('case-card', true);
  $('case-id').textContent = item.id.slice(0, 8);
  $('system-prompt').textContent = item.system_prompt;
  $('user-message').textContent = item.user_message;
  renderMarkdown('option-a', item.option_a_answer, 'A');
  renderMarkdown('option-b', item.option_b_answer, 'B');
  renderInlineFeedback();
}

function kindLabel(kind) {
  return kind === 'mold' ? 'Mold agent' : 'Simple agent';
}

function scoreText(scores = {}) {
  const a = Number(scores.A || 0).toFixed(1).replace(/\.0$/, '');
  const b = Number(scores.B || 0).toFixed(1).replace(/\.0$/, '');
  return `A ${a} · B ${b}`;
}

function selectedChoice(item) {
  const choiceId = state.case?.feedback_votes?.[item.id];
  return (item.choices || []).find(choice => choice.id === choiceId);
}

function renderReveal(item) {
  show('reveal-card', true);
  const winnerKind = item.human_winner === 'A' ? item.option_a_kind : item.human_winner === 'B' ? item.option_b_kind : null;
  const judge = item.judge_suggestion || '—';
  const agreed = item.judge_agreed ? 'agreed' : 'missed';
  const winnerLabel = item.human_winner === 'tie' ? 'Tie' : `Option ${esc(item.human_winner)} · ${kindLabel(winnerKind)}`;
  const feedbackRows = feedbackItems().map(item => {
    const choice = selectedChoice(item);
    return `<div class="item"><strong>${esc(item.question)}</strong><div class="muted" style="margin-top:6px">${esc(choice?.label || '—')}</div></div>`;
  }).join('');
  $('reveal').innerHTML = `
    <div class="item"><strong>Net feedback signal:</strong> ${winnerLabel} <span class="badge">${esc(scoreText(item.feedback_scores))}</span></div>
    <div class="item"><strong>Option A:</strong> ${kindLabel(item.option_a_kind)}</div>
    <div class="item"><strong>Option B:</strong> ${kindLabel(item.option_b_kind)}</div>
    <div class="item"><strong>Hidden signal prediction:</strong> Option ${esc(judge)} · ${agreed} · confidence ${Math.round((item.judge_confidence || 0) * 100)}%</div>
    <div class="item muted">${esc(item.judge_rationale || '')}</div>
    <div class="stack">${feedbackRows}</div>
  `;
  $('mold-state').textContent = JSON.stringify(item.mold_state || {}, null, 2);
  $('mold-trace').textContent = JSON.stringify(item.mold_trace || [], null, 2);
}

async function submitVote() {
  if (!state.case || state.case.voted) return;
  $('submit-vote').disabled = true;
  try {
    const { case: item } = await api(`/api/evals/cases/${state.case.id}/vote`, {
      method: 'POST',
      body: JSON.stringify({ feedback_votes: state.votes }),
    });
    state.case = item;
    state.votes = item.feedback_votes || state.votes;
    renderInlineFeedback();
    await loadStatsBadge();
    renderReveal(item);
  } catch (err) {
    alert(err.message);
    updateVoteProgress();
  }
}

function stopBatchPolling() {
  if (state.batchPoll) clearInterval(state.batchPoll);
  state.batchPoll = null;
  state.activeBatchId = null;
}

function renderBatchProgress(batch) {
  if (!batch) {
    $('generate').disabled = false;
    $('generate').textContent = 'Generate 5 blind tests';
    if (!$('generate-status').textContent.includes('failed')) $('generate-status').textContent = '';
    return;
  }
  const created = batch.created_cases || 0;
  const target = batch.target_count || 5;
  if (batch.status === 'running') {
    $('generate').disabled = true;
    $('generate').textContent = 'Generating…';
    $('generate-status').textContent = `Generating ${created} / ${target} blind tests. Safe to refresh — progress is saved.`;
    return;
  }
  $('generate').disabled = false;
  $('generate').textContent = 'Generate 5 blind tests';
  if (batch.status === 'failed') {
    $('generate-status').textContent = batch.error || 'Batch generation failed.';
  } else {
    $('generate-status').textContent = `Batch ready: ${created} / ${target} tests generated.`;
  }
}

async function pollBatch(batchId) {
  const { batch } = await api(`/api/evals/batch/${batchId}`);
  renderBatchProgress(batch);
  if (!batch || batch.status !== 'running') {
    stopBatchPolling();
    await loadStatsBadge();
    if (batch?.status === 'done') await loadNext();
  }
}

function startBatchPolling(batch) {
  if (!batch) return;
  stopBatchPolling();
  state.activeBatchId = batch.id;
  renderBatchProgress(batch);
  if (batch.status !== 'running') return;
  state.batchPoll = setInterval(() => {
    pollBatch(batch.id).catch(err => {
      stopBatchPolling();
      $('generate').disabled = false;
      $('generate').textContent = 'Generate 5 blind tests';
      $('generate-status').textContent = err.message;
    });
  }, 2500);
}

async function loadActiveBatch() {
  const { batch } = await api('/api/evals/batch/active');
  if (batch) startBatchPolling(batch);
  else renderBatchProgress(null);
}

async function generateBatch() {
  $('generate').disabled = true;
  $('generate').textContent = 'Starting…';
  $('generate-status').textContent = 'Starting background generation…';
  try {
    const { batch } = await api('/api/evals/batch', {
      method: 'POST',
      body: JSON.stringify({}),
    });
    startBatchPolling(batch);
  } catch (err) {
    $('generate-status').textContent = err.message;
    $('generate').disabled = false;
    $('generate').textContent = 'Generate 5 blind tests';
  }
}

$('generate').addEventListener('click', generateBatch);
$('submit-vote').addEventListener('click', submitVote);
$('next').addEventListener('click', loadNext);

await loadNext();
await loadActiveBatch();
