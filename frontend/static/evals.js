const $ = id => document.getElementById(id);
const state = { case: null, votes: {} };

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

function feedbackItems() {
  return state.case?.feedback_items || state.case?.rubric || [];
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
  renderFeedback();
  renderMarkdown('option-a', item.option_a_answer, 'A');
  renderMarkdown('option-b', item.option_b_answer, 'B');
}

function anchorLine(item) {
  if (!item.anchor_option || !item.anchor_text) return '';
  return `
    <button class="quote-jump" data-jump="${esc(item.id)}">
      <span class="badge">Option ${esc(item.anchor_option)} quote</span>
      “${esc(item.anchor_text)}”
    </button>
  `;
}

function renderFeedback() {
  const items = feedbackItems();
  $('vote-score').textContent = `${Object.keys(state.votes).length} / ${items.length}`;
  $('rubric').innerHTML = items.map((item, i) => {
    const itemId = item.id || `q${i + 1}`;
    const choices = item.choices || [];
    return `
      <div class="criterion feedback-item" data-feedback-item="${esc(itemId)}">
        <div class="criterion-text">
          <div class="row" style="margin-bottom:8px">
            ${item.focus ? `<span class="badge">${esc(item.focus)}</span>` : ''}
            ${item.kind ? `<span class="badge">${esc(item.kind)}</span>` : ''}
          </div>
          <strong>${esc(item.question)}</strong>
          ${anchorLine({ ...item, id: itemId })}
          ${item.why_ask ? `<div class="muted" style="margin-top:8px">${esc(item.why_ask)}</div>` : ''}
        </div>
        <div class="criterion-actions vertical">
          ${choices.map(choice => `
            <button class="btn feedback-choice ${state.votes[itemId] === choice.id ? 'primary' : ''}" data-item-id="${esc(itemId)}" data-choice-id="${esc(choice.id)}">
              ${esc(choice.label)}
            </button>
          `).join('')}
        </div>
      </div>
    `;
  }).join('');

  $('rubric').querySelectorAll('[data-choice-id]').forEach(btn => btn.addEventListener('click', () => {
    state.votes[btn.dataset.itemId] = btn.dataset.choiceId;
    $('submit-vote').disabled = Object.keys(state.votes).length !== items.length;
    renderFeedback();
  }));
  $('rubric').querySelectorAll('[data-jump]').forEach(btn => btn.addEventListener('click', () => focusAnchor(btn.dataset.jump)));
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
  const itemId = item.id;
  const choiceId = state.case?.feedback_votes?.[itemId];
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
  if (!state.case) return;
  $('submit-vote').disabled = true;
  try {
    const { case: item } = await api(`/api/evals/cases/${state.case.id}/vote`, {
      method: 'POST',
      body: JSON.stringify({ feedback_votes: state.votes }),
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
  $('generate-status').textContent = 'Creating 5 blind tests and LLM-scanned feedback prompts. This can take a few minutes.';
  try {
    await api('/api/evals/batch', {
      method: 'POST',
      body: JSON.stringify({}),
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

await loadNext();
