const $ = id => document.getElementById(id);

async function api(url) {
  const r = await fetch(url);
  const data = await r.json();
  if (!r.ok) throw new Error(data.detail || 'Request failed');
  return data;
}

function esc(value) {
  return String(value ?? '').replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
}

function pct(value) {
  return `${Math.round((value || 0) * 100)}%`;
}

function signed(value) {
  const n = Number(value || 0);
  const text = n.toFixed(1).replace(/\.0$/, '');
  return n > 0 ? `+${text}` : text;
}

function winningKind(item) {
  if (!item.human_winner) return '—';
  if (item.human_winner === 'tie') return 'tie';
  return item.human_winner === 'A' ? item.option_a_kind : item.option_b_kind;
}

async function load() {
  const data = await api('/api/evals/stats');
  const s = data.summary || {};
  $('total').textContent = s.total_cases || 0;
  $('pending').textContent = s.pending_cases || 0;
  $('mold-rate').textContent = signed(s.mold_signal_avg);
  $('judge-rate').textContent = pct(s.judge_accuracy);

  $('per-mold').innerHTML = data.per_mold.length ? data.per_mold.map(m => `
    <div class="item row" style="justify-content: space-between">
      <div>
        <strong>${esc(m.mold_name)}</strong>
        <div class="muted">${m.voted} voted · ${m.signal_cases || 0} signal-scored</div>
      </div>
      <div class="row">
        <span class="badge">mold ${signed(m.mold_signal_avg)}</span>
        <span class="badge">simple ${signed(m.simple_signal_avg)}</span>
        <span class="badge">delta ${signed(m.mold_signal_delta)}</span>
        <span class="badge">predictor ${pct(m.judge_accuracy)}</span>
      </div>
    </div>
  `).join('') : '<p class="muted">No feedback yet.</p>';

  $('recent').innerHTML = data.recent.length ? data.recent.map(r => `
    <div class="item">
      <div class="row" style="justify-content: space-between">
        <strong>Signal: ${r.human_winner === 'tie' ? 'Tie' : `Option ${esc(r.human_winner)} · ${esc(winningKind(r))}`}</strong>
        <span class="badge">predictor ${r.judge_suggestion === r.human_winner ? 'agreed' : 'missed'}</span>
      </div>
      <div class="muted" style="margin-top:6px">${esc(r.user_message).slice(0, 220)}</div>
    </div>
  `).join('') : '<p class="muted">No recent feedback.</p>';
}

load().catch(err => { document.body.innerHTML = `<pre>${esc(err.message)}</pre>`; });
