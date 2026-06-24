const $ = id => document.getElementById(id);

async function api(url) {
  const r = await fetch(url);
  const data = await r.json();
  if (!r.ok) throw new Error(data.detail || 'Request failed');
  return data;
}

function esc(value) {
  return String(value ?? '').replace(/[&<>"]/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
}

function pct(value) {
  return `${Math.round((value || 0) * 100)}%`;
}

function winningKind(item) {
  if (!item.human_winner) return '—';
  return item.human_winner === 'A' ? item.option_a_kind : item.option_b_kind;
}

async function load() {
  const data = await api('/api/evals/stats');
  const s = data.summary || {};
  $('total').textContent = s.total_cases || 0;
  $('pending').textContent = s.pending_cases || 0;
  $('mold-rate').textContent = pct(s.mold_win_rate);
  $('judge-rate').textContent = pct(s.judge_accuracy);

  $('per-mold').innerHTML = data.per_mold.length ? data.per_mold.map(m => `
    <div class="item row" style="justify-content: space-between">
      <div>
        <strong>${esc(m.mold_name)}</strong>
        <div class="muted">${m.voted} voted · ${m.mold_wins || 0} mold wins</div>
      </div>
      <div class="row">
        <span class="badge">mold ${pct(m.mold_win_rate)}</span>
        <span class="badge">judge ${pct(m.judge_accuracy)}</span>
      </div>
    </div>
  `).join('') : '<p class="muted">No votes yet.</p>';

  $('recent').innerHTML = data.recent.length ? data.recent.map(r => `
    <div class="item">
      <div class="row" style="justify-content: space-between">
        <strong>Winner: Option ${esc(r.human_winner)} · ${esc(winningKind(r))}</strong>
        <span class="badge">judge ${r.judge_suggestion === r.human_winner ? 'agreed' : 'missed'}</span>
      </div>
      <div class="muted" style="margin-top:6px">${esc(r.user_message).slice(0, 220)}</div>
    </div>
  `).join('') : '<p class="muted">No recent votes.</p>';
}

load().catch(err => { document.body.innerHTML = `<pre>${esc(err.message)}</pre>`; });
