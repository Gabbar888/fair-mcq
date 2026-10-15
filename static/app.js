const app = document.getElementById('app');
const ACTIVE = 'activeExam';      // in-progress test (survives refresh)
const PENDING = 'pendingResult';  // finished test not yet saved to server

const md = s => marked.parse(s || '');
const esc = s => String(s ?? '').replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
const load = k => { try { return JSON.parse(localStorage.getItem(k)); } catch { return null; } };
const store = (k, v) => v == null ? localStorage.removeItem(k) : localStorage.setItem(k, JSON.stringify(v));

const stem = f => f.replace(/\.md$/i, '').replace(/_/g, ' ');
const bmKey = (file, num) => file + '#' + num;
// "Question 23" for a mock; "Mock Exam 1 · Q23" for a custom test.
const qLabel = (q, custom) => custom ? `${esc(stem(q.file))} · Q${q.num}` : `Question ${q.num}`;
function starButton(file, num, marks) {
  const btn = document.createElement('button');
  btn.className = 'star';
  const draw = () => { const on = marks.includes(bmKey(file, num)); btn.textContent = on ? '★ Bookmarked' : '☆ Bookmark'; btn.classList.toggle('on', on); };
  btn.onclick = async () => { marks.splice(0, marks.length, ...await api('/api/bookmark', { file, num, on: !marks.includes(bmKey(file, num)) })); draw(); };
  draw();
  return btn;
}

function fmt(ms) {
  const t = Math.max(0, Math.round(ms / 1000));
  const h = Math.floor(t / 3600), m = Math.floor(t % 3600 / 60), s = t % 60;
  return (h ? h + ':' : '') + String(m).padStart(h ? 2 : 1, '0') + ':' + String(s).padStart(2, '0');
}

async function api(path, body) {
  const res = await fetch(path, body === undefined ? {} : {
    method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body),
  });
  const data = await res.json();
  if (!res.ok) throw new Error(data.error || res.statusText);
  return data;
}

let timer = null;
function stopTimer() { clearInterval(timer); timer = null; }

function route() {
  stopTimer();
  const h = location.hash;
  if (h === '#/exam') return renderExam();
  if (h === '#/custom') return renderBuilder();
  if (h.startsWith('#/review/')) return renderReview(decodeURIComponent(h.slice(9)));
  renderDashboard();
}
window.addEventListener('hashchange', route);
route();

/* ---------------- Dashboard ---------------- */

async function renderDashboard() {
  if (load(PENDING)) return savePending();
  const [exams, results] = await Promise.all([api('/api/exams'), api('/api/results')]);
  const active = load(ACTIVE);
  app.innerHTML = `
  <div class="page">
    <h1>Mock Exams</h1>
    ${active ? `<div class="card warn"><span>Test in progress: <b>${esc(active.exam.title)}</b> (Q${active.current + 1} of ${active.exam.questions.length})</span>
      <button class="primary" onclick="location.hash='#/exam'">Resume</button></div>` : ''}
    <section class="card">
      <h2>Attempt a new test</h2>
      <div class="pick">
        <select id="examPick">${exams.map(f => `<option value="${esc(f)}">${esc(stem(f))}</option>`).join('')}</select>
        <button class="primary" id="startPick" ${active || !exams.length ? 'disabled' : ''}>Start</button>
      </div>
      <label class="upload muted">Upload a new .md exam <input type="file" accept=".md" id="upload"></label>
      <div id="upmsg"></div>
    </section>
    <section class="card">
      <h2>Custom practice</h2>
      <p class="muted">Pick questions by bookmark, status and domain, and set your own time.</p>
      <button class="primary" onclick="location.hash='#/custom'" ${active ? 'disabled' : ''}>Build custom test</button>
    </section>
    <section class="card">
      <h2>Previous attempts</h2>
      ${results.length ? `<table class="results">
        <tr><th>Finished</th><th>Exam</th><th>Score</th><th>Time used</th><th>Ended by</th><th></th></tr>
        ${results.map(r => `<tr class="row" data-id="${esc(r.id)}">
          <td>${new Date(r.finishedAt).toLocaleString()}</td><td>${esc(r.examTitle)}</td>
          <td><b>${r.score}</b> / ${r.total}</td><td>${fmt(r.elapsedMs)}</td>
          <td>${r.endReason === 'timeout' ? 'Time up' : 'Completed'}</td>
          <td><button data-del="${esc(r.id)}">Delete</button></td></tr>`).join('')}
      </table>` : '<p class="muted">No attempts yet.</p>'}
    </section>
  </div>`;

  document.getElementById('startPick').onclick = () => startExam(document.getElementById('examPick').value);
  app.querySelectorAll('[data-id]').forEach(r => r.onclick = () => location.hash = '#/review/' + encodeURIComponent(r.dataset.id));
  app.querySelectorAll('[data-del]').forEach(b => b.onclick = async e => {
    e.stopPropagation();
    if (!confirm('Delete this attempt permanently?')) return;
    await api('/api/delete-result', { id: b.dataset.del });
    renderDashboard();
  });
  document.getElementById('upload').onchange = async e => {
    const file = e.target.files[0];
    const msg = document.getElementById('upmsg');
    try {
      await api('/api/upload', { name: file.name, content: await file.text() });
      renderDashboard();
    } catch (err) { msg.innerHTML = `<p class="error">${esc(err.message)}</p>`; }
  };
}

async function startExam(file) {
  beginExam(file, await api('/api/exam?file=' + encodeURIComponent(file)));
}

function beginExam(file, exam) {
  if (!confirm(`Start "${exam.title}"?\n\n${exam.questions.length} questions, ${exam.minutes} minutes.\nYou can only move forward. The timer starts now.`)) return;
  store(ACTIVE, {
    file, exam, startedAt: new Date().toISOString(),
    current: 0, draft: null, answers: [],
    accMs: 0, runningSince: Date.now(), pauses: [],
  });
  location.hash = '#/exam';
}

/* ---------------- Exam ---------------- */

async function renderExam() {
  const s = load(ACTIVE);
  if (!s) { location.hash = '#/'; return; }
  const marks = await api('/api/bookmarks');
  const qs = s.exam.questions, limit = s.exam.minutes * 60000;
  const elapsed = () => s.accMs + (s.runningSince ? Date.now() - s.runningSince : 0);

  app.innerHTML = `
  <div id="exam">
    <div class="topbar">
      <div class="title">${esc(s.exam.title)}</div>
      <div id="qcount"></div>
      <div id="timer" class="timer"></div>
      <button id="pause">Pause</button>
    </div>
    <div id="strip" class="strip">${qs.map(() => '<div></div>').join('')}</div>
    <div class="panes">
      <div class="pane" id="qpane"></div>
      <div class="pane">
        <div id="opts"></div>
        <div class="actions">
          <button id="skip">Skip</button>
          <button id="submit" class="primary">Submit &amp; Next</button>
        </div>
      </div>
    </div>
  </div>
  <div id="overlay" class="overlay" hidden>
    <div class="big">Paused</div>
    <div class="muted">Reason: <span id="preason"></span></div>
    <div>Paused for <span id="pfor"></span> &middot; Time left <span id="pleft"></span></div>
    <button class="primary" id="resume">Resume</button>
  </div>
  <dialog id="pdlg">
    <h2>Why are you pausing?</h2>
    <textarea id="ptext" placeholder="Reason (required)"></textarea>
    <div class="row"><button id="pcancel">Cancel</button><button class="primary" id="pok">Pause test</button></div>
  </dialog>`;

  const $ = id => document.getElementById(id);

  function drawQuestion() {
    const q = qs[s.current];
    $('qcount').textContent = `Q ${s.current + 1} / ${qs.length}`;
    [...$('strip').children].forEach((c, i) => {
      c.className = i === s.current ? 'current' : (s.answers[i]?.status || '');
      c.title = 'Q' + (i + 1);
    });
    $('qpane').innerHTML = `<div class="qnum">${qLabel(q, s.exam.custom)} </div>${md(q.body)}`;
    $('qpane').firstChild.append(starButton(q.file || s.file, q.num, marks));
    $('opts').innerHTML = Object.entries(q.options).map(([L, text]) => `
      <label class="option ${s.draft === L ? 'selected' : ''}">
        <input type="radio" name="opt" value="${L}" ${s.draft === L ? 'checked' : ''}>
        <span class="letter">${L}.</span><div class="text">${md(text)}</div>
      </label>`).join('');
    $('opts').querySelectorAll('input').forEach(inp => inp.onchange = () => {
      s.draft = inp.value; store(ACTIVE, s);
      $('opts').querySelectorAll('.option').forEach(o => o.classList.toggle('selected', o.contains(inp)));
      $('submit').disabled = false;
    });
    $('submit').disabled = !s.draft;
    $('qpane').scrollTop = 0;
  }

  function advance(status) {
    s.answers[s.current] = { chosen: status === 'answered' ? s.draft : null, status };
    s.draft = null;
    s.current++;
    store(ACTIVE, s);
    if (s.current >= qs.length) return finish(s, 'completed', elapsed());
    drawQuestion();
  }

  function showPaused(on) {
    $('overlay').hidden = !on;
    $('exam').style.visibility = on ? 'hidden' : '';
  }

  function tick() {
    const left = limit - elapsed();
    if (left <= 0) return finish(s, 'timeout', limit);
    $('timer').textContent = fmt(left);
    $('timer').classList.toggle('low', left < 5 * 60000);
    if (!s.runningSince) {
      const p = s.pauses[s.pauses.length - 1];
      $('preason').textContent = p.reason;
      $('pfor').textContent = fmt(Date.now() - p.startedAt);
      $('pleft').textContent = fmt(left);
    }
  }

  $('submit').onclick = () => s.draft && advance('answered');
  $('skip').onclick = () => advance('skipped');
  $('pause').onclick = () => { $('ptext').value = ''; $('pdlg').showModal(); };
  $('pcancel').onclick = () => $('pdlg').close();
  $('pok').onclick = () => {
    const reason = $('ptext').value.trim();
    if (!reason) return $('ptext').focus();
    s.accMs = elapsed();
    s.runningSince = null;
    s.pauses.push({ reason, atQuestion: s.current + 1, startedAt: Date.now(), elapsedAtPauseMs: s.accMs, durationMs: null });
    store(ACTIVE, s);
    $('pdlg').close();
    showPaused(true);
    tick();
  };
  $('resume').onclick = () => {
    const p = s.pauses[s.pauses.length - 1];
    p.durationMs = Date.now() - p.startedAt;
    s.runningSince = Date.now();
    store(ACTIVE, s);
    showPaused(false);
  };

  drawQuestion();
  showPaused(!s.runningSince);
  tick();
  timer = setInterval(tick, 250);
}

function finish(s, endReason, elapsedMs) {
  stopTimer();
  const qs = s.exam.questions;
  // On timeout, a selected-but-not-submitted option on the current question counts.
  if (endReason === 'timeout' && s.current < qs.length && s.draft)
    s.answers[s.current] = { chosen: s.draft, status: 'answered' };
  const items = qs.map((q, i) => {
    const a = s.answers[i];
    const chosen = a?.chosen ?? null;
    return { chosen, status: a?.status ?? 'unseen', correct: chosen !== null && chosen === q.answer };
  });
  store(PENDING, {
    examFile: s.file, examTitle: s.exam.title, minutes: s.exam.minutes, custom: !!s.exam.custom,
    startedAt: s.startedAt, finishedAt: new Date().toISOString(), endReason,
    elapsedMs: Math.min(elapsedMs, s.exam.minutes * 60000),
    score: items.filter(x => x.correct).length, total: qs.length,
    items, pauses: s.pauses, exam: s.exam,
  });
  store(ACTIVE, null);
  savePending();
}

async function savePending() {
  const r = load(PENDING);
  try {
    const { id } = await api('/api/results', r);
    store(PENDING, null);
    location.hash = '#/review/' + encodeURIComponent(id);
  } catch (err) {
    app.innerHTML = `<div class="page"><div class="card">
      <h2>Test finished — score ${r.score} / ${r.total}</h2>
      <p class="error">Could not save the result to the server: ${esc(err.message)}</p>
      <p>The result is kept in this browser. Start the server and retry.</p>
      <button class="primary" id="retry">Retry save</button></div></div>`;
    document.getElementById('retry').onclick = savePending;
  }
}

/* ---------------- Review ---------------- */

async function renderReview(id) {
  const [r, marks] = await Promise.all([api('/api/result?id=' + encodeURIComponent(id)), api('/api/bookmarks')]);
  const qs = r.exam.questions;
  const cat = i => r.items[i].correct ? 'correct' : r.items[i].status === 'answered' ? 'wrong' : 'unanswered';
  const all = qs.map((_, i) => i);
  const groups = {
    all, wrong: all.filter(i => cat(i) === 'wrong'),
    unanswered: all.filter(i => cat(i) === 'unanswered'), correct: all.filter(i => cat(i) === 'correct'),
  };
  const labels = { all: 'All', wrong: 'Wrong', unanswered: 'Skipped / not reached', correct: 'Correct' };
  let filter = groups.wrong.length ? 'wrong' : 'all';
  let pos = 0;

  // Per-domain breakdown, only if the exam has domain tags.
  const domains = {};
  qs.forEach((q, i) => {
    if (!q.domain) return;
    const d = domains[q.domain] ??= { total: 0, correct: 0, wrong: 0 };
    d.total++;
    if (cat(i) === 'correct') d.correct++;
    if (cat(i) === 'wrong') d.wrong++;
  });

  app.innerHTML = `
  <div class="topbar">
    <div class="title">${esc(r.examTitle)} — review</div>
    <button onclick="location.hash='#/'">Dashboard</button>
  </div>
  <div class="page" style="max-width:none">
    <div class="card">
      <div class="summary">
        <span class="score">${r.score} / ${r.total}</span>
        <span class="badge ok">${groups.correct.length} correct</span>
        <span class="badge bad">${groups.wrong.length} wrong</span>
        <span class="badge">${groups.unanswered.length} skipped / not reached</span>
        <span class="muted">Time used ${fmt(r.elapsedMs)} of ${r.minutes} min &middot;
          ${r.endReason === 'timeout' ? 'Time ran out' : 'Completed'} &middot; ${new Date(r.finishedAt).toLocaleString()}</span>
      </div>
      ${Object.keys(domains).length ? `<details><summary>By domain</summary><table>
        <tr><th>Domain</th><th>Correct</th><th>Wrong</th><th>Total</th></tr>
        ${Object.entries(domains).map(([d, v]) => `<tr><td>${esc(d)}</td><td>${v.correct}</td><td>${v.wrong}</td><td>${v.total}</td></tr>`).join('')}
      </table></details>` : ''}
      ${r.pauses.length ? `<details><summary>Pauses (${r.pauses.length})</summary><table>
        <tr><th>At question</th><th>Exam time</th><th>Paused for</th><th>Reason</th></tr>
        ${r.pauses.map(p => `<tr><td>Q${p.atQuestion}</td><td>${fmt(p.elapsedAtPauseMs)}</td>
          <td>${p.durationMs == null ? '—' : fmt(p.durationMs)}</td><td>${esc(p.reason)}</td></tr>`).join('')}
      </table></details>` : ''}
      <div class="filters" id="filters"></div>
      <div class="grid" id="grid"></div>
    </div>
  </div>
  <div class="review-nav topbar">
    <button id="prev">&larr; Prev</button>
    <span id="navpos"></span>
    <button id="next">Next &rarr;</button>
  </div>
  <div class="panes review">
    <div class="pane" id="qpane"></div>
    <div class="pane" id="opts"></div>
  </div>`;

  const $ = id => document.getElementById(id);

  function draw() {
    const list = groups[filter];
    $('filters').innerHTML = Object.keys(groups).map(k =>
      `<button data-f="${k}" class="${k === filter ? 'on' : ''}">${labels[k]} (${groups[k].length})</button>`).join('');
    $('filters').querySelectorAll('button').forEach(b => b.onclick = () => { filter = b.dataset.f; pos = 0; draw(); });
    $('grid').innerHTML = list.map((i, p) =>
      `<button data-p="${p}" class="${cat(i)} ${p === pos ? 'current' : ''}" title="${r.custom ? esc(stem(qs[i].file)) + ' · Q' + qs[i].num : ''}">${r.custom ? i + 1 : qs[i].num}</button>`).join('');
    $('grid').querySelectorAll('button').forEach(b => b.onclick = () => { pos = +b.dataset.p; draw(); });
    $('prev').disabled = pos <= 0;
    $('next').disabled = pos >= list.length - 1;

    if (!list.length) {
      $('navpos').textContent = 'No questions in this filter';
      $('qpane').innerHTML = $('opts').innerHTML = '';
      return;
    }
    const i = list[pos], q = qs[i], it = r.items[i];
    $('navpos').textContent = `${pos + 1} of ${list.length} (${labels[filter]})`;
    const status = cat(i) === 'correct' ? '<span class="badge ok">Correct</span>'
      : cat(i) === 'wrong' ? '<span class="badge bad">Wrong</span>'
      : `<span class="badge">${it.status === 'skipped' ? 'Skipped' : 'Not reached'}</span>`;
    $('qpane').innerHTML = `<div class="qnum">${qLabel(q, r.custom)} ${status}
      ${q.domain ? `<span class="badge">${esc(q.domain)}</span>` : ''} </div>${md(q.body)}`;
    $('qpane').firstChild.append(starButton(q.file || r.examFile, q.num, marks));
    $('opts').innerHTML = Object.entries(q.options).map(([L, text]) => {
      const cls = L === q.answer ? 'is-correct' : L === it.chosen ? 'is-wrong' : 'plain';
      const tags = (L === q.answer ? '<span class="badge ok">Correct answer</span>' : '')
        + (L === it.chosen ? `<span class="badge ${it.correct ? 'ok' : 'bad'}">Your answer</span>` : '');
      return `<div class="option ${cls}"><span class="letter">${L}.</span><div class="text">${md(text)}${tags}</div></div>`;
    }).join('') + `
      <div class="explain"><b>Answer: ${q.answer ?? '—'}</b>
        ${q.erratum ? `<div class="muted">Answer key said ${q.erratum}; corrected by errata to ${q.answer}.</div>` : ''}
        ${md(q.explanation || '_No explanation given._')}</div>`;
    $('qpane').scrollTop = $('opts').scrollTop = 0;
  }

  $('prev').onclick = () => { pos--; draw(); };
  $('next').onclick = () => { pos++; draw(); };
  draw();
}

/* ---------------- Custom test builder ---------------- */

async function renderBuilder() {
  if (load(ACTIVE)) { location.hash = '#/exam'; return; }
  const [pool, marks] = await Promise.all([api('/api/pool'), api('/api/bookmarks')]);
  pool.forEach(q => { if (marks.includes(bmKey(q.file, q.num))) q.statuses.push('bookmarked'); });
  const TYPES = {
    bookmarked: 'Bookmarked', wrong: 'Wrong', skipped: 'Skipped', notReached: 'Not reached',
    neverAttempted: 'Never attempted (exam not taken)', correct: 'Correct',
  };
  const dom = q => q.domain || '(none)';
  const files = [...new Set(pool.map(q => q.file))];
  const domains = [...new Set(pool.map(dom))].sort();
  const sel = { files: new Set(files), types: new Set(['bookmarked', 'wrong']), domains: new Set(domains) };
  const excluded = new Set();
  let minutesEdited = false;

  app.innerHTML = `
  <div class="topbar"><div class="title">Build custom test</div><button onclick="location.hash='#/'">Dashboard</button></div>
  <div class="page">
    <section class="card"><h2>1. Source exams</h2><div class="checks" id="files"></div></section>
    <section class="card"><h2>2. Question types <span class="muted">(any ticked type)</span></h2><div class="checks" id="types"></div></section>
    <section class="card"><h2>3. Domains</h2><div class="checks" id="domains"></div></section>
    <section class="card">
      <h2>4. Matching questions <span class="muted">(click to exclude / include)</span></h2>
      <div id="chips" class="chips"></div>
    </section>
    <section class="card">
      <h2>5. Settings</h2>
      <div class="settings">
        <label>Order <select id="order"><option value="orig">Original</option><option value="shuffle">Shuffled</option></select></label>
        <label>Use at most <input type="number" id="limit" min="1" placeholder="all"> questions (random pick)</label>
        <label>Time <input type="number" id="minutes" min="1" step="1"> minutes</label>
      </div>
      <p id="summary"></p>
      <button class="primary" id="start">Start custom test</button>
    </section>
  </div>`;
  const $ = id => document.getElementById(id);

  const checks = (id, items, set, count) => {
    $(id).innerHTML = items.map(([k, label]) => `<label><input type="checkbox" value="${esc(k)}" ${set.has(k) ? 'checked' : ''}>
      ${esc(label)}${count ? ` <span class="muted">(${count(k)})</span>` : ''}</label>`).join('');
    $(id).querySelectorAll('input').forEach(i => i.onchange = () => { i.checked ? set.add(i.value) : set.delete(i.value); draw(); });
  };
  const inFiles = q => sel.files.has(q.file);
  const inTypes = q => q.statuses.some(t => sel.types.has(t));
  const inDomains = q => sel.domains.has(dom(q));
  const picked = () => pool.filter(q => inFiles(q) && inTypes(q) && inDomains(q) && !excluded.has(bmKey(q.file, q.num)));
  const finalCount = n => { const lim = parseInt($('limit').value); return lim > 0 && lim < n ? lim : n; };

  function draw() {
    checks('files', files.map(f => [f, stem(f)]), sel.files);
    checks('types', Object.entries(TYPES), sel.types,
      t => pool.filter(q => inFiles(q) && inDomains(q) && q.statuses.includes(t)).length);
    checks('domains', domains.map(d => [d, d]), sel.domains,
      d => pool.filter(q => inFiles(q) && inTypes(q) && dom(q) === d).length);
    const match = pool.filter(q => inFiles(q) && inTypes(q) && inDomains(q));
    $('chips').innerHTML = match.map(q => {
      const k = bmKey(q.file, q.num);
      return `<button class="chip ${excluded.has(k) ? 'off' : ''}" data-k="${esc(k)}">${esc(stem(q.file))} · Q${q.num} · ${esc(dom(q))}
        <span class="muted">${q.statuses.map(t => t === 'bookmarked' ? '★' : TYPES[t].split(' (')[0].toLowerCase()).join(', ')}</span></button>`;
    }).join('') || '<p class="muted">No questions match.</p>';
    $('chips').querySelectorAll('.chip').forEach(c => c.onclick = () => {
      excluded.has(c.dataset.k) ? excluded.delete(c.dataset.k) : excluded.add(c.dataset.k); draw();
    });
    drawSummary();
  }

  function drawSummary() {
    const all = picked(), n = finalCount(all.length);
    if (!minutesEdited) $('minutes').value = Math.ceil(n * 100 / 60);
    $('summary').innerHTML = `<b>${n}</b> questions in the test` + (n < all.length ? ` (random ${n} of ${all.length})` : '')
      + ` &middot; ${$('minutes').value || 0} min`;
    $('start').disabled = !n || !(+$('minutes').value > 0);
  }
  $('limit').oninput = drawSummary;
  $('minutes').oninput = () => { minutesEdited = true; drawSummary(); };

  $('start').onclick = async () => {
    let qs = picked();
    const n = finalCount(qs.length);
    if (n < qs.length) qs = shuffle(qs).slice(0, n).sort((a, b) => pool.indexOf(a) - pool.indexOf(b));
    if ($('order').value === 'shuffle') qs = shuffle(qs);
    const exam = await api('/api/custom', { picks: qs.map(q => ({ file: q.file, num: q.num })), minutes: +$('minutes').value });
    beginExam('custom', exam);
  };
  draw();
}

function shuffle(a) {
  a = [...a];
  for (let i = a.length - 1; i > 0; i--) { const j = Math.floor(Math.random() * (i + 1)); [a[i], a[j]] = [a[j], a[i]]; }
  return a;
}
