/* Observatorium — the logbook: chapters, the mission card (lecture → predict → exercise → explain), the notebook
 * of what you can measure so far, a before/after concept check, and the data sources. */
(function (G) {
  'use strict';
  var OBS = G.OBS, $ = OBS.$, esc = OBS.esc;
  var LB = OBS.logbook = {};
  var P = OBS.store.get('progress', { done: {}, concepts: {}, quiz: {} });
  OBS.state.concepts = P.concepts || {};
  OBS.state.sandbox = !!OBS.store.get('sandbox', false);
  var ml = OBS.store.get('mlslope', null); if (ml) OBS.astro.ML_SLOPE = ml;
  function save() { P.concepts = OBS.state.concepts; OBS.store.set('progress', P); }
  var cur = { ch: OBS.store.get('chapter', 1), m: null, step: 0, st: null, pred: null, timer: null };

  LB.current = function () { return cur; };   // for tests and for a teacher's console
  function byId(id) { for (var i = 0; i < OBS.MISSIONS.length; i++) if (OBS.MISSIONS[i].id === id) return OBS.MISSIONS[i]; return null; }
  function number(m) { var list = OBS.MISSIONS.filter(function (x) { return x.ch === m.ch; }); return m.ch + '.' + (list.indexOf(m) + 1); }
  LB.number = number;
  function needsMet(m) { return OBS.learned(m.needs); }
  function missingNames(m) {
    return (m.needs || []).filter(function (k) { return !OBS.state.concepts[k]; }).map(function (k) { var by = byId(OBS.CONCEPTS[k].by); return by ? number(by) + ' ' + by.title : k; });
  }

  // ------------------------------------------------------------------------------------------ chapters
  LB.renderChapters = function () {
    var nav = $('chapters');
    nav.innerHTML = OBS.CHAPTERS.map(function (c) {
      var ms = OBS.MISSIONS.filter(function (m) { return m.ch === c.n; }), d = ms.filter(function (m) { return P.done[m.id]; }).length;
      return '<button type="button" class="chap" data-ch="' + c.n + '" aria-pressed="' + (c.n === cur.ch) + '"' + (c.locked ? ' disabled' : '') + ' title="' + esc(c.lecture) + '">' +
        '<span class="chap-n">' + c.n + '</span><span class="chap-t">' + esc(c.title) + '</span>' + (ms.length ? '<span class="chap-p">' + d + '/' + ms.length + '</span>' : '<span class="chap-p">soon</span>') + '</button>';
    }).join('');
    [].forEach.call(nav.querySelectorAll('.chap'), function (b) { b.addEventListener('click', function () { cur.ch = +b.dataset.ch; OBS.store.set('chapter', cur.ch); LB.renderChapters(); LB.renderList(); LB.tab('missions'); }); });
    var all = OBS.MISSIONS.length, done = OBS.MISSIONS.filter(function (m) { return P.done[m.id]; }).length;
    $('progress').textContent = done + ' of ' + all + ' missions';
  };
  LB.renderList = function () {
    var c = OBS.CHAPTERS[cur.ch - 1], ms = OBS.MISSIONS.filter(function (m) { return m.ch === cur.ch; });
    $('lb-list').innerHTML = '<p class="eyebrow">Chapter ' + c.n + (c.hw ? ' · ' + esc(c.hw) : '') + '</p><h2 class="chap-title">' + esc(c.title) + '</h2><p class="muted small">' + esc(c.lecture) + '</p>' +
      '<ol class="mlist">' + ms.map(function (m) {
        var d = P.done[m.id], ok = needsMet(m);
        return '<li><button type="button" class="mitem' + (cur.m === m ? ' on' : '') + '" data-id="' + m.id + '">' +
          '<span class="mnum">' + number(m) + '</span><span class="mtitle">' + esc(m.title) + '<span class="mera">' + esc(m.era) + ' · ' + esc(m.who) + '</span></span>' +
          '<span class="mstate ' + (d ? 'done' : ok ? '' : 'needs') + '">' + (d ? 'done' : ok ? '' : 'needs ' + esc(missingNames(m)[0].split(' ')[0])) + '</span></button></li>';
      }).join('') + '</ol>';
    [].forEach.call($('lb-list').querySelectorAll('.mitem'), function (b) { b.addEventListener('click', function () { LB.start(byId(b.dataset.id)); }); });
  };

  // ------------------------------------------------------------------------------------------ the mission card
  var STEPS = ['Lecture', 'Predict', 'Exercise', 'Explain'];
  /** Lend the mission's own concepts while it is open (or take them back with m = null). */
  function lend(m) {
    OBS.state.lend = {};
    if (m) (m.unlocks || []).forEach(function (k) { OBS.state.lend[k] = true; });
    OBS.emit('unlock');
  }
  LB.start = function (m) {
    stopTimer(); OBS.runCleanups();
    cur.m = m; lend(m); cur.step = 0; cur.pred = (P.done[m.id] && P.done[m.id].pred) || null; cur.st = {};
    if (m.ch !== cur.ch) { cur.ch = m.ch; LB.renderChapters(); }
    LB.renderList(); LB.tab('missions');
    OBS.lecture.open(m, 0);      // the dome turns to the mission's part of the sky straight away
    renderCard();
    $('lb-card').scrollIntoView({ block: 'nearest' });
  };
  /** From the lecture's last step to the prediction. */
  LB.toPredict = function () { if (!cur.m || OBS.lecture.blocked(cur.m) && !OBS.state.sandbox) return; cur.step = 1; renderCard(); };
  OBS.on('lecture', function () { if (cur.m && cur.step === 0) renderCard(); });
  /** Leave the mission card; if the lens still shows a lecture, give it back its data. */
  LB.leave = leave;
  function leave() {
    cur.m = null; stopTimer(); OBS.runCleanups(); lend(null); renderCard(); LB.renderList();
    if (OBS.app.lensMode() === 'lecture') OBS.app.open(OBS.app.chart.ds && OBS.DATASETS[OBS.app.chart.ds.id] ? OBS.app.chart.ds.id : 'sky');
  }
  function stepper() {
    return '<ol class="steps">' + STEPS.map(function (s, k) { return '<li class="' + (k === cur.step ? 'on' : k < cur.step ? 'past' : '') + '">' + s + '</li>'; }).join('') + '</ol>';
  }
  function renderCard() {
    var m = cur.m, el = $('lb-card');
    if (!m) { el.hidden = true; return; }
    el.hidden = false;
    var head = '<header><p class="eyebrow">Mission ' + number(m) + ' · ' + esc(m.era) + '</p><h3>' + esc(m.title) + '</h3><p class="who">' + esc(m.who) + '</p></header>' + stepper();
    var body = '';
    if (cur.step === 0) {
      var miss = missingNames(m), beats = OBS.lecture.beats(m), k = OBS.lecture.m === m ? OBS.lecture.k : 0;
      body = '<p class="muted small">The lecture is in the right-hand pane; the sky follows it. Steps:</p>' +
        '<ol class="toc">' + beats.map(function (b, i) { return '<li><button type="button" class="toc-b' + (i === k ? ' on' : '') + '" data-k="' + i + '">' + esc(b.h) + '</button></li>'; }).join('') + '</ol>' +
        (miss.length && !OBS.state.sandbox ? '<p class="warn">This mission uses ideas from ' + miss.map(esc).join(' and ') + '. Do that first, or turn on teacher mode in the Notebook.</p>' : '') +
        '<div class="actions"><button type="button" class="btn ghost" id="m-skip"' + (miss.length && !OBS.state.sandbox ? ' disabled' : '') + '>Skip to the prediction</button></div>';
    } else if (cur.step === 1) {
      body = '<p class="question">' + esc(m.question) + '</p><div class="choices" role="radiogroup">' +
        m.choices.map(function (c) { return '<label class="choice"><input type="radio" name="pred" value="' + c[0] + '"' + (cur.pred === c[0] ? ' checked' : '') + '><span>' + esc(c[1]) + '</span></label>'; }).join('') +
        '</div><p class="muted small">Commit before you look. A wrong prediction you can explain afterwards teaches more than a right guess.</p>' +
        '<div class="actions"><button type="button" class="btn ghost" id="m-back">Back</button><button type="button" class="btn primary" id="m-next"' + (cur.pred ? '' : ' disabled') + '>Lock in and observe</button></div>';
    } else if (cur.step === 2) {
      body = '<p class="your-pred">Your prediction: <b>' + esc(choiceText(m, cur.pred)) + '</b></p>' +
        '<div class="task"><h4>Your task</h4><p>' + m.task + '</p></div>' +
        '<p class="status" id="m-status" role="status">Setting up the instruments…</p>' +
        '<div class="actions"><button type="button" class="btn ghost" id="m-reset">Reset the view</button><button type="button" class="btn primary" id="m-next" disabled>What does it mean?</button></div>';
    } else {
      var right = cur.pred === m.answer, meas = m.measured ? safe(function () { return m.measured(OBS.app, cur.st); }) : m.answer;
      body = '<p class="verdict ' + (right ? 'ok' : 'miss') + '">' + (right ? 'Your prediction held: ' : 'You predicted “' + esc(choiceText(m, cur.pred)) + '”. The data say: ') + '<b>' + esc(choiceText(m, meas || m.answer)) + '</b></p>' +
        '<h4>Your measurement</h4><div class="result">' + (safe(function () { return m.result(OBS.app, cur.st); }) || '') + '</div>' +
        '<h4>What it means</h4><div class="prose"><p>' + m.explain + '</p></div>' +
        '<h4>What really happened</h4><div class="prose"><p>' + m.real + '</p></div>' +
        '<p class="course">In your course: ' + esc(m.course) + (m.hw ? ' · ' + esc(m.hw) : '') + '</p>' +
        (m.unlocks ? '<p class="unlock">Unlocked: ' + m.unlocks.map(function (k) { return '<span class="chip">' + esc(OBS.CONCEPTS[k].label) + '</span>'; }).join(' ') + (m.unlockText ? ' ' + esc(m.unlockText) : '') + '</p>' : '') +
        practice(m) +
        '<div class="actions"><button type="button" class="btn ghost" id="m-free">Keep exploring</button><button type="button" class="btn primary" id="m-next">' + (nextMission(m) ? 'Next mission' : 'Back to the chapter') + '</button></div>';
    }
    el.innerHTML = head + '<div class="mbody">' + body + '</div>';
    bindCard();
    if (cur.step === 1) OBS.lecture.showSummary(m);
    if (cur.step === 2) observe();
    if (cur.step === 3) { bindPractice(m); OBS.typeset(el); }
  }
  function safe(fn) { try { return fn(); } catch (e) { if (G.console) console.error(e); return ''; } }
  function choiceText(m, k) { for (var i = 0; i < m.choices.length; i++) if (m.choices[i][0] === k) return m.choices[i][1]; return k; }
  function nextMission(m) { var i = OBS.MISSIONS.indexOf(m), n = OBS.MISSIONS[i + 1]; return n && !OBS.CHAPTERS[n.ch - 1].locked ? n : null; }
  function bindCard() {
    var m = cur.m, next = $('m-next'), back = $('m-back'), reset = $('m-reset'), free = $('m-free'), skip = $('m-skip');
    [].forEach.call(document.querySelectorAll('.toc-b'), function (b) { b.addEventListener('click', function () { OBS.lecture.go(+b.dataset.k); }); });
    if (skip) skip.addEventListener('click', function () { LB.toPredict(); });
    [].forEach.call(document.querySelectorAll('input[name="pred"]'), function (r) { r.addEventListener('change', function () { cur.pred = r.value; next.disabled = false; }); });
    if (back) back.addEventListener('click', function () { cur.step--; if (cur.step === 0) OBS.lecture.open(m, OBS.lecture.beats(m).length - 1); renderCard(); });
    if (reset) reset.addEventListener('click', function () { stopTimer(); cur.st = {}; renderCard(); });
    if (free) free.addEventListener('click', leave);
    if (next) next.addEventListener('click', function () {
      if (cur.step === 3) { var n = nextMission(m); if (n) LB.start(n); else leave(); return; }
      if (cur.step === 2) {
        // check once more: the measurement may have been undone since the button lit up
        next.disabled = true;
        Promise.resolve(safe(function () { return m.progress(OBS.app, cur.st); })).then(function (r) {
          if (cur.m !== m || cur.step !== 2) return;
          if (!r || !r.done) { showStatus(r, true); return; }
          finish(); cur.step++; renderCard();
        });
        return;
      }
      cur.step++; renderCard();
    });
  }
  // ------------------------------------------------------------------------------------------ practice
  function practice(m) {
    if (!m.practice || !m.practice.length) return '';
    return '<h4>Practice</h4><ol class="practice">' + m.practice.map(function (p, i) {
      return '<li><p>' + p.q + '</p><div class="pr-row"><input type="number" step="any" inputmode="decimal" id="pr-' + i + '" aria-label="Your answer"><span class="unit">' + esc(p.unit || '') + '</span>' +
        '<button type="button" class="btn small" data-pr="' + i + '">Check</button><span class="pr-msg" id="pr-msg-' + i + '" role="status"></span></div>' +
        '<details class="pr-sol"><summary>Show the solution</summary><p>' + p.sol + '</p></details></li>';
    }).join('') + '</ol>';
  }
  function bindPractice(m) {
    [].forEach.call(document.querySelectorAll('[data-pr]'), function (b) {
      b.addEventListener('click', function () {
        var i = +b.dataset.pr, p = m.practice[i], x = parseFloat($('pr-' + i).value), msg = $('pr-msg-' + i);
        if (!isFinite(x)) { msg.textContent = 'Type a number first.'; msg.className = 'pr-msg'; return; }
        var ok = Math.abs(x - p.a) <= Math.abs(p.a) * (p.tol || 0.02);
        msg.textContent = ok ? 'Right.' : (x > p.a ? 'Too high.' : 'Too low.') + ' Try again, or open the solution.';
        msg.className = 'pr-msg ' + (ok ? 'ok' : 'miss');
      });
    });
  }

  // ------------------------------------------------------------------------------------------ observing
  function stopTimer() { if (cur.timer) { clearInterval(cur.timer); cur.timer = null; } cur.busy = false; }
  function observe() {
    var m = cur.m, st = cur.st, app = OBS.app;
    app.reset();
    Promise.resolve().then(function () { return m.setup(app, st); }).then(function () {
      if (cur.m !== m) return;
      check();
      cur.timer = setInterval(check, 400);
    }).catch(function (e) { $('m-status').textContent = 'Could not set up this mission: ' + e.message; });
    function check() {
      if (cur.m !== m || cur.step !== 2 || cur.busy) return;
      var r = safe(function () { return m.progress(app, st); });
      if (r && typeof r.then === 'function') {
        cur.busy = true;
        r.then(function (x) { cur.busy = false; show(x); }, function (e) { cur.busy = false; $('m-status').textContent = e.message; });
      } else show(r);
    }
    function show(r) { showStatus(r); }
  }
  /** Show the exercise's progress; the button follows it both ways (done → enabled, undone → disabled again). */
  function showStatus(r, after) {
    var s = $('m-status'), b = $('m-next'); if (!s || !b || !r) return;
    s.innerHTML = (r.done ? '<span class="pill ok">done</span> ' : '<span class="pill">' + (after ? 'not yet' : 'in progress') + '</span> ') + esc(r.text || '');
    if (r.done && b.disabled) { b.disabled = false; b.classList.add('pulse'); }
    else if (!r.done && !b.disabled) { b.disabled = true; b.classList.remove('pulse'); }
  }
  function finish() {
    var m = cur.m; stopTimer();
    P.done[m.id] = { pred: cur.pred, right: cur.pred === m.answer, at: Date.now() };
    (m.unlocks || []).forEach(function (k) { OBS.state.concepts[k] = true; });
    save(); OBS.emit('unlock'); LB.renderChapters(); LB.renderList(); LB.renderNotebook();
  }
  OBS.on('selection', function () { if (cur.step === 2 && cur.timer) setTimeout(function () { }, 0); });

  // ------------------------------------------------------------------------------------------ tabs
  LB.tab = function (t) {
    ['missions', 'notebook', 'quiz', 'data'].forEach(function (k) {
      $('lb-' + k).hidden = k !== t;
      var b = document.querySelector('[data-tab="' + k + '"]'); if (b) b.setAttribute('aria-selected', k === t);
    });
    if (t === 'notebook') LB.renderNotebook();
    if (t === 'quiz') LB.renderQuiz();
    if (t === 'data') LB.renderData();
  };

  // ------------------------------------------------------------------------------------------ notebook
  LB.renderNotebook = function () {
    var el = $('lb-notebook'); if (el.hidden) return;
    var ds = OBS.app && OBS.app.chart.ds && OBS.app.chart.ds.vars && OBS.DATASETS[OBS.app.chart.ds.id] ? OBS.app.chart.ds : null;
    el.innerHTML = '<h2 class="chap-title">Notebook</h2><p class="muted small">What you can measure so far. Each relation opens new axes in every dataset.</p>' +
      '<ul class="concepts">' + Object.keys(OBS.CONCEPTS).map(function (k) {
        var c = OBS.CONCEPTS[k], on = OBS.state.concepts[k], by = byId(c.by);
        return '<li class="' + (on ? 'on' : '') + '"><b>' + esc(c.label) + '</b><span class="rel">' + esc(c.rel) + '</span><span class="src">' + (on ? 'from mission ' + number(by) : OBS.state.sandbox ? 'open in teacher mode' : 'locked: mission ' + number(by) + ', ' + esc(by.title)) + '</span></li>';
      }).join('') + '</ul>' +
      (ds ? '<h4>Variables in “' + esc(ds.title) + '”</h4><table class="vars"><tbody>' + ds.vars.map(function (v) {
        var ok = OBS.unlocked(v.needs);
        return '<tr class="' + (ok ? '' : 'locked') + '"><td>' + esc(v.label) + (v.unit ? ' <span class="unit">[' + esc(v.unit) + ']</span>' : '') + '</td><td>' + (ok ? esc(v.how || '') : 'Locked until: ' + v.needs.filter(function (k) { return !OBS.state.concepts[k]; }).map(function (k) { return esc(OBS.CONCEPTS[k].label.toLowerCase()); }).join(', ')) + '</td></tr>';
      }).join('') + '</tbody></table>' : '') +
      '<h4>Teacher mode</h4><label class="check"><input type="checkbox" id="sandbox"' + (OBS.state.sandbox ? ' checked' : '') + '> Unlock every variable and mission, for exploring or preparing a class</label>' +
      '<div class="actions"><button type="button" class="btn ghost" id="reset-progress">Reset my progress</button></div><p class="small muted" id="reset-confirm"></p>';
    $('sandbox').addEventListener('change', function (e) { OBS.state.sandbox = e.target.checked; OBS.store.set('sandbox', e.target.checked); OBS.emit('unlock'); LB.renderNotebook(); LB.renderList(); if (cur.m && cur.step === 0) renderCard(); });
    $('reset-progress').addEventListener('click', function () {
      var c = $('reset-confirm');
      c.innerHTML = 'This clears your missions, predictions and quiz answers in this browser. <button type="button" class="btn small" id="reset-yes">Clear everything</button>';
      $('reset-yes').addEventListener('click', function () { P = { done: {}, concepts: {}, quiz: {} }; OBS.state.concepts = {}; OBS.store.set('mlslope', null); OBS.astro.ML_SLOPE = 3.86; save(); OBS.emit('unlock'); LB.renderChapters(); LB.renderList(); LB.renderNotebook(); });
    });
  };
  OBS.on('unlock', function () { LB.renderNotebook(); });
  OBS.on('dataset', function () { LB.renderNotebook(); });

  // ------------------------------------------------------------------------------------------ check yourself
  LB.renderQuiz = function () {
    var el = $('lb-quiz'), Q = OBS.QUIZ, q = P.quiz || {}, which = q.before && q.before.done ? 'after' : 'before';
    if (q.after && q.after.done) which = q.mode || 'after';
    var ans = (q[which] && q[which].ans) || [];
    function score(k) { var a = q[k] && q[k].ans; if (!a) return null; var s = 0; Q.forEach(function (x, i) { if (a[i] === x.a) s++; }); return s; }
    var b = score('before'), a = score('after');
    var gain = b !== null && a !== null && b < Q.length ? (a - b) / (Q.length - b) : null;
    el.innerHTML = '<h2 class="chap-title">Check yourself</h2><p class="muted small">' + Q.length + ' questions, one per big idea. Take them once before the missions and once after; the page works out your normalised gain, the share of the questions you missed before that you now get right.</p>' +
      '<p>' + (b === null ? 'No “before” attempt yet.' : 'Before: <span class="num">' + b + '/' + Q.length + '</span>') + (a === null ? '' : ' · After: <span class="num">' + a + '/' + Q.length + '</span>') + (gain !== null ? ' · Gain <span class="num">' + gain.toFixed(2) + '</span>' : '') + '</p>' +
      '<form id="quiz-form"><p class="eyebrow">Now taking: the “' + which + '” attempt</p><ol class="quiz">' + Q.map(function (x, i) {
        return '<li><p>' + esc(x.q) + '</p>' + x.c.map(function (c, j) { return '<label class="choice"><input type="radio" name="q' + i + '" value="' + j + '"' + (ans[i] === j ? ' checked' : '') + '><span>' + esc(c) + '</span></label>'; }).join('') + '</li>';
      }).join('') + '</ol><div class="actions"><button type="submit" class="btn primary">Save the “' + which + '” answers</button><button type="button" class="btn ghost" id="quiz-copy">Copy results</button></div><p class="small muted" id="quiz-msg"></p></form>';
    $('quiz-form').addEventListener('submit', function (e) {
      e.preventDefault();
      var got = Q.map(function (x, i) { var r = el.querySelector('input[name="q' + i + '"]:checked'); return r ? +r.value : null; });
      P.quiz = P.quiz || {}; P.quiz[which] = { ans: got, done: true, at: Date.now() }; save(); LB.renderQuiz();
      $('quiz-msg').textContent = 'Saved.';
    });
    $('quiz-copy').addEventListener('click', function () {
      var lines = ['Observatorium results', 'Before: ' + (b === null ? '–' : b + '/' + Q.length), 'After: ' + (a === null ? '–' : a + '/' + Q.length), 'Normalised gain: ' + (gain === null ? '–' : gain.toFixed(2)), 'Missions:'];
      OBS.MISSIONS.forEach(function (m) { var d = P.done[m.id]; if (d) lines.push('  ' + number(m) + ' ' + m.title + ': predicted ' + (d.right ? 'right' : 'wrong') + ' (' + choiceText(m, d.pred) + ')'); });
      var txt = lines.join('\n'), msg = $('quiz-msg');
      if (navigator.clipboard) navigator.clipboard.writeText(txt).then(function () { msg.textContent = 'Copied.'; }, function () { msg.innerHTML = '<textarea readonly rows="6">' + esc(txt) + '</textarea>'; });
      else msg.innerHTML = '<textarea readonly rows="6">' + esc(txt) + '</textarea>';
    });
  };

  // ------------------------------------------------------------------------------------------ data sources
  LB.renderData = function () {
    var el = $('lb-data');
    OBS.loadJSON('manifest').then(function (man) {
      el.innerHTML = '<h2 class="chap-title">Where the data come from</h2><p class="muted small">Every point in the lens is a real measurement. These are the catalogues, the cuts applied and the papers to cite.</p>' +
        Object.keys(man).filter(function (k) { return k[0] !== '_'; }).map(function (k) {
          var d = man[k];
          return '<details class="src"><summary><b>' + esc(d.title || k) + '</b> <span class="num muted">' + (d.rows ? d.rows.toLocaleString('en-US') + ' rows' : '') + '</span></summary>' +
            '<dl>' + [['Source', d.source], ['Query', d.query], ['Cuts', d.cuts], ['Columns', d.columns_note], ['Cite', d.citation], ['Licence', d.license], ['Fetched', d.fetched]].filter(function (r) { return r[1]; }).map(function (r) { return '<dt>' + r[0] + '</dt><dd>' + esc(r[1]) + '</dd>'; }).join('') + '</dl></details>';
        }).join('');
    });
  };
})(typeof window !== 'undefined' ? window : globalThis);
