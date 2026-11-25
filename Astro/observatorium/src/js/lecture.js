/* Observatorium — the lecture hall. A mission opens with its lecture in the right-hand pane: one step at a time,
 * with the equation typeset, an explainer to play with, and the dome turning to whatever is discussed. The last
 * step lists what you will measure, then hands over to the prediction. */
(function (G) {
  'use strict';
  var OBS = G.OBS, $ = OBS.$, esc = OBS.esc;
  var LEC = OBS.lecture = { m: null, k: 0, ex: null };

  /** Typeset TeX in an element with MathJax, now or as soon as MathJax has loaded. */
  var pending = [];
  OBS.typeset = function (el) {
    var MJ = G.MathJax;
    if (MJ && MJ.typesetPromise) { MJ.typesetPromise([el]).catch(function () { }); return; }
    pending.push(el);
  };
  OBS.mathReady = function () { var p = pending; pending = []; p.forEach(OBS.typeset); };

  LEC.beats = function (m) { return (m.lecture || []).concat([{ h: 'Your mission', summary: true }]); };
  function summary(m) {
    var miss = (m.needs || []).filter(function (k) { return !OBS.state.concepts[k]; });
    return '<div class="prose"><p>' + m.brief + '</p></div>' +
      '<h4>Variables</h4><div class="scroll-x"><table class="vars"><thead><tr><th>Symbol</th><th>Quantity</th><th>Unit</th><th>How it is measured</th></tr></thead><tbody>' +
      m.vars.map(function (v) { return '<tr><td class="sym">' + v[0] + '</td><td>' + v[1] + '</td><td class="unit">' + v[2] + '</td><td>' + v[3] + '</td></tr>'; }).join('') + '</tbody></table></div>' +
      '<h4>The relation you will test</h4><p class="relation">' + m.relation + '</p>' +
      '<h4>What it lets you deduce</h4><p>' + m.deduce + '</p>' +
      '<p class="course">In your course: ' + esc(m.course) + (m.hw ? ' · ' + esc(m.hw) : '') + '</p>' +
      (miss.length && !OBS.state.sandbox ? '<p class="warn">This mission uses what you learn in ' + miss.map(function (k) { var b = OBS.MISSIONS.filter(function (x) { return x.id === OBS.CONCEPTS[k].by; })[0]; return esc(OBS.logbook.number(b) + ' ' + b.title); }).join(' and ') + '. Do that first, or turn on teacher mode in the Notebook.</p>' : '');
  }
  LEC.blocked = function (m) { return !OBS.unlocked(m.needs); };

  LEC.open = function (m, k) {
    LEC.m = m; LEC.k = k || 0;
    OBS.app.sky.layers.grid = false;
    OBS.app.lectureMode();
    render(true);
  };
  LEC.go = function (k) {
    var n = LEC.beats(LEC.m).length;
    LEC.k = Math.max(0, Math.min(n - 1, k));
    render(false);
    OBS.emit('lecture', { m: LEC.m, k: LEC.k });
  };
  LEC.stop = function () { if (LEC.ex) { try { LEC.ex.stop(); } catch (e) { /* already gone */ } LEC.ex = null; } };
  /** Show the summary step again (during the prediction), without moving the sky. */
  LEC.showSummary = function (m) { LEC.m = m; LEC.k = LEC.beats(m).length - 1; OBS.app.lectureMode(); render(false, true); };

  function cue(s) {
    if (!s) return;
    var sky = OBS.app.sky;
    if (s.grid !== undefined) sky.layers.grid = s.grid;
    OBS.app.markers(s.markers || []);
    sky.goTo(s.ra, s.dec, s.fov, { mode: s.mode, frame: s.frame, photo: s.photo });
  }
  function render(first, quiet) {
    LEC.stop();
    var m = LEC.m, beats = LEC.beats(m), k = LEC.k, b = beats[k], el = $('lecture'), last = k === beats.length - 1;
    var eqs = b.eq ? (Array.isArray(b.eq) ? b.eq : [b.eq]) : [];
    el.innerHTML = '<header class="lec-head"><p class="eyebrow">Lecture · Mission ' + OBS.logbook.number(m) + ' · ' + esc(m.title) + '</p><h3>' + esc(b.h) + '</h3></header>' +
      (b.summary ? summary(m) : '<div class="lec-body prose">' + b.html + '</div>') +
      eqs.map(function (e) { return '<div class="lec-eq">\\[' + e + '\\]</div>'; }).join('') +
      (b.ex ? '<div class="lec-ex" id="lec-ex"></div>' : '') +
      '<footer class="lec-foot"><span class="dots" aria-label="Step ' + (k + 1) + ' of ' + beats.length + '">' + beats.map(function (x, i) { return '<button type="button" class="dot' + (i === k ? ' on' : '') + '" data-k="' + i + '" aria-label="Step ' + (i + 1) + ': ' + esc(x.h) + '"></button>'; }).join('') + '</span>' +
      '<span class="lec-nav">' + (k ? '<button type="button" class="btn ghost" id="lec-back">Back</button>' : '') +
      (last ? '<button type="button" class="btn primary" id="lec-pred"' + (LEC.blocked(m) && !OBS.state.sandbox ? ' disabled' : '') + '>Make a prediction</button>' : '<button type="button" class="btn primary" id="lec-next">Next</button>') +
      '</span></footer>';
    el.scrollTop = 0;
    [].forEach.call(el.querySelectorAll('.dot'), function (d) { d.addEventListener('click', function () { LEC.go(+d.dataset.k); }); });
    if ($('lec-back')) $('lec-back').addEventListener('click', function () { LEC.go(k - 1); });
    if ($('lec-next')) $('lec-next').addEventListener('click', function () { LEC.go(k + 1); });
    if ($('lec-pred')) $('lec-pred').addEventListener('click', function () { OBS.logbook.toPredict(); });
    if (b.ex && OBS.EXPLAINERS[b.ex.id]) LEC.ex = OBS.EXPLAINERS[b.ex.id]($('lec-ex'), b.ex.opts || {});
    OBS.typeset(el);
    if (quiet) return;
    if (b.sky) cue(b.sky);
    else if (first && m.target) cue(m.target);
  }
})(typeof window !== 'undefined' ? window : globalThis);
