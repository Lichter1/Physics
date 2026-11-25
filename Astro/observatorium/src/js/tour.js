/* Observatorium — the first-visit tour. A spotlight walks through the page one part at a time: what each part
 * is for, how to work it, and two small "try it" moments (turn the sky, brush the lens). It starts by itself on a
 * first visit, can be left at any step (Esc), and is replayed from the Tour button.
 *   step: {at: selector or function -> element (none: a centred card), h, html, pane (unfold first), tab,
 *          lens: true (the lens must show data, not a lecture), try: {say, start(state), done(state) -> bool, ok},
 *          enter / exit: set up and undo a demonstration} */
(function (G) {
  'use strict';
  var OBS = G.OBS, $ = OBS.$, esc = OBS.esc;
  var T = OBS.tour = { on: false, k: 0 };

  function first() { return OBS.MISSIONS[0]; }
  function unfold(p) { var pane = $('pane-' + p); if (pane && pane.classList.contains('folded')) { var b = pane.querySelector('[data-unfold]'); if (b) b.click(); } }
  function visible(el) { if (!el) return false; var r = el.getBoundingClientRect(); return r.width > 4 && r.height > 4 && getComputedStyle(el).visibility !== 'hidden'; }

  var STEPS = [
    { h: 'Welcome to the Observatorium', wide: true,
      html: function () {
        return '<p>A planetarium and a lab in one page, built for General Astronomy 1. You retrace the course’s discoveries yourself, with the measurements astronomers actually made: Gaia, Hipparcos, Kepler, OGLE, SDSS and more.</p>' +
          '<div class="tour-map" aria-hidden="true"><span><b>Logbook</b>chapters, missions, notebook</span><span><b>Sky</b>the real stars, drawn or photographed</span><span><b>Data lens</b>the same objects as numbers; the lecture hall</span></div>' +
          '<p>This walk-through takes about two minutes. Leave it at any step with <kbd>Esc</kbd>, and replay it from <b>Tour</b> at the top.</p>';
      }, next: 'Show me around', skip: 'Skip, I’ll explore' },
    { at: '#chapters', h: 'Chapters follow the lectures',
      html: '<p>Chapter 1 is Lecture 1, and so on. Each chapter holds a few missions, and the small counter shows how many you have finished. The chapters zoom out as the course does: from single stars to clusters, the Milky Way and other galaxies. Chapter 6, the Universe, opens with the cosmology lectures.</p>' },
    { at: '#lb-list', pane: 'logbook', tab: 'missions', h: 'Missions: one discovery each',
      html: '<p>Every mission retraces one real measurement, with the year and the people who made it. Later missions build on earlier ones: <span class="tour-tag">needs 1.2</span> means mission 1.2 teaches something this one uses. (Teacher mode in the Notebook opens everything at once.)</p><p>Click a mission to start it. Finished ones are marked <span class="tour-tag ok">done</span>.</p>' },
    { at: '#pane-logbook', pane: 'logbook', h: 'Every mission runs in four steps',
      html: '<ol class="tour-steps"><li><b>Lecture.</b> The theory, step by step, in the right-hand pane, while the sky turns to whatever is being discussed.</li>' +
        '<li><b>Predict.</b> Commit to an answer before you see the data. Guessing first is what makes the result stick, right or wrong.</li>' +
        '<li><b>Exercise.</b> Make the measurement yourself on the real data. The card tells you what to do and says when you have it.</li>' +
        '<li><b>Explain.</b> Your numbers, what they mean, what really happened in history, and practice problems with worked solutions.</li></ol>' +
        '<p>Finishing a mission unlocks a new quantity, such as distance, luminosity or mass, for every dataset.</p>' },
    { at: '.skycanvas', pane: 'sky', h: 'The sky',
      html: '<p>Every point is a real star (or galaxy, or asteroid) from the open catalogue. <b>Drag</b> to turn, <b>scroll</b> or pinch to zoom, <b>click</b> an object for its card. Shift-drag selects a circle of sky.</p>',
      try: { say: 'Try it: drag the sky a little, or zoom.', ok: 'That’s it. The readout under the sky tracks where you are looking.',
        start: function (s) { var c = OBS.app.sky.centre(); s.c = c; s.fov = OBS.app.sky.fov; },
        done: function (s) { var k = OBS.app.sky, c = k.centre(); return Math.abs(c[0] - s.c[0]) + Math.abs(c[1] - s.c[1]) > 1 || Math.abs(k.fov - s.fov) > 2; } } },
    { at: '#pane-sky .ctl-bar', pane: 'sky', h: 'How the sky is shown',
      html: '<dl class="tour-dl"><dt>Drawn · Photo</dt><dd>Catalogue stars as points of light, or real photographs: ESO’s Milky Way panorama with telescope close-ups of clusters, nebulae, galaxies and lenses. Credits appear under the sky.</dd>' +
        '<dt>Dome · Whole sky</dt><dd>A view through a telescope dome, or the entire sky as one map.</dd>' +
        '<dt>Equatorial · Galactic</dt><dd>Line the map up with the Earth’s equator, or with the Milky Way.</dd>' +
        '<dt>Milky Way · Lines · Grid · Names</dt><dd>Layers you can switch on and off.</dd>' +
        '<dt>Turn · Select circle</dt><dd>What dragging does: turn the sky, or draw a circle that selects everything inside it.</dd></dl>' },
    { at: '#ladder', pane: 'sky', h: 'The scale ladder',
      html: '<p>Jump between scales: the naked-eye sky, the Solar System, the stars within 100 pc, star clusters, the Milky Way and other galaxies. Each rung opens that scale’s data in the lens and turns the sky to match.</p>' },
    { at: '#pane-lens .chartwrap', pane: 'lens', lens: true, h: 'The data lens',
      html: '<p>The same objects as the sky, as numbers: each dot is one object, placed by any two measured quantities. Right now it is the naked-eye sky, colour against brightness.</p><p>The lens and the sky are linked. Select objects in one and they light up in the other.</p>',
      try: { say: 'Try it: drag a box around a group of dots.', ok: 'Those objects are now highlighted on the sky as well. Clear, under the plot, removes the selection.',
        start: function (s) { s.n = OBS.app.chart.selected().length; s.hit = false; s.off = OBS.on('selection', function () { s.hit = true; }); },
        done: function (s) { return s.hit && OBS.app.chart.selected().length > 0; } } },
    { at: '#lens-ctl', pane: 'lens', lens: true, h: 'Choose what to plot',
      html: '<p><b>Dataset</b> picks one of 21 real catalogues. <b>Across</b> and <b>Up</b> pick the two axes; <i>Number of objects</i> turns the plot into a histogram. <b>Colour by</b> paints the dots by another quantity.</p>' +
        '<p>Quantities you have not learned to measure yet are not in the menus. They appear when a mission teaches the method, the way they did in history: no distances before parallax, no luminosities before distances.</p>' },
    { at: '#pane-lens', pane: 'lens', h: 'The lecture hall',
      enter: function () { var m = first(), k = OBS.lecture.beats(m).map(function (b) { return b.h; }).indexOf('Magnitudes'); OBS.lecture.open(m, Math.max(0, k)); },
      exit: function () { if (OBS.app.lensMode() === 'lecture') OBS.app.open('sky'); },
      html: '<p>When you start a mission, this pane becomes the lecture. Here is a real step from mission 1.1: the idea, its equation, a live chart of real data, a diagram to play with and a worked example (scroll the pane). Other steps have photographs (click one to enlarge it). The sky turns to whatever is being discussed: it has just turned to Sirius.</p><p>Move with <b>Next</b>, <b>Back</b> or the dots. The last step lists exactly what you will measure, and hands over to your prediction.</p>' },
    { at: '#pane-logbook .tabs', pane: 'logbook', h: 'More in the logbook',
      html: '<dl class="tour-dl"><dt>Notebook</dt><dd>What you can measure so far, and how each quantity of the open dataset is measured. Teacher mode unlocks everything.</dd>' +
        '<dt>Check yourself</dt><dd>Twelve concept questions. Answer them before you start and again at the end, and the page shows how much you gained.</dd>' +
        '<dt>Data</dt><dd>Every source, query, cut, citation and licence behind the numbers.</dd></dl>' },
    { at: function () { return visible($('split')) ? $('split') : document.querySelector('#pane-sky [data-fold]'); }, h: 'Make room',
      html: '<p><b>Hide</b> folds any pane into a slim tab; click the tab to bring it back. Drag the divider between the sky and the lens to share the space differently (double-click resets it).</p><p>Your layout, view and progress are saved in this browser only.</p>' },
    { at: '.top-right', h: 'Night view, plate view',
      html: '<p><b>Plate view</b> shows the sky as a photographic negative, dark stars on pale glass, the way Henrietta Leavitt and the Harvard computers saw it. <b>Tour</b> replays this walk-through.</p>' },
    { h: 'Where to start', wide: true,
      html: function () {
        var m = first();
        return '<p>Begin with mission ' + OBS.logbook.number(m) + ', <b>' + esc(m.title) + '</b>: what can be learned from where stars are, how bright they look and what colour they are, before anyone knew their distances?</p>' +
          '<p>A tip: answer the twelve questions in <b>Check yourself</b> first. Doing them again at the end shows how far you have come.</p>';
      }, last: true }
  ];
  T.steps = STEPS;

  // ------------------------------------------------------------------------------------------ drawing
  var root, spot, pop, raf = 0, st = {}, done = false, lastRect = '';
  function build() {
    root = document.createElement('div'); root.className = 'tour';
    root.innerHTML = '<div class="tour-spot"></div><div class="tour-pop" role="dialog" aria-modal="false" aria-labelledby="tour-h" aria-describedby="tour-b"></div>';
    document.body.appendChild(root);
    spot = root.firstChild; pop = root.lastChild;
    pop.addEventListener('keydown', function (e) {
      if (e.key === 'ArrowRight' && !e.target.matches('input, select, textarea')) { e.preventDefault(); T.go(T.k + 1); }
      if (e.key === 'ArrowLeft' && !e.target.matches('input, select, textarea')) { e.preventDefault(); T.go(T.k - 1); }
    });
  }
  function target(s) {
    if (!s.at) return null;
    var el = typeof s.at === 'function' ? s.at() : document.querySelector(s.at);
    return visible(el) ? el : null;
  }
  var shown = null;   // the step on screen, so its demonstration can be undone
  function leaveStep() { if (shown && shown.exit) shown.exit(); shown = null; }
  function render() {
    var s = STEPS[T.k], n = STEPS.length, html = typeof s.html === 'function' ? s.html() : s.html;
    if (shown !== s) { leaveStep(); shown = s; if (s.enter) s.enter(); }
    if (s.pane) unfold(s.pane);
    if (s.tab) OBS.logbook.tab(s.tab);
    if (s.lens && OBS.app.lensMode() !== 'data') OBS.app.open('sky');
    if (st.off) st.off(); st = {}; done = false;
    if (s.try) s.try.start(st);
    var first = T.k === 0;
    pop.className = 'tour-pop' + (s.wide ? ' wide' : '');
    pop.innerHTML = '<p class="tour-count">' + (first ? 'Tour' : 'Step ' + T.k + ' of ' + (n - 1)) + '</p>' +
      '<h3 id="tour-h">' + esc(s.h) + '</h3><div class="tour-body" id="tour-b">' + html + '</div>' +
      (s.try ? '<p class="tour-try" aria-live="polite"><span class="tour-tick" aria-hidden="true"></span><span class="tour-say">' + esc(s.try.say) + '</span></p>' : '') +
      '<div class="tour-foot">' +
      (s.last ? '<button type="button" class="btn ghost" data-t="explore">Explore on my own</button><button type="button" class="btn" data-t="quiz">Check yourself first</button><button type="button" class="btn primary" data-t="begin">Start mission ' + OBS.logbook.number(OBS.MISSIONS[0]) + '</button>'
        : (first ? '<button type="button" class="btn ghost" data-t="end">' + esc(s.skip) + '</button>' : '<button type="button" class="linkish" data-t="end">End tour</button><button type="button" class="btn ghost" data-t="back">Back</button>') +
        '<button type="button" class="btn primary" data-t="next">' + esc(s.next || 'Next') + '</button>') +
      '</div>';
    [].forEach.call(pop.querySelectorAll('[data-t]'), function (b) {
      b.addEventListener('click', function () {
        var a = b.dataset.t;
        if (a === 'next') T.go(T.k + 1);
        else if (a === 'back') T.go(T.k - 1);
        else if (a === 'end' || a === 'explore') T.end();
        else if (a === 'quiz') { T.end(); unfold('logbook'); OBS.logbook.tab('quiz'); }
        else if (a === 'begin') { T.end(); OBS.logbook.start(OBS.MISSIONS[0]); }
      });
    });
    var el = target(s);
    if (el && el.scrollIntoView) el.scrollIntoView({ block: 'nearest', inline: 'nearest' });
    lastRect = '';
    place();
    var focus = pop.querySelector('.btn.primary'); if (focus) focus.focus({ preventScroll: true });
  }
  /** Put the spotlight on the step's element and the card beside it, wherever there is room. */
  function place() {
    var s = STEPS[T.k], el = target(s), W = G.innerWidth, H = G.innerHeight, M = 12;
    var r = el ? el.getBoundingClientRect() : null;
    var key = r ? [r.left, r.top, r.width, r.height, W, H].map(Math.round).join() : 'c' + W + 'x' + H;
    if (key === lastRect) return; lastRect = key;
    root.classList.toggle('centred', !el);
    if (!el) { spot.style.cssText = ''; pop.style.left = Math.max(M, (W - pop.offsetWidth) / 2) + 'px'; pop.style.top = Math.max(M, (H - pop.offsetHeight) / 2) + 'px'; return; }
    var p = 6, x0 = Math.max(4, r.left - p), y0 = Math.max(4, r.top - p), x1 = Math.min(W - 4, r.right + p), y1 = Math.min(H - 4, r.bottom + p);
    spot.style.cssText = 'left:' + x0 + 'px;top:' + y0 + 'px;width:' + Math.max(0, x1 - x0) + 'px;height:' + Math.max(0, y1 - y0) + 'px';
    var pw = pop.offsetWidth, ph = pop.offsetHeight, x, y;
    if (W < 640) {   // phones: a sheet at whichever end of the screen the spotlight leaves free
      x = M; y = (y0 + y1) / 2 > H / 2 ? M : H - ph - M;
    } else if (W - x1 - M >= pw + M) { x = x1 + M; y = OBS.clamp(y0, M, H - ph - M); }
    else if (x0 - M >= pw + M) { x = x0 - pw - M; y = OBS.clamp(y0, M, H - ph - M); }
    else if (H - y1 - M >= ph + M) { y = y1 + M; x = OBS.clamp(x0, M, W - pw - M); }
    else if (y0 - M >= ph + M) { y = y0 - ph - M; x = OBS.clamp(x0, M, W - pw - M); }
    else { x = OBS.clamp(x1 - pw - M, M, W - pw - M); y = H - ph - M; }   // a big target: sit on top of its corner
    pop.style.left = Math.round(x) + 'px'; pop.style.top = Math.round(y) + 'px';
  }
  function tick() {
    raf = G.requestAnimationFrame(tick);
    place();
    var s = STEPS[T.k];
    if (s.try && !done && s.try.done(st)) {
      done = true;
      var t = pop.querySelector('.tour-try'); if (t) { t.classList.add('ok'); t.querySelector('.tour-say').textContent = s.try.ok; }
    }
  }
  function key(e) { if (e.key === 'Escape' && T.on && !document.querySelector('.lightbox')) T.end(); }

  // ------------------------------------------------------------------------------------------ control
  T.start = function () {
    if (T.on) return;
    // start from a clean slate: no mission open, the lens showing data
    if (OBS.logbook.current().m) OBS.logbook.leave();
    if (!root) build();
    T.on = true; root.hidden = false; document.body.classList.add('touring');
    document.addEventListener('keydown', key);
    T.k = 0; render(); tick();
  };
  T.go = function (k) { if (!T.on) return; T.k = OBS.clamp(k, 0, STEPS.length - 1); render(); };
  T.end = function () {
    if (!T.on) return;
    T.on = false; G.cancelAnimationFrame(raf); if (st.off) st.off(); st = {};
    leaveStep();
    root.hidden = true; document.body.classList.remove('touring');
    document.removeEventListener('keydown', key);
    OBS.store.set('toured', true);
    var b = $('tour-btn'); if (b) b.focus({ preventScroll: true });
  };
  /** First visit: start by itself, once. */
  T.maybe = function () { if (!OBS.store.get('toured', false)) setTimeout(T.start, 500); };
})(typeof window !== 'undefined' ? window : globalThis);
