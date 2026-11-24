/* cGħ World — a short concept quiz, taken before and after the missions, to see what changed.
 * Answers stay in this browser. Distractors follow misconceptions reported in physics-education research
 * (for measurement: Zhu and Singh 2012). Answers are revealed only after the "after" attempt. */
(function () {
  'use strict';
  var A = CGH.app, $ = A.$;
  var Q = [
    { m: 'lamps', q: 'Light is slow. Two lamps flash at the same moment, one 10 m from you and one 40 m away. What do you see?',
      o: ['Both flashes at once', 'The near flash first', 'The far flash first'], a: 1 },
    { m: 'twins', q: 'One twin sprints away and back while the other waits. When they meet, whose watch shows less time?',
      o: ['The one who waited', 'The one who ran', 'Neither: motion is relative, so they agree'], a: 1 },
    { m: 'upstairs', q: 'Compared with an identical clock at the foot of a tower, a clock at the top runs…',
      o: ['Slower', 'At the same rate', 'Faster'], a: 2 },
    { m: 'tower', q: 'Seen from the top of the tower, the light of a lamp on the ground is…',
      o: ['Shifted to the blue', 'Unchanged', 'Shifted to the red'], a: 2 },
    { m: 'horizon', q: 'On a more compact planet of the same size you can see further round it, because…',
      o: ['The planet’s surface flattens', 'Light bends down, following the curve of the ground', 'Your eyes are effectively higher'], a: 1 },
    { m: 'satellite', q: 'Compared with a clock on the ground, the clock of a satellite in a very low orbit runs…',
      o: ['Faster, because it is higher', 'Slower, because its speed matters more than its height', 'At the same rate, because it is in free fall'], a: 1 },
    { m: 'doors', q: 'Particles are sent one at a time through two doors and detected on a screen behind. The stripes…',
      o: ['Appear faintly in every single run', 'Build up only from many runs; each run gives one spot', 'Never appear with one particle at a time'], a: 1 },
    { m: 'bell', q: 'A detector at one door records which way each particle went, but nobody ever looks at its record. The stripes…',
      o: ['Stay, because nobody looked', 'Fade, because the record exists', 'Fade only after someone reads the record'], a: 1 },
    { m: 'microscope', q: 'You measure a particle’s position more precisely. Afterwards its momentum is…',
      o: ['More uncertain', 'Unaffected', 'Less uncertain'], a: 0 },
    { m: 'doors', q: 'You measure a particle’s position, then measure it again immediately, then again a while later. Which is right?',
      o: ['Each result is unrelated to the previous one', 'The second agrees with the first; the third may not, as the wave spreads again',
          'All three agree, because the wave stays collapsed'], a: 1 }
  ];
  var store = { pre: null, post: null };
  try { store = Object.assign(store, JSON.parse(localStorage.getItem('cgh-quiz') || '{}')); } catch (e) { /* fresh */ }
  function save() { try { localStorage.setItem('cgh-quiz', JSON.stringify(store)); } catch (e) { /* optional */ } }
  function score(ans) { return ans ? ans.filter(function (x, i) { return x === Q[i].a; }).length : null; }
  function esc(s) { return String(s).replace(/[&<>"]/g, function (c) { return { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]; }); }

  function form(which) {
    var el = $('quiz-body');
    el.innerHTML = '<p class="note">' + (which === 'pre' ? 'Answer before trying the missions. You will see your score, but not the answers.' : 'Answer after the missions. Then you will see what changed.') + '</p>' +
      Q.map(function (q, i) {
        return '<fieldset class="qz"><legend>' + (i + 1) + '. ' + esc(q.q) + '</legend>' + q.o.map(function (o, j) {
          return '<label class="ms-choice"><input type="radio" name="qz' + i + '" value="' + j + '"> ' + esc(o) + '</label>';
        }).join('') + '</fieldset>';
      }).join('') + '<div class="ms-actions"><button class="btn" type="button" id="qz-submit" disabled>Submit</button><button class="btn quiet" type="button" id="qz-cancel">Cancel</button></div>';
    function picks() { return Q.map(function (_, i) { var c = el.querySelector('input[name="qz' + i + '"]:checked'); return c ? Number(c.value) : null; }); }
    el.addEventListener('change', function () { $('qz-submit').disabled = picks().some(function (x) { return x === null; }); });
    $('qz-submit').addEventListener('click', function () { store[which] = picks(); store[which + 'At'] = new Date().toISOString(); save(); render(); });
    $('qz-cancel').addEventListener('click', render);
  }

  function summary() {
    var n = Q.length, pre = score(store.pre), post = score(store.post), lines = ['cGħ World concept quiz (' + n + ' questions)'];
    if (pre !== null) lines.push('Before: ' + pre + '/' + n + '  answers ' + store.pre.join(',') + '  at ' + store.preAt);
    if (post !== null) lines.push('After:  ' + post + '/' + n + '  answers ' + store.post.join(',') + '  at ' + store.postAt);
    if (pre !== null && post !== null && pre < n) lines.push('Normalised gain (after − before) / (' + n + ' − before): ' + ((post - pre) / (n - pre)).toFixed(2));
    var done = {}; try { done = JSON.parse(localStorage.getItem('cgh-missions') || '{}') || {}; } catch (e) { /* none */ }
    var ids = Object.keys(done);
    lines.push('Missions finished (prediction right?): ' + (ids.map(function (k) { var d = done[k]; return k + (d && d.pick ? (d.right ? ' ✓' : ' ✗') : ''); }).join(', ') || 'none'));
    return lines.join('\n');
  }

  function render() {
    var el = $('quiz-body'), n = Q.length, pre = score(store.pre), post = score(store.post), html = '';
    html += '<div class="ms-actions">' +
      '<button class="btn' + (pre === null ? '' : ' quiet') + '" type="button" id="qz-pre">' + (pre === null ? 'Take the “before” quiz' : 'Retake “before”') + '</button>' +
      '<button class="btn' + (pre !== null && post === null ? '' : ' quiet') + '" type="button" id="qz-post">' + (post === null ? 'Take the “after” quiz' : 'Retake “after”') + '</button>' +
      '<button class="btn quiet" type="button" id="qz-copy"' + (pre === null && post === null ? ' disabled' : '') + '>Copy results</button>' +
      '<button class="btn quiet" type="button" id="qz-clear"' + (pre === null && post === null ? ' disabled' : '') + '>Clear</button></div>';
    if (pre !== null || post !== null) {
      html += '<dl class="m-meter"><div><dt>Before</dt><dd>' + (pre === null ? '—' : pre + ' of ' + n) + '</dd></div><div><dt>After</dt><dd>' + (post === null ? '—' : post + ' of ' + n) + '</dd></div>' +
        (pre !== null && post !== null && pre < n ? '<div><dt>Normalised gain</dt><dd>' + ((post - pre) / (n - pre)).toFixed(2) + ' (of the questions you missed before, the share you now get right)</dd></div>' : '') + '</dl>';
    }
    if (post !== null) {
      html += '<ol class="qz-review">' + Q.map(function (q, i) {
        var ok = store.post[i] === q.a, was = store.pre ? store.pre[i] === q.a : null;
        return '<li class="' + (ok ? 'ok' : 'no') + '"><strong>' + esc(q.q) + '</strong><br>' + (ok ? '✓ ' : '✗ ') + 'Answer: ' + esc(q.o[q.a]) +
          (was === false && ok ? ' · newly right' : was === true && !ok ? ' · was right before' : '') + ' · mission: ' + esc(CGH.missions.titleOf(q.m)) + '</li>';
      }).join('') + '</ol>';
    }
    html += '<textarea id="qz-text" class="qz-text" readonly hidden></textarea>';
    el.innerHTML = html;
    $('qz-pre').addEventListener('click', function () { form('pre'); });
    $('qz-post').addEventListener('click', function () { form('post'); });
    $('qz-clear').addEventListener('click', function () { store = { pre: null, post: null }; save(); render(); });
    $('qz-copy').addEventListener('click', function () {
      var text = summary(), ta = $('qz-text');
      function fallback() { ta.hidden = false; ta.value = text; ta.focus(); ta.select(); $('qz-copy').textContent = 'Select and copy the text below'; }
      try { navigator.clipboard.writeText(text).then(function () { $('qz-copy').textContent = 'Copied'; }, fallback); } catch (e) { fallback(); }
    });
  }

  CGH.quiz = { init: render, Q: Q, summary: summary };
})();
