/* cGħ World — missions. Each one: a brief, a prediction you commit to, an exploration with only the needed
 * dials unlocked and a goal the world checks for you, then the explanation and the real experiment. */
(function () {
  'use strict';
  var G = CGH.geo, A = CGH.app, WV = CGH.wave, $ = A.$;
  var R0 = A.REAL;
  function S() { return CGH.world.state; }
  function probe() { return CGH.probe.last || {}; }
  function dom(id, prop, value, ev) { var el = $(id); el[prop] = value; el.dispatchEvent(new Event(ev || 'change')); }
  function quantumKit(bell) {
    dom('m-screen', 'checked', true); dom('m-bell', 'value', bell, 'input'); dom('m-sharp', 'value', '0.5'); dom('m-env', 'value', '0');
    WV.W.marks = [];
  }
  function noQuantum() { dom('m-screen', 'checked', false); dom('m-bell', 'value', 0, 'input'); }
  function world(v) { A.set(Object.assign({ cw: R0.cw, comp: R0.comp, lam: R0.lam, photons: R0.photons, alpha: 1, cosmo: R0.cosmo }, v)); CGH.world.setSky(0, 0); }
  // Share of the marks that sit straight behind the wall between the doors, where no straight path leads.
  function middleShare() {
    var m = WV.W.marks, n = 0, mid = 0;
    m.forEach(function (p) { if (p.a < G.WALL.a - 0.5) { n++; if (Math.abs(p.l) < 0.9) mid++; } });
    return n ? mid / n : 0;
  }

  var M = [
    {
      id: 'lamps', title: 'The lamps flash together', dials: ['cw'],
      brief: 'Light here is slow: three walking speeds. Every lamp on this street flashes at exactly the same moment by its own clock.',
      question: 'Standing still, which lamps do you see flash first?',
      choices: [['all', 'All of them at once'], ['near', 'The nearest, then a ripple running away from me'], ['far', 'The farthest, then a ripple coming towards me']],
      answer: 'near',
      setup: function () { world({ cw: 3 }); CGH.world.place(G.START.a, 3, 0); noQuantum(); },
      task: 'Stand still and watch the lamps along the right of the street for a few flashes. Put the crosshair on a distant lamp to see how old its light is.',
      progress: function (st) { var t = S().tau - st.t0; return { done: t >= 10 && S().effort === 0, text: 'Watched for ' + Math.min(10, t).toFixed(0) + ' of 10 s' + (S().effort ? ' (stop first: Space)' : '') }; },
      result: function () { var c = A.derived().c; return 'Light travels ' + A.num(c, 2) + ' m every second here, so a lamp 40 m away is seen ' + A.num(40 / c, 2) + ' s late.'; },
      explain: 'Every lamp flashed at the same moment. You saw each flash when its light arrived, d/c later, so nearer flashes came first and the flashing seemed to run away down the street. Nothing was out of step except the light. In the real world the same delay is why the Sun you see is 8 minutes old.',
      real: 'Ole Rømer, 1676: Jupiter’s moon Io seemed to run late when Jupiter was far from Earth. The lateness was light’s travel time: the first measurement of c.'
    },
    {
      id: 'twins', title: 'The twin trip', dials: ['cw'],
      brief: 'Light is four walking speeds. Your watch and the street clocks start together. You will sprint to the next block’s clock and back.',
      question: 'When you are back, how does your watch compare with the street clocks?',
      choices: [['ahead', 'My watch is ahead'], ['same', 'They agree'], ['behind', 'My watch is behind']],
      answer: 'behind',
      setup: function () { world({ cw: 4 }); CGH.world.place(G.START.a, 3, 0); noQuantum(); },
      task: 'Hold W to sprint past the tower to the clock 130 m ahead. Press “Turn around”, sprint back, and stop (Space) beside where you started.',
      progress: function (st) {
        var a = S().a;
        if (!st.out && a > 115) st.out = true;
        var back = st.out && a < G.START.a + 6 && S().effort === 0;
        return { done: back, text: st.out ? 'Turned. Now come back and stop near the start: ' + A.num(Math.max(0, a - G.START.a), 3) + ' m to go' : 'Out: ' + A.num(Math.max(0, a - G.START.a), 3) + ' of 125 m' };
      },
      measured: function () { return S().street - S().tau > 0.5 ? 'behind' : S().tau - S().street > 0.5 ? 'ahead' : 'same'; },
      result: function () { return 'Street clocks: ' + A.clock(S().street) + '. Your watch: ' + A.clock(S().tau) + '. You are ' + A.duration(S().street - S().tau, 3) + ' younger.'; },
      explain: 'While you ran, your watch ticked slower than the street’s by the factor γ = 1/√(1 − v²/c²). But from your point of view the street was moving: why is it not the street that aged less? Because only you turned round. Changing direction changes which events count as “now” for you, and that is where the street’s extra time comes from.',
      real: 'Hafele and Keating, 1971: atomic clocks flown round the world on airliners came back nanoseconds out of step with clocks on the ground, as predicted.'
    },
    {
      id: 'tower', title: 'Colours from the tower', dials: ['comp'],
      brief: 'The planet is now heavy: its horizon radius is 0.45 of its size, about a neutron star’s. A tower stands in the street.',
      question: 'From the top of the tower you look at a street lamp below. Compared with how it looks from the street, its light is…',
      choices: [['bluer', 'Bluer'], ['same', 'The same'], ['redder', 'Redder']],
      answer: 'redder',
      setup: function () { world({ cw: 7, comp: 0.45 }); CGH.world.place(G.TOWER.a - 4, 0, 0); noQuantum(); },
      task: 'Fly up with E (or ▲) to about 46 m. Turn round (“Turn around”), drag the view down, and put the crosshair on the street or a lamp below. Read the spectrometer.',
      progress: function () {
        var p = probe(), h = S().h, low = p.kind === 1 && p.h < 6 && p.eyeH > 40;
        return { done: h > 40 && low, text: h > 40 ? (low ? 'On the street' : 'Now aim down at the street') : 'Height ' + A.num(h, 2) + ' of 46 m' };
      },
      measured: function () { var p = probe(); return p.Dg < 0.999 ? 'redder' : p.Dg > 1.001 ? 'bluer' : 'same'; },
      result: function () { var p = probe(); return 'Light that left the street at 589 nm (a sodium lamp’s yellow) arrives at ' + A.num(589 / p.D, 4) + ' nm. Gravity’s share: ×' + (p.Dg || 1).toFixed(4) + ' in frequency.'; },
      explain: 'Light climbing out of gravity loses frequency. Equivalently, a clock lower down runs slow, so everything it does, including the light it gives off, is slowed when seen from above. The factor is √((1 − rs/r_lamp)/(1 − rs/r_you)). Fly down and look up at the tower clock: its light is bluer.',
      real: 'Pound and Rebka, 1960: gamma rays sent up a 22.5 m tower at Harvard were shifted by 2.5 parts in 10¹⁵. Chou et al., 2010, saw optical clocks disagree over a 33 cm rise.'
    },
    {
      id: 'upstairs', title: 'Older upstairs', dials: ['comp'],
      brief: 'Same heavy planet. You start at the top of the tower with your watch set to the street clocks.',
      question: 'You wait one minute on your watch, then fly down. Compared with the street clocks, your watch is…',
      choices: [['ahead', 'Ahead'], ['same', 'The same'], ['behind', 'Behind']],
      answer: 'ahead',
      setup: function () { world({ cw: 7, comp: 0.45 }); CGH.world.place(G.TOWER.a - 4, 0, 0, 46); noQuantum(); },
      task: 'Wait until your watch shows 1:00, then fly down with Q (or ▼) to the street.',
      progress: function (st) {
        if (!st.waited && S().tau >= 60) st.waited = true;
        return { done: st.waited && S().h < 2.5, text: st.waited ? 'Now fly down: ' + A.num(S().h, 2) + ' m up' : 'Waiting: ' + A.clock(S().tau) + ' of 1:00' + (S().h < 40 ? ' (stay at the top)' : '') };
      },
      measured: function () { return S().tau - S().street > 0.3 ? 'ahead' : S().street - S().tau > 0.3 ? 'behind' : 'same'; },
      result: function () { return 'Your watch: ' + A.clock(S().tau) + '. Street clocks: ' + A.clock(S().street) + '. You aged ' + A.duration(S().tau - S().street, 3) + ' more.'; },
      explain: 'Clocks higher up run faster: here by √((1 − rs/r_top)/(1 − rs/R)) ≈ 1.04. Your minute at the top was 58 s of street time. Nothing felt different to you; time runs at its own pace wherever you are, and only comparison shows the difference.',
      real: 'GPS: satellite clocks gain about 45 µs a day from being higher up and lose about 7 from their speed. Without the correction positions would drift by kilometres a day.'
    },
    {
      id: 'horizon', title: 'How far can you see?', dials: ['comp'],
      brief: 'Light is seven walking speeds and the planet is ordinary. Its horizon is only a few dozen metres away: it is a small planet.',
      question: 'You make the planet more and more compact. What happens to your horizon?',
      choices: [['closer', 'It comes closer: the planet looks smaller'], ['same', 'Nothing changes'], ['further', 'It moves away and rises, until the street seems endless']],
      answer: 'further',
      setup: function () { world({ cw: 7 }); CGH.world.place(G.START.a, 3, 0); noQuantum(); },
      task: 'Raise the G dial slowly, watching the horizon readout. Near 0.65 aim the crosshair along the street, just below the horizon, until the probe finds something more than 250 m away.',
      progress: function () { var p = probe(), c = A.state.comp; return { done: c > 0.55 && p.kind === 1 && p.s > 250, text: 'Compactness ' + A.num(c, 2) + ' · probe sees ' + (p.kind === 1 ? A.num(p.s, 3) + ' m away' : 'the sky') }; },
      result: function () { var p = probe(); return 'You are seeing ' + (G.MATS[p.mat] || 'something') + ' ' + A.num(p.s, 3) + ' m away along the light’s path, round the curve of the planet.'; },
      explain: 'Light falls toward the planet. The more compact it is, the more the light curves down, following the ground round. At rs/R = 2/3 your eyes sit on the photon sphere, r = 1.5 rs, where a horizontal ray circles the planet: the street looks flat and endless. Beyond that, the ground seems to rise around you like a bowl.',
      real: 'Neutron stars are compact enough that more than half the surface is visible at once (Nemiroff 1993; Kraus and Zahn’s films at spacetimetravel.org). The bright ring in black-hole images comes from light near the photon orbit.'
    },
    {
      id: 'doors', title: 'Two doors', dials: ['lam'],
      brief: 'You are a wave with a 2 m wavelength. Ahead is a wall with two doors. A detector screen counts where you are found behind the wall.',
      question: 'Each run you are found at one spot behind the wall. After many runs, where do the marks pile up?',
      choices: [['two', 'In two bands, one behind each door'], ['stripes', 'In stripes, with plenty straight behind the wall between the doors'], ['even', 'Evenly spread']],
      answer: 'stripes',
      setup: function () { world({ lam: 2 }); document.dispatchEvent(new CustomEvent('cgh-pose', { detail: 'doors' })); quantumKit(0); },
      task: 'Walk at the doors (W) and stop when about 30% of you is past the wall (see “Measuring you”). Look (F), then press “Rewind and look again” until there are 40 marks.',
      progress: function () { var n = WV.W.marks.length; return { done: n >= 40, text: n + ' of 40 marks · ' + $('m-behind').textContent + ' of you past the wall' }; },
      measured: function () { return middleShare() > 0.12 ? 'stripes' : 'two'; },
      result: function () { var m = middleShare(); M.doorsShare = m; return Math.round(100 * m) + '% of the marks landed straight behind the solid wall between the doors.'; },
      explain: 'Each look found you in one place: the stripes exist only in the tally. They come from the two routes through the doors adding up or cancelling. The middle stripe sits behind solid wall, where no straight path from either door leads. Lower your wavelength and the stripes crowd together until nothing could resolve them: that is why footballs look classical.',
      real: 'Electrons sent one at a time build up stripes (Tonomura et al., 1989); so do whole molecules of 60 carbon atoms (Arndt et al., 1999).'
    },
    {
      id: 'bell', title: 'The bell nobody hears', dials: ['lam'],
      brief: 'The same doors. Now a bell at the left door rings for anyone passing near it. Nobody listens to it, ever.',
      question: 'Do the stripes survive?',
      choices: [['same', 'Yes: nobody heard the bell'], ['fade', 'No, they fade, even though nobody listens'], ['listen', 'They fade only if someone listens']],
      answer: 'fade',
      setup: function () { world({ lam: 2 }); document.dispatchEvent(new CustomEvent('cgh-pose', { detail: 'doors' })); quantumKit(100); },
      task: 'Do the same run: walk at the doors, stop when about 30% of you is past the wall, look, and rewind until there are 40 marks. Watch the map: cyan is the part of you that rang the bell.',
      progress: function () { var n = WV.W.marks.length; return { done: n >= 40, text: n + ' of 40 marks · ' + $('m-rang').textContent }; },
      result: function () {
        var m = middleShare(), d = M.doorsShare;
        var noisy = d !== undefined && m >= 0.8 * d;
        return Math.round(100 * m) + '% of the marks landed behind the wall between the doors' + (d !== undefined ? ' (without the bell: ' + Math.round(100 * d) + '%).' : '.') +
          ' The bell rang for ' + Math.round(100 * WV.W.rang) + '% of you.' + (noisy ? ' With 40 marks the difference can hide in the noise; take more, or compare the maps.' : '');
      },
      explain: 'Passing the bell left a record: the part of you that went left is now tied to “bell rang”, the rest to “bell silent”. Parts tied to different records cannot interfere, so the stripes fade. Nobody has to listen; the record existing is enough. Now press “Erase the bell’s record”: it reads the bell in a way that cannot tell left from right, and the stripes come back. The trade is exact: how well a record tells the paths apart (D) and how strong the stripes are (V) obey D² + V² ≤ 1.',
      real: 'Hot molecules lose their stripes by giving off light that records their path (Hackermüller et al., 2004). The trade-off D² + V² ≤ 1 was measured with photons by Schwindt, Kwiat and Englert in 1999.'
    },
    {
      id: 'microscope', title: 'Look closer', dials: ['lam'],
      brief: 'You are a wave on an open street. You can check where you are with a microscope of any sharpness.',
      question: 'After a sharp look (0.25 m) compared with a blurry one (2 m), you spread out…',
      choices: [['faster', 'Faster'], ['same', 'At the same speed'], ['slower', 'Slower']],
      answer: 'faster',
      setup: function () { world({ lam: 2 }); CGH.world.place(G.START.a, 0, 0); noQuantum(); dom('m-sharp', 'value', '2'); WV.W.marks = []; },
      task: 'Look (F) with the microscope at 2 m and note Δv in “Measuring you”. Then set it to 0.25 m and look again.',
      progress: function (st) {
        var n = WV.W.marks.length;
        if (n > (st.n || 0)) { st.n = n; st.dv = st.dv || {}; var k = WV.kstats(); st.dv[S().sharp] = WV.W.hbar * Math.hypot(k.dk[0], k.dk[1]) / A.MASS; }
        var dv = st.dv || {};
        return { done: dv[2] !== undefined && dv[0.25] !== undefined, text: 'Blurry look: ' + (dv[2] !== undefined ? 'done' : 'not yet') + ' · sharp look: ' + (dv[0.25] !== undefined ? 'done' : 'not yet') };
      },
      measured: function (st) { return st.dv[0.25] > st.dv[2] * 1.2 ? 'faster' : st.dv[0.25] < st.dv[2] / 1.2 ? 'slower' : 'same'; },
      result: function (st) { return 'Spread in velocity after the 2 m look: ' + A.num(st.dv[2], 2) + ' m/s; after the 0.25 m look: ' + A.num(st.dv[0.25], 2) + ' m/s.'; },
      explain: 'To pin your position to within Δx, the look must leave your momentum uncertain by at least ħ/(2Δx). With ħ this large and Δx = 0.25 m that is a velocity spread of nearly 1 m/s, so afterwards you smear out fast. A blurry look disturbs you gently and teaches you little. You cannot have both.',
      real: 'This back-action has been measured: light bouncing off a tiny drum shakes it (Purdy, Peterson and Regal, 2013), and the same quantum noise limits gravitational-wave detectors.'
    }
    ,{
      id: 'satellite', title: 'Satellite clocks', dials: [],
      brief: 'Same heavy planet. A satellite circles 30 m above the street, blinking once a second by its own clock.',
      question: 'Compared with your watch on the street, the satellite’s clock runs…',
      choices: [['faster', 'Faster: it is higher up'], ['same', 'At the same rate'], ['slower', 'Slower: it is moving fast']],
      answer: 'slower',
      setup: function () { world({ cw: 7, comp: 0.45 }); CGH.world.place(G.START.a, 0, 0); CGH.world.setSky(30, 0); noQuantum(); },
      task: 'Drag the view up to find the satellite. Then raise its orbit with the “Satellite orbit” slider until its clock keeps pace with yours: “Satellite clock against yours” reaches ×1.000.',
      progress: function (st) {
        var d = A.derived(), H = S().satH, o = G.orbit(G.R + H, d.rs, d.c, d.lam), you = Math.sqrt(1 - d.rs / (G.R + G.EYE) - d.lam * Math.pow(G.R + G.EYE, 2));
        var ratio = o ? 1 / (o.ut * you) : 0;
        if (st.r0 === undefined && o) st.r0 = ratio;
        st.ratio = ratio; st.H = H;
        return { done: o && Math.abs(ratio - 1) < 0.004, text: o ? 'Orbit ' + H + ' m up: its clock runs ×' + ratio.toFixed(4) + ' of yours' : 'No circular orbit there' };
      },
      measured: function (st) { return st.r0 < 0.999 ? 'slower' : st.r0 > 1.001 ? 'faster' : 'same'; },
      result: function (st) { return 'At 30 m it ran ×' + (st.r0 || 0).toFixed(4) + '. It keeps pace at ' + st.H + ' m up, where its orbit is ' + ((G.R + st.H) / G.R).toFixed(2) + ' times the planet’s radius.'; },
      explain: 'Height makes a clock run fast; speed makes it run slow. For a circular orbit the clock rate is √(1 − 1.5 rs/r), against √(1 − rs/R) on the ground. They match when r = 1.5 R, for any planet and any mass. Lower satellites lose time; higher ones gain.',
      real: 'GPS satellites orbit at about 4.2 Earth radii, far above 1.5, so their clocks gain about 38 microseconds a day. They are built to tick slightly slow on the ground to make up for it.'
    },
    {
      id: 'radar', title: 'The radar echo', dials: [],
      brief: 'A compact planet (0.6) and slow light. You have a radar: it times a pulse to whatever is under the crosshair and back.',
      question: 'Compared with light going straight there and back at c, the echo comes back…',
      choices: [['earlier', 'Earlier'], ['same', 'On time'], ['later', 'Later']],
      answer: 'later',
      setup: function () { world({ cw: 7, comp: 0.6 }); CGH.world.place(G.START.a, 3, 0); noQuantum(); },
      task: 'Aim along the street at something more than 100 m away, just below the horizon, and read “Radar echo” in the light probe.',
      progress: function () { var p = probe(); return { done: p.kind === 1 && p.s > 100 && p.echo > 0, text: p.kind === 1 ? 'Target ' + A.num(p.s, 3) + ' m along the light’s path' : 'Aim lower: the probe sees the sky' }; },
      measured: function () { var p = probe(); return p.echo > p.flat * 1.001 ? 'later' : p.echo < p.flat * 0.999 ? 'earlier' : 'same'; },
      result: function () { var p = probe(); return 'Echo: ' + A.duration(p.echo, 3) + '. Straight there and back at c: ' + A.duration(p.flat, 3) + '.'; },
      explain: 'Two things make the echo late. The path bends, so it is longer; and, timed from far away, light near a mass covers ground more slowly. Anyone standing next to the light measures it passing at exactly c; only the comparison between places shows the delay.',
      real: 'Irwin Shapiro predicted it in 1964; radar echoes from Venus and Mercury confirmed it. In 2002 signals to the Cassini spacecraft matched Einstein’s prediction to about 20 parts in a million.'
    },
    {
      id: 'rosette', title: 'Rosette orbits', dials: ['comp'],
      brief: 'A heavy planet. In the Orbits lab below you can throw a ball sideways hard enough to go round.',
      question: 'You throw the ball into a slightly stretched orbit. Each time round, it…',
      choices: [['same', 'Retraces exactly the same ellipse'], ['turns', 'Comes round a little further, so the orbit slowly turns'], ['spiral', 'Spirals down to the ground']],
      answer: 'turns',
      setup: function () { world({ cw: 7, comp: 0.3 }); CGH.world.place(G.START.a, 0, 0); noQuantum(); CGH.labs.state.show('orbits'); },
      task: 'In the Orbits lab, press Throw. Then try throwing from lower down, slower, or on a heavier planet: too low or too slow and the ball hits the ground.',
      progress: function (st) {
        var O = CGH.labs.state.O;
        if (st.n0 === undefined) st.n0 = O.thrown;
        var ok = O.thrown > st.n0 && O.last && !O.last.E.crashed && O.last.E.peri.length >= 3;
        return { done: ok, text: O.thrown > st.n0 ? (O.last.E.crashed ? 'That one crashed. Throw from higher, or faster.' : 'Orbit drawn') : 'Throw a ball in the Orbits lab' };
      },
      measured: function () { var E = CGH.labs.state.O.last.E; return E.precess > 1e-3 ? 'turns' : 'same'; },
      result: function () { var E = CGH.labs.state.O.last.E; return 'Each time round, the closest point moved on by ' + A.num(E.precess * 180 / Math.PI, 3) + '°.'; },
      explain: 'Near a mass, gravity pulls slightly harder than Newton’s 1/r²: the extra term grows as 3L²/(c²r²). The ball reaches its closest point a little late each time, so the ellipse rotates and draws a rosette. Closer and more compact means more turning; inside three horizon radii there is no stable orbit at all.',
      real: 'Mercury’s orbit turns 43 seconds of arc per century more than Newton allows: Einstein’s first success, in 1915. The star S2, circling the black hole at the centre of our galaxy, shows the same turning (GRAVITY collaboration, 2020).'
    },
    {
      id: 'hover', title: 'Hover', dials: [],
      brief: 'You are a wave (2 m wavelength) on a planet with Earth-like gravity. The Hover lab below treats your height as a wave too.',
      question: 'You stop jumping and settle down. Where do your feet end up?',
      choices: [['floor', 'On the floor'], ['hover', 'A little above the floor, and only certain heights are allowed'], ['sink', 'Partly inside the floor']],
      answer: 'hover',
      setup: function () { world({ cw: 80, comp: 0.45, lam: 2 }); CGH.world.place(G.START.a, 0, 0); noQuantum(); CGH.labs.state.show('hover'); },
      task: 'In the Hover lab, press “Settle to the lowest state” and watch for five seconds. Your view rises by however far your feet hover. Then try “Drop from 2 m”.',
      progress: function () { var H = CGH.labs.state.H; return { done: H.on && H.mode === 'settle' && H.settled > 5, text: H.on && H.mode === 'settle' ? 'Settled for ' + Math.min(5, H.settled).toFixed(1) + ' of 5 s' : 'Press “Settle to the lowest state”' }; },
      result: function () { var b = CGH.labs.state.H.b; return 'Your feet hover ' + A.length(2 / 3 * b.heights[0], 2) + ' above the floor on average. Allowed levels reach ' + b.heights.slice(0, 3).map(function (z) { return A.length(z, 2); }).join(', ') + '.'; },
      explain: 'Squeezing your wave flat onto the floor would cost kinetic energy, about ħ²/(2mΔz²); lifting it costs potential energy, mgz. The lowest total is a compromise: a hover height set by z₀ = (ħ²/2m²g)^⅓. Above it only certain energies are allowed, fixed by the zeros of the Airy function.',
      real: 'Ultracold neutrons above a mirror do exactly this, with levels about ten micrometres apart (Nesvizhevsky et al., 2002; the qBounce experiments).'
    },
    {
      id: 'blackhole', title: 'A pocket black hole', dials: [],
      brief: 'Slow light, ordinary planet. A tiny black hole, half a metre across its horizon, hovers 3.5 m above the street between you and the tower.',
      question: 'Looking at the tower through it, you see…',
      choices: [['disc', 'A black disc in front of the tower, nothing else'], ['ring', 'The tower smeared into a ring around a black disc'], ['none', 'Nothing: it is too small to notice']],
      answer: 'ring',
      setup: function () { world({ cw: 7 }); CGH.world.place(G.START.a, 0, 0); CGH.world.setSky(0, 0.5); noQuantum(); },
      task: 'Look along the street at the black hole, 30 m ahead. Put the crosshair just outside its dark disc: the probe shows how far that light bent.',
      progress: function () { var p = probe(); return { done: p.kind === 1 && p.bend > 20, text: p.kind === 3 ? 'That is the dark disc: aim just outside it' : 'Light under the crosshair bent by ' + A.num(p.bend || 0, 2) + '°' }; },
      result: function () { var p = probe(); return 'The light under your crosshair bent by ' + A.num(p.bend, 3) + '° on its way past the black hole, and came from ' + (G.MATS[p.mat] || 'something') + '.'; },
      explain: 'Light passing a black hole bends more the closer it passes. Light from the tower straight behind can come round any side, so it arrives as a ring: an Einstein ring. Inside it is the shadow, about 2.6 times the horizon radius: light from there would have had to start inside the photon orbit.',
      real: 'Hubble photographs Einstein rings around distant galaxies. The Event Horizon Telescope imaged the shadows of the black holes in M87 (2019) and at the centre of our galaxy (2022).'
    },
    {
      id: 'alpha', title: 'The colour of atoms', dials: ['alpha'],
      brief: 'The fine-structure constant α sets how strongly electrons and light interact. It decides the size and colours of atoms. Open “More constants” in the console.',
      question: 'Make α 15% larger. The yellow street lamps look…',
      choices: [['bluer', 'Bluer'], ['same', 'The same'], ['redder', 'Redder']],
      answer: 'bluer',
      setup: function () { world({ cw: 7 }); CGH.world.place(G.START.a, 3, 0); noQuantum(); var d = document.querySelector('.more-dials'); if (d) d.open = true; },
      task: 'Raise α to about 1.15 and put the crosshair on a street lamp. Read the spectrometer.',
      progress: function () { var p = probe(); return { done: p.kind === 1 && p.mat === 4 && A.state.alpha >= 1.1, text: 'α = ' + A.num(A.state.alpha, 3) + ' × today’s · ' + (p.mat === 4 ? 'on a lamp' : 'aim at a lamp') }; },
      measured: function () { return A.state.alpha > 1 ? 'bluer' : A.state.alpha < 1 ? 'redder' : 'same'; },
      result: function () { return 'Sodium’s yellow line moves from 589 nm to ' + A.num(589 / (A.state.alpha * A.state.alpha), 3) + ' nm.'; },
      explain: 'Atomic energy levels scale as α²mₑc², so every spectral line moves together: the lamps, paint and leaves. Atoms also shrink as 1/α. Larger α also means heavy elements cannot hold their innermost electrons once Zα approaches 1, so the periodic table gets shorter.',
      real: 'Astronomers compare lines in quasar light that is billions of years old with lines measured today; any change in α is at most a few parts per million, and claimed hints of change are disputed. Atomic clocks limit any change now to less than about a part in 10¹⁶ per year.'
    },
    {
      id: 'stairs', title: 'Two staircases', dials: [],
      brief: 'You are a wave on a heavy planet. In the Two staircases lab below, your wave splits: one part climbs a staircase and waits, the other waits at the bottom. Then they rejoin.',
      question: 'You carry a ticking watch. Do the stripes survive?',
      choices: [['yes', 'Yes: nobody measured which way I went'], ['shift', 'They shift, but stay sharp'], ['fade', 'They fade, and come back when the two watch readings differ by a whole tick']],
      answer: 'fade',
      setup: function () { world({ cw: 7, comp: 0.45, lam: 2 }); CGH.world.place(G.START.a, 0, 0); noQuantum(); CGH.labs.state.show('stairs'); CGH.labs.state.T2.faded = false; CGH.labs.state.T2.revived = false; },
      task: 'In the lab, increase “Time apart” until the stripes vanish, then keep going until they come back.',
      progress: function () { var T2 = CGH.labs.state.T2; return { done: T2.faded && T2.revived, text: 'Stripes vanished: ' + (T2.faded ? 'yes' : 'not yet') + ' · came back: ' + (T2.revived ? 'yes' : 'not yet') }; },
      result: function () { var s = CGH.labs.state.stairs(); return 'Now the watches differ by ' + A.num(Math.abs(s.r.dtau) * s.tick, 3) + ' ticks and the stripe strength is ' + s.r.V.toFixed(2) + '.'; },
      explain: 'The upper half of you ages faster, so the two halves carry different watch readings. Different readings are a record of the path, like the bell, so the stripes fade: D² + V² = 1. When the readings differ by exactly one tick they look alike again, the record is gone and the stripes return. With no clock the stripes only shift: that shift is gravity acting on quantum matter, and it has been measured.',
      real: 'Neutron stripes shift with height in Earth’s gravity (Colella, Overhauser and Werner, 1975). The fading with a clock was proposed by Zych and colleagues in 2011 and imitated with atoms and a magnetic field by Margalit et al. in 2015. With real gravity it has not yet been done.'
    }
  ];

  /* ---------- controller ---------- */
  var cur = null;         // { m, step, pick, st }
  var done = {};
  try { done = JSON.parse(localStorage.getItem('cgh-missions') || '{}') || {}; } catch (e) { done = {}; }
  function save() { try { localStorage.setItem('cgh-missions', JSON.stringify(done)); } catch (e) { /* optional */ } }
  var STEPS = ['Brief', 'Predict', 'Explore', 'Explain'];
  function esc(s) { return String(s).replace(/[&<>]/g, function (c) { return { '&': '&amp;', '<': '&lt;', '>': '&gt;' }[c]; }); }
  function sym(k) { return '<span class="sym sym-' + ({ cw: 'c', comp: 'g', lam: 'h', photons: 'h', alpha: 'c', cosmo: 'g' })[k] + '">' + ({ cw: 'c', comp: 'G', lam: 'ħ', photons: 'ħ', alpha: 'α', cosmo: 'Λ' })[k] + '</span>'; }

  function render() {
    var el = $('missions');
    if (!cur) {
      el.innerHTML = '<div class="ms-list"><span class="ms-label">Missions</span>' + M.map(function (m, i) {
        return '<button type="button" class="ms-chip' + (done[m.id] ? ' done' : '') + '" data-i="' + i + '">' + (done[m.id] ? '✓ ' : (i + 1) + '. ') +
          esc(m.title) + ' ' + m.dials.map(sym).join('') + '</button>';
      }).join('') + '<span class="ms-free">or explore freely with every dial</span></div>';
      Array.prototype.forEach.call(el.querySelectorAll('.ms-chip'), function (b) { b.addEventListener('click', function () { start(Number(b.getAttribute('data-i'))); }); });
      return;
    }
    var m = cur.m, i = M.indexOf(m), body = '', act = '';
    if (cur.step === 0) {
      body = '<p>' + esc(m.brief) + '</p>';
      act = '<button class="btn" type="button" data-go="1">Next: predict</button>';
    } else if (cur.step === 1) {
      body = '<p class="ms-q">' + esc(m.question) + '</p><div class="ms-choices" role="radiogroup">' + m.choices.map(function (c) {
        return '<label class="ms-choice"><input type="radio" name="ms-pick" value="' + c[0] + '"' + (cur.pick === c[0] ? ' checked' : '') + '> ' + esc(c[1]) + '</label>';
      }).join('') + '</div>';
      act = '<button class="btn" type="button" data-go="2"' + (cur.pick ? '' : ' disabled') + '>Lock in my prediction</button>';
    } else if (cur.step === 2) {
      body = '<p>' + esc(m.task) + '</p><p class="ms-status" id="ms-status" aria-live="polite"></p>';
      act = '<button class="btn quiet" type="button" data-go="3">Skip to the explanation</button>';
    } else {
      var meas = cur.finished && m.measured ? m.measured(cur.st) : m.answer, right = cur.pick === meas;
      var text = function (id) { var c = m.choices.filter(function (x) { return x[0] === id; })[0]; return c ? c[1] : '—'; };
      body = '<p class="ms-verdict ' + (right ? 'ok' : 'no') + '">You predicted: “' + esc(text(cur.pick)) + '”. ' +
        (cur.finished ? (right ? 'That is what happened.' : 'What happened: “' + esc(text(meas)) + '”.') : 'The answer: “' + esc(text(m.answer)) + '”.') + '</p>' +
        (cur.finished ? '<p class="ms-result">' + esc(m.result(cur.st)) + '</p>' : '') +
        '<p>' + esc(m.explain) + '</p><p class="ms-real"><strong>In the real world.</strong> ' + esc(m.real) + '</p>';
      act = (i < M.length - 1 ? '<button class="btn" type="button" data-next="1">Next mission</button>' : '') + '<button class="btn quiet" type="button" data-again="1">Try again</button>';
    }
    el.innerHTML = '<div class="ms-card"><div class="ms-head"><ol class="ms-steps">' + STEPS.map(function (s, k) {
      return '<li class="' + (k === cur.step ? 'now' : k < cur.step ? 'past' : '') + '">' + s + '</li>';
    }).join('') + '</ol><button class="link-btn" type="button" data-leave="1">Leave mission</button></div>' +
      '<h3>' + (i + 1) + '. ' + esc(m.title) + ' ' + m.dials.map(sym).join('') + '</h3>' + body + '<div class="ms-actions">' + act + '</div></div>';
    Array.prototype.forEach.call(el.querySelectorAll('input[name="ms-pick"]'), function (r) {
      r.addEventListener('change', function () { cur.pick = r.value; el.querySelector('[data-go="2"]').disabled = false; });
    });
    var go = el.querySelector('[data-go]');
    if (go) go.addEventListener('click', function () { step(Number(go.getAttribute('data-go'))); });
    var lv = el.querySelector('[data-leave]'); if (lv) lv.addEventListener('click', leave);
    var nx = el.querySelector('[data-next]'); if (nx) nx.addEventListener('click', function () { start(i + 1); });
    var ag = el.querySelector('[data-again]'); if (ag) ag.addEventListener('click', function () { start(i); });
    if (cur.step === 2) tick();
  }
  function start(i) {
    cur = { m: M[i], step: 0, pick: null, st: {} };
    A.lock(cur.m.dials);
    CGH.world.setFx(true);
    cur.m.setup();
    render();
    $('missions').scrollIntoView({ block: 'nearest' });
  }
  function step(k) {
    cur.step = k;
    if (k === 2) { cur.st = { t0: S().tau }; }
    render();
  }
  function leave() { cur = null; A.lock(null); render(); }
  function tick() {
    if (!cur || cur.step !== 2) return;
    var p = cur.m.progress(cur.st), el = $('ms-status');
    if (el) el.textContent = p.text;
    if (p.done) {
      cur.finished = true;
      var meas = cur.m.measured ? cur.m.measured(cur.st) : cur.m.answer;
      done[cur.m.id] = { pick: cur.pick, right: cur.pick === meas };
      save(); step(3);
    }
  }

  CGH.missions = {
    init: function () { render(); setInterval(tick, 250); },
    list: M, start: start, leave: leave,
    titleOf: function (id) { var m = M.filter(function (x) { return x.id === id; })[0]; return m ? m.title : id; }
  };
})();
