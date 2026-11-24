/* cGħ World — start-up. */
(function () {
  'use strict';
  function go() { CGH.app.init(); CGH.world.init(); CGH.labs.init(); CGH.missions.init(); CGH.quiz.init(); }
  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', go); else go();
})();
