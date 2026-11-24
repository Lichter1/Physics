/* cGħ World — start-up. */
(function () {
  'use strict';
  function go() { CGH.app.init(); CGH.world.init(); CGH.missions.init(); }
  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', go); else go();
})();
