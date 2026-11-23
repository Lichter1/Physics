/* cGħ Lab — start-up. Scenes register themselves, then the shell opens the tab named in the URL. */
(function () {
  'use strict';
  function go() {
    CGH.app.init();
    CGH.light.init();
    CGH.gravity.init();
    CGH.quantum.init();
    CGH.app.start();
  }
  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', go); else go();
})();
