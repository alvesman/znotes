// Remembers option checkboxes per browser (localStorage). A checkbox opts in with
// data-persist="<key>"; the page's own default applies until the person changes it once.
// Storage can be unavailable (private windows, blocked site data): the page then simply
// keeps its defaults.
(function () {
  "use strict";

  var PREFIX = "mynotes.pref.";

  function read(key) {
    try {
      return window.localStorage.getItem(PREFIX + key);
    } catch (err) {
      return null;
    }
  }

  function write(key, value) {
    try {
      window.localStorage.setItem(PREFIX + key, value);
    } catch (err) {
      /* storage blocked: nothing to remember */
    }
  }

  function init() {
    Array.prototype.forEach.call(
      document.querySelectorAll('input[type="checkbox"][data-persist]'),
      function (box) {
        var key = box.dataset.persist;
        var stored = read(key);
        if (stored === "1" || stored === "0") {
          box.checked = stored === "1";
        }
        box.addEventListener("change", function () {
          write(key, box.checked ? "1" : "0");
        });
      }
    );
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
