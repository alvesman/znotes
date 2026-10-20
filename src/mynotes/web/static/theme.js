/* Theme bootstrap. Loaded in <head> before the stylesheet and without defer, so the
   remembered theme is on <html> before the first paint (no flash of the other theme). The
   CSP forbids inline scripts, hence this file. Also marks <html class="js"> for progressive
   enhancement. Exposes window.MynotesTheme: choice() is "system" | "light" | "dark",
   effective() is the theme in use, set(choice) applies and remembers one and dispatches
   "mynotes:theme-changed" on document with {choice, effective}. Storage can be unavailable
   (private windows, blocked site data): the choice then lasts for the page only. */
(function () {
  "use strict";

  var KEY = "mynotes.pref.theme";
  var root = document.documentElement;
  var choice = "system";

  root.classList.add("js");

  function normalize(value) {
    return value === "light" || value === "dark" ? value : "system";
  }

  function apply(next) {
    if (next === "system") {
      root.removeAttribute("data-theme");
    } else {
      root.setAttribute("data-theme", next);
    }
  }

  function effective() {
    if (choice !== "system") {
      return choice;
    }
    var dark = window.matchMedia && window.matchMedia("(prefers-color-scheme: dark)").matches;
    return dark ? "dark" : "light";
  }

  function announce() {
    document.dispatchEvent(
      new CustomEvent("mynotes:theme-changed", {
        detail: { choice: choice, effective: effective() }
      })
    );
  }

  try {
    choice = normalize(window.localStorage.getItem(KEY));
  } catch (err) {
    choice = "system";
  }
  apply(choice);

  if (window.matchMedia) {
    var query = window.matchMedia("(prefers-color-scheme: dark)");
    var onChange = function () {
      if (choice === "system") {
        announce();
      }
    };
    if (query.addEventListener) {
      query.addEventListener("change", onChange);
    } else if (query.addListener) {
      query.addListener(onChange);
    }
  }

  window.MynotesTheme = {
    choice: function () {
      return choice;
    },
    effective: effective,
    set: function (next) {
      choice = normalize(next);
      apply(choice);
      try {
        if (choice === "system") {
          window.localStorage.removeItem(KEY);
        } else {
          window.localStorage.setItem(KEY, choice);
        }
      } catch (err) {
        /* storage blocked: the choice lasts for this page */
      }
      announce();
    }
  };
})();
