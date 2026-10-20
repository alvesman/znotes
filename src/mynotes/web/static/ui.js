/* Shell behaviour shared by every page (loaded deferred from base.html): the theme switch, the
   phone menu, the panels that collapse on phones and the tab state on Reviews. theme.js has already applied the remembered theme; this file only wires
   the buttons. Everything works without it: the menu links are visible when <html> lacks
   the "js" class that theme.js adds. */
(function () {
  "use strict";

  function initThemeSwitch(root) {
    var group = root.querySelector("[data-theme-switch]");
    var theme = window.MynotesTheme;
    if (!group || !theme) {
      return;
    }
    var buttons = Array.prototype.slice.call(group.querySelectorAll("[data-theme-choice]"));

    function reflect() {
      var current = theme.choice();
      buttons.forEach(function (button) {
        button.setAttribute("aria-pressed", button.dataset.themeChoice === current ? "true" : "false");
      });
    }

    buttons.forEach(function (button) {
      button.addEventListener("click", function () {
        theme.set(button.dataset.themeChoice);
      });
    });
    document.addEventListener("mynotes:theme-changed", reflect);
    reflect();
  }

  // Phone menu: the button before #nav-links toggles it (CSS shows the list when
  // aria-expanded is true); Escape closes it and returns focus to the button.
  function initMenu(root) {
    var toggle = root.querySelector(".nav-toggle[aria-controls]");
    if (!toggle) {
      return;
    }
    var links = document.getElementById(toggle.getAttribute("aria-controls"));
    toggle.addEventListener("click", function () {
      var open = toggle.getAttribute("aria-expanded") === "true";
      toggle.setAttribute("aria-expanded", open ? "false" : "true");
    });
    if (links) {
      links.addEventListener("keydown", function (event) {
        if (event.key === "Escape" && toggle.getAttribute("aria-expanded") === "true") {
          toggle.setAttribute("aria-expanded", "false");
          toggle.focus();
        }
      });
    }
  }

  // Panels marked data-collapse-narrow are open in the HTML (so they work without JS) and
  // start collapsed on phones; they reopen when the window grows, where the summary is hidden.
  function initCollapsibles(root) {
    var panels = Array.prototype.slice.call(root.querySelectorAll("details[data-collapse-narrow]"));
    if (!panels.length || !window.matchMedia) {
      return;
    }
    var narrow = window.matchMedia("(max-width: 48rem)");
    function apply() {
      panels.forEach(function (panel) {
        if (narrow.matches) {
          panel.removeAttribute("open");
        } else {
          panel.setAttribute("open", "");
        }
      });
    }
    apply();
    if (narrow.addEventListener) {
      narrow.addEventListener("change", apply);
    } else if (narrow.addListener) {
      narrow.addListener(apply);
    }
  }

  // Tabs (Reviews): htmx loads the panel; this keeps aria-selected on the chosen tab and lets
  // Left/Right move between tabs.
  function initTabs(root) {
    Array.prototype.forEach.call(root.querySelectorAll('[role="tablist"]'), function (list) {
      var tabs = Array.prototype.slice.call(list.querySelectorAll('[role="tab"]'));
      function select(tab) {
        tabs.forEach(function (other) {
          other.setAttribute("aria-selected", other === tab ? "true" : "false");
        });
      }
      tabs.forEach(function (tab, index) {
        tab.addEventListener("click", function () {
          select(tab);
        });
        tab.addEventListener("keydown", function (event) {
          var step = event.key === "ArrowRight" ? 1 : event.key === "ArrowLeft" ? -1 : 0;
          if (!step) {
            return;
          }
          event.preventDefault();
          var next = tabs[(index + step + tabs.length) % tabs.length];
          next.focus();
          next.click();
        });
      });
    });
  }

  function init() {
    initThemeSwitch(document);
    initMenu(document);
    initCollapsibles(document);
    initTabs(document);
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
