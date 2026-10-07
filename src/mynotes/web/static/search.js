// Search page (task 47): tag picker filtering and automatic re-search.
// Lives in a static file because the UI CSP forbids inline scripts.
(function () {
  "use strict";

  function fold(text) {
    return String(text || "")
      .normalize("NFD")
      .replace(/[̀-ͯ]/g, "")
      .toLowerCase()
      .trim();
  }

  function init() {
    var form = document.getElementById("search-form");
    if (!form) {
      return;
    }
    var query = form.querySelector('input[name="query"]');
    var filter = document.getElementById("tag-filter");
    var clear = document.getElementById("tag-clear");
    var count = document.getElementById("tag-selected-count");
    var none = document.getElementById("tag-none");
    var options = Array.prototype.slice.call(form.querySelectorAll(".tag-option"));

    function checkedTags() {
      return form.querySelectorAll('input[name="tags"]:checked');
    }

    function refreshSelection() {
      var selected = checkedTags().length;
      if (count) {
        count.textContent = selected ? "· " + selected + " selected" : "";
      }
      if (clear) {
        clear.hidden = selected === 0;
      }
    }

    // Show the tags whose name contains the typed text (case and accents
    // ignored); selected tags always stay visible.
    function applyFilter() {
      if (!filter) {
        return;
      }
      var needle = fold(filter.value);
      var shown = 0;
      options.forEach(function (option) {
        var box = option.querySelector("input");
        var match = !needle || fold(option.dataset.name).indexOf(needle) !== -1 || box.checked;
        option.hidden = !match;
        if (match) {
          shown += 1;
        }
      });
      if (none) {
        none.hidden = shown !== 0;
      }
    }

    // Re-run the search when a filter changes, once there is something to search for.
    function research() {
      if (!query || !query.value.trim()) {
        return;
      }
      if (typeof form.requestSubmit === "function") {
        form.requestSubmit();
      } else if (window.htmx) {
        window.htmx.trigger(form, "submit");
      }
    }

    if (filter) {
      filter.addEventListener("input", applyFilter);
      filter.addEventListener("keydown", function (event) {
        if (event.key === "Enter") {
          event.preventDefault();
        } else if (event.key === "Escape") {
          filter.value = "";
          applyFilter();
        }
      });
    }

    form.addEventListener("change", function (event) {
      if (event.target === filter || event.target === query) {
        return;
      }
      if (event.target.name === "tags") {
        refreshSelection();
      }
      research();
    });

    if (clear) {
      clear.addEventListener("click", function () {
        Array.prototype.forEach.call(checkedTags(), function (box) {
          box.checked = false;
        });
        refreshSelection();
        applyFilter();
        research();
      });
    }

    refreshSelection();
  }

  // Snippets arrive as plain text inside the HTMX partial; show them formatted, like chat answers.
  function renderSnippets(root) {
    if (!window.MynotesMarkdown) {
      return;
    }
    Array.prototype.forEach.call(root.querySelectorAll(".hit-snippet"), function (node) {
      if (node.dataset.rendered) {
        return;
      }
      node.dataset.rendered = "1";
      window.MynotesMarkdown.render(node.textContent, node);
    });
  }

  document.addEventListener("htmx:afterSwap", function (event) {
    renderSnippets(event.target instanceof Element ? event.target : document);
  });

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
