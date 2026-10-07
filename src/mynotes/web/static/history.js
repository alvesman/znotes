// History page: render stored answers as Markdown (like the chat page) after load and after
// every HTMX swap of the list. Lives in a static file because the UI CSP forbids inline scripts.
(function () {
  "use strict";

  function renderAnswers(root) {
    if (!window.MynotesMarkdown) {
      return;
    }
    Array.prototype.forEach.call(root.querySelectorAll("[data-markdown]"), function (node) {
      if (node.dataset.rendered) {
        return;
      }
      node.dataset.rendered = "1";
      window.MynotesMarkdown.render(node.textContent, node, {
        citePrefix: node.dataset.citePrefix || "cite-"
      });
    });
  }

  document.addEventListener("htmx:afterSwap", function (event) {
    renderAnswers(event.target instanceof Element ? event.target : document);
  });

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", function () {
      renderAnswers(document);
    });
  } else {
    renderAnswers(document);
  }
})();
