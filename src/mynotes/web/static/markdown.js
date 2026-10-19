/* Markdown rendering shared by the chat answers and the search snippets. Built from DOM nodes
   only (no innerHTML): raw HTML in the text stays text. Exposes
   window.MynotesMarkdown.render(text, rootElement). Also gives the code blocks the server
   renders on the note page (<div class="code-block" data-copy-code>) the same Copy button. */
"use strict";

(function () {
  function el(tag, className, text) {
    var node = document.createElement(tag);
    if (className) {
      node.className = className;
    }
    if (text !== undefined) {
      node.textContent = text;
    }
    return node;
  }

  /* Markdown for answers, built from DOM nodes only (no innerHTML): headings, paragraphs,
     bullet/numbered lists (nested by indentation), tables, block quotes, code blocks, bold,
     italic, inline code, and [n] source markers linked to the citation list. Raw HTML in the
     text stays text, and no other links are created. */
  var INLINE_SOURCE = /(`[^`\n]+`)|(\*\*[^*\n]+?\*\*|__[^_\n]+?__)|(\*[^*\s][^*\n]*?\*|(?:^|\W)_[^_\s][^_\n]*?_(?=\W|$))|(\[\d+\])/;

  function inline(text, parent) {
    var last = 0;
    var match;
    // a fresh expression per call: the recursive calls below must not reset this one's position
    var pattern = new RegExp(INLINE_SOURCE.source, "g");
    while ((match = pattern.exec(text)) !== null) {
      var token = match[0];
      var offset = 0;
      if (match[3] && token.charAt(0) !== "*") {
        offset = 1; // the leading non-word character of the _italic_ form
        token = token.slice(1);
      }
      if (match.index + offset > last) {
        parent.appendChild(document.createTextNode(text.slice(last, match.index + offset)));
      }
      if (match[1]) {
        parent.appendChild(el("code", null, token.slice(1, -1)));
      } else if (match[2]) {
        var strong = el("strong");
        inline(token.slice(2, -2), strong);
        parent.appendChild(strong);
      } else if (match[3]) {
        var em = el("em");
        inline(token.slice(1, -1), em);
        parent.appendChild(em);
      } else {
        var anchor = el("a", null, token);
        anchor.href = "#" + citePrefix + token.slice(1, -1);
        parent.appendChild(anchor);
      }
      last = match.index + match[0].length;
    }
    if (last < text.length) {
      parent.appendChild(document.createTextNode(text.slice(last)));
    }
  }

  function copyText(text) {
    if (navigator.clipboard && window.isSecureContext) {
      return navigator.clipboard.writeText(text);
    }
    return new Promise(function (resolve, reject) {
      // plain-http pages (not localhost) have no clipboard API: fall back to a hidden textarea
      var area = document.createElement("textarea");
      area.value = text;
      area.setAttribute("readonly", "");
      area.style.position = "fixed";
      area.style.opacity = "0";
      document.body.appendChild(area);
      area.select();
      var ok = false;
      try {
        ok = document.execCommand("copy");
      } catch (err) {
        ok = false;
      }
      document.body.removeChild(area);
      if (ok) {
        resolve();
      } else {
        reject(new Error("copy failed"));
      }
    });
  }

  function copyButton(text) {
    var button = el("button", "copy-code", "Copy");
    button.type = "button";
    button.title = "Copy to clipboard";
    button.addEventListener("click", function () {
      copyText(text).then(
        function () {
          button.textContent = "Copied";
          button.classList.add("copied");
        },
        function () {
          button.textContent = "Copy failed";
        }
      ).then(function () {
        window.setTimeout(function () {
          button.textContent = "Copy";
          button.classList.remove("copied");
        }, 1500);
      });
    });
    return button;
  }

  function tableCells(line) {
    var trimmed = line.trim().replace(/^\|/, "").replace(/\|$/, "");
    return trimmed.split("|").map(function (cell) {
      return cell.trim();
    });
  }

  var LIST_ITEM = /^(\s*)([-*+]|\d+[.)])\s+(.*)$/;
  var TABLE_RULE = /^\s*\|?\s*:?-{2,}:?\s*(\|\s*:?-{2,}:?\s*)*\|?\s*$/;

  function renderMarkdown(text, root) {
    root.textContent = "";
    var lines = text.replace(/\r\n?/g, "\n").split("\n");
    var i = 0;
    while (i < lines.length) {
      var line = lines[i];
      var m;
      if (!line.trim()) {
        i += 1;
      } else if (/^\s*```/.test(line)) {
        var code = [];
        i += 1;
        while (i < lines.length && !/^\s*```/.test(lines[i])) {
          code.push(lines[i]);
          i += 1;
        }
        var closed = i < lines.length; // an unfinished block (still streaming) gets no button
        i += 1;
        var wrap = el("div", "code-block");
        var pre = el("pre");
        pre.appendChild(el("code", null, code.join("\n")));
        wrap.appendChild(pre);
        if (closed) {
          wrap.appendChild(copyButton(code.join("\n")));
        }
        root.appendChild(wrap);
      } else if ((m = /^(#{1,6})\s+(.*?)\s*#*\s*$/.exec(line))) {
        var heading = el("h" + Math.min(m[1].length + 2, 6));
        heading.className = "answer-heading";
        inline(m[2], heading);
        root.appendChild(heading);
        i += 1;
      } else if (/^\s*([-*_])(\s*\1){2,}\s*$/.test(line)) {
        root.appendChild(el("hr"));
        i += 1;
      } else if (line.indexOf("|") !== -1 && i + 1 < lines.length && TABLE_RULE.test(lines[i + 1])) {
        var table = el("table", "answer-table");
        var head = el("tr");
        tableCells(line).forEach(function (cell) {
          var th = el("th");
          inline(cell, th);
          head.appendChild(th);
        });
        var thead = el("thead");
        thead.appendChild(head);
        table.appendChild(thead);
        var body = el("tbody");
        i += 2;
        while (i < lines.length && lines[i].trim() && lines[i].indexOf("|") !== -1) {
          var row = el("tr");
          tableCells(lines[i]).forEach(function (cell) {
            var td = el("td");
            inline(cell, td);
            row.appendChild(td);
          });
          body.appendChild(row);
          i += 1;
        }
        table.appendChild(body);
        root.appendChild(table);
      } else if (/^\s*>/.test(line)) {
        var quote = [];
        while (i < lines.length && /^\s*>/.test(lines[i])) {
          quote.push(lines[i].replace(/^\s*>\s?/, ""));
          i += 1;
        }
        var blockquote = el("blockquote");
        renderMarkdown(quote.join("\n"), blockquote);
        root.appendChild(blockquote);
      } else if (LIST_ITEM.test(line)) {
        i = renderList(lines, i, root);
      } else {
        var para = [];
        while (
          i < lines.length &&
          lines[i].trim() &&
          !/^\s*(```|#{1,6}\s|>)/.test(lines[i]) &&
          !LIST_ITEM.test(lines[i]) &&
          !(lines[i].indexOf("|") !== -1 && i + 1 < lines.length && TABLE_RULE.test(lines[i + 1]))
        ) {
          para.push(lines[i].trim());
          i += 1;
        }
        var paragraph = el("p");
        inline(para.join("\n"), paragraph);
        root.appendChild(paragraph);
      }
    }
  }

  function renderList(lines, start, root) {
    var i = start;
    var stack = []; // {indent, list, item}
    while (i < lines.length) {
      var m = LIST_ITEM.exec(lines[i]);
      if (!m) {
        // a wrapped continuation line of the last item (indented text), else the list ends
        if (lines[i].trim() && /^\s{2,}\S/.test(lines[i]) && stack.length) {
          var current = stack[stack.length - 1].item;
          current.appendChild(document.createTextNode(" " + lines[i].trim()));
          i += 1;
          continue;
        }
        break;
      }
      var indent = m[1].replace(/\t/g, "    ").length;
      var ordered = /\d/.test(m[2].charAt(0));
      while (stack.length && indent < stack[stack.length - 1].indent) {
        stack.pop();
      }
      if (
        stack.length &&
        indent === stack[stack.length - 1].indent &&
        stack[stack.length - 1].ordered !== ordered
      ) {
        stack.pop(); // a bulleted list followed by a numbered one (or the reverse) is a new list
      }
      if (!stack.length || indent > stack[stack.length - 1].indent) {
        var list = el(ordered ? "ol" : "ul");
        var parent = stack.length ? stack[stack.length - 1].item : root;
        parent.appendChild(list);
        stack.push({ indent: indent, list: list, item: null, ordered: ordered });
      }
      var top = stack[stack.length - 1];
      top.item = el("li");
      inline(m[3], top.item);
      top.list.appendChild(top.item);
      i += 1;
    }
    return i;
  }

  var citePrefix = "cite-";

  // render(text, root[, {citePrefix}]): [n] markers link to "#<citePrefix><n>" ("cite-" by
  // default, the chat page's citation ids; the History page gives each chat its own prefix).
  function render(text, root, options) {
    var previous = citePrefix;
    citePrefix = (options && options.citePrefix) || "cite-";
    try {
      renderMarkdown(text, root);
    } finally {
      citePrefix = previous;
    }
  }

  // Copy buttons for server-rendered code blocks (the note page, web/render.py)
  function addCopyButtons(root) {
    var blocks = (root || document).querySelectorAll(".code-block[data-copy-code]");
    Array.prototype.forEach.call(blocks, function (block) {
      var pre = block.querySelector("pre");
      if (!pre || block.querySelector(".copy-code")) {
        return;
      }
      block.appendChild(copyButton(pre.textContent.replace(/\n$/, "")));
    });
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", function () {
      addCopyButtons(document);
    });
  } else {
    addCopyButtons(document);
  }

  window.MynotesMarkdown = { render: render, addCopyButtons: addCopyButtons };
})();
