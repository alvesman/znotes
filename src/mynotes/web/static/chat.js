/* Chat page (task 47): streams /v1/ask/stream SSE and renders citations as DOM
   nodes. Model output is only ever inserted as text, never as HTML. */
"use strict";

(function () {
  var OFFICE_FORMATS = ["docx", "pptx", "xlsx", "epub"];

  var FOLD_MAP = {
    "ø": "o", "æ": "ae", "œ": "oe", "ł": "l",
    "đ": "d", "ð": "d", "þ": "th", "ı": "i"
  };

  function slug(text) {
    return String(text)
      .normalize("NFD")
      .replace(/[\u0300-\u036f]/g, "")
      .replace(/[øæœłđðþı]/g, function (ch) { return FOLD_MAP[ch] || ch; })
      .toLowerCase()
      .replace(/ß/g, "ss")
      .replace(/\s+/g, " ")
      .trim()
      .replace(/ /g, "-")
      .replace(/[^a-z0-9-]/g, "")
      .replace(/^-+|-+$/g, "");
  }

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

  function secFragment(citation) {
    return citation.section ? "#sec-" + slug(citation.section) : "";
  }

  function citationActions(citation) {
    var actions = el("span", "cite-actions");
    var format = citation.format || "";
    var raw = "/v1/files/" + citation.document_id + "/raw";
    var link;
    if (format === "pdf") {
      link = el("a", null, "Open");
      link.href = citation.page ? raw + "#page=" + citation.page : raw;
      link.target = "_blank";
      link.rel = "noopener";
      actions.appendChild(link);
    } else if (format === "md" || format === "txt") {
      link = el("a", null, "View note");
      link.href = "/ui/documents/" + citation.document_id + secFragment(citation);
      link.target = "_blank";
      link.rel = "noopener";
      actions.appendChild(link);
    } else if (format === "html") {
      link = el("a", null, "Open");
      link.href = raw;
      link.target = "_blank";
      link.rel = "noopener";
      actions.appendChild(link);
    } else if (OFFICE_FORMATS.indexOf(format) !== -1) {
      if (document.body.dataset.canOpenFiles === "true") {
        var button = el("button", "open-in-app", "Open in app");
        button.type = "button";
        button.addEventListener("click", function () {
          fetch("/v1/files/" + citation.document_id + "/open", { method: "POST" });
        });
        actions.appendChild(button);
      }
      link = el("a", null, "Download");
      link.href = raw;
      actions.appendChild(link);
      link = el("a", null, "View text");
      link.href = "/ui/documents/" + citation.document_id + secFragment(citation);
      link.target = "_blank";
      link.rel = "noopener";
      actions.appendChild(link);
    }
    return actions;
  }

  function citationNode(citation) {
    var node = el("div", "citation");
    if (citation.n) {
      node.id = "cite-" + citation.n;
      node.appendChild(el("span", "cite-n", "[" + citation.n + "] "));
    }
    node.appendChild(el("span", "cite-title", citation.title || citation.path));
    node.appendChild(el("span", "cite-sep", " \u2014 "));
    node.appendChild(el("span", "cite-path", citation.path));
    if (citation.page) {
      node.appendChild(el("span", "cite-loc", " (p. " + citation.page + ")"));
    } else if (citation.slide) {
      node.appendChild(el("span", "cite-loc", " (slide " + citation.slide + ")"));
    }
    if (citation.doc_date) {
      node.appendChild(el("span", "cite-date", " \u2014 " + String(citation.doc_date).split("T")[0]));
    }
    if (!citation.is_current) {
      node.appendChild(el("span", "badge", "historical"));
    }
    node.appendChild(citationActions(citation));
    return node;
  }

  var renderMarkdown = window.MynotesMarkdown.render;

  function handleFrame(frame, block) {
    var event = "message";
    var dataLines = [];
    frame.split("\n").forEach(function (line) {
      if (line.indexOf("event:") === 0) {
        event = line.slice(6).trim();
      } else if (line.indexOf("data:") === 0) {
        dataLines.push(line.slice(5).trim());
      }
    });
    if (!dataLines.length) {
      return;
    }
    var payload;
    try {
      payload = JSON.parse(dataLines.join("\n"));
    } catch (err) {
      return;
    }
    if (event === "rewritten") {
      // a follow-up searched as a standalone question: show it, so a misreading is visible
      block.searched.textContent = "Searched as: " + payload.question;
      block.searched.hidden = false;
    } else if (event === "token") {
      block.raw += payload.text;
      renderMarkdown(block.raw, block.answer);
    } else if (event === "sources") {
      (payload.citations || []).forEach(function (citation) {
        block.citations.appendChild(citationNode(citation));
      });
    } else if (event === "done") {
      renderMarkdown(block.raw, block.answer);
      if (!payload.relevant) {
        block.status.textContent = "No relevant notes were found for this question.";
      }
    } else if (event === "error") {
      block.status.textContent = payload.message || "The request failed. Please try again.";
    }
  }

  function readStream(response, block) {
    var reader = response.body.getReader();
    var decoder = new TextDecoder();
    var buffer = "";
    function step() {
      return reader.read().then(function (read) {
        if (read.done) {
          if (buffer.trim()) {
            handleFrame(buffer, block);
          }
          return;
        }
        buffer += decoder.decode(read.value, { stream: true });
        var frames = buffer.split("\n\n");
        buffer = frames.pop();
        frames.forEach(function (frame) {
          handleFrame(frame, block);
        });
        return step();
      });
    }
    return step();
  }

  // One id per chat page: every question asked here belongs to the same conversation in the
  // History page. Reloading the page (or opening Chat again) starts a new conversation.
  function newConversationId() {
    if (window.crypto && typeof window.crypto.randomUUID === "function") {
      return window.crypto.randomUUID();
    }
    return "c" + Date.now().toString(36) + Math.random().toString(36).slice(2, 10);
  }

  var conversationId = null; // set in init(): the resumed chat's id, or a new one

  function ask(question, filters, block) {
    return fetch("/v1/ask/stream", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        question: question,
        filters: filters,
        conversation_id: conversationId
      })
    }).then(function (response) {
      if (!response.ok || !response.body) {
        throw new Error("HTTP " + response.status);
      }
      return readStream(response, block);
    });
  }

  function init() {
    var form = document.getElementById("ask-form");
    if (!form) {
      return;
    }
    var conversation = document.getElementById("conversation");
    conversationId = conversation.dataset.conversationId || newConversationId();
    // a chat resumed from History: its stored answers are shown formatted, like new ones
    Array.prototype.forEach.call(conversation.querySelectorAll("[data-markdown]"), function (node) {
      renderMarkdown(node.textContent, node, { citePrefix: node.dataset.citePrefix });
    });
    if (conversation.lastElementChild) {
      conversation.lastElementChild.scrollIntoView({ block: "end" });
    }
    var questionInput = document.getElementById("question");
    var tagsInput = document.getElementById("tags");
    var historyInput = document.getElementById("history");
    var button = document.getElementById("ask-button");

    // Enter asks; Shift+Enter inserts a newline (not while an input method is composing).
    questionInput.addEventListener("keydown", function (event) {
      if (event.key === "Enter" && !event.shiftKey && !event.isComposing) {
        event.preventDefault();
        if (!button.disabled) {
          button.click();
        }
      }
    });

    form.addEventListener("submit", function (event) {
      event.preventDefault();
      var question = questionInput.value.trim();
      if (!question) {
        return;
      }
      var filters = { history: historyInput.checked };
      var tags = tagsInput.value
        .split(",")
        .map(function (tag) {
          return tag.trim();
        })
        .filter(Boolean);
      if (tags.length) {
        filters.tags = tags;
      }
      questionInput.value = "";
      var block = {
        root: el("article", "qa"),
        raw: "",
        searched: el("p", "searched muted"),
        answer: el("div", "answer md"),
        citations: el("div", "citations"),
        status: el("p", "status")
      };
      block.root.appendChild(el("p", "question", question));
      block.searched.hidden = true;
      block.root.appendChild(block.searched);
      block.root.appendChild(block.answer);
      block.root.appendChild(block.citations);
      block.root.appendChild(block.status);
      conversation.appendChild(block.root);
      button.disabled = true;
      ask(question, filters, block)
        .catch(function () {
          block.status.textContent = "The request failed. Please try again.";
        })
        .then(function () {
          button.disabled = false;
          questionInput.focus();
        });
    });
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
