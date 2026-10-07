/* Graph explorer (task 48): renders the entity neighbourhood with the vendored
   Cytoscape build. All rendering goes through DOM APIs and cytoscape's own
   canvas; model data is never inserted as HTML. */
"use strict";

(function () {
  var TYPE_COLORS = {
    Project: "#4f7cc9",
    Idea: "#8a6fc9",
    Person: "#c94f7c",
    Organization: "#c98a4f",
    Technology: "#4fc98a",
    Concept: "#7c7c7c",
    Decision: "#4fb9c9",
    Requirement: "#c9c14f",
    Milestone: "#914fc9",
    Risk: "#c94f4f",
    Metric: "#4f9ec9"
  };
  var FALLBACK_COLOR = "#7c9ec9";
  var EDGE_LABEL_MAX = 28;

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

  function colorFor(type) {
    return TYPE_COLORS[type] || FALLBACK_COLOR;
  }

  function edgeLabel(edge) {
    var label = edge.attribute !== null && edge.attribute !== undefined
      ? edge.attribute + "=" + edge.value
      : edge.type;
    label = String(label || "");
    return label.length > EDGE_LABEL_MAX ? label.slice(0, EDGE_LABEL_MAX - 1) + "…" : label;
  }

  function nodeLabel(name) {
    var label = String(name || "");
    return label.length > 24 ? label.slice(0, 23) + "…" : label;
  }

  function showEdgeEvidence(edge) {
    var info = document.getElementById("graph-edge-info");
    if (!info) {
      return;
    }
    info.textContent = "";
    var label = el("span", "cite-title", edgeLabel(edge) + " · " + edge.status);
    info.appendChild(label);
    var count = edge.evidence_count || (edge.evidence || []).length;
    info.appendChild(el("span", "muted", " evidence: " + count + " document(s)"));
    (edge.evidence || []).forEach(function (document_) {
      var link = el("a", null, document_.title || document_.path);
      link.href = "/ui/documents/" + document_.document_id;
      info.appendChild(link);
    });
    info.hidden = false;
  }

  function buildElements(payload) {
    var elements = [];
    var entity = payload.entity || {};
    elements.push({
      data: {
        id: "e" + entity.id,
        label: nodeLabel(entity.name),
        type: entity.type || "",
        color: colorFor(entity.type),
        center: true
      }
    });
    (payload.neighbors || []).forEach(function (neighbor) {
      elements.push({
        data: {
          id: "e" + neighbor.id,
          label: nodeLabel(neighbor.name),
          type: neighbor.type || "",
          color: colorFor(neighbor.type),
          center: false
        }
      });
    });
    (payload.edges || []).forEach(function (edge) {
      elements.push({
        data: {
          id: "r" + edge.relation_id,
          source: "e" + edge.src,
          target: "e" + edge.dst,
          label: edgeLabel(edge),
          status: edge.status || "active",
          "line-style": (edge.status || "active") === "active" ? "solid" : "dashed",
          edge: edge
        }
      });
    });
    return elements;
  }

  function cytoscapeStyle() {
    return [
      {
        selector: "node",
        style: {
          "background-color": "data(color)",
          label: "data(label)",
          color: getComputedStyle(document.body).getPropertyValue("--fg") || "#1c1c1c",
          "font-size": 10,
          "text-valign": "bottom",
          "text-margin-y": 4
        }
      },
      {
        selector: "node[center]",
        style: {
          width: 34,
          height: 34,
          "border-width": 2,
          "border-color": getComputedStyle(document.body).getPropertyValue("--accent") || "#0b62c4"
        }
      },
      {
        selector: "edge",
        style: {
          label: "data(label)",
          "font-size": 8,
          "text-background-color": getComputedStyle(document.body).getPropertyValue("--bg") || "#fbfbfa",
          "text-background-opacity": 0.85,
          width: 1.5,
          "curve-style": "bezier",
          "target-arrow-shape": "triangle",
          "line-color": "#8a8a86",
          "line-style": "data(line-style)"
        }
      }
    ];
  }

  function renderGraph(container, payload) {
    container.textContent = "";
    if (typeof window.cytoscape !== "function") {
      container.appendChild(el("p", "empty", "The graph library failed to load."));
      return;
    }
    var elements = buildElements(payload);
    var graph = window.cytoscape({
      container: container,
      elements: elements,
      style: cytoscapeStyle(),
      layout: { name: "cose", animate: false, randomize: true }
    });
    graph.on("tap", "node", function (event) {
      var id = event.target.id().slice(1);
      var name = event.target.data("label");
      container.dataset.entityId = id;
      container.dataset.entityName = name;
      loadNeighbors(container, container.dataset.hops || "1");
    });
    graph.on("tap", "edge", function (event) {
      showEdgeEvidence(event.target.data("edge"));
    });
  }

  function loadNeighbors(container, hops) {
    container.dataset.hops = hops;
    var entityId = container.dataset.entityId;
    if (!entityId) {
      return;
    }
    container.appendChild(el("p", "empty", "Loading neighbourhood…"));
    fetch("/v1/graph/entities/" + encodeURIComponent(entityId) + "/neighbors?hops=" + hops)
      .then(function (response) {
        if (!response.ok) {
          throw new Error("HTTP " + response.status);
        }
        return response.json();
      })
      .then(function (payload) {
        container.dataset.entityId = payload.entity && payload.entity.id;
        renderGraph(container, payload);
      })
      .catch(function () {
        container.textContent = "";
        container.appendChild(el("p", "empty", "The neighbourhood could not be loaded."));
      });
  }

  function init(root) {
    var container = root.querySelector("#graph-view");
    if (!container || !container.dataset.entityId) {
      return;
    }
    var hops = root.querySelector("#graph-hops");
    if (hops) {
      hops.addEventListener("change", function () {
        loadNeighbors(container, hops.value);
      });
    }
    loadNeighbors(container, hops ? hops.value : "1");
  }

  document.addEventListener("DOMContentLoaded", function () {
    init(document);
  });
  document.addEventListener("htmx:afterSwap", function (event) {
    if (event.target && event.target.id === "entity-panel") {
      init(event.target);
    }
  });
})();
