/* The service worker of the web UI, served by GET /sw.js (web/pwa.py), which fills in the two
   placeholders below. It makes the app installable and shows the offline notice when a page
   cannot be loaded. It caches static assets only: pages are per user and behind the sign-in
   cookie, the API streams answers, and the htmx partials are per user too, so none of those
   is ever stored. The raw file under /static/sw.js is never registered. */
"use strict";

var CACHE = "__CACHE_NAME__";
var PRECACHE = __PRECACHE__;
var OFFLINE_URL = PRECACHE[0];
var BYPASS = ["/v1/", "/ui/partials/"];

self.addEventListener("install", function (event) {
  event.waitUntil(
    caches
      .open(CACHE)
      .then(function (cache) {
        return cache.addAll(PRECACHE);
      })
      .then(function () {
        return self.skipWaiting();
      })
  );
});

self.addEventListener("activate", function (event) {
  event.waitUntil(
    caches
      .keys()
      .then(function (names) {
        return Promise.all(
          names
            .filter(function (name) {
              return name.indexOf("mynotes-") === 0 && name !== CACHE;
            })
            .map(function (name) {
              return caches.delete(name);
            })
        );
      })
      .then(function () {
        return self.clients.claim();
      })
  );
});

self.addEventListener("fetch", function (event) {
  var request = event.request;
  if (request.method !== "GET") {
    return; // 1. writes go straight to the server
  }
  var url = new URL(request.url);
  if (url.origin !== self.location.origin) {
    return; // 2. nothing from other origins
  }
  if (
    BYPASS.some(function (prefix) {
      return url.pathname.indexOf(prefix) === 0;
    })
  ) {
    return; // 3. the API (answers stream through it) and the htmx partials: untouched
  }
  if (request.mode === "navigate") {
    // 4. pages always come from the server (the sign-in redirect included); only a network
    //    failure shows the offline notice
    event.respondWith(
      fetch(request).catch(function () {
        return caches.match(OFFLINE_URL);
      })
    );
    return;
  }
  if (url.pathname.indexOf("/static/") === 0) {
    // 5. assets: cache first, keyed by the full URL (the ?v= stamp changes with the file)
    event.respondWith(
      caches.match(request).then(function (hit) {
        if (hit) {
          return hit;
        }
        return fetch(request).then(function (response) {
          if (response.ok) {
            var copy = response.clone();
            caches.open(CACHE).then(function (cache) {
              cache.put(request, copy);
            });
          }
          return response;
        });
      })
    );
    return;
  }
  // 6. anything else (/healthz, /favicon.ico, /sw.js itself) goes to the network
});
