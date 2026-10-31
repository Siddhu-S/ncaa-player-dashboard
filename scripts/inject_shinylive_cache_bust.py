#!/usr/bin/env python3
"""Inject build-aware cache busting into a shinylive-exported index.html."""

from __future__ import annotations

import sys
from pathlib import Path


CACHE_BUST_TEMPLATE = """
    <meta name="ncaa-dashboard-build" content="{build_id}" />
    <script>
      (function() {{
        var buildId = "{build_id}";
        var storageKey = "ncaa-dashboard-build-id";
        var priorBuild = null;
        try {{
          priorBuild = localStorage.getItem(storageKey);
        }} catch (err) {{
          priorBuild = null;
        }}
        var needsRefresh = priorBuild !== buildId;

        window.__DASHBOARD_BUILD_ID__ = buildId;

        var originalFetch = window.fetch ? window.fetch.bind(window) : null;
        if (originalFetch) {{
          window.fetch = function(resource, init) {{
            try {{
              var request = resource;
              var url = null;
              if (typeof resource === "string") {{
                url = new URL(resource, window.location.href);
              }} else if (resource instanceof Request) {{
                url = new URL(resource.url, window.location.href);
              }}
              if (
                url &&
                (url.pathname.endsWith("/app.json") ||
                 url.pathname.endsWith("/shinylive-sw.js") ||
                 url.pathname.endsWith("/load-shinylive-sw.js"))
              ) {{
                url.searchParams.set("v", buildId);
                request = resource instanceof Request ? new Request(url.toString(), resource) : url.toString();
              }}
              return originalFetch(request, init);
            }} catch (err) {{
              return originalFetch(resource, init);
            }}
          }};
        }}

        if (!needsRefresh) {{
          return;
        }}

        function done() {{
          try {{
            localStorage.setItem(storageKey, buildId);
          }} catch (err) {{}}
          window.location.replace(window.location.pathname + "?v=" + encodeURIComponent(buildId));
        }}

        var unregisterPromise = Promise.resolve();
        if ("serviceWorker" in navigator && navigator.serviceWorker.getRegistrations) {{
          unregisterPromise = navigator.serviceWorker.getRegistrations()
            .then(function(registrations) {{
              return Promise.all(registrations.map(function(reg) {{
                return reg.unregister();
              }}));
            }})
            .catch(function() {{ return []; }});
        }}

        unregisterPromise
          .then(function() {{
            if (!("caches" in window) || !window.caches.keys) {{
              return [];
            }}
            return window.caches.keys().then(function(keys) {{
              return Promise.all(
                keys
                  .filter(function(key) {{
                    return key.toLowerCase().indexOf("shinylive") >= 0;
                  }})
                  .map(function(key) {{
                    return window.caches.delete(key);
                  }})
              );
            }});
          }})
          .catch(function() {{ return []; }})
          .finally(done);
      }})();
    </script>
"""


def main() -> None:
    if len(sys.argv) != 3:
        print("usage: inject_shinylive_cache_bust.py <index.html> <build-id>", file=sys.stderr)
        raise SystemExit(1)

    index_path = Path(sys.argv[1])
    build_id = sys.argv[2]

    html = index_path.read_text()
    script_tag = '<script\n      src="./shinylive/load-shinylive-sw.js"\n      type="module"\n    ></script>'
    if script_tag not in html:
        print("could not find shinylive service worker loader tag", file=sys.stderr)
        raise SystemExit(1)

    cache_bust_html = CACHE_BUST_TEMPLATE.format(build_id=build_id) + script_tag.replace(
        './shinylive/load-shinylive-sw.js"',
        f'./shinylive/load-shinylive-sw.js?v={build_id}"',
    )
    html = html.replace(script_tag, cache_bust_html, 1)
    index_path.write_text(html)


if __name__ == "__main__":
    main()
