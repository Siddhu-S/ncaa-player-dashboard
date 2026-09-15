#!/usr/bin/env python3
"""Injects a client-side password gate into a shinylive-exported index.html.

Run after `shinylive export`, e.g.:
    python scripts/inject_password_gate.py docs/index.html <sha256-hex-of-password>

The gate blocks the page with an overlay until the visitor enters a password
whose SHA-256 hash matches the one baked in at build time. This deters casual
visitors but is not real security: the hash ships in the page source, so a
motivated visitor could brute-force it client-side or just read the app
source directly out of docs/app.json.
"""
import sys
from pathlib import Path

GATE_TEMPLATE = """
<style>
  #pw-gate {{
    position: fixed; inset: 0; z-index: 999999;
    display: flex; align-items: center; justify-content: center;
    background: #0b1220; font-family: system-ui, sans-serif;
  }}
  #pw-gate form {{
    background: #151d2e; padding: 2rem 2.5rem; border-radius: 12px;
    display: flex; flex-direction: column; gap: 0.75rem; min-width: 280px;
    box-shadow: 0 10px 40px rgba(0,0,0,0.4);
  }}
  #pw-gate h1 {{ color: #e8ecf4; font-size: 1.1rem; margin: 0 0 0.25rem; }}
  #pw-gate input {{
    padding: 0.6rem 0.75rem; border-radius: 8px; border: 1px solid #2c3a54;
    background: #0b1220; color: #e8ecf4; font-size: 1rem;
  }}
  #pw-gate button {{
    padding: 0.6rem 0.75rem; border-radius: 8px; border: none;
    background: #3b82f6; color: white; font-size: 1rem; cursor: pointer;
  }}
  #pw-gate button:hover {{ background: #2563eb; }}
  #pw-gate .pw-error {{ color: #f87171; font-size: 0.85rem; min-height: 1.1em; }}
  body.pw-locked > *:not(#pw-gate) {{ display: none !important; }}
</style>
<div id="pw-gate">
  <form id="pw-gate-form" autocomplete="off">
    <h1>This dashboard is password protected</h1>
    <input type="password" id="pw-gate-input" placeholder="Password" autofocus />
    <button type="submit">Enter</button>
    <div class="pw-error" id="pw-gate-error"></div>
  </form>
</div>
<script>
(function() {{
  var STORAGE_KEY = "ncaa-dashboard-authed-v1";
  var EXPECTED_HASH = "{password_hash}";

  function unlock() {{
    document.body.classList.remove("pw-locked");
    var gate = document.getElementById("pw-gate");
    if (gate) {{
      gate.remove();
    }}
  }}

  function sha256Hex(text) {{
    var enc = new TextEncoder().encode(text);
    return crypto.subtle.digest("SHA-256", enc).then(function(buf) {{
      return Array.from(new Uint8Array(buf))
        .map(function(b) {{ return b.toString(16).padStart(2, "0"); }})
        .join("");
    }});
  }}

  if (sessionStorage.getItem(STORAGE_KEY) === EXPECTED_HASH) {{
    unlock();
    return;
  }}

  document.body.classList.add("pw-locked");

  document.getElementById("pw-gate-form").addEventListener("submit", function(e) {{
    e.preventDefault();
    var val = document.getElementById("pw-gate-input").value;
    sha256Hex(val).then(function(hash) {{
      if (hash === EXPECTED_HASH) {{
        sessionStorage.setItem(STORAGE_KEY, EXPECTED_HASH);
        unlock();
      }} else {{
        document.getElementById("pw-gate-error").textContent = "Incorrect password.";
      }}
    }});
  }});
}})();
</script>
"""


def main():
    if len(sys.argv) != 3:
        print("usage: inject_password_gate.py <index.html> <sha256-hex>", file=sys.stderr)
        sys.exit(1)

    index_path = Path(sys.argv[1])
    password_hash = sys.argv[2]

    html = index_path.read_text()
    marker = "<body>"
    if marker not in html:
        print("could not find <body> tag to inject into", file=sys.stderr)
        sys.exit(1)

    gate_html = GATE_TEMPLATE.format(password_hash=password_hash)
    html = html.replace(marker, marker + gate_html, 1)
    index_path.write_text(html)


if __name__ == "__main__":
    main()
