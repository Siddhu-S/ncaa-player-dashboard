"""Inject the client-side password gate (scripts/password_gate.html) into
docs/index.html.

Run this after `shinylive export . docs` -- that command regenerates
docs/index.html from scratch every time and would otherwise silently wipe
the gate out. .github/workflows/rebuild-pages.yml calls this automatically.

To set/change the password, edit the PASSWORD constant in
scripts/password_gate.html and push -- no need to touch this file.
"""
from pathlib import Path

HERE = Path(__file__).resolve().parent
INDEX_PATH = HERE.parent / "docs" / "index.html"
GATE_PATH = HERE / "password_gate.html"
MARKER = "<body>"


def main() -> None:
    index_html = INDEX_PATH.read_text()
    gate_html = GATE_PATH.read_text()

    if MARKER not in index_html:
        raise SystemExit(f"could not find {MARKER!r} in {INDEX_PATH} -- shinylive's output format may have changed")
    if "id=\"pw-gate\"" in index_html:
        raise SystemExit(f"{INDEX_PATH} already contains a password gate -- run shinylive export again first")

    index_html = index_html.replace(MARKER, MARKER + "\n" + gate_html, 1)
    INDEX_PATH.write_text(index_html)
    print(f"Injected password gate into {INDEX_PATH}")


if __name__ == "__main__":
    main()
