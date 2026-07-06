"""Probe a Praat script: print its form, parse defaults, run it on one clip.

Usage:
  python scripts/probe_praat_script.py [praat/YourScript.praat]

With no path it picks the first praat/*.praat whose name mentions 'nuclei'/'syll'.
It prints the form definition, the default arguments it will pass (in order),
and the captured output (or the Praat error) from running on the first interim
clip. Use the printed form + output to finalize the real feature wrapper.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def find_script() -> Path | None:
    cands = sorted((ROOT / "praat").glob("*.praat"))
    for c in cands:
        if "nuclei" in c.name.lower() or "syll" in c.name.lower():
            return c
    return cands[0] if cands else None


def extract_form(text: str) -> list[str]:
    out, in_form = [], False
    for ln in text.splitlines():
        s = ln.strip().lower()
        if s.startswith("form "):
            in_form = True
        if in_form:
            out.append(ln)
        if s == "endform":
            break
    return out


def parse_defaults(form_lines: list[str], interim: Path):
    """Return (args, fields) using each field's default value, in form order."""
    args, fields = [], []
    body = form_lines[1:] if form_lines else []
    for idx, ln in enumerate(body):
        s = ln.strip()
        if not s or s.startswith("#") or s.lower() == "endform":
            continue
        toks = s.split(None, 1)
        kind = toks[0].lower()
        rest = toks[1] if len(toks) > 1 else ""
        if kind in ("comment", "option"):
            continue
        name, _, default = rest.partition(" ")
        default = default.strip()
        if kind in ("real", "positive", "integer", "natural"):
            try:
                val = int(default) if kind in ("integer", "natural") else float(default)
            except ValueError:
                val = 0
        elif kind == "boolean":
            val = bool(int(default or "0"))
        elif kind in ("word", "sentence", "text"):
            # Substitute the interim dir for directory-like fields.
            if any(k in name.lower() for k in ("dir", "folder", "path")):
                val = str(interim) + "/"
            else:
                val = default
        elif kind in ("optionmenu", "choice"):
            didx = int(default or "1")
            opts = []
            for ln2 in body[idx + 1 :]:
                s2 = ln2.strip()
                if s2.startswith("#"):
                    continue
                if s2.lower().startswith("option "):
                    opts.append(s2.split(None, 1)[1].strip())
                elif s2.lower().startswith("comment"):
                    continue
                else:
                    break
            val = opts[didx - 1] if 0 < didx <= len(opts) else (opts[0] if opts else "")
        else:
            continue
        args.append(val)
        fields.append((kind, name, val))
    return args, fields


def main():
    import parselmouth
    from parselmouth.praat import run_file

    script = Path(sys.argv[1]) if len(sys.argv) > 1 else find_script()
    if not script or not script.exists():
        print("No .praat script found. Put it in praat/ or pass its path as an argument.")
        return

    interim = ROOT / "data" / "interim"
    form = extract_form(script.read_text(errors="ignore"))
    print("=== FORM BLOCK ===")
    print("\n".join(form) if form else "(no form block found)")

    args, fields = parse_defaults(form, interim)
    print("\n=== default args to pass (in order) ===")
    for f in fields:
        print("  ", f)

    wavs = sorted(interim.rglob("*.wav"))
    if not wavs:
        print("\nNo interim wavs found.")
        return

    snd = parselmouth.Sound(str(wavs[0]))
    print(f"\n=== running on {wavs[0].name} ===")
    try:
        out = run_file([snd], str(script), *args, capture_output=True)
        txt = out[1] if isinstance(out, tuple) else str(out)
        print(txt[:4000] if txt.strip() else "(no captured output -- may write to a file/TextGrid)")
    except Exception as e:  # noqa: BLE001  (probe: surface any Praat error verbatim)
        print("Praat error:", e)
        print("\nUse the FORM BLOCK above to set the correct args.")


if __name__ == "__main__":
    main()
