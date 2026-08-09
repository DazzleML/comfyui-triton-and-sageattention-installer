"""Build native tool shims that emit German (cp850) output — reproduces Issue #36.

Why this exists
---------------
Issue #36 was reported from a German Windows install:

    UnicodeDecodeError: 'charmap' codec can't decode byte 0x81 ...
    ERROR - Installation failed: argument of type 'NoneType' is not iterable

The trigger is NOT the console codepage. `subprocess.run(..., text=True)` with no
`encoding=` decodes using the PARENT's ANSI codepage (cp1252 on German Windows),
while native tools like nvcc/cl/nmake emit the OEM console codepage (cp850).
Byte 0x81 is "u-umlaut" in cp850 and undefined in cp1252, so one umlaut anywhere
in a tool's output killed the install.

Reproducing that needs a CHILD PROCESS that emits those bytes — not a different
console setting, and not a mocked function. Hence real compiled .exe shims.

Two false starts worth recording so nobody repeats them:
  1. `.bat` shims DO NOT WORK. subprocess with a list argv uses CreateProcess,
     which only appends `.exe` — the real tool on PATH is found instead and the
     test silently passes while proving nothing.
  2. Shimming `nvcc` alone reproduces A DIFFERENT crash than the reported one.
     nvcc is probed during plan_installation(), which sits OUTSIDE install()'s
     try block, so the exception escapes and the destructive cleanup never runs.
     The reporter's crash is in `nmake /?` at the build-tools check, INSIDE the
     try — which is why cleanup ran and their PyTorch was uninstalled.

Usage
-----
    python tests/one-offs/make_german_locale_shims.py            # build into %TEMP%\\de_shim
    python tests/one-offs/make_german_locale_shims.py --out DIR  # elsewhere

Then prepend the output directory to PATH and run the installer against a
torch-less venv:

    PATH="<outdir>:$PATH" python comfyui_triton_sageattention.py --install \\
        --non-interactive --base-path <scratch ComfyUI> --python <empty venv>/Scripts/python.exe

Expected on FIXED code: build tools detected, plan produced, install proceeds.
Expected on code before the fix: the two errors quoted at the top.

Requires MSVC (`cl`) — the same Build Tools this installer sets up.
"""
import argparse
import os
import subprocess
import sys
from pathlib import Path

# Real localized output as the German tools emit it. The umlauts are the point:
# 'ü' is 0x81 in cp850, which cp1252 cannot decode.
SHIMS = {
    "nvcc": (
        "nvcc: NVIDIA (R) Cuda compiler driver\r\n"
        "Copyright (c) 2005-2026 NVIDIA Corporation\r\n"
        "Ungültig: Gerät wird für Größe geprüft\r\n"
        "Cuda compilation tools, release 13.2, V13.2.51\r\n"
    ),
    # This one reproduces the REPORTED crash. The installer tests
    #   "Microsoft (R) Program Maintenance Utility" in result.stdout
    # and German nmake says "Programmwartungsprogramm", so even after the decode
    # is fixed the check correctly fails to match -- it just no longer explodes.
    "nmake": (
        "Microsoft (R) Programmwartungsprogramm Version 14.44.35207.1\r\n"
        "Copyright (C) Microsoft Corporation. Alle Rechte vorbehalten.\r\n\r\n"
        "Ungültige Option. Grösse wird geprüft.\r\n"
    ),
}

C_TEMPLATE = """#include <stdio.h>
#include <io.h>
#include <fcntl.h>
/* Stands in for {name} on a German Windows box: writes raw OEM cp850 bytes,
   including 0x81 (u-umlaut), exactly as the real tool would. Binary mode so
   the CRT does not transform anything on the way out. */
int main(void) {{
    static const unsigned char out[] = {{{body}}};
    _setmode(_fileno(stdout), _O_BINARY);
    fwrite(out, 1, sizeof(out), stdout);
    return 0;
}}
"""

VCVARS_CANDIDATES = [
    r"C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat",
    r"C:\Program Files\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat",
    r"C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat",
    r"C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat",
]


def find_vcvars():
    for candidate in VCVARS_CANDIDATES:
        if Path(candidate).exists():
            return candidate
    return None


def build(outdir: Path) -> int:
    if sys.platform != "win32":
        print("This harness reproduces a Windows codepage bug; skipping.")
        return 0

    vcvars = find_vcvars()
    if not vcvars:
        print("ERROR: no vcvars64.bat found. Install MSVC Build Tools first.")
        return 1

    outdir.mkdir(parents=True, exist_ok=True)
    for name, text in SHIMS.items():
        raw = text.encode("cp850")
        assert b"\x81" in raw, f"{name}: fixture lost the 0x81 byte"
        body = ",".join(str(b) for b in raw)
        (outdir / f"{name}.c").write_text(
            C_TEMPLATE.format(name=name, body=body), encoding="ascii"
        )
        # Build through a .bat rather than an inline `cmd /c "..."` string.
        # Passing the command as a list element makes Python's list2cmdline
        # escape the embedded quotes to \" before cmd sees them, and the vcvars
        # path (which contains spaces) then fails to resolve. A script file has
        # no quoting layer to get wrong.
        bat = outdir / f"_build_{name}.bat"
        bat.write_text(
            "@echo off\r\n"
            f'call "{vcvars}" >nul 2>&1\r\n'
            f'cd /d "{outdir}"\r\n'
            f"cl /nologo /O1 {name}.c /Fe:{name}.exe\r\n",
            encoding="ascii",
        )
        result = subprocess.run([str(bat)], capture_output=True,
                                text=True, encoding="utf-8", errors="replace")
        bat.unlink(missing_ok=True)
        exe = outdir / f"{name}.exe"
        if not exe.exists():
            print(f"ERROR building {name}.exe:\n{result.stdout}\n{result.stderr}")
            return 1
        print(f"  built {exe}  ({len(raw)} cp850 bytes, 0x81 present)")

    print(f"\nShims in: {outdir}")
    print("Prepend that to PATH, then run the installer against a torch-less venv.")
    print("NOTE: rename nvcc.exe out of the way to reproduce the REPORTED crash "
          "path (nmake, inside install()'s try block).")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", type=Path,
                    default=Path(os.environ.get("TEMP", ".")) / "de_shim")
    args = ap.parse_args()
    return build(args.out)


if __name__ == "__main__":
    sys.exit(main())
