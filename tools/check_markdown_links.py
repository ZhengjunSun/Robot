from __future__ import annotations

import argparse
import re
from pathlib import Path
from urllib.parse import unquote


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LINK_RE = re.compile(r"!?\[[^\]]*\]\((?P<target><[^>]+>|[^)\s]+)")
EXCLUDED_PARTS = {
    ".git",
    "3d_modeling/outputs",
    "没有用的文件",
}


def is_excluded(path: Path) -> bool:
    relative = path.relative_to(PROJECT_ROOT).as_posix()
    return (
        any(relative == item or relative.startswith(f"{item}/") for item in EXCLUDED_PARTS)
        or any(part.startswith("build_") for part in path.parts)
    )


def local_target(source: Path, raw_target: str) -> Path | None:
    target = raw_target.strip("<>")
    if (
        not target
        or target.startswith("#")
        or target.startswith("/")
        or re.match(r"^[A-Za-z][A-Za-z0-9+.-]*:", target)
    ):
        return None
    target = unquote(target.split("#", 1)[0].split("?", 1)[0])
    if not target:
        return None
    return (source.parent / target).resolve()


def check() -> list[tuple[Path, int, str]]:
    missing: list[tuple[Path, int, str]] = []
    for source in PROJECT_ROOT.rglob("*.md"):
        if is_excluded(source):
            continue
        text = source.read_text(encoding="utf-8")
        for line_number, line in enumerate(text.splitlines(), start=1):
            for match in LINK_RE.finditer(line):
                target = local_target(source, match.group("target"))
                if target is not None and not target.exists():
                    missing.append((source, line_number, match.group("target")))
    return missing


def main() -> int:
    parser = argparse.ArgumentParser(description="Check local Markdown links in the active repository.")
    parser.parse_args()
    missing = check()
    for source, line_number, target in missing:
        relative = source.relative_to(PROJECT_ROOT)
        print(f"{relative}:{line_number}: missing local target {target}")
    print(f"markdown_link_check: {len(missing)} missing")
    return 1 if missing else 0


if __name__ == "__main__":
    raise SystemExit(main())
