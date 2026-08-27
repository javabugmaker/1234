from __future__ import annotations

import re
import sys
from pathlib import Path


LOCAL_REFERENCE = re.compile(r'''(?:src|href)=["']([^"']+)["']''')


def main() -> int:
    docs = Path(sys.argv[1] if len(sys.argv) > 1 else "docs")
    docs = docs.resolve()
    required = [
        docs / "index.html",
        docs / "daily.html",
        docs / "weekly.html",
        docs / "publish.html",
    ]
    failures = []
    for path in required:
        if not path.exists():
            failures.append(f"missing {path}")
            continue
        content = path.read_text(encoding="utf-8")
        if len(content) < 300 or "</html>" not in content.lower():
            failures.append(f"invalid {path}")
        if "http://" in content or "https://cdn" in content:
            failures.append(f"external CDN/reference in core page {path}")
        for reference in LOCAL_REFERENCE.findall(content):
            if reference.startswith(("http://", "https://", "#", "mailto:")):
                continue
            clean = reference.split("?", 1)[0].split("#", 1)[0]
            if not clean.startswith("assets/"):
                continue
            target = (docs / clean).resolve()
            try:
                target.relative_to(docs)
            except ValueError:
                failures.append(f"unsafe asset path in {path}: {reference}")
                continue
            if not target.exists():
                failures.append(f"missing asset for {path}: {clean}")
    if failures:
        print("\n".join(failures), file=sys.stderr)
        return 1
    print("Pages validation PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
