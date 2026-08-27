from __future__ import annotations

import sys
from pathlib import Path


def main() -> int:
    docs = Path(sys.argv[1] if len(sys.argv) > 1 else "docs")
    required = [docs / "index.html", docs / "daily.html", docs / "weekly.html"]
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
    if failures:
        print("\n".join(failures), file=sys.stderr)
        return 1
    print("Pages validation PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
