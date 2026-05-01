"""Download data/invoices/ from the upstream galatiq-case-invoices repo."""
from __future__ import annotations

import sys
import urllib.error
import urllib.request
from pathlib import Path

FILES = [
    "invoice_1001.txt",
    "invoice_1002.txt",
    "invoice_1003.txt",
    "invoice_1004.json",
    "invoice_1004_revised.json",
    "invoice_1005.json",
    "invoice_1006.csv",
    "invoice_1007.csv",
    "invoice_1008.txt",
    "invoice_1009.json",
    "invoice_1010.txt",
    "invoice_1011.pdf",
    "invoice_1011.txt",
    "invoice_1012.pdf",
    "invoice_1012.txt",
    "invoice_1013.json",
    "invoice_1013.pdf",
    "invoice_1014.xml",
    "invoice_1015.csv",
    "invoice_1016.json",
]
BASE = "https://raw.githubusercontent.com/galatiq-ai/galatiq-case-invoices/main/data/invoices/"


def main() -> int:
    out_dir = Path(__file__).resolve().parents[1] / "data" / "invoices"
    out_dir.mkdir(parents=True, exist_ok=True)
    for name in FILES:
        dst = out_dir / name
        if dst.exists() and dst.stat().st_size > 0:
            print(f"  ok  {name}")
            continue
        url = BASE + name
        try:
            urllib.request.urlretrieve(url, dst)
            print(f"  got {name}")
        except urllib.error.URLError as e:
            print(f"  err {name}: {e}", file=sys.stderr)
            return 1
    print(f"\n{len(FILES)} invoices in {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
