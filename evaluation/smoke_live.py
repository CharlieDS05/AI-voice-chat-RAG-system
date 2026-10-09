"""Smoke test for the deployed demo.

Sends every golden_demo question to the live /ask endpoint and checks the
outcome: answerable questions must be answered, cite a source, and retrieve an
expected page; unanswerable questions must be refused.

Usage:
    API_KEY=... python evaluation/smoke_live.py https://<host>
"""
import os
import sys

import httpx
import yaml

base = sys.argv[1].rstrip("/")
key = os.environ["API_KEY"]
with open("evaluation/golden_demo.yaml") as f:
    cases = yaml.safe_load(f)["cases"]

failures = 0
for c in cases:
    r = httpx.post(f"{base}/ask", json={"question": c["question"]},
                   headers={"X-API-Key": key}, timeout=60)
    r.raise_for_status()
    body = r.json()
    pages = {s["page"] for s in body["sources"]}
    if c["answerable"]:
        ok = (not body["refused"]
              and "[source:" in body["answer"]
              and bool(pages & set(c["expected_pages"])))
    else:
        ok = body["refused"]
    failures += not ok
    print(f"{'PASS' if ok else 'FAIL'}  {c['id']:<28} refused={body['refused']!s:<5} "
          f"top_score={body['top_score']:+6.2f}  pages={sorted(pages)}")

print(f"\n{len(cases) - failures}/{len(cases)} passed")
sys.exit(1 if failures else 0)
