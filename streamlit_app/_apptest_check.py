"""Headless render check for every page using Streamlit's AppTest harness."""
import os

from streamlit.testing.v1 import AppTest

ok = True
for name in ["dashboard", "insights", "scorer", "intelligence", "business"]:
    os.environ["PAGE"] = name
    at = AppTest.from_file("_run_page.py", default_timeout=90)
    at.run()
    if at.exception:
        ok = False
        print(f"[FAIL] {name}")
        for e in at.exception:
            print("    ", repr(e.value)[:300])
    else:
        n_charts = len(at.get("plotly_chart")) if at.get("plotly_chart") else 0
        print(f"[OK]   {name}  (markdown={len(at.markdown)}, plotly={n_charts})")

print("ALL PASS" if ok else "SOME FAILED")
