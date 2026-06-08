"""Tiny runner so AppTest.from_file can exercise one page with real imports."""
import os
import streamlit as st  # noqa: F401

from theme import inject_css, register_plotly_template
from views import dashboard, insights, scorer, intelligence, business

inject_css()
register_plotly_template()

_PAGES = {
    "dashboard": dashboard.render,
    "insights": insights.render,
    "scorer": scorer.render,
    "intelligence": intelligence.render,
    "business": business.render,
}
_PAGES[os.environ.get("PAGE", "dashboard")]()
