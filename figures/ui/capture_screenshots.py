"""Capture raw panel screenshots for Extended Data Fig. 3.

Requires the FastAPI backend on :8000 and the vinext dev server on :5173.
prefers-reduced-motion makes the pathway reveal render all nodes at once.
"""
from playwright.sync_api import sync_playwright

URL = "http://localhost:5173/?run=a6b1629ace8346f4"
OUT = "figures/ui"
# Illustrative English input text shown in the entry box (the loaded run is
# the same event: tirzepatide injection).
INPUT_TEXT = "I received a tirzepatide injection (Mounjaro)."

with sync_playwright() as p:
    b = p.chromium.launch()
    pg = b.new_page(viewport={"width": 1440, "height": 940}, device_scale_factor=2,
                    reduced_motion="reduce")
    pg.goto(URL, wait_until="networkidle")
    pg.wait_for_selector(".pathway-node", timeout=60000)
    pg.wait_for_timeout(2500)
    pg.add_style_tag(content=".recharts-tooltip-wrapper{display:none!important}")

    # Panel a: header + event input (English entry) + post-run review line
    pg.fill("#event-input", INPUT_TEXT)
    pg.evaluate("window.scrollTo(0,0)")
    pg.wait_for_timeout(400)
    pg.screenshot(path=f"{OUT}/panel_a_input.png")

    # Panel b (summary variant): whole-body response cards + pathway heading
    pg.locator(".flow-heading").scroll_into_view_if_needed()
    pg.evaluate("window.scrollBy(0,-60)")
    pg.wait_for_timeout(600)
    pg.screenshot(path=f"{OUT}/panel_b_pathway.png")

    # Panel b (graph variant used in the composite): input -> exposure ->
    # branched propagation; Observable vs MODEL EXTENSION badges visible
    btn = pg.locator('button.pathway-node[aria-label^="Lipolysis"]')
    btn.scroll_into_view_if_needed()
    pg.evaluate("window.scrollBy(0,-620)")
    pg.wait_for_timeout(600)
    pg.screenshot(path=f"{OUT}/panel_b_graph.png")

    # Panel c: Blood glucose detail (native published-model state, real
    # units, no LLM wording anywhere in the sheet)
    gb = pg.locator('button.pathway-node[aria-label^="Blood glucose"]')
    gb.scroll_into_view_if_needed()
    pg.wait_for_timeout(400)
    gb.click()
    pg.wait_for_selector(".research-detail", timeout=15000)
    pg.wait_for_timeout(1200)
    pg.screenshot(path=f"{OUT}/panel_c_detail.png")
    pg.locator(".research-detail").screenshot(path=f"{OUT}/panel_c_glucose_sheet.png")
    b.close()
print("done")
