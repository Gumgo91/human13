# Extended Data Fig. 3 — Human13 web interface and example state-tracing workflow

Composite: `extended_data_fig3.png` / `extended_data_fig3.pdf` (2880 px wide, 2x scale)
Separate panels: `panel_a.png`, `panel_b.png`, `panel_c.png`

## Case
- Saved run: `backend/data/runs/a6b1629ace8346f4.json` (run_id `a6b1629ace8346f4`)
- Event: tirzepatide injection (Mounjaro); 438 nodes, 104 changing systemic
  observables, 240 min horizon
- Loaded via `http://localhost:5173/?run=a6b1629ace8346f4`
  (workspace fetches `/api/runs/{id}?language=en`)
- The entry box text in panel a is shown as illustrative English input
  ("I received a tirzepatide injection (Mounjaro)."); the loaded result is the
  saved run for the same event.

## Panel mapping (implemented UI only, web/app/workspace.tsx)
- **a** Input and settings: `panel_a.png` — cropped from `panel_a_input.png`.
  Page header, `Event` textarea, `Trace response` button, post-run review line
  ("Added 3 model states. Updated 98 existing observables.").
- **b** State and result exploration: `panel_b.png` — cropped from
  `panel_b_graph.png`. Response pathway graph (`event-flow.tsx`): `INPUT` node,
  route-of-administration exposure, propagated native states carrying
  `Observable` badges, and generated states carrying `MODEL EXTENSION` badges.
- **c** Detailed result: `panel_c.png` = `panel_c_glucose_sheet.png`.
  Node detail sheet (`research-node-detail.tsx`) for `glucose` ("Blood glucose",
  method badge "Published model"): peak change −0.487 mg/dL, simulated time
  course 0–240 min in real units, related cells, and `Connections & feedback`
  (← liver production, ← GLUT4 utilization, → insulin secretion).

## Capture conditions
- Playwright Chromium headless shell; viewport 1440x940, deviceScaleFactor 2,
  `prefers-reduced-motion: reduce` (renders all pathway nodes immediately).
- Frontend: vinext dev server on :5173 (`npm run dev` in `web/`);
  backend: FastAPI on :8000 (`uvicorn backend.app:app`).
- `.recharts-tooltip-wrapper` hidden via injected style to suppress a transient
  hover tooltip; no other visual modification.
- Only features rendered by the routed page are shown. `body-atlas.tsx`,
  `physiology-graph.tsx`, `node-inspector.tsx`, `generated-model.tsx`,
  `study-report.tsx` exist in the tree but are not wired into the routed page
  and are not shown.
- Scripts: `capture_screenshots.py` (raw captures), `compose_figure.py`
  (panel crops + composite).
