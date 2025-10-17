# Authors : Millan Das, Arthur de Leusse & Alexis Vannson
#  Plant Capacity Dashboard (Streamlit + Plotly)

Interactive dashboard to explore **global iron & steel plant data**.  
Filter by company, region, country and capacity; visualize plants as **points**, a **density heatmap**, or **company aggregates**; view KPIs and export filtered data. Optionally render a companion **Jupyter notebook** inside the app.

---

##  Features

- **Rich filtering** (sidebar): Company, Region, Country, **Capacity metric & range**.
- **KPIs**: total plants, total capacity (ttpa), number of companies, number of regions.
- **Interactive maps** (tabs):
  - **Points** — markers per plant; optional size by capacity; color by Region/Company/Country.
  - **Heatmap** — capacity‑weighted density (uses *Total capacity (ttpa)* when available).
  - **Companies** — bubbles at capacity‑weighted **company centroids**; size = total capacity; color = simple environmental score (mean of available ISO flags).
- **Automatic coordinates parsing**: uses existing `lat/lon`, or `Latitude/Longitude`, or parses a `"Coordinates"` field like `"36.75, 6.24"`.
- **Capacity handling**: computes **Total capacity (ttpa)** by summing present capacity columns if not provided.
- **Data table** for the filtered subset + **CSV export**.
- **Notebook tab (optional)**: render a `.ipynb` (markdown, code cells collapsed, outputs incl. Plotly).

---

##  Repository Layout

```
.
├── dashboard.py                 # Streamlit app
├── requirements.txt             # Project dependencies
├── cleaned_plant.csv            # Example plant-level dataset (optional at runtime)
├── company_level_agg.csv        # Optional: company aggregates (centroids & totals)
├── lab_1.ipynb           # Optional: analysis notebook to render in-app
└── Plant-level-data-Global-Iron-and-Steel-Tracker-September-2025-V1.xlsx  # Optional source data
```

> You can load data via the app’s sidebar (Upload CSV) if you don’t keep CSVs in the repo root.

---

##  Quickstart

**Python 3.9+** recommended.

```bash
# 1) (Optional) create and activate a virtual env
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

# 2) Install deps
pip install -r requirements.txt

# 3) Run the app
streamlit run dashboard.py
```

By default the app looks for `cleaned_plant.csv` (and optionally `company_level_agg.csv`) in the project root. If not found, choose **Upload CSV** in the sidebar.

> Maps use **OpenStreetMap** via Plotly — **no Mapbox token** required.

---

##  Data Expectations

**Minimum** for plant‑level CSV (“cleaned_plant.csv”):

- **Location**: either `lat` & `lon`, or `Latitude` & `Longitude`, **or** a single `Coordinates` column with `"lat, lon"`.
- **Categoricals** (recommended): `Owner` or `Parent GEM ID`, `Region`, `Country/Area`.
- **Capacity columns** (any subset):
  - `Nominal crude steel capacity (ttpa)`
  - `Ferronickel capacity (ttpa)`
  - `Pelletizing plant capacity (ttpa)`
  - `Sinter plant capacity (ttpa)`
  - `Coking plant capacity (ttpa)`
  - or a generic `Capacity`
- The app will coerce these to numeric and compute **`Total capacity (ttpa)`** = row‑wise sum of present capacity columns.

**Optional company aggregates** (“company_level_agg.csv”):
- `Owner`, `centroid_lat`, `centroid_lon`, `total_capacity` (and optionally `avg_iso14001`, `avg_iso50001`, `avg_responsible`).  
- If not provided, the app **computes aggregates on the fly** from plant‑level data, including capacity‑weighted centroids.

---

##  How to Use

1. **Data Source (sidebar)**: choose *Default path* or *Upload CSV* for plants (and optionally aggregates).
2. **Filters (sidebar)**: select Company/Region/Country; pick **Capacity metric** and set a **range**.
3. **KPIs**: sanity‑check counts and totals for the current filter.
4. **Maps**:
   - **Points**: explore plant locations; hover for details; size by selected capacity metric.
   - **Heatmap**: visualize density; weighted by Total capacity (if available).
   - **Companies**: see firm footprints; size by total capacity; color by environmental score.
5. **Data Table**: preview the filtered rows and **Download filtered CSV**.

---

##  Notebook Tab (Optional)

The **Notebook** tab renders a `.ipynb` (default path: `lab_1_Arthur.ipynb`) including markdown, code cells (collapsed), and outputs (text/HTML/Plotly).  
Install `nbformat` (already in `requirements.txt`). You can also upload a notebook via the tab.

> Want to **execute** the notebook inside the app? Add `nbclient` and we can wire a “Run notebook” button.

---
