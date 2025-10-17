# Plant Capacity Dashboard

A Streamlit application to explore global iron & steel plant data with interactive maps, KPIs, and rich filtering. The app supports plant‑level and company‑level views, plus a data table and CSV export.

## Features

- **Sidebar filters**: filter by Company, Region, Country, and **Capacity range**.
- **KPIs**: total plants, total capacity (ttpa), number of companies & regions.
- **Interactive maps (tabs)**  
  - **Points**: plant markers; optional size by a chosen capacity metric.  
  - **Heatmap**: capacity-weighted density map (uses *Total capacity (ttpa)* when available).  
  - **Companies**: bubbles at capacity‑weighted company centroids; sized by total capacity and colored by an environmental score (when inputs available).
- **Coordinates parsing**: derives `lat`/`lon` automatically from `lat/lon`, `Latitude/Longitude`, or a single `Coordinates` field like `"36.75, 6.24"`.
- **Capacity handling**: computes **Total capacity (ttpa)** by summing present capacity columns if not already provided.
- **Data table** of filtered rows and **Download filtered CSV** button.
- **Notebook integration (optional)**: render a Jupyter notebook in a “Notebook” tab if you enable it (requires `nbformat`).

##  Repo Contents

```
.
├── dashboard.py                 # Streamlit app
├── cleaned_plant.csv            # Example plant-level dataset (required at runtime or upload via UI)
├── company_level_agg.csv        # Optional company-level aggregates (centroids, totals)
├── lab_1_Arthur.ipynb           # Notebook with analyses (optional)
└── Plant-level-data-Global-Iron-and-Steel-Tracker-September-2025-V1.xlsx  # Source data (optional)
```

> You can also upload CSV files directly from the app’s sidebar if you prefer not to place them in the repo root.

##  Installation

> **Python**: 3.9+ recommended

```bash
# (Optional) create and activate a virtual environment
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

# Install dependencies
pip install streamlit pandas numpy plotly nbformat
```

> `nbformat` is only needed if you plan to render a Jupyter notebook inside the app.

##  Run

From the repository root:

```bash
streamlit run dashboard.py
```

By default the app looks for `cleaned_plant.csv` (and optional `company_level_agg.csv`) in the project root. If they’re not found, use the **Upload CSV** option in the sidebar.

##  Data Expectations

At minimum, the plant‑level CSV should include:

- **Location**: either `lat` and `lon`; or `Latitude` and `Longitude`; or a **single** `Coordinates` column with `"lat, lon"`.
- **Categoricals** (recommended): `Owner` or `Parent GEM ID`, `Region`, `Country/Area`.
- **Capacity columns** (any subset):  
  `Nominal crude steel capacity (ttpa)`, `Ferronickel capacity (ttpa)`, `Pelletizing plant capacity (ttpa)`, `Sinter plant capacity (ttpa)`, `Coking plant capacity (ttpa)`, or a generic `Capacity`.
  - The app will coerce to numeric and compute **Total capacity (ttpa)** as the row‑wise sum of available capacity columns.

**Optional company aggregates (`company_level_agg.csv`)**: provide columns like `Owner`, `centroid_lat`, `centroid_lon`, and `total_capacity`. If not provided, the app computes aggregates and a simple environmental score from plant‑level fields (e.g., `ISO 14001`, `ISO 50001`, `ResponsibleSteel Certification`) when available.

## 🗺 Maps & Styling

- Maps use **OpenStreetMap** styles via Plotly, so **no Mapbox token** is required.
- The heatmap uses a default radius and centers on the median of visible points; tune these in `dashboard.py` if desired.

##  Notebook Tab (Optional)

If you enable the Notebook tab in the code, the app can display a `.ipynb` file (default path `lab_1_Arthur.ipynb`) including markdown, code (collapsed), and outputs (text/HTML/Plotly). Install `nbformat` to enable:

```bash
pip install nbformat
```

> You can upload a notebook from the UI as well.
