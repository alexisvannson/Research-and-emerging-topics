# dashboard.py
# Streamlit dashboard for plant data with Plotly Express interactive maps (points, heatmap, companies)

import os
from pathlib import Path
from typing import Tuple, Optional, List

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

# ---------------------------
# Page configuration
# ---------------------------
st.set_page_config(
    page_title="Plant Capacity Dashboard",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------
# Helpers
# ---------------------------
DEFAULT_DATA_PATH = Path("cleaned_plant.csv")
COMPANY_AGG_PATH = Path("company_level_agg.csv")

@st.cache_data(show_spinner=False)
def load_csv(csv_source) -> pd.DataFrame:
    """Load a CSV from path or uploaded file-like object."""
    try:
        if isinstance(csv_source, (str, os.PathLike, Path)) and Path(csv_source).exists():
            return pd.read_csv(csv_source)
        # Fallback: uploaded file-like via Streamlit
        return pd.read_csv(csv_source)
    except Exception as e:
        st.warning(f"Could not load CSV: {e}")
        return pd.DataFrame()

def numeric_coerce(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")

def ensure_lat_lon(df: pd.DataFrame) -> pd.DataFrame:
    """Create df['lat'] and df['lon'] from common column patterns or 'Coordinates' string."""
    if df is None or df.empty:
        return df
    df = df.copy()
    lower_map = {c.lower().strip(): c for c in df.columns}

    def find_col(*cands):
        for c in cands:
            if c in lower_map:
                return lower_map[c]
        return None

    lat_src = find_col("lat", "latitude", "y")
    lon_src = find_col("lon", "longitude", "lng", "long", "x")
    if lat_src is not None and lon_src is not None:
        df["lat"] = numeric_coerce(df[lat_src])
        df["lon"] = numeric_coerce(df[lon_src])
        return df

    coord_src = find_col("coordinates", "coord", "geocoordinates", "geo")
    if coord_src is not None:
        def parse_coord(val) -> Tuple[float, float]:
            if pd.isna(val):
                return (np.nan, np.nan)
            s = str(val)
            s = (s.replace("[", "").replace("]", "")
                   .replace("(", "").replace(")", "")
                   .replace(";", ","))
            parts = [p.strip() for p in s.split(",") if p.strip()]
            if len(parts) >= 2:
                try:
                    return (float(parts[0]), float(parts[1]))
                except Exception:
                    return (np.nan, np.nan)
            return (np.nan, np.nan)

        latlon = df[coord_src].apply(parse_coord)
        df["lat"] = latlon.map(lambda t: t[0])
        df["lon"] = latlon.map(lambda t: t[1])
        return df

    return df

def pick_company_column(df: pd.DataFrame) -> Optional[str]:
    for col in ["Parent GEM ID", "Owner", "Company", "Parent", "Owner GEM ID"]:
        if col in df.columns:
            return col
    return None

def candidate_capacity_columns(df: pd.DataFrame) -> List[str]:
    candidates = [
        "Nominal crude steel capacity (ttpa)",
        "Ferronickel capacity (ttpa)",
        "Pelletizing plant capacity (ttpa)",
        "Sinter plant capacity (ttpa)",
        "Coking plant capacity (ttpa)",
        "Capacity",
        "Total capacity (ttpa)",  # allow precomputed total
    ]
    return [c for c in candidates if c in df.columns]

def compute_total_capacity(df: pd.DataFrame) -> pd.DataFrame:
    """Add 'Total capacity (ttpa)' as the sum of known capacity columns (if not present)."""
    total_col = "Total capacity (ttpa)"
    if total_col in df.columns:
        df[total_col] = numeric_coerce(df[total_col])
        return df
    cap_cols = [
        "Ferronickel capacity (ttpa)",
        "Sinter plant capacity (ttpa)",
        "Coking plant capacity (ttpa)",
        "Pelletizing plant capacity (ttpa)",
        "Nominal crude steel capacity (ttpa)",
        "Capacity",
    ]
    present = [c for c in cap_cols if c in df.columns]
    if present:
        for c in present:
            df[c] = numeric_coerce(df[c])
        df[total_col] = df[present].sum(axis=1, skipna=True)
    return df

def to_boolish(series: pd.Series) -> pd.Series:
    """Map typical yes/true strings to 1.0 else 0.0."""
    if series is None:
        return pd.Series(dtype=float)
    s = series.astype(str).str.lower().str.strip()
    trueish = {"yes", "y", "true", "1", "certified", "present"}
    return s.apply(lambda x: 1.0 if x in trueish else 0.0)

def compute_company_agg(df: pd.DataFrame, company_col: str, cap_col: str) -> pd.DataFrame:
    """Aggregate to company-level metrics with centroid and simple env score."""
    work = df.dropna(subset=["lat", "lon"]).copy()
    work[cap_col] = numeric_coerce(work[cap_col]).fillna(0)

    # Weight by capacity when available; otherwise unweighted mean
    def wmean(x, w):
        w = np.asarray(w, dtype=float)
        x = np.asarray(x, dtype=float)
        if np.all(np.isnan(w)) or np.nansum(w) == 0:
            return float(np.nanmean(x))
        return float(np.nansum(x * w) / (np.nansum(w) + 1e-12))

    # Environmental flags to numeric if present
    if "ISO 14001" in work.columns:
        work["iso14001_num"] = to_boolish(work["ISO 14001"])
    if "ISO 50001" in work.columns:
        work["iso50001_num"] = to_boolish(work["ISO 50001"])
    if "ResponsibleSteel Certification" in work.columns:
        work["responsible_num"] = to_boolish(work["ResponsibleSteel Certification"])

    grp = work.groupby(company_col, dropna=True)

    agg = pd.DataFrame({
        company_col: grp.size().index,
        "total_capacity": grp[cap_col].sum().values,
        "num_plants": grp.size().values,
        "num_countries": grp["Country/Area"].nunique().values if "Country/Area" in work.columns else np.nan,
        "centroid_lat": grp.apply(lambda g: wmean(g["lat"], g[cap_col])),
        "centroid_lon": grp.apply(lambda g: wmean(g["lon"], g[cap_col])),
    })

    # Environmental averages
    if "iso14001_num" in work.columns:
        agg["avg_iso14001"] = grp["iso14001_num"].mean().values
    if "iso50001_num" in work.columns:
        agg["avg_iso50001"] = grp["iso50001_num"].mean().values
    if "responsible_num" in work.columns:
        agg["avg_responsible"] = grp["responsible_num"].mean().values

    # Env score = mean of available env columns
    env_parts = [c for c in ["avg_iso14001", "avg_iso50001", "avg_responsible"] if c in agg.columns]
    if env_parts:
        agg["env_score"] = agg[env_parts].mean(axis=1)
    else:
        agg["env_score"] = np.nan

    return agg

# ---------------------------
# Title & Description
# ---------------------------
st.title("🌍 Plant Capacity Dashboard")
st.markdown(
    "Explore plant-level data by company, region, and capacity. "
    "Use the sidebar to filter and the map to explore locations."
)

# ---------------------------
# Data loading section
# ---------------------------
with st.sidebar:
    st.header("🔧 Data Source")
    data_choice = st.radio(
        "Choose how to load data:",
        options=["Default path", "Upload CSV"],
        index=0,
    )

    if data_choice == "Default path":
        df = load_csv(DEFAULT_DATA_PATH)
        if df.empty:
            st.info("Default data not found. Please upload a CSV instead.")
            uploaded = st.file_uploader("Upload plant CSV", type=["csv"])
            if uploaded:
                df = load_csv(uploaded)
    else:
        uploaded = st.file_uploader("Upload plant CSV", type=["csv"])
        df = load_csv(uploaded) if uploaded else pd.DataFrame()

# Optional company aggregation file (used if present; otherwise computed on the fly)
company_agg = pd.DataFrame()
if COMPANY_AGG_PATH.exists():
    company_agg = load_csv(COMPANY_AGG_PATH)

if df.empty:
    st.stop()

# Clean columns (trim spaces)
df.columns = df.columns.str.strip()

# Ensure lat/lon and compute total capacity
df = ensure_lat_lon(df)
df = compute_total_capacity(df)

# Capacity column selection
cap_cols = candidate_capacity_columns(df)
if cap_cols:
    default_idx = cap_cols.index("Total capacity (ttpa)") if "Total capacity (ttpa)" in cap_cols else 0
else:
    st.warning("No capacity columns detected. You can still explore the map and table.")
    cap_cols = []
    default_idx = 0

# ---------------------------
# Sidebar Filters
# ---------------------------
st.sidebar.header("🎛️ Filters")

# Debug helper (optional): show columns
with st.sidebar.expander("🔎 Debug: Show columns"):
    st.write(list(df.columns))

company_col = pick_company_column(df)
region_col = "Region" if "Region" in df.columns else None
country_col = "Country/Area" if "Country/Area" in df.columns else None

# Company filter
if company_col:
    all_companies = sorted([c for c in df[company_col].dropna().astype(str).unique()])
    selected_companies = st.sidebar.multiselect("Company", options=all_companies, default=[])
else:
    selected_companies = []
    st.sidebar.caption("ℹ️ No company column found. Skipping company filter.")

# Region / Country filters
if region_col:
    all_regions = sorted([r for r in df[region_col].dropna().astype(str).unique()])
    selected_regions = st.sidebar.multiselect("Region", options=all_regions, default=[])
else:
    selected_regions = []

if country_col:
    all_countries = sorted([c for c in df[country_col].dropna().astype(str).unique()])
    selected_countries = st.sidebar.multiselect("Country/Area", options=all_countries, default=[])
else:
    selected_countries = []

# Capacity metric selection & range slider
if cap_cols:
    cap_choice = st.sidebar.selectbox("Capacity metric", cap_cols, index=default_idx)
    df[cap_choice] = numeric_coerce(df[cap_choice])
    cap_min = float(np.nanmin(df[cap_choice].values)) if df[cap_choice].notna().any() else 0.0
    cap_max = float(np.nanmax(df[cap_choice].values)) if df[cap_choice].notna().any() else 0.0
    cap_low, cap_high = st.sidebar.slider(
        "Capacity range (ttpa)",
        min_value=float(np.floor(cap_min)),
        max_value=float(np.ceil(cap_max)) if cap_max > cap_min else float(np.ceil(cap_min + 1)),
        value=(float(np.floor(cap_min)), float(np.ceil(cap_max))) if cap_max > cap_min else (0.0, float(np.ceil(cap_max))),
        step=1.0,
    )
    size_max = st.sidebar.slider("Marker max size", 5, 40, 18)
else:
    cap_choice = None
    cap_low, cap_high = None, None
    size_max = 18

# Color by selection
color_options = []
for c in [region_col, company_col, country_col]:
    if c is not None:
        color_options.append(c)
color_by = st.sidebar.selectbox("Color by", options=["(none)"] + color_options, index=0)
if color_by == "(none)":
    color_by = None

# ---------------------------
# Apply filters
# ---------------------------
filtered = df.copy()

if company_col and selected_companies:
    filtered = filtered[filtered[company_col].astype(str).isin(selected_companies)]

if region_col and selected_regions:
    filtered = filtered[filtered[region_col].astype(str).isin(selected_regions)]

if country_col and selected_countries:
    filtered = filtered[filtered[country_col].astype(str).isin(selected_countries)]

if cap_choice is not None and cap_low is not None:
    filtered = filtered[(filtered[cap_choice] >= cap_low) & (filtered[cap_choice] <= cap_high)]

# Guard: ensure lat/lon exist
if "lat" not in filtered.columns or "lon" not in filtered.columns:
    st.error("No latitude/longitude columns found. "
             "Make sure your CSV has either 'lat'/'lon', 'Latitude'/'Longitude', "
             "or a 'Coordinates' column like 'lat, lon'.")
    st.stop()

# Drop rows with missing coordinates for the maps
filtered_geo = filtered.dropna(subset=["lat", "lon"])

# ---------------------------
# KPI Metrics
# ---------------------------
st.subheader("📈 KPIs")

kpi_cols = st.columns(4)
with kpi_cols[0]:
    st.metric("Total plants", int(len(filtered)))
with kpi_cols[1]:
    if cap_choice:
        total_cap = float(np.nansum(filtered[cap_choice].values))
        st.metric("Total capacity (ttpa)", f"{total_cap:,.0f}")
    else:
        st.metric("Total capacity (ttpa)", "N/A")
with kpi_cols[2]:
    if company_col:
        st.metric("Companies", int(filtered[company_col].nunique()))
    else:
        st.metric("Companies", "N/A")
with kpi_cols[3]:
    if region_col:
        st.metric("Regions", int(filtered[region_col].nunique()))
    else:
        st.metric("Regions", "N/A")

# ---------------------------
# Maps (Tabs): Points | Heatmap | Companies
# ---------------------------
st.subheader("🗺️ Interactive Maps")
tab_points, tab_heatmap, tab_companies = st.tabs(["Points", "Heatmap", "Companies"])

with tab_points:
    if not filtered_geo.empty:
        size_arg = None
        if cap_choice:
            size_series = numeric_coerce(filtered_geo[cap_choice]).fillna(0)
            if (size_series > 0).any():
                size_arg = cap_choice
            else:
                st.caption("⚠️ Capacity column has no positive values; marker size will be constant.")

        fig = px.scatter_mapbox(
            filtered_geo,
            lat="lat",
            lon="lon",
            color=color_by,
            size=size_arg,
            size_max=size_max,
            hover_name=("Plant name (English)" if "Plant name (English)" in filtered_geo.columns else None),
            hover_data=[c for c in ["Owner", "Parent GEM ID", "Region", "Country/Area", cap_choice] if c in filtered_geo.columns],
            zoom=1,
            height=640,
        )
        fig.update_layout(
            mapbox_style="open-street-map",
            margin=dict(l=0, r=0, t=0, b=0),
            legend=dict(orientation="h", yanchor="bottom", y=0.01, xanchor="left", x=0.01),
        )
        fig.update_layout(mapbox=dict(center=dict(lat=float(filtered_geo["lat"].median()),
                                                 lon=float(filtered_geo["lon"].median()))))
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No geolocated rows to display after filtering.")

with tab_heatmap:
    if not filtered_geo.empty:
        # Use total capacity as default weight if available, else cap_choice, else unweighted
        z_col = "Total capacity (ttpa)" if "Total capacity (ttpa)" in filtered_geo.columns else cap_choice
        fig_hm = px.density_mapbox(
            filtered_geo,
            lat="lat",
            lon="lon",
            z=z_col if z_col in filtered_geo.columns else None,
            radius=15,
            center=dict(lat=float(filtered_geo["lat"].median()), lon=float(filtered_geo["lon"].median())),
            zoom=1,
            height=640,
            mapbox_style="open-street-map",
            title=None,
        )
        fig_hm.update_layout(margin=dict(l=0, r=0, t=0, b=0))
        st.plotly_chart(fig_hm, use_container_width=True)
    else:
        st.info("No data for heatmap after filtering.")

with tab_companies:
    if company_col and not filtered_geo.empty:
        # Prefer provided company_agg file if it looks valid
        if not company_agg.empty and all(c in company_agg.columns for c in ["centroid_lat", "centroid_lon", "total_capacity", "Owner"]):
            agg_df = company_agg.copy()
            owner_name_col = "Owner" if "Owner" in agg_df.columns else company_col
        else:
            # Compute on the fly from filtered plants
            owner_name_col = company_col
            agg_df = compute_company_agg(filtered_geo, company_col=company_col, cap_col=(cap_choice or "Total capacity (ttpa)"))

        # Ensure env_score exists
        if "env_score" not in agg_df.columns:
            env_parts = [c for c in ["avg_iso14001", "avg_iso50001", "avg_responsible"] if c in agg_df.columns]
            if env_parts:
                agg_df["env_score"] = agg_df[env_parts].mean(axis=1)
            else:
                agg_df["env_score"] = np.nan

        # Drop rows without centroids
        agg_geo = agg_df.dropna(subset=["centroid_lat", "centroid_lon"]).copy()
        if agg_geo.empty:
            st.info("No company centroids available. Check that latitude/longitude exist for plants.")
        else:
            # Build hover
            hover_data = {}
            for c in ["total_capacity", "num_countries", "avg_iso14001", "avg_iso50001", "avg_responsible", "env_score"]:
                if c in agg_geo.columns:
                    hover_data[c] = ":.2f" if c != "num_countries" else True

            fig_comp = px.scatter_geo(
                agg_geo,
                lat="centroid_lat",
                lon="centroid_lon",
                size="total_capacity" if "total_capacity" in agg_geo.columns else None,
                color="env_score" if "env_score" in agg_geo.columns else None,
                hover_name=owner_name_col,
                hover_data=hover_data if hover_data else None,
                projection="natural earth",
                title=None,
                color_continuous_scale="YlGn",
                height=640,
            )
            fig_comp.update_traces(marker=dict(line=dict(width=0.5, color="black")))
            fig_comp.update_layout(
                coloraxis_colorbar=dict(title="Env. Score"),
                margin=dict(l=0, r=0, t=0, b=0),
                geo=dict(showland=True, landcolor="lightgray"),
            )
            st.plotly_chart(fig_comp, use_container_width=True)
    else:
        st.info("Company column not found or no geolocated plants to aggregate.")

# ---------------------------
# Data Table
# ---------------------------
st.subheader("📋 Data Table")
st.dataframe(filtered, use_container_width=True, height=420)

# Optional: CSV download of filtered data
@st.cache_data(show_spinner=False)
def to_csv_bytes(df_in: pd.DataFrame) -> bytes:
    return df_in.to_csv(index=False).encode("utf-8")

st.download_button(
    "Download filtered CSV",
    data=to_csv_bytes(filtered),
    file_name="filtered_plants.csv",
    mime="text/csv",
)

# ---------------------------
# Footer
# ---------------------------
st.markdown("---")
st.caption("Data sources: cleaned_plant.csv (and optional company_level_agg.csv if present). "
           "Notes: Capacity values are treated as tTpa when available. "
           "Heatmap weights default to Total capacity.")

# ---------------------------
# Notes cell
# ---------------------------
with st.expander("🗒️ Notes / Observations"):
    st.write("""
- What works well?
    - Filters + three map views (points, heatmap, company aggregates).
- What could be improved?
    - Add date filters for plant lifecycle, and a top-companies bar chart.
- Performance with large datasets?
    - Consider Parquet/Arrow, server-side filtering, and sampling for map layers.
""")
