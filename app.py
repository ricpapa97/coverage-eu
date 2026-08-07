"""
WWRR Coverage Tool — EU (France, Spain)
"""
import io
import os
import sys
import time
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.neighbors import BallTree

APP_DIR = os.path.dirname(os.path.abspath(__file__))
st.set_page_config(page_title="WWRR Coverage EU", page_icon="🗺️", layout="wide")

EARTH_RADIUS_KM = 6371.0
DEFAULT_RADII_M = [300, 500, 1000, 3000, 5000, 10000]
FINE_GRID_M = 50


@st.cache_data(show_spinner=False)
def load_grid_cached(path):
    """Load and cache grid parquet — survives app sleep."""
    return pd.read_parquet(path)
DEG_PER_M_LAT = 1.0 / 111_320.0
FINE_GRID_DEG_LAT = FINE_GRID_M * DEG_PER_M_LAT

COUNTRIES = {}
for code in ["ES", "FR"]:
    grid_path = os.path.join(APP_DIR, "data", code, "customer_grid.parquet")
    if os.path.exists(grid_path):
        COUNTRIES[code] = grid_path

COUNTRY_NAMES = {"ES": "🇪🇸 Spain", "FR": "🇫🇷 France"}
COUNTRY_CENTERS = {"ES": [40.4168, -3.7038], "FR": [46.6034, 1.8883]}


# ============================================================
# HELPER FUNCTIONS
# ============================================================
def parse_file(uploaded):
    name = uploaded.name.lower()
    if name.endswith(".csv"):
        return pd.read_csv(uploaded)
    else:
        return pd.read_excel(uploaded)


def detect_columns(df):
    cols = [c.lower().strip() for c in df.columns]
    result = {"lat": None, "lon": None, "carrier": None, "store_id": None}
    for i, c in enumerate(cols):
        real = df.columns[i]
        if c in ["lat", "latitude", "latitud"]:
            result["lat"] = real
        elif c in ["lon", "lng", "longitude", "longitud", "long"]:
            result["lon"] = real
        elif c in ["carrier", "program", "partner", "network", "brand"]:
            result["carrier"] = real
        elif c in ["store_id", "id", "name", "store_name", "store"]:
            result["store_id"] = real
    return result


def clean_stores(raw_df, lat_col, lon_col, carrier_col=None, id_col=None):
    df = raw_df.copy()
    df["lat"] = pd.to_numeric(df[lat_col].astype(str).str.replace(",", "."), errors="coerce")
    df["lon"] = pd.to_numeric(df[lon_col].astype(str).str.replace(",", "."), errors="coerce")
    df = df.dropna(subset=["lat", "lon"])
    df = df[(df["lat"].between(-90, 90)) & (df["lon"].between(-180, 180))]
    out = df[["lat", "lon"]].copy()
    if carrier_col and carrier_col in df.columns:
        out["carrier"] = df[carrier_col].fillna("Unknown")
    else:
        out["carrier"] = "All"
    if id_col and id_col in df.columns:
        out["store_id"] = df[id_col].astype(str)
    else:
        out["store_id"] = [f"store_{i}" for i in range(len(out))]
    return out.reset_index(drop=True)


def store_upload_widget(key, label="Upload store file"):
    uploaded = st.file_uploader(label, type=["csv", "xlsx", "xls"], key=key)
    if uploaded is None:
        return None
    raw_df = parse_file(uploaded)
    detected = detect_columns(raw_df)
    all_cols = ["(none)"] + raw_df.columns.tolist()
    with st.expander("Column mapping", expanded=detected["lat"] is None):
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            lat_col = st.selectbox("Latitude", all_cols,
                index=all_cols.index(detected["lat"]) if detected["lat"] else 0, key=f"{key}_lat")
        with c2:
            lon_col = st.selectbox("Longitude", all_cols,
                index=all_cols.index(detected["lon"]) if detected["lon"] else 0, key=f"{key}_lon")
        with c3:
            carrier_col = st.selectbox("Carrier", all_cols,
                index=all_cols.index(detected["carrier"]) if detected["carrier"] else 0, key=f"{key}_car")
        with c4:
            id_col = st.selectbox("ID", all_cols,
                index=all_cols.index(detected["store_id"]) if detected["store_id"] else 0, key=f"{key}_id")
    if lat_col == "(none)" or lon_col == "(none)":
        st.warning("Select at least Latitude and Longitude.")
        return None
    carrier_col = None if carrier_col == "(none)" else carrier_col
    id_col = None if id_col == "(none)" else id_col
    stores = clean_stores(raw_df, lat_col, lon_col, carrier_col, id_col)
    if len(stores) == 0:
        st.error("No valid stores found.")
        return None
    st.caption(f"✓ **{len(stores):,}** stores loaded")
    return stores


def load_default_stores(country):
    """Load default store file for a country. Returns DataFrame or None."""
    stores_path = os.path.join(APP_DIR, "data", country, f"stores_{country.lower()}.csv")
    if not os.path.exists(stores_path):
        return None
    df = pd.read_csv(stores_path)
    df = df.dropna(subset=["lat", "lon"])
    return df


def get_stores_with_carrier_selection(country, key_prefix):
    """Show store source selector with carrier checkboxes. Returns stores DataFrame."""
    default_stores = load_default_stores(country)
    if default_stores is not None:
        source = st.radio("Store source",
            ["Use current store network", "Upload a different file"],
            key=f"{key_prefix}_source", horizontal=True)
        if source == "Use current store network":
            carrier_col = "carrier" if "carrier" in default_stores.columns else None
            if carrier_col:
                carriers = sorted(default_stores[carrier_col].dropna().unique())
                st.markdown("**Select carriers:**")
                sel = []
                cols = st.columns(min(len(carriers), 4))
                for i, cr in enumerate(carriers):
                    with cols[i % len(cols)]:
                        if st.checkbox(cr.strip(), value=True, key=f"{key_prefix}_c_{i}"):
                            sel.append(cr)
                if sel:
                    default_stores = default_stores[default_stores[carrier_col].isin(sel)]
                else:
                    st.warning("Select at least one carrier.")
                    return None
            stores = default_stores[["lat", "lon"]].copy()
            if carrier_col:
                stores["carrier"] = default_stores[carrier_col]
            stores["store_id"] = default_stores.get("store_id",
                pd.Series([f"store_{i}" for i in range(len(stores))]))
            st.caption(f"✓ **{len(stores):,}** stores loaded")
            return stores
        else:
            return store_upload_widget(f"{key_prefix}_upload")
    else:
        return store_upload_widget(f"{key_prefix}_upload")


def compute_coverage(grid_df, stores_df, radii_m):
    """Compute coverage at multiple radii. Returns dict with DataFrames."""
    grid_coords = np.radians(grid_df[["lat", "lon"]].to_numpy(np.float64))
    store_coords = np.radians(stores_df[["lat", "lon"]].to_numpy(np.float64))
    weights = grid_df["weight"].to_numpy(np.int64)
    categories = grid_df["category"].to_numpy()
    total_w = int(weights.sum())
    store_tree = BallTree(store_coords, metric="haversine")
    dist_rad, _ = store_tree.query(grid_coords, k=1)
    dist_km = dist_rad[:, 0] * EARTH_RADIUS_KM

    country_rows = []
    for r_m in radii_m:
        cov = int(weights[dist_km <= r_m / 1000.0].sum())
        country_rows.append({"radius_m": r_m, "range": f"0-{r_m}m",
                             "coverage_pct": round(cov / total_w * 100, 2)})
    cat_rows = []
    for cat in ["Urban", "Suburban", "Rural"]:
        m = categories == cat
        cat_tot = int(weights[m].sum())
        if cat_tot == 0:
            continue
        for r_m in radii_m:
            cov = int(weights[m & (dist_km <= r_m / 1000.0)].sum())
            cat_rows.append({"category": cat, "radius_m": r_m,
                             "coverage_pct": round(cov / cat_tot * 100, 2)})
    return {
        "country": pd.DataFrame(country_rows),
        "by_category": pd.DataFrame(cat_rows),
        "dist_km": dist_km,
        "weights": weights,
        "categories": categories,
        "total_w": total_w,
    }


def make_excel(sheets_dict):
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as w:
        for name, df in sheets_dict.items():
            if df is not None and len(df) > 0:
                df.to_excel(w, sheet_name=name[:31], index=False)
    return buf.getvalue()


def build_hotspots(grid_df, dist_km, cell_radius_km, weights, n=100):
    """Find top N uncovered hotspot zones (aggregated to ~2km resolution)."""
    uncovered = dist_km > cell_radius_km
    if not uncovered.any():
        return pd.DataFrame()
    unc_df = grid_df.iloc[np.where(uncovered)[0]].copy()
    unc_df["weight"] = weights[uncovered]
    # Aggregate to ~2km grid
    unc_df["lat_r"] = (unc_df["lat"] * 50).round() / 50
    unc_df["lon_r"] = (unc_df["lon"] * 50).round() / 50
    hotspots = unc_df.groupby(["lat_r", "lon_r"]).agg(
        lat=("lat", "mean"), lon=("lon", "mean"), customers=("weight", "sum")
    ).reset_index(drop=True)
    hotspots = hotspots.nlargest(n, "customers").reset_index(drop=True)
    return hotspots


def show_map_coverage(stores, grid_df, dist_km, cell_radius_km, weights, country):
    """Map for Coverage model: stores (blue) + uncovered hotspots (red)."""
    hotspots = build_hotspots(grid_df, dist_km, cell_radius_km, weights)
    fig = go.Figure()
    fig.add_trace(go.Scattermapbox(
        lat=stores["lat"], lon=stores["lon"],
        mode="markers", marker=dict(size=4, color="#1f77b4"),
        name="Stores", hoverinfo="skip",
    ))
    if not hotspots.empty:
        fig.add_trace(go.Scattermapbox(
            lat=hotspots["lat"], lon=hotspots["lon"],
            mode="markers",
            marker=dict(size=hotspots["customers"] / hotspots["customers"].max() * 20 + 5,
                        color="red", opacity=0.6),
            name="Uncovered hotspots",
            text=[f"{int(c):,} customers" for c in hotspots["customers"]],
            hoverinfo="text",
        ))
    center = COUNTRY_CENTERS.get(country, [46.5, 2.0])
    fig.update_layout(
        mapbox=dict(style="open-street-map", center=dict(lat=center[0], lon=center[1]), zoom=5),
        margin=dict(l=0, r=0, t=0, b=0), height=500, showlegend=True,
        legend=dict(x=0, y=1, bgcolor="rgba(255,255,255,0.8)"),
    )
    st.plotly_chart(fig, use_container_width=True)


def show_map_rationalize(stores, essential_stores, country):
    """Map for Rationalize: green = essential, red = removable."""
    n_stores = len(stores)
    colors = ["green" if i in essential_stores else "red" for i in range(n_stores)]
    labels = ["Essential" if i in essential_stores else "Removable" for i in range(n_stores)]
    fig = go.Figure()
    # Essential
    ess_mask = [i in essential_stores for i in range(n_stores)]
    rem_mask = [i not in essential_stores for i in range(n_stores)]
    ess_df = stores.iloc[ess_mask]
    rem_df = stores.iloc[rem_mask]
    fig.add_trace(go.Scattermapbox(
        lat=ess_df["lat"], lon=ess_df["lon"],
        mode="markers", marker=dict(size=4, color="green"),
        name=f"Essential ({len(ess_df):,})", hoverinfo="skip",
    ))
    fig.add_trace(go.Scattermapbox(
        lat=rem_df["lat"], lon=rem_df["lon"],
        mode="markers", marker=dict(size=5, color="red", opacity=0.7),
        name=f"Removable ({len(rem_df):,})", hoverinfo="skip",
    ))
    center = COUNTRY_CENTERS.get(country, [46.5, 2.0])
    fig.update_layout(
        mapbox=dict(style="open-street-map", center=dict(lat=center[0], lon=center[1]), zoom=5),
        margin=dict(l=0, r=0, t=0, b=0), height=500, showlegend=True,
        legend=dict(x=0, y=1, bgcolor="rgba(255,255,255,0.8)"),
    )
    st.plotly_chart(fig, use_container_width=True)


def show_map_candidates(current, candidates, country):
    """Map for Rank Candidates: current (grey) + candidates (green)."""
    fig = go.Figure()
    fig.add_trace(go.Scattermapbox(
        lat=current["lat"], lon=current["lon"],
        mode="markers", marker=dict(size=4, color="grey", opacity=0.5),
        name=f"Current ({len(current):,})", hoverinfo="skip",
    ))
    fig.add_trace(go.Scattermapbox(
        lat=candidates["lat"], lon=candidates["lon"],
        mode="markers", marker=dict(size=6, color="green"),
        name=f"Candidates ({len(candidates):,})", hoverinfo="skip",
    ))
    center = COUNTRY_CENTERS.get(country, [46.5, 2.0])
    fig.update_layout(
        mapbox=dict(style="open-street-map", center=dict(lat=center[0], lon=center[1]), zoom=5),
        margin=dict(l=0, r=0, t=0, b=0), height=500, showlegend=True,
        legend=dict(x=0, y=1, bgcolor="rgba(255,255,255,0.8)"),
    )
    st.plotly_chart(fig, use_container_width=True)


def show_map_rollout(phases_stores, country):
    """Map for Rollout: baseline (grey) + each list in different color."""
    colors = ["grey", "#1f77b4", "#ff7f0e", "#2ca02c"]
    fig = go.Figure()
    for i, (name, df) in enumerate(phases_stores):
        fig.add_trace(go.Scattermapbox(
            lat=df["lat"], lon=df["lon"],
            mode="markers", marker=dict(size=4 if i == 0 else 6,
                                        color=colors[i % len(colors)],
                                        opacity=0.5 if i == 0 else 0.8),
            name=f"{name} ({len(df):,})", hoverinfo="skip",
        ))
    center = COUNTRY_CENTERS.get(country, [46.5, 2.0])
    fig.update_layout(
        mapbox=dict(style="open-street-map", center=dict(lat=center[0], lon=center[1]), zoom=5),
        margin=dict(l=0, r=0, t=0, b=0), height=500, showlegend=True,
        legend=dict(x=0, y=1, bgcolor="rgba(255,255,255,0.8)"),
    )
    st.plotly_chart(fig, use_container_width=True)


def coverage_rules_widget(key_prefix):
    """Show coverage rules (radius + category). Returns list of criteria and radii."""
    st.markdown("**Coverage rules**")
    st.caption("Define radius per category, or use one radius for all.")
    n = st.number_input("Number of rules", min_value=1, max_value=10,
                        value=3, key=f"{key_prefix}_n_rules")
    all_radii_opts = sorted(DEFAULT_RADII_M)
    cat_options = ["All categories", "Urban", "Suburban", "Rural"]
    cat_map = {"All categories": None, "Urban": "Urban", "Suburban": "Suburban", "Rural": "Rural"}
    criteria = []
    defaults = [(300, "Urban"), (1000, "Suburban"), (3000, "Rural")]
    for i in range(int(n)):
        rc1, rc2 = st.columns([1, 1])
        d_r = defaults[i][0] if i < len(defaults) else 500
        d_c = defaults[i][1] if i < len(defaults) else "All categories"
        with rc1:
            cr_r = st.selectbox(f"Radius" if i == 0 else f"Radius #{i+1}",
                all_radii_opts,
                index=all_radii_opts.index(d_r) if d_r in all_radii_opts else 0,
                key=f"{key_prefix}_r_{i}", format_func=lambda x: f"{x}m")
        with rc2:
            cr_cat = st.selectbox(f"For customers" if i == 0 else f"For #{i+1}",
                cat_options,
                index=cat_options.index(d_c) if d_c in cat_options else 0,
                key=f"{key_prefix}_cat_{i}")
        criteria.append({"radius_m": cr_r, "category": cat_map[cr_cat]})
    radii = sorted(set(cr["radius_m"] for cr in criteria))
    return criteria, radii


# ============================================================
# MAIN
# ============================================================
def main():
    st.markdown("<h1>Coverage Analysis — EU</h1>", unsafe_allow_html=True)
    st.caption("Store network optimization for WW Returns & ReCommerce")

    if not COUNTRIES:
        st.error("No data found. Check the data folder.")
        return

    # Country selector
    country = st.selectbox("Country", list(COUNTRIES.keys()),
        format_func=lambda c: COUNTRY_NAMES.get(c, c), key="country_sel")

    # Load grid
    cache_key = f"grid_{country}"
    with st.spinner(f"Loading {COUNTRY_NAMES.get(country, country)}..."):
        grid_df = load_grid_cached(COUNTRIES[country])
    total_w = int(grid_df["weight"].sum())

    # Stats
    cat_totals = grid_df.groupby("category")["weight"].sum()
    cols = st.columns(4)
    cols[0].metric("Customers", f"{total_w:,}")
    for i, (cat, w) in enumerate(cat_totals.items()):
        cols[i + 1].metric(cat, f"{int(w):,}")

    st.divider()

    # Analysis picker
    analysis = st.radio("What do you need?", options=[
        "📊 See the coverage of my store network",
        "✂️ Rationalize my network (remove stores without losing coverage)",
        "🎯 Find the best stores to add from a candidate list",
        "📋 Build a prioritized rollout plan (baseline + multiple lists)",
        "🔢 Maximize store options (improve density of choices)",
    ], key="analysis_picker")

    st.divider()


    # ═══════════ ANALYSIS 1: COVERAGE ═══════════
    if "coverage of my store network" in analysis:
        stores = get_stores_with_carrier_selection(country, "a1")
        if stores is not None:
            criteria, radii = coverage_rules_widget("a1")
            if st.button("🚀 Calculate Coverage", type="primary", key="a1_run"):
                with st.spinner("Computing coverage..."):
                    t0 = time.time()
                    # Compute nearest-store distance for each grid cell
                    grid_coords = np.radians(grid_df[["lat", "lon"]].to_numpy(np.float64))
                    store_coords = np.radians(stores[["lat", "lon"]].to_numpy(np.float64))
                    store_tree = BallTree(store_coords, metric="haversine")
                    d, _ = store_tree.query(grid_coords, k=1)
                    dist_km = d[:, 0] * EARTH_RADIUS_KM
                    weights = grid_df["weight"].to_numpy(np.int64)
                    categories = grid_df["category"].to_numpy()
                    total_w = int(weights.sum())

                    # Coverage per rule
                    rule_rows = []
                    for cr in criteria:
                        r_km = cr["radius_m"] / 1000.0
                        cat_label = cr["category"] if cr["category"] else "All"
                        if cr["category"] is None:
                            mask = dist_km <= r_km
                            denom = total_w
                        else:
                            mask = (dist_km <= r_km) & (categories == cr["category"])
                            denom = int(weights[categories == cr["category"]].sum())
                        cov = int(weights[mask].sum())
                        rule_rows.append({
                            "Radius": f"{cr['radius_m']}m",
                            "Category": cat_label,
                            "Covered": f"{cov:,}",
                            "Total": f"{denom:,}",
                            "Coverage %": round(cov / denom * 100, 2) if denom > 0 else 0,
                        })

                    elapsed = time.time() - t0
                    st.success(f"Done in {elapsed:.1f}s")
                    st.markdown("### Coverage")
                    st.dataframe(pd.DataFrame(rule_rows), use_container_width=True, hide_index=True)
                    st.download_button("📥 Download Results (Excel)",
                        data=make_excel({"Coverage": pd.DataFrame(rule_rows)}),
                        file_name=f"coverage_{country}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

                    # Map button
                    if st.button("🗺️ Show Map", key="a1_map"):
                        cell_radius_km = np.zeros(len(grid_df), np.float64)
                        for cr in criteria:
                            r_km = cr["radius_m"] / 1000.0
                            if cr["category"] is None:
                                cell_radius_km = np.maximum(cell_radius_km, r_km)
                            else:
                                m = grid_df["category"].to_numpy() == cr["category"]
                                cell_radius_km[m] = np.maximum(cell_radius_km[m], r_km)
                        show_map_coverage(stores, grid_df, dist_km, cell_radius_km, weights, country)


    # ═══════════ ANALYSIS 2: RATIONALIZE ═══════════
    elif "Rationalize my network" in analysis:
        st.write("Upload your store network. I'll identify which stores you can **remove** "
                 "without losing coverage.")
        stores = get_stores_with_carrier_selection(country, "a5")
        if stores is not None:
            criteria, radii = coverage_rules_widget("a5")
            tolerance = st.slider("Max coverage loss (pp)", 0.0, 5.0, 0.0,
                                  step=0.1, key="a5_tol",
                                  help="0 = only remove stores with zero impact.")
            if st.button("✂️ Rationalize Network", type="primary", key="a5_run"):
                with st.spinner("Computing..."):
                    t0 = time.time()
                    grid_rad_full = np.radians(grid_df[["lat", "lon"]].to_numpy(np.float64))
                    weights_full = grid_df["weight"].to_numpy(np.int64)
                    cats_full = grid_df["category"].to_numpy()
                    total_full = int(weights_full.sum())
                    n_stores = len(stores)

                    # Build per-cell radius (full grid)
                    cell_radius_km_full = np.zeros(len(grid_df), np.float64)
                    for cr in criteria:
                        r_km = cr["radius_m"] / 1000.0
                        if cr["category"] is None:
                            cell_radius_km_full = np.maximum(cell_radius_km_full, r_km)
                        else:
                            mask = cats_full == cr["category"]
                            cell_radius_km_full[mask] = np.maximum(cell_radius_km_full[mask], r_km)

                    # Compute distances on FULL grid (for coverage display)
                    store_coords = np.radians(stores[["lat", "lon"]].to_numpy(np.float64))
                    all_tree = BallTree(store_coords, metric="haversine")
                    all_d_full, idx_full = all_tree.query(grid_rad_full, k=1)
                    all_km_full = all_d_full[:, 0] * EARTH_RADIUS_KM
                    nearest_idx_full = idx_full[:, 0]

                    # For cells that are covered (nearest store within their radius)
                    covered_full = all_km_full <= cell_radius_km_full

                    # Find essential stores using query_radius (exact)
                    essential_stores = set()
                    unique_radii_km = np.unique(cell_radius_km_full[covered_full])
                    for r_km in unique_radii_km:
                        r_rad = r_km / EARTH_RADIUS_KM
                        cell_mask = covered_full & (cell_radius_km_full == r_km)
                        if not cell_mask.any():
                            continue
                        cell_coords = grid_rad_full[cell_mask]
                        counts = all_tree.query_radius(cell_coords, r=r_rad, count_only=True)
                        single_mask = counts == 1
                        if single_mask.any():
                            single_store_ids = nearest_idx_full[cell_mask][single_mask]
                            essential_stores.update(single_store_ids.tolist())

                    # VALIDATION: check essential-only coverage matches full coverage
                    # For any cell that loses coverage, add its nearest store (from full list)
                    ess_list = sorted(essential_stores)
                    ess_coords = store_coords[ess_list]
                    ess_tree = BallTree(ess_coords, metric="haversine")
                    ess_d, _ = ess_tree.query(grid_rad_full, k=1)
                    ess_km = ess_d[:, 0] * EARTH_RADIUS_KM
                    lost = covered_full & (ess_km > cell_radius_km_full)
                    if lost.any():
                        # Add the nearest store (from full list) for each lost cell
                        lost_stores = nearest_idx_full[lost]
                        essential_stores.update(lost_stores.tolist())

                    # Build removed list
                    removed_order = []
                    remaining = np.ones(n_stores, dtype=bool)
                    for s in range(n_stores):
                        if s not in essential_stores:
                            remaining[s] = False
                            removed_order.append({
                                "removal_order": len(removed_order) + 1,
                                "store_id": stores.iloc[s].get("store_id", f"store_{s}"),
                                "carrier": stores.iloc[s].get("carrier", ""),
                                "lat": stores.iloc[s]["lat"],
                                "lon": stores.iloc[s]["lon"],
                            })

                    elapsed = time.time() - t0
                    n_essential = int(remaining.sum())
                    n_removable = n_stores - n_essential

                    st.success(f"Done in {elapsed:.1f}s")

                    # Coverage per rule (SAME logic as Coverage tool — full grid)
                    st.markdown("### Coverage")
                    rule_rows = []
                    for cr in criteria:
                        r_km = cr["radius_m"] / 1000.0
                        cat_label = cr["category"] if cr["category"] else "All"
                        if cr["category"] is None:
                            mask = all_km_full <= r_km
                            denom = total_full
                        else:
                            mask = (all_km_full <= r_km) & (cats_full == cr["category"])
                            denom = int(weights_full[cats_full == cr["category"]].sum())
                        cov = int(weights_full[mask].sum())
                        rule_rows.append({
                            "Radius": f"{cr['radius_m']}m",
                            "Category": cat_label,
                            "Covered": f"{cov:,}",
                            "Total": f"{denom:,}",
                            "Coverage %": round(cov / denom * 100, 2) if denom > 0 else 0,
                        })
                    st.dataframe(pd.DataFrame(rule_rows), use_container_width=True, hide_index=True)

                    st.markdown("### Rationalization")
                    mc1, mc2, mc3 = st.columns(3)
                    mc1.metric("Total Stores", f"{n_stores:,}")
                    mc2.metric("Essential", f"{n_essential:,}")
                    mc3.metric("Removable", f"{n_removable:,}")
                    st.caption(f"Removing {n_removable:,} stores has zero impact on coverage.")

                    # Build essential list
                    essential_list = []
                    for s in range(n_stores):
                        if s in essential_stores:
                            essential_list.append({
                                "store_id": stores.iloc[s].get("store_id", f"store_{s}"),
                                "carrier": stores.iloc[s].get("carrier", ""),
                                "lat": stores.iloc[s]["lat"],
                                "lon": stores.iloc[s]["lon"],
                            })
                    essential_df = pd.DataFrame(essential_list) if essential_list else pd.DataFrame()

                    if removed_order:
                        removed_df = pd.DataFrame(removed_order)
                        st.markdown("### Removable Stores")
                        st.caption(f"{n_removable:,} stores that can be removed with zero coverage loss.")
                        st.dataframe(removed_df, use_container_width=True, hide_index=True)
                    else:
                        removed_df = pd.DataFrame()

                    if not essential_df.empty:
                        st.markdown("### Essential Stores")
                        st.caption(f"{n_essential:,} stores that are critical — removing any would reduce coverage.")
                        st.dataframe(essential_df, use_container_width=True, hide_index=True)

                    st.download_button("📥 Download Results (Excel)",
                        data=make_excel({"Removable": removed_df, "Essential": essential_df}),
                        file_name=f"rationalize_{country}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

                    # Map button
                    if st.button("🗺️ Show Map", key="a5_map"):
                        show_map_rationalize(stores, essential_stores, country)


    # ═══════════ ANALYSIS 3: BEST STORES TO ADD ═══════════
    elif "best stores to add" in analysis:
        st.write("Upload your **current network** and a **list of candidates**. "
                 "I'll rank each candidate by how much new coverage it adds.")
        st.subheader("Current network")
        current = get_stores_with_carrier_selection(country, "a2_cur")
        st.subheader("Candidate stores")
        candidates = store_upload_widget("a2_cand")

        if current is not None and candidates is not None:
            criteria, radii = coverage_rules_widget("a2")
            if st.button("🚀 Rank Candidates", type="primary", key="a2_run"):
                with st.spinner("Ranking candidates..."):
                    t0 = time.time()
                    grid_rad = np.radians(grid_df[["lat", "lon"]].to_numpy(np.float64))
                    weights = grid_df["weight"].to_numpy(np.int64)
                    cats = grid_df["category"].to_numpy()
                    total = int(weights.sum())

                    # Sample for speed
                    MAX_CELLS = 200_000
                    if len(grid_df) > MAX_CELLS:
                        np.random.seed(42)
                        idx = np.random.choice(len(grid_df), MAX_CELLS, replace=False,
                            p=weights.astype(np.float64) / weights.sum())
                        grid_rad = grid_rad[idx]
                        weights = weights[idx]
                        cats = cats[idx]
                        total = int(weights.sum())

                    # Build cell radius
                    cell_radius_km = np.zeros(len(weights), np.float64)
                    for cr in criteria:
                        r_km = cr["radius_m"] / 1000.0
                        if cr["category"] is None:
                            cell_radius_km = np.maximum(cell_radius_km, r_km)
                        else:
                            mask = cats == cr["category"]
                            cell_radius_km[mask] = np.maximum(cell_radius_km[mask], r_km)

                    # Baseline coverage
                    cur_coords = np.radians(current[["lat", "lon"]].to_numpy(np.float64))
                    cur_tree = BallTree(cur_coords, metric="haversine")
                    cur_d, _ = cur_tree.query(grid_rad, k=1)
                    cur_km = cur_d[:, 0] * EARTH_RADIUS_KM
                    baseline_covered = cur_km <= cell_radius_km
                    uncovered = ~baseline_covered

                    # Score each candidate
                    scores = []
                    cand_coords = np.radians(candidates[["lat", "lon"]].to_numpy(np.float64))
                    for i in range(len(candidates)):
                        c_rad = cand_coords[i:i+1]
                        d = BallTree(c_rad, metric="haversine").query(grid_rad, k=1)[0][:, 0] * EARTH_RADIUS_KM
                        newly_covered = uncovered & (d <= cell_radius_km)
                        score = int(weights[newly_covered].sum())
                        scores.append(score)

                    candidates = candidates.copy()
                    candidates["incremental_coverage"] = scores
                    candidates["pct_of_uncovered"] = [
                        round(s / max(int(weights[uncovered].sum()), 1) * 100, 2) for s in scores]
                    candidates = candidates.sort_values("incremental_coverage", ascending=False)

                    elapsed = time.time() - t0
                    st.success(f"Done in {elapsed:.1f}s")
                    st.dataframe(candidates.head(50), use_container_width=True, hide_index=True)
                    st.download_button("📥 Download Ranked Candidates",
                        data=make_excel({"Candidates": candidates}),
                        file_name=f"candidates_ranked_{country}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

                    # Map button
                    if st.button("🗺️ Show Map", key="a2_map"):
                        show_map_candidates(current, candidates, country)


    # ═══════════ ANALYSIS 4: PRIORITIZED ROLLOUT ═══════════
    elif "prioritized rollout" in analysis:
        st.write("Upload a **baseline** and up to 3 **priority lists**. "
                 "Shows coverage progression as each list is added.")
        st.subheader("Baseline (current network)")
        baseline = get_stores_with_carrier_selection(country, "a3_base")
        st.subheader("Priority Lists")
        list1 = store_upload_widget("a3_l1")
        list2 = store_upload_widget("a3_l2")
        list3 = store_upload_widget("a3_l3")

        if baseline is not None:
            criteria, radii = coverage_rules_widget("a3")
            if st.button("🚀 Run Rollout Model", type="primary", key="a3_run"):
                with st.spinner("Computing rollout..."):
                    t0 = time.time()
                    grid_rad = np.radians(grid_df[["lat", "lon"]].to_numpy(np.float64))
                    weights = grid_df["weight"].to_numpy(np.int64)
                    cats = grid_df["category"].to_numpy()
                    total = int(weights.sum())

                    # Sample
                    MAX_CELLS = 200_000
                    if len(grid_df) > MAX_CELLS:
                        np.random.seed(42)
                        idx = np.random.choice(len(grid_df), MAX_CELLS, replace=False,
                            p=weights.astype(np.float64) / weights.sum())
                        grid_rad = grid_rad[idx]
                        weights = weights[idx]
                        cats = cats[idx]
                        total = int(weights.sum())

                    cell_radius_km = np.zeros(len(weights), np.float64)
                    for cr in criteria:
                        r_km = cr["radius_m"] / 1000.0
                        if cr["category"] is None:
                            cell_radius_km = np.maximum(cell_radius_km, r_km)
                        else:
                            mask = cats == cr["category"]
                            cell_radius_km[mask] = np.maximum(cell_radius_km[mask], r_km)

                    # Compute coverage for each phase
                    phases = [("Baseline", baseline)]
                    phase_only_stores = [("Baseline", baseline)]
                    if list1 is not None:
                        phases.append(("+ List 1", pd.concat([baseline, list1], ignore_index=True)))
                        phase_only_stores.append(("List 1", list1))
                    if list2 is not None:
                        prev = phases[-1][1]
                        phases.append(("+ List 2", pd.concat([prev, list2], ignore_index=True)))
                        phase_only_stores.append(("List 2", list2))
                    if list3 is not None:
                        prev = phases[-1][1]
                        phases.append(("+ List 3", pd.concat([prev, list3], ignore_index=True)))
                        phase_only_stores.append(("List 3", list3))

                    summary = []
                    for phase_name, phase_stores in phases:
                        s_coords = np.radians(phase_stores[["lat", "lon"]].to_numpy(np.float64))
                        s_tree = BallTree(s_coords, metric="haversine")
                        s_d, _ = s_tree.query(grid_rad, k=1)
                        s_km = s_d[:, 0] * EARTH_RADIUS_KM
                        covered = s_km <= cell_radius_km
                        cov_w = int(weights[covered].sum())
                        row = {"phase": phase_name, "stores": len(phase_stores),
                               "coverage_pct": round(cov_w / total * 100, 1)}
                        for cat in ["Urban", "Suburban", "Rural"]:
                            cm = cats == cat
                            ct = int(weights[cm].sum())
                            if ct > 0:
                                row[f"{cat}_pct"] = round(int(weights[cm & covered].sum()) / ct * 100, 1)
                        summary.append(row)

                    elapsed = time.time() - t0
                    st.success(f"Done in {elapsed:.1f}s")
                    summary_df = pd.DataFrame(summary)
                    st.dataframe(summary_df, use_container_width=True, hide_index=True)
                    st.download_button("📥 Download Rollout",
                        data=make_excel({"Rollout": summary_df}),
                        file_name=f"rollout_{country}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

                    # Map button
                    if st.button("🗺️ Show Map", key="a3_map"):
                        show_map_rollout(phase_only_stores, country)


    # ═══════════ ANALYSIS 5: MAXIMIZE OPTIONS ═══════════
    elif "Maximize store options" in analysis:
        st.write("Given a baseline and candidates, greedily add stores to maximize "
                 "customers with **multiple store options** within radius.")
        st.subheader("Baseline (current network)")
        a4_baseline = get_stores_with_carrier_selection(country, "a4_base")
        st.subheader("Candidate stores to add")
        a4_candidates = store_upload_widget("a4_cand")

        if a4_baseline is not None and a4_candidates is not None:
            radius_m = st.selectbox("Radius", DEFAULT_RADII_M, index=2, key="a4_radius",
                                    format_func=lambda x: f"{x}m")
            min_options = st.number_input("Min store options required", 2, 10, 2, key="a4_min_opt")
            n_to_add = st.number_input("How many to add?", 1, len(a4_candidates), min(50, len(a4_candidates)), key="a4_n")

            if st.button("🚀 Maximize Options", type="primary", key="a4_run"):
                with st.spinner("Computing..."):
                    t0 = time.time()
                    grid_rad = np.radians(grid_df[["lat", "lon"]].to_numpy(np.float64))
                    weights = grid_df["weight"].to_numpy(np.int64)
                    total = int(weights.sum())
                    r_km = radius_m / 1000.0

                    # Sample
                    MAX_CELLS = 200_000
                    if len(grid_df) > MAX_CELLS:
                        np.random.seed(42)
                        idx = np.random.choice(len(grid_df), MAX_CELLS, replace=False,
                            p=weights.astype(np.float64) / weights.sum())
                        grid_rad = grid_rad[idx]
                        weights = weights[idx]
                        total = int(weights.sum())

                    # Count options per cell with baseline
                    base_coords = np.radians(a4_baseline[["lat", "lon"]].to_numpy(np.float64))
                    base_tree = BallTree(base_coords, metric="haversine")
                    r_rad = r_km / EARTH_RADIUS_KM
                    counts = base_tree.query_radius(grid_rad, r=r_rad, count_only=True)

                    # Greedy: add candidate that helps most under-served customers
                    added = []
                    cand_coords = np.radians(a4_candidates[["lat", "lon"]].to_numpy(np.float64))
                    current_counts = counts.copy()
                    cand_available = np.ones(len(a4_candidates), dtype=bool)

                    for step in range(int(n_to_add)):
                        best_score = -1
                        best_idx = -1
                        for ci in range(len(a4_candidates)):
                            if not cand_available[ci]:
                                continue
                            d = BallTree(cand_coords[ci:ci+1], metric="haversine").query(
                                grid_rad, k=1)[0][:, 0] * EARTH_RADIUS_KM
                            within = d <= r_km
                            # Score = customers currently below min_options that this helps
                            helps = within & (current_counts < min_options)
                            score = int(weights[helps].sum())
                            if score > best_score:
                                best_score = score
                                best_idx = ci
                        if best_idx == -1:
                            break
                        # Add it
                        cand_available[best_idx] = False
                        d = BallTree(cand_coords[best_idx:best_idx+1], metric="haversine").query(
                            grid_rad, k=1)[0][:, 0] * EARTH_RADIUS_KM
                        current_counts += (d <= r_km).astype(int)
                        added.append({
                            "rank": len(added) + 1,
                            "store_id": a4_candidates.iloc[best_idx].get("store_id", f"cand_{best_idx}"),
                            "lat": a4_candidates.iloc[best_idx]["lat"],
                            "lon": a4_candidates.iloc[best_idx]["lon"],
                            "customers_helped": best_score,
                        })

                    elapsed = time.time() - t0
                    st.success(f"Done in {elapsed:.1f}s — added {len(added)} stores")
                    if added:
                        added_df = pd.DataFrame(added)
                        st.dataframe(added_df, use_container_width=True, hide_index=True)
                        st.download_button("📥 Download",
                            data=make_excel({"Added": added_df}),
                            file_name=f"options_density_{country}.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")


if __name__ == "__main__":
    main()
