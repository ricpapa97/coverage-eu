"""
OSRM routing module — fast driving distance computation.
Uses parallel async requests + spatial sampling for speed.
"""
import time
import numpy as np
import pandas as pd
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from sklearn.neighbors import BallTree

EARTH_RADIUS_KM = 6371.0
OSRM_SERVER = "http://router.project-osrm.org"
MAX_COORDS_PER_REQUEST = 100
PARALLEL_WORKERS = 10
SAMPLE_RATE = 5  # compute 1 in N cells, interpolate the rest


def _osrm_table_batch(store_lon, store_lat, cell_coords):
    """
    Single OSRM table request: 1 store → N cells.
    Returns list of distances in km (None if failed).
    """
    coords_parts = [f"{store_lon},{store_lat}"]
    for lat, lon in cell_coords:
        coords_parts.append(f"{lon},{lat}")
    coords_str = ";".join(coords_parts)
    dests = ";".join(str(i) for i in range(1, len(coords_parts)))
    url = (f"{OSRM_SERVER}/table/v1/driving/{coords_str}"
           f"?sources=0&destinations={dests}&annotations=distance")
    try:
        resp = requests.get(url, timeout=15)
        if resp.status_code == 200:
            data = resp.json()
            if data.get("code") == "Ok" and data.get("distances"):
                return [d / 1000.0 if d is not None else None
                        for d in data["distances"][0]]
    except Exception:
        pass
    return [None] * len(cell_coords)


def compute_driving_distances(
    grid_df: pd.DataFrame,
    stores_df: pd.DataFrame,
    max_radius_km: float = 3.0,
    progress_callback=None,
) -> np.ndarray:
    """
    Compute driving distance from each grid cell to nearest store.

    Strategy:
    1. Aerial pre-filter: only cells within (max_radius_km + 1.5km) aerial
    2. Sample 1 in SAMPLE_RATE cells, route those
    3. Interpolate remaining cells using nearest sampled cell's ratio
    4. Parallel requests (PARALLEL_WORKERS simultaneous)

    Returns: array of driving distances in km, same length as grid_df
    """
    grid_coords = grid_df[["lat", "lon"]].to_numpy(np.float64)
    store_coords = stores_df[["lat", "lon"]].to_numpy(np.float64)

    # Step 1: Aerial distances (instant)
    grid_rad = np.radians(grid_coords)
    store_rad = np.radians(store_coords)
    store_tree = BallTree(store_rad, metric="haversine")
    aerial_dist_rad, nearest_idx = store_tree.query(grid_rad, k=1)
    aerial_dist_km = (aerial_dist_rad[:, 0] * EARTH_RADIUS_KM)
    nearest_idx = nearest_idx[:, 0]

    # Default: aerial × 1.3 (correction factor)
    driving_dist_km = aerial_dist_km * 1.3

    # Pre-filter: cells within reach (aerial < max_radius + buffer)
    # Buffer accounts for road detours: 3km driving ≈ 2-2.5km aerial
    prefilter_km = max_radius_km + 1.5
    close_mask = aerial_dist_km <= prefilter_km
    close_indices = np.where(close_mask)[0]

    if len(close_indices) == 0:
        if progress_callback:
            progress_callback(1.0)
        return driving_dist_km

    # Step 2: Sample — pick 1 in SAMPLE_RATE cells to route
    # Use spatial sampling: pick cells spread across the area
    np.random.seed(42)
    sample_mask = np.zeros(len(close_indices), dtype=bool)
    sample_mask[::SAMPLE_RATE] = True
    # Always include some random ones for better coverage
    extra = np.random.choice(len(close_indices),
                             size=min(len(close_indices) // SAMPLE_RATE, len(close_indices)),
                             replace=False)
    sample_mask[extra] = True
    sample_indices = close_indices[sample_mask]

    # Step 3: Build batch jobs — group by nearest store
    jobs = []  # list of (store_idx, cell_indices_batch)
    batch_size = MAX_COORDS_PER_REQUEST - 1

    for s_idx in np.unique(nearest_idx[sample_indices]):
        cells_for_store = sample_indices[nearest_idx[sample_indices] == s_idx]
        for batch_start in range(0, len(cells_for_store), batch_size):
            batch = cells_for_store[batch_start:batch_start + batch_size]
            jobs.append((int(s_idx), batch))

    # Step 4: Execute in parallel
    completed = 0
    total_jobs = len(jobs)

    def run_job(job):
        s_idx, cell_batch = job
        s_lat, s_lon = store_coords[s_idx]
        cell_coords = [(grid_coords[ci, 0], grid_coords[ci, 1]) for ci in cell_batch]
        distances = _osrm_table_batch(s_lon, s_lat, cell_coords)
        return cell_batch, distances

    with ThreadPoolExecutor(max_workers=PARALLEL_WORKERS) as executor:
        futures = {executor.submit(run_job, job): job for job in jobs}
        for future in as_completed(futures):
            cell_batch, distances = future.result()
            for i, ci in enumerate(cell_batch):
                if distances[i] is not None:
                    driving_dist_km[ci] = distances[i]
            completed += 1
            if progress_callback:
                progress_callback(completed / total_jobs * 0.8)  # 80% for routing

    # Step 5: Interpolate non-sampled close cells
    # For each non-sampled cell, use the ratio from its nearest sampled cell
    sampled_set = set(sample_indices)
    non_sampled = close_indices[~sample_mask]

    if len(non_sampled) > 0 and len(sample_indices) > 0:
        # Build a tree of sampled cells for quick lookup
        sampled_coords_rad = np.radians(grid_coords[sample_indices])
        sampled_tree = BallTree(sampled_coords_rad, metric="haversine")

        non_sampled_rad = np.radians(grid_coords[non_sampled])
        _, nearest_sampled = sampled_tree.query(non_sampled_rad, k=1)
        nearest_sampled = nearest_sampled[:, 0]

        for i, ci in enumerate(non_sampled):
            ref_ci = sample_indices[nearest_sampled[i]]
            ref_aerial = aerial_dist_km[ref_ci]
            ref_driving = driving_dist_km[ref_ci]
            if ref_aerial > 0.01:
                ratio = ref_driving / ref_aerial
            else:
                ratio = 1.3
            driving_dist_km[ci] = aerial_dist_km[ci] * ratio

    if progress_callback:
        progress_callback(1.0)

    return driving_dist_km


def estimate_time(n_close_cells: int) -> str:
    """Estimate time with optimizations."""
    sampled = n_close_cells // SAMPLE_RATE + n_close_cells // SAMPLE_RATE
    batches = sampled / (MAX_COORDS_PER_REQUEST - 1)
    # With parallel workers, effective time is batches / workers * avg_response
    seconds = (batches / PARALLEL_WORKERS) * 0.5  # 0.5s avg with parallelism
    if seconds < 60:
        return f"~{max(10, int(seconds))}s"
    elif seconds < 3600:
        return f"~{int(seconds/60)} min"
    else:
        return f"~{seconds/3600:.1f} hours"
