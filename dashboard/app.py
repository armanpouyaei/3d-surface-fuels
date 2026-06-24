"""3D Surface Fuels dashboard.

Two modes:
  * Synthetic longleaf demo — *truth* vs *FastFuels* uniform baseline vs *ours*
    (recovered), with the full validation report.
  * Real — Eglin 3DEP LiDAR — a measured 1 m surface-fuel grid built from real
    USGS 3DEP point clouds (scripts/build_eglin.py), shown over a real Esri
    World Imagery basemap, vs the FastFuels-style uniform layer.

The 3D view is a semi-transparent volume render of the full fuel column.

Run:  streamlit run dashboard/app.py   (from the project root, venv active)
"""

import os
import sys

import numpy as np
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from surface_fuels import FuelVoxelGrid, fusion, metrics  # noqa: E402
from surface_fuels.synthetic import make_demo_scenario, make_fusion_scenario  # noqa: E402

st.set_page_config(page_title="3D Surface Fuels", layout="wide")

PROC = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
COLORSCALE = "YlOrRd"
SYN_SCENARIOS = {
    "Truth (measured)": "truth",
    "FastFuels baseline (uniform per class)": "fastfuels",
    "Ours (SAR+LiDAR recovered)": "ours",
}


# ── data loaders ────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_synthetic(nx, seed):
    return make_demo_scenario(nx=nx, ny=nx, nz=4, seed=int(seed))


@st.cache_data(show_spinner=False)
def load_fusion(nx, seed):
    sc = make_fusion_scenario(nx=nx, ny=nx, seed=int(seed))
    pred = fusion.spatial_block_cv(sc, blocks=4)
    return sc, pred, fusion.evaluate(pred, sc)


@st.cache_data(show_spinner=False)
def load_real_sar():
    p = os.path.join(PROC, "eglin_sar.npz")
    if not os.path.exists(p):
        return None
    d = np.load(p, allow_pickle=True)
    return {k: d[k] for k in d.files}


@st.cache_data(show_spinner=False)
def load_real():
    """Load the real Eglin grids + basemap if scripts/build_eglin.py has run."""
    mpath = os.path.join(PROC, "eglin_measured.nc")
    upath = os.path.join(PROC, "eglin_uniform.nc")
    bpath = os.path.join(PROC, "eglin_basemap.png")
    if not (os.path.exists(mpath) and os.path.exists(upath)):
        return None
    measured = FuelVoxelGrid.from_netcdf(mpath)
    uniform = FuelVoxelGrid.from_netcdf(upath)
    rgb = None
    if os.path.exists(bpath):
        from PIL import Image
        rgb = np.asarray(Image.open(bpath).convert("RGB"))
    return {"measured": measured, "uniform": uniform, "rgb": rgb}


# ── figures ─────────────────────────────────────────────────────────────────
def _coarsen(bd, max_cells=45_000):
    """Block-mean horizontally so go.Volume stays responsive on big grids."""
    nz, ny, nx = bd.shape
    f = 1
    while nz * (ny // (f + 1)) * (nx // (f + 1)) > max_cells and (ny // (f + 1)) > 8:
        f += 1
    if f == 1:
        return bd, 1
    ny2, nx2 = ny // f, nx // f
    return bd[:, : ny2 * f, : nx2 * f].reshape(nz, ny2, f, nx2, f).mean(axis=(2, 4)), f


def voxel_figure(grid, threshold, opacity, title):
    """Semi-transparent volume render of the full fuel column (all altitude)."""
    bd, f = _coarsen(grid.bulk_density.astype(float))
    nz, ny, nx = bd.shape
    zc = (np.arange(nz) + 0.5) * grid.dz
    yc = (np.arange(ny) + 0.5) * grid.dy * f
    xc = (np.arange(nx) + 0.5) * grid.dx * f
    Z, Y, X = np.meshgrid(zc, yc, xc, indexing="ij")
    vmax = float(bd.max())
    iso = max(threshold, 1e-3)
    note = f"  (display coarsened {f}×)" if f > 1 else ""
    fig = go.Figure(
        go.Volume(
            x=X.ravel(), y=Y.ravel(), z=Z.ravel(), value=bd.ravel(),
            isomin=iso, isomax=max(vmax, iso * 1.01),
            opacity=opacity, surface_count=20,
            colorscale=COLORSCALE, cmin=0, cmax=max(vmax, 1e-3),
            colorbar=dict(title="kg/m³"),
            opacityscale=[[0, 0.0], [0.08, 0.3], [0.35, 0.7], [1, 1.0]],
            caps=dict(x_show=False, y_show=False, z_show=False),
            hovertemplate="x=%{x:.0f} m<br>y=%{y:.0f} m<br>h=%{z:.1f} m<br>ρ=%{value:.2f} kg/m³<extra></extra>",
        )
    )
    north_y, mid_x, top_z = float(yc.max()), float(xc.mean()), nz * grid.dz
    fig.update_layout(
        title=f"{title} — full column, height 0–{nz * grid.dz:.0f} m (vertical exaggerated){note}",
        scene=dict(
            xaxis_title="x — East (m)", yaxis_title="y — North (m)", zaxis_title="height (m)",
            zaxis=dict(range=[0, top_z]),
            aspectmode="manual", aspectratio=dict(x=2, y=2, z=0.9),
            # camera south of the scene, elevated, looking toward +y (North), inclined down
            camera=dict(eye=dict(x=0.9, y=-1.9, z=0.85), up=dict(x=0, y=0, z=1),
                        center=dict(x=0, y=0, z=-0.1)),
            annotations=[dict(x=mid_x, y=north_y, z=top_z, text="▲ North",
                              showarrow=False, font=dict(size=13, color="#444"),
                              bgcolor="rgba(255,255,255,0.6)")],
        ),
        height=560, margin=dict(l=0, r=0, t=40, b=0),
    )
    return fig


def synthetic_aerial_rgb(grid):
    """Synthetic natural-color render of a scene (synthetic mode only)."""
    bd = grid.bulk_density.astype(float)
    load = grid.fuel_load()
    col = bd.sum(axis=0)
    live = grid.extra.get("live_fraction")
    with np.errstate(invalid="ignore", divide="ignore"):
        livefrac = np.where(col > 0, (bd * (live if live is not None else 0)).sum(axis=0) / col, 0.0)
    p90 = np.percentile(load[load > 0], 90) if np.any(load > 0) else 1.0
    cover = np.clip(load / (p90 + 1e-9), 0, 1)[..., None]
    soil, litter, green = (np.array([0.80, 0.73, 0.55]),
                           np.array([0.55, 0.45, 0.28]), np.array([0.20, 0.45, 0.17]))
    veg = litter * (1 - livefrac[..., None]) + green * livefrac[..., None]
    rgb = soil * (1 - cover) + veg * cover
    rng = np.random.default_rng(0)
    rgb = np.clip(rgb + rng.normal(0, 0.02, rgb.shape), 0, 1)
    # row 0 = south, same convention as the load arrays (rendered north-up together)
    return (rgb * 255).astype(np.uint8)


def _resample_rgb(rgb, ny, nx):
    """Resample an RGB array to the fuel grid resolution so it shares the panels'
    coordinate system (needed for aligned, synchronized zoom)."""
    if rgb.shape[0] == ny and rgb.shape[1] == nx:
        return rgb
    from PIL import Image
    return np.asarray(Image.fromarray(rgb.astype("uint8")).resize((nx, ny)))


def load_maps(panels, vmax, rgb_northup, rgb_title):
    """Row of top-down maps: an RGB panel then load heatmaps, all on a *shared,
    synchronized* coordinate system — zooming/panning any panel zooms them all.

    ``panels`` = list of (title, load2d) with load2d row 0 = south. ``rgb_northup``
    is row 0 = north. Everything is rendered north-up: the RGB stays as-is and the
    heatmaps are flipped, all on a reversed y-axis, with axes matched across panels.
    """
    ny, nx = panels[0][1].shape
    titles = [rgb_title] + [t for t, _ in panels]
    n = len(titles)
    fig = make_subplots(rows=1, cols=n, subplot_titles=titles, horizontal_spacing=0.03)
    # All input arrays use row 0 = south (verified: LiDAR canopy and the basemap
    # share this orientation, corr +0.54 greenness). go.Image forces the shared
    # y-axis to 'reversed' (row 0 at top), so we flip every array to row 0 = north
    # -> north is at the top and all panels line up. Axes are matched so zooming
    # or panning any panel zooms them all together.
    rgb = _resample_rgb(rgb_northup, ny, nx)
    fig.add_trace(go.Image(z=np.flipud(rgb)), row=1, col=1)
    for i, (_, f) in enumerate(panels, start=2):
        fig.add_trace(
            go.Heatmap(z=np.flipud(f), zmin=0, zmax=vmax, colorscale=COLORSCALE,
                       colorbar=dict(title="kg/m²") if i == n else None, showscale=(i == n)),
            row=1, col=i)
    fig.update_xaxes(matches="x", showticklabels=False)
    fig.update_yaxes(matches="y", autorange="reversed", showticklabels=False)
    fig.update_layout(height=320, margin=dict(l=0, r=0, t=30, b=0), dragmode="zoom")
    return fig


def synced_heatmaps(panels):
    """Row of heatmaps on shared, synchronized axes (zoom one -> all zoom),
    rendered north-up. ``panels`` = list of dict(title, z, cmin, cmax, colorscale,
    cbar?, cbar_title?). z is row 0 = south."""
    n = len(panels)
    fig = make_subplots(rows=1, cols=n, subplot_titles=[p["title"] for p in panels],
                        horizontal_spacing=0.02)
    cbar_panels = [i for i, p in enumerate(panels) if p.get("cbar")]
    m = max(1, len(cbar_panels))
    for i, p in enumerate(panels, start=1):
        cbar = None
        if p.get("cbar"):
            k = cbar_panels.index(i - 1)          # stack colorbars vertically so they don't overlap
            cbar = dict(title=p.get("cbar_title", ""), x=1.01,
                        y=1 - (k + 0.5) / m, len=1.0 / m - 0.06, thickness=12)
        fig.add_trace(go.Heatmap(
            z=np.flipud(p["z"]), zmin=p["cmin"], zmax=p["cmax"], colorscale=p["colorscale"],
            showscale=p.get("cbar", False), colorbar=cbar),
            row=1, col=i)
    fig.update_xaxes(matches="x", showticklabels=False)
    fig.update_yaxes(matches="y", autorange="reversed", showticklabels=False)
    fig.update_layout(height=300, margin=dict(l=0, r=0, t=30, b=0), dragmode="zoom")
    return fig


def profile_figure(series):
    """series = list of (name, grid, color)."""
    fig = go.Figure()
    for name, grid, color in series:
        z = grid.heights()
        fig.add_trace(go.Scatter(x=grid.vertical_profile(), y=z, mode="lines+markers",
                                 name=name, line=dict(color=color)))
    fig.update_layout(title="Mean vertical bulk-density profile",
                      xaxis_title="bulk density (kg/m³)", yaxis_title="height (m)",
                      height=360, margin=dict(l=0, r=0, t=40, b=0))
    return fig


# ── UI ──────────────────────────────────────────────────────────────────────
st.title("🔥 3D Surface Fuels — measured beats modeled")
st.caption(
    "Meter-scale 3D surface-fuel mapping for the "
    "[Surface Fuels Prize Challenge](https://centralfloridatechgrove.org/surface-fuels-prize-challenge/)."
)

with st.sidebar:
    st.header("Data source")
    source = st.radio("", ["Synthetic longleaf demo", "SAR + LiDAR fusion (synthetic)",
                           "Real — Eglin 3DEP LiDAR"], index=0)
    st.divider()
    threshold = st.slider("Min bulk density shown (kg/m³)", 0.0, 1.0, 0.02, step=0.02)
    opacity = st.slider("Volume opacity", 0.05, 0.6, 0.2, step=0.05)

# ===================== SYNTHETIC MODE =====================
if source.startswith("Synthetic"):
    with st.sidebar:
        st.header("Synthetic scene")
        nx = st.slider("Domain size (m)", 40, 160, 100, step=20)
        seed = st.number_input("Random seed", 0, 9999, 42, step=1)
        scenario = st.radio("3D scenario", list(SYN_SCENARIOS.keys()), index=0)
    scene = load_synthetic(nx, seed)
    grids = {"truth": scene.truth, "fastfuels": scene.fastfuels, "ours": scene.ours}
    grid = grids[SYN_SCENARIOS[scenario]]

    st.subheader("3D voxel field")
    st.plotly_chart(voxel_figure(grid, threshold, opacity, scenario), use_container_width=True)

    st.subheader("Fuel loading (kg/m²) — top-down")
    vmax = float(max(scene.truth_load.max(), scene.fastfuels_load.max(), scene.ours_load.max()))
    st.plotly_chart(load_maps(
        [("Truth", scene.truth_load), ("FastFuels", scene.fastfuels_load), ("Ours", scene.ours_load)],
        vmax, synthetic_aerial_rgb(scene.truth), "Aerial RGB (synthetic)",
    ), use_container_width=True)
    st.caption("FastFuels paints one value across the whole class (flat). Truth and Ours show the sub-metre clumping that drives fire behavior. "
               "Zoom or pan any panel — they all move together.")

    st.subheader("Validation — vs. truth")
    ours = metrics.compare(scene.ours, scene.truth)
    ff = metrics.compare(scene.fastfuels, scene.truth)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Load R² (ours)", f"{ours['load_r2']:.2f}", f"{ours['load_r2'] - ff['load_r2']:+.2f} vs FastFuels")
    c2.metric("Load RMSE (kg/m²)", f"{ours['load_rmse_kg_m2']:.3f}", f"{ours['load_rmse_kg_m2'] - ff['load_rmse_kg_m2']:+.3f}", delta_color="inverse")
    c3.metric("Heterogeneity kept (CV ratio)", f"{ours['cv_ratio']:.2f}", f"FastFuels {ff['cv_ratio']:.2f}")
    c4.metric("3D bulk-density RMSE", f"{ours['bulk_density_rmse_3d_kg_m3']:.3f}", f"{ours['bulk_density_rmse_3d_kg_m3'] - ff['bulk_density_rmse_3d_kg_m3']:+.3f}", delta_color="inverse")
    col_a, col_b = st.columns([1, 1])
    with col_a:
        st.plotly_chart(profile_figure([("Truth", scene.truth, "#2c7fb8"),
                                        ("FastFuels", scene.fastfuels, "#d95f0e"),
                                        ("Ours", scene.ours, "#31a354")]), use_container_width=True)
    with col_b:
        st.markdown("**Full metric comparison** (closer to truth is better)")
        keys = ["load_r2", "load_rmse_kg_m2", "load_bias_kg_m2", "bulk_density_rmse_3d_kg_m3",
                "depth_rmse_m", "occupancy_iou", "cv_truth", "cv_pred", "cv_ratio", "mass_error_pct"]
        st.dataframe([{"metric": k, "Ours": ours.get(k), "FastFuels": ff.get(k)} for k in keys],
                     use_container_width=True, hide_index=True)
    st.caption("On synthetic data — real numbers come from RxCADRE/NEON truth. See research/VALIDATION.md.")

# ===================== SAR + LiDAR FUSION MODE =====================
elif source.startswith("SAR"):
    with st.sidebar:
        st.header("Fusion scene")
        fnx = st.slider("Domain size (m)", 80, 160, 120, step=20)
        fseed = st.number_input("Random seed", 0, 9999, 42, step=1)
    sc, pred, rep = load_fusion(fnx, fseed)
    vmax = float(sc.truth_load.max())

    st.subheader("SAR + LiDAR fusion — filling LiDAR's under-canopy blind spot")
    st.plotly_chart(synced_heatmaps([
        {"title": "Truth load", "z": sc.truth_load, "cmin": 0, "cmax": vmax, "colorscale": COLORSCALE},
        {"title": "Canopy cover", "z": sc.canopy, "cmin": 0, "cmax": 1, "colorscale": "Greens"},
        {"title": "LiDAR-only", "z": pred["lidar_only"], "cmin": 0, "cmax": vmax, "colorscale": COLORSCALE},
        {"title": "SAR-only", "z": pred["sar_only"], "cmin": 0, "cmax": vmax, "colorscale": COLORSCALE},
        {"title": "Fusion", "z": pred["fusion"], "cmin": 0, "cmax": vmax, "colorscale": COLORSCALE,
         "cbar": True, "cbar_title": "kg/m²"},
    ]), use_container_width=True)
    st.caption("LiDAR is accurate in the open but **occluded under canopy** (note the washed-out LiDAR-only "
               "patches where canopy cover is high). SAR penetrates the canopy. Fusion = (1−cover)·LiDAR + "
               "cover·SAR recovers fuel under canopy. Zoom any panel — all move together.")

    st.subheader("Validation — spatially-blocked cross-validation (vs. truth)")
    f, l, u = rep["fusion"], rep["lidar_only"], rep["fastfuels_uniform"]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Fusion R²", f"{f['r2']:.2f}", f"{f['r2']-l['r2']:+.2f} vs LiDAR-only")
    c2.metric("Fusion R² under canopy", f"{f['r2_under_canopy']:.2f}",
              f"{f['r2_under_canopy']-l['r2_under_canopy']:+.2f} vs LiDAR-only")
    c3.metric("LiDAR-only R² under canopy", f"{l['r2_under_canopy']:.2f}", "occluded", delta_color="off")
    c4.metric("FastFuels uniform R²", f"{u['r2']:.2f}", "no skill", delta_color="off")

    methods = ["lidar_only", "sar_only", "fusion", "fastfuels_uniform"]
    labels = {"lidar_only": "LiDAR-only", "sar_only": "SAR-only", "fusion": "Fusion",
              "fastfuels_uniform": "FastFuels unif."}
    bar = go.Figure()
    for strat, key in [("Open", "r2_open"), ("Under canopy", "r2_under_canopy"), ("Overall", "r2")]:
        bar.add_bar(name=strat, x=[labels[m] for m in methods], y=[rep[m][key] for m in methods])
    bar.update_layout(barmode="group", height=340, yaxis_title="R²",
                      margin=dict(l=0, r=0, t=10, b=0), legend=dict(orientation="h", y=1.12))
    st.plotly_chart(bar, use_container_width=True)
    st.caption("Fusion matches LiDAR in the open and far exceeds it under canopy, beating SAR-only and the "
               "uniform baseline overall. Synthetic sensors here; real Sentinel-1 features are in the Real Eglin mode.")

# ===================== REAL EGLIN MODE =====================
else:
    real = load_real()
    if real is None:
        st.warning("No real Eglin grids found. Run `python scripts/download_eglin_lidar.py` "
                   "then `python scripts/build_eglin.py` to build them.")
        st.stop()
    measured, uniform, rgb = real["measured"], real["uniform"], real["rgb"]
    with st.sidebar:
        st.header("Real scene")
        which = st.radio("3D scenario", ["Measured (3DEP LiDAR)", "Uniform baseline (FastFuels-style)"], index=0)
    grid = measured if which.startswith("Measured") else uniform

    st.subheader("3D voxel field — real Eglin AFB surface fuels (USGS 3DEP LiDAR)")
    st.plotly_chart(voxel_figure(grid, threshold, opacity, which), use_container_width=True)
    st.caption(f"Source: {measured.attrs.get('source','3DEP')}. {measured.attrs.get('note','')}. "
               "Sparse occupancy reflects real ALS under-sampling of the understory — the gap SAR fusion addresses.")

    st.subheader("Fuel loading (kg/m²) — top-down, over real satellite imagery")
    ml, ul = measured.fuel_load(), uniform.fuel_load()
    vmax = float(max(ml.max(), ul.max()))
    rgb_img, rgb_title = (rgb, "Esri World Imagery (real)") if rgb is not None else \
                         (synthetic_aerial_rgb(measured), "Aerial (offline — synthetic)")
    st.plotly_chart(load_maps([("Measured (3DEP)", ml), ("Uniform baseline", ul)], vmax, rgb_img, rgb_title),
                    use_container_width=True)
    st.caption("Left: **real Esri imagery** for the AOI (Eglin AFB, ~−86.63°, 30.47°N — verified same footprint). "
               "Middle: our measured 1 m surface-fuel loading. Right: the FastFuels/SB40-style uniform layer (flat). "
               "Zoom any panel — all move together. Note the LiDAR is 2007; the imagery is recent, so individual trees "
               "in this frequently-burned, managed forest won't match one-to-one (temporal gap), but the footprint is identical.")

    st.subheader("Heterogeneity — measured vs. uniform")
    hm, hu = metrics.heterogeneity_stats(measured), metrics.heterogeneity_stats(uniform)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Measured CV", f"{hm['cv']:.2f}", f"uniform {hu['cv']:.2f}")
    c2.metric("Moran's I (clustering)", f"{hm['moran_i']:.2f}")
    c3.metric("Mean load (kg/m²)", f"{hm['mean_load_kg_m2']:.2f}")
    c4.metric("Occupied surface voxels", f"{measured.occupancy(1e-6).mean()*100:.0f}%")
    col_a, col_b = st.columns([1, 1])
    with col_a:
        st.plotly_chart(profile_figure([("Measured", measured, "#31a354"),
                                        ("Uniform", uniform, "#d95f0e")]), use_container_width=True)
    with col_b:
        st.markdown("**Grid summary**")
        st.dataframe([{"property": k, "value": v} for k, v in measured.summary().items()],
                     use_container_width=True, hide_index=True)
    st.caption("Real-data status: the spatial pattern is *measured* from 3DEP LiDAR; magnitude is anchored to a "
               "literature mean load pending co-located destructive/TLS calibration (RxCADRE clip plots). "
               "FastFuels' uniform layer has CV≈0 by construction — the heterogeneity we recover is the whole point.")

    sar_d = load_real_sar()
    if sar_d is not None:
        st.subheader("Real Sentinel-1 SAR + fusion (Eglin)")
        ll = sar_d["lidar_load"]
        vmaxL = float(max(ll.max(), sar_d["fused"].max()))
        vh_vv = sar_d["vh_vv"]
        st.plotly_chart(synced_heatmaps([
            {"title": "VH/VV ratio (Sentinel-1, real)", "z": vh_vv,
             "cmin": float(np.nanpercentile(vh_vv, 2)), "cmax": float(np.nanpercentile(vh_vv, 98)),
             "colorscale": "Viridis", "cbar": True, "cbar_title": "VH/VV"},
            {"title": "Canopy cover (LiDAR)", "z": sar_d["canopy"], "cmin": 0, "cmax": 1, "colorscale": "Greens"},
            {"title": "LiDAR load", "z": ll, "cmin": 0, "cmax": vmaxL, "colorscale": COLORSCALE},
            {"title": "SAR + LiDAR fused", "z": sar_d["fused"], "cmin": 0, "cmax": vmaxL,
             "colorscale": COLORSCALE, "cbar": True, "cbar_title": "kg/m²"},
        ]), use_container_width=True)
        st.caption(f"**Real Sentinel-1 RTC** ({int(sar_d['n_items'])} acquisitions, {str(sar_d['dates'])}, "
                   "Planetary Computer) over the AOI. VH/VV cross-pol ratio is the understory / volume-scattering "
                   "proxy; the fused product leans on SAR where canopy occludes the LiDAR. Uncalibrated blend "
                   "(magnitude pending destructive truth) — demonstrates the real SAR on-ramp. Run "
                   "`scripts/build_sar_eglin.py` to (re)build.")
