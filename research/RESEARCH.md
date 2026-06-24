# RESEARCH.md — A Measured 3D Surface-Fuel Product for the 3D Surface Fuels Prize Challenge

## The Challenge

The **3D Surface Fuels & Vegetation Modeling Prize Challenge** (Central Florida Tech Grove / NAWCTSD, in partnership with SERDP-ESTCP) asks competitors to produce **high-resolution 3D maps of surface and understory fuels** — grasses, shrubs, litter, downed woody debris, and other near-ground/under-canopy vegetation — at **meter scale (nominally 1 m × 1 m × 1 m voxels)**, suitable for ingestion by next-generation, physics-based fire models (QUIC-Fire, FIRETEC/HIGRAD, FDS). The submission deadline is **July 20, 2026**.

### Required fuel properties

Per voxel (or per cell, depending on pathway), the challenge wants:

- **Fuel loading** (kg/m²)
- **Bulk density** (kg/m³)
- **Depth / height** (m)
- **Live-vs-dead fraction** (%)
- **Surface-area-to-volume ratio (SAVR/SAV)** (m²/m³, often reported as 1/m)
- Ideally also **species**, **fuel moisture** (%), and **patch/clump heterogeneity metrics**

### Submission pathways

There are three accepted delivery formats, two of which are explicitly **FastFuels-compatible**:

1. **Georeferenced arrays** (generic gridded output).
2. **FastFuels-compatible inputs**, of which there are three sub-options:
   - (2a) a **2D raster fuel classification** (FCCS fuelbed IDs or Scott & Burgan FBFM40) at up to 10 m;
   - (2b) a **3D voxelized netCDF** of bulk density at 1 m;
   - (2c) a **statistical "layerset"** delivered as GeoJSON.
3. (Subsumed under 2c above.)

### Judging weights

In rough descending order of importance: **validation / credibility** (most heavily weighted), then **utility**, **generality across ecosystems**, **data cost**, **clarity**, **relevance to surface/understory fuels specifically**, and **scalability**. The single most important takeaway is that *validation against measured ground truth dominates the score* — this directly shapes our strategy.

### Why surface/understory fuels are hard

Surface fuels are the part of the fuel complex that operational remote-sensing pipelines handle worst, for physical and methodological reasons:

- **Canopy occlusion.** Airborne LiDAR (ALS) and optical sensors cannot see the ground reliably beneath a closed canopy; the few returns that reach the surface are sparse and biased. The state of the art predicts litter/duff loads only *indirectly* from **overstory structural proxies**, which caps accuracy. The Kaibab Plateau multitemporal-LiDAR study achieved only **R² ≈ 0.39–0.59** for surface components (litter+duff R²=0.59, total surface fuel R²=0.48, 1–1000 h woody R²=0.39), and noted that litter/duff "LiDAR cannot directly penetrate" had to be predicted from overstory correlates ([PMC9361251](https://pmc.ncbi.nlm.nih.gov/articles/PMC9361251/)). Critically, surface fuel at that site averaged **82.7 Mg/ha vs. 6.9 Mg/ha** of available canopy fuel — roughly 12× larger — so error in the surface stratum dominates fire-model error.
- **Adding optical does not fix it.** A temperate-forest study fusing ALS with Sentinel-2 reached only **R²=0.27–0.41** for litter and fine dead woody fuels, and effectively failed (≈0) on coarse woody debris ([Remote Sensing of Environment, 2023](https://www.sciencedirect.com/science/article/abs/pii/S0034425723002626)). Optical sensors see the canopy top, not the fuel beneath it.
- **Extreme sub-meter heterogeneity.** Surface fuels in frequently burned ecosystems reorganize at scales finer than a meter (detailed in [The Gap We Exploit](#the-gap-we-exploit)), so any product that assigns one value to a 30 m pixel discards the controlling variability.
- **Dimensional mismatch in validation.** The gold-standard truth (destructive clip-plot harvest) is a 2D/integrated mass over ~1 m² quadrats, while the target is a 3D voxel column — making rigorous validation itself a methodological problem.

These constraints are precisely why the challenge exists, and why a **measured** surface-fuel product is valuable.

---

## How FastFuels Works Today

FastFuels (silvxlabs, currently v2-beta) is a **cloud API plus a Python SDK** that assembles meter-scale 3D fuel grids for next-generation fire models. It is both the named integration target of the challenge (FastFuels-compatible inputs are an accepted pathway) and the baseline our thesis targets. The authoritative description of its data sources, endpoints, and defaults comes from its OpenAPI 3.1 schema ([api-v2 openapi.json](https://api-v2-prod-782971006568.us-west1.run.app/openapi.json)), the open-source core library ([silvxlabs/fastfuels-core](https://github.com/silvxlabs/fastfuels-core)), and the methods paper (Marcozzi et al., *Environmental Modelling & Software*, 2024/2025; [open-access PDF](https://www.fs.usda.gov/rm/pubs_journals/2025/rmrs_2025_marcozzi_a001.pdf)).

### Data sources

- **TreeMap** (a 30 m CONUS raster mapping each pixel to a representative FIA plot) and the **FIA** tree/plot database — the backbone of the canopy product.
- **LANDFIRE** — the backbone of the *surface* product, via the Scott & Burgan **FBFM40** (40 fire-behavior fuel models, 46 valid codes) or **FCCS** fuelbed rasters at **30 m**. Default LANDFIRE versions in the API: FBFM40 = 2024, FCCS = 2023, canopy = 2022, topography = 2020.
- Optional **LiDAR/CHM**, uploaded inventories, and **NAIP** imagery as canopy on-ramps.

### How the 3D CANOPY grid is built

This is FastFuels' genuine strength and is reasonably "measured-like":

1. **Imputation (TreeMap PIM).** `POST /grids/pim/treemap` maps each 30 m cell to a TreeMap/FIA plot.
2. **Spatialization (inhomogeneous Poisson point process).** `POST /inventories/tree/pim` interpolates trees-per-area onto a **15 m sub-cell grid**, assigns plot IDs by Voronoi nearest-neighbor, draws a Poisson count of trees per sub-cell, samples individual trees from the assigned FIA plot weighted by trees-per-acre, and places each at a random in-cell coordinate.
3. **Crown voxelization.** `POST /grids/voxelize/inventory/tree` discretizes each tree's crown onto the voxel grid using a **species-specific crown-profile model** (`purves` default, `beta` alternative). Foliage biomass from **NSVB allometry** is distributed (uniformly) over stochastically occupied crown voxels (`distribute_biomass()` in [voxelization.py](https://github.com/silvxlabs/fastfuels-core/blob/main/fastfuels_core/voxelization.py)), yielding per-voxel **bulk density, moisture, and SAVR**. **Default voxel resolution is 2 m horizontal × 1 m vertical** (coarser than the challenge's 1 m³, though 0.5 m demos exist).

FastFuels also has measured-data canopy on-ramps that bypass TreeMap — individual-tree detection from a CHM via Local-Maximum / Variable-Window Filtering (`POST /inventories/tree/chm`) and the **GDAM** masked-autoencoder to impute dbh/crown/species from LiDAR stems (`POST /inventories/tree/allometry/gdam`). **Notably, every one of these on-ramps is for canopy — there is no analogous measured surface-fuel on-ramp.**

### How SURFACE fuels are built — the critical detail

By default, FastFuels does **not measure surface fuels**. It performs a **categorical lookup**:

1. `POST /grids/fbfm40/landfire` creates a single categorical band `fbfm` of SB40 codes at **30 m** (46 valid codes: NB 91–99; GR 101–109; GS 121–124; SH 141–149; TU 161–165; TL 181–189; SB 201–204).
2. `POST /grids/lookup/fbfm40` converts each code to continuous bands using **"Scott-Burgan 40 lookup tables"**: `fuel_load.{1hr,10hr,100hr,live_herb,live_woody}` (kg/m²), `savr.{1hr,10hr,100hr,live}` (1/m), and `fuel_depth` (m). The FCCS path (`POST /grids/fccs/landfire`) is analogous with fuelbed IDs.

The consequence: **every 30 m cell sharing an SB40 code receives identical surface fuel parameters** — loading, bulk density, depth, and SAVR are *constant within each fuel-model class and spatially uniform across the landscape*. There is no within-class loading variation, no sub-30 m heterogeneity, and no actual measurement of grass/shrub/litter/CWD at the site.

The FastFuels paper is explicit about this. In its FDS demo it modeled surface fuels "as a uniform layer of Long-needle forest floor litter, drawing upon the characteristics of Anderson Fire Behavior Fuel Model 9" (**loading 0.717 kg/m², fuel-bed depth 6.1 cm**), and stated this "does not capture the full complexity of surface fuel heterogeneity (Vakili et al., 2016)." In the QUIC-Fire demo it converted the LANDFIRE FBFM40 30 m raster "from discrete categorical values to continuous fuel properties using a lookup table," with SAVR "assigned uniformly" (e.g., 8000 /m litter, 9770 /m grass) and uniform moisture. Improving surface fuel models is listed as future work ([Marcozzi et al. 2025](https://www.fs.usda.gov/rm/pubs_journals/2025/rmrs_2025_marcozzi_a001.pdf)).

### The only heterogeneous-surface pathway: user-supplied layersets

FastFuels *can* represent heterogeneous surface fuels — but only if the **user brings the data**. The `layerset` pathway ([layersets.py](https://github.com/silvxlabs/fastfuels-core/blob/main/fastfuels_core/layersets.py); `POST /domains/{id}/features/layerset/geojson` → `POST /grids/rasterize/layerset`) rasterizes a user-uploaded GeoDataFrame of fuel polygons. **Required input columns the user must supply:** `fuel_type`, `fuel_loading` (kg/m²), `fuel_height` (m), `percent_cover` (0–100), and `distribution`. Distribution modes are `homogeneous`, `uniform_random`, and `random_clusters` (circular patches of diameter `patch_size`, optional `patch_std_dev`). Output is a multi-band surface grid (loading, height, live/dead moisture). The logic follows the CAPSIS/StandFire pipeline.

This is purely a **container** — FastFuels supplies none of the loading, cover, patch size, or moisture. The challenge's required per-voxel properties map almost 1:1 onto these layerset columns, which makes a measured product a clean, drop-in delivery (and aligns directly with submission pathway 2c).

### v2 API/SDK structure, output formats, and resolution

- **Architecture:** a REST API (domains → features → inventories → grids → exports) with a Python client ([FastFuels SDK](https://silvxlabs.github.io/fastfuels-sdk-python/)). 64 endpoints in the v2 schema.
- **Grid storage:** chunked, band-addressable arrays in a **zarr-style** band/chunk store (`GET /grids/{id}/data/{band}/{chunk_index}`); netCDF and GeoTIFF can be uploaded (`/grids/upload/netcdf`, `/upload/geotiff`) and grids exported in those formats. `/grids/compose`, `/grids/resample`, and `/grids/uniform` exist for derivation.
- **QUIC-Fire export:** `POST /grids/exports/quicfire` writes a zip containing `treesrhof.dat` (bulk density), `treesmoist.dat`, `treesfueldepth.dat`, `metadata.json`, `domain.geojson` (always), plus `topo.dat` and `treesss.dat` (SAVR) when the relevant roles are present. **Surface and canopy are produced as separate grids and only merged at export** — the surface remains a flat 2D layer extruded into the grid.
- **Resolution:** canopy default **2 m × 2 m × 1 m**; surface inherited from the **30 m** LANDFIRE source then resampled.

---

## The Gap We Exploit

The scientific argument is precise and well-supported.

**1. FastFuels' surface fuels are MODELED, not MEASURED.** As documented above, the default surface layer is a static lookup from a 30 m LANDFIRE FBFM40/FCCS class to fixed Scott & Burgan parameters. The bulk density, loading, depth, and SAVR of a 30 m pixel are determined entirely by *which of 46 categories it falls into*, not by any observation of the fuel actually present. FastFuels' authors say so directly, flagging the uniform-surface assumption as a known limitation ([Marcozzi et al. 2025](https://www.fs.usda.gov/rm/pubs_journals/2025/rmrs_2025_marcozzi_a001.pdf)).

**2. The surface layer is spatially uniform within each fuel-model class.** Every pixel of a given SB40 code gets identical parameters. Yet within-stand surface-fuel variance is large even within one treatment class: Vakili et al. (2016) documented high spatial variability of surface fuel loads in treated and untreated ponderosa pine — the very paper FastFuels cites as the reason its uniform approach is a limitation ([IJWF 25(11):1156–1168](https://www.publish.csiro.au/wf/WF15139)).

**3. Real surface fuel mass is highly heterogeneous at SUB-METER scale.** The foundational longleaf-pine/wiregrass work from the Tall Timbers / USFS Center for Forest Disturbance Science group establishes this quantitatively:

- The **"wildland fuel cell"** concept (Hiers, O'Brien, Mitchell, Grego & Loudermilk, 2009): using ground-based LiDAR plus inventory across 0.1–10 m sampling scales, within-fuelbed fuel composition and architecture form discrete patches that become **spatially independent beyond ~0.5 m** — and fire temperatures/residence times varied at the same sub-meter scale ([IJWF 18(3):315–325](https://www.publish.csiro.au/wf/WF08084)).
- Loudermilk et al. (2012) showed fire behavior and effects are spatially correlated with vegetation structure at **grain finer than 0.25 m²** ([USFS Treesearch 46764](https://research.fs.usda.gov/treesearch/46764)).
- Rowell et al. (2020, 2021) demonstrated the measured alternative: TLS-derived **occupied voxel volume** predicts fine-scale surface biomass and bulk density across grass, litter, and shrub types at <0.25 m² grain in longleaf pine ([Fire 4(3):36](https://doi.org/10.3390/fire4030036); [FE&M 2020](https://www.sciencedirect.com/science/article/abs/pii/S0378112719320523)).

A single value per 30 m fuel class cannot represent variability whose correlation length is sub-meter.

**4. Physics-based fire models are nonlinearly sensitive to exactly this variability.** QUIC-Fire, FIRETEC, and FDS take **per-voxel bulk density as a direct input**, and rate of spread is a power-law function of bulk density in both lab burns and FIRETEC simulations; QUIC-Fire's 3D fuel structure modifies local winds and fire behavior ([Linn et al. 2020, EMS](https://www.sciencedirect.com/science/article/abs/pii/S1364815219307388)). Feeding these models spatially uniform, class-averaged bulk density therefore discards precisely the heterogeneity they were designed to resolve, degrading predicted spread, consumption, and downwind behavior.

> *Note on uncertainty:* the longleaf/wiregrass evidence is strongest in southeastern US frequently burned ecosystems; the magnitude of sub-meter heterogeneity (and thus the size of our improvement) will differ in other fuel types, which is why we plan cross-ecosystem validation rather than claiming a single universal gain.

### Thesis statement

**FastFuels paints surface fuel as one number per 30 m fuel-model class, while the literature shows real surface fuel mass, bulk density, and architecture reorganize at sub-meter scales that physics-based fire models consume nonlinearly. We propose a *measured*, spatially heterogeneous surface-fuel product that fuses canopy-penetrating SAR with near-ground LiDAR returns, calibrated to destructively sampled and TLS-derived ground truth, and delivered as a FastFuels-compatible 1 m voxelized netCDF (bulk density) or layerset GeoJSON. It is simple to explain — *radar sees under the canopy where LiDAR is blind, and LiDAR sharpens the surface where the canopy is open* — uses only free/public data (Sentinel-1, NISAR, USGS 3DEP, NEON, plus open RxCADRE/SERDP 3D-Fuels truth), and is validated head-to-head against FastFuels' own uniform surface layer, attacking its weakest link on the single most heavily weighted judging axis.**

---

## Remote Sensing of Surface Fuels: LiDAR

LiDAR is the de-facto measurement basis for 3D wildland fuels, but its usefulness for the *surface/understory* stratum is strongly stratified by platform. The core tension is between **spatial coverage** (spaceborne > airborne > UAV > terrestrial) and **near-ground fidelity** (terrestrial > UAV > airborne > spaceborne). The challenge's 1 m³ voxel target sits at the resolution where airborne and terrestrial methods overlap.

### Platform capabilities and resolution

| Platform | Sensor / source | Point density | Footprint / resolution | Surface-fuel utility |
|---|---|---|---|---|
| Spaceborne (full-waveform) | GEDI (ISS) | n/a (waveform) | ~25 m footprint; ~60 m along-track, ~600 m across-track | Poor for surface stratum (R² 0.15–0.46); good for woody/total load and scaling |
| Spaceborne (photon) | ICESat-2 ATL08 | n/a | ~100 m segment canopy metrics | Canopy/terrain, scaling layer only |
| Airborne (ALS) | USGS 3DEP, NEON AOP | ~2–8 pts/m² (QL1/QL2); up to ~10–65 pts/m² in research flights | 1 m grid products | Wall-to-wall; weak near ground (occlusion) |
| UAV | drone LiDAR / SfM | hundreds–thousands pts/m² | sub-meter | Strong upper-canopy; 20–80% gaps in lower canopy (5–13 m) |
| Handheld / backpack (MLS) | SLAM scanners | thousands pts/m² | cm-scale | Best understory capture (~20% gap below 7 m) |
| Terrestrial (TLS) | tripod scanners | thousands pts/m² | mm–cm | "Truth" for 3D surface structure |

Sources: [GEDI / Leite et al. 2022](https://www.fs.usda.gov/rm/pubs_journals/2022/rmrs_2022_leite_r001.pdf); [NEON Discrete LiDAR DP1.30003.001](https://data.neonscience.org/data-products/DP1.30003.001); [Martin-Ducup et al. 2025 (lidarforfuel)](https://hal.inrae.fr/hal-04908881v1); [UAV+handheld fusion study (Sensors)](https://pmc.ncbi.nlm.nih.gov/articles/PMC12845990/).

A 2025/2026 voxel-fusion study quantified the occlusion trade-off: handheld scanners retained roughly a 20% coverage gap below 7 m (good understory), UAV-LiDAR had 20–80% gaps in the lower canopy (5–13 m) but good upper canopy, and **fused UAV + handheld achieved a <10% gap ratio above 3 m** ([PMC12845990](https://pmc.ncbi.nlm.nih.gov/articles/PMC12845990/)). That same study found **50–100 cm voxels optimal** for balancing structural detail against statistical stability (tested 20/50/100/200 cm).

### From point cloud to voxel bulk density / PAD

Two established conversion families exist:

1. **Voxel occupancy → bulk density (TLS / mobile).** The point cloud is voxelized (typically 1 cm = 0.001 m³, or 2 cm), each voxel flagged occupied/empty, and the **occupied-volume fraction** (occupied voxels / total possible) or a **porosity index** is calibrated to destructively harvested dry mass. Rowell et al. (2020) found occupied-voxel density related to destructive 3D bulk density by a logarithmic fit (adj. R² = 0.32 for sparkleberry shrub), with stronger relationships for clip-plot mass vs. occupied volume/porosity/surface area across grasses and shrubs ([Rowell et al. 2020, FEM](https://research.fs.usda.gov/treesearch/59822)). The SERDP RC19-1064 "3D Fuels" project formalized this: a porosity index estimated per **10×10×10 cm voxel** is "directly related to the bulk density of vegetation and scaled to gridded inputs to QUICFire, FIRETEC and FDS," using a hierarchical sampling frame of whole plot (0.25 m³), stratum (0.025 m³), and voxel (0.001 m³) ([SERDP RC1064 Final Report, Aug 2024](https://depts.washington.edu/flame/docs/SERDP_RC1064_Final_Report_Aug2024.pdf)).

2. **Beer-Lambert PAD inversion → bulk-density profile (ALS).** Airborne discrete-return clouds are inverted to a vertical **Plant Area Density (PAD)** profile using the Beer-Lambert light-extinction law, then PAD is converted to bulk density using species-specific plant traits (PAD-to-mass ratios) and integrated to layered load (kg/m²), CBH, and CBD. Martin-Ducup et al. (2025) validated this in France/Spain/Portugal: for simply-stratified structure, **R² = 0.42 (canopy fuel load), 0.68 (canopy bulk density), 0.60 (CBH)**, bias −2% to −5%; strata above 0.5 m gave R² 0.39–0.70. The method is released as the open R package `lidarforfuel` ([Martin-Ducup et al. 2025](https://hal.inrae.fr/hal-04908881v1)).

A documented bias in Boolean-occupancy voxelization is **stem oversampling**, which inflates apparent occupied volume ([Marcozzi et al. 2023 / voxelization methods, RS](https://www.mdpi.com/2072-4292/17/3/552)).

### The understory occlusion problem

This is the central limitation for **every** LiDAR platform that looks down from above. Because the canopy intercepts pulses, near-ground returns thin out and surface fuels must be inferred indirectly. Concrete evidence:

- **Airborne LiDAR predicts litter/duff only via *overstory correlates*.** The Kaibab Plateau study reached only **R² = 0.59 (litter+duff), 0.48 (total surface), 0.39 (1–1000 h woody)** at 20 m resolution, with surface fuel (mean 82.7 Mg/ha) ~12× larger than canopy fuel (6.9 Mg/ha) — so surface error dominates fire-model error ([PMC9361251](https://pmc.ncbi.nlm.nih.gov/articles/PMC9361251/)).
- **Adding optical does not solve it.** LiDAR + Sentinel-2 fusion explained only **R² = 0.27–0.41** for litter/fine dead woody and effectively zero for coarse woody ([Remote Sensing of Environment 2023](https://www.sciencedirect.com/science/article/abs/pii/S0034425723002626)).
- **Spaceborne GEDI** mapped woody (R² 0.88) and total load (R² 0.71) but only **R² 0.15–0.46** for the surface/herbaceous/shrub strata ([Leite et al. 2022](https://www.fs.usda.gov/rm/pubs_journals/2022/rmrs_2022_leite_r001.pdf)).

This occlusion ceiling is precisely the gap that motivates either (a) below-canopy/TLS measurement, or (b) an independent penetrating sensor such as SAR (next section).

### Calibration to fuel load via destructive sampling

LiDAR metrics are not fuel mass; they must be regressed against ground truth. The accepted hierarchy: **destructive clip-plot harvest** is the gold standard for loading (kg/m²) and live/dead fraction, while **TLS voxel grids** are the reference for 3D structure. The strongest precedent links the two: TLS occupied volume (2 cm voxels, clipped to 15 m radius) explained **~85% of variance in destructively weighed surface-fuel mass at ~16% relative RMSE** — one of the first direct 3D-field-vs-3D-TLS comparisons ([Rowell et al. 2020, FEM](https://www.sciencedirect.com/science/article/abs/pii/S0378112719320523)). Because a 1 m² clip plot does not map cleanly onto cm-scale voxels, modern protocols use **3D stratified destructive sampling** — e.g. FERA's 0.5×0.5 m columns segmented into 10 cm vertical strata to 1–2 m, weighed by component, yielding layer-resolved bulk density ([FERA 3D Fuels Project](https://depts.washington.edu/fera/3dfuels/)). Peer-reviewed work confirms TLS voxel metrics linearly map to fine-fuel mass in SE-US longleaf systems (162 voxel metrics regressed on layered clip-plot mass at Fort Stewart; [bioRxiv 2023](https://www.biorxiv.org/content/10.1101/2023.01.15.524107v1.full)).

---

## Remote Sensing of Surface Fuels: SAR

Synthetic Aperture Radar is physically attractive for fuels because backscatter responds to vegetation **biomass, structure, and moisture (dielectric)**, and — critically — longer wavelengths **penetrate the canopy** to sense the understory and ground where LiDAR goes blind. SAR is also free (Sentinel-1, NISAR), global, all-weather, and day/night with frequent revisit. The honest limitation is that spaceborne SAR is coarse (3–20 m) and saturates at low biomass, so it is best framed as a **continuous structure/moisture proxy fused with LiDAR/field data**, not a standalone 1 m fuel sensor.

### Wavelength, penetration, and band selection

| Band | Wavelength | Frequency | Penetration | Fuel relevance |
|---|---|---|---|---|
| X | ~3 cm | ~9.6 GHz | Canopy tops only | Commercial sub-meter (Capella/ICEYE 25 cm, Umbra 16 cm); poor under-canopy fit |
| C | ~5.5 cm | 5.405 GHz | ~1 cm soil under sparse cover; upper canopy in dense veg | Sentinel-1 (free); moisture-confounded for grass |
| S | ~9.4 cm | 3.20 GHz | Light vegetation | NISAR S-band |
| L | ~24 cm | 1.25 GHz | Meters in dry media; reaches understory/shrubs | NISAR, ALOS-2; workhorse for understory biomass |
| P | ~70 cm | 435 MHz | Trunks/large branches (~75% of tree biomass) | ESA BIOMASS; 3D tomography, but coarse products |

Penetration scales with wavelength: C-band penetrates only ~1 cm of soil under low vegetation and is dominated by the upper canopy in dense stands; L-band penetration of ~2.84–2.98 m has been measured in extremely arid media; P-band reaches main trunks ([L-band review, PMC11397620](https://pmc.ncbi.nlm.nih.gov/articles/PMC11397620/); [ESA BIOMASS](https://earth.esa.int/eogateway/missions/biomass/description)). **This is exactly why L/P-band sense the understory better than C-band**, and why the commercial sub-meter X-band sensors — despite 16–25 cm resolution — image only canopy/ground geometry, not fuel beneath cover ([Umbra remote sensing](https://umbra.space/remote-sensing/)).

### Backscatter–biomass relationships and saturation

Cross-polarized **L-band HV** is the single most informative SAR channel for understory/shrub structure, more sensitive than HH; one multi-frequency study fit shrubland biomass at **r² = 0.572 with L-band** vs. lower for C-band, attributed to deeper shrub penetration ([PMC11397620](https://pmc.ncbi.nlm.nih.gov/articles/PMC11397620/)). The core limitation is **saturation**: the L-band HH backscatter–AGB curve saturates around **~105 Mg/ha** (slope falling to ~0.01 dB per Mg/ha). For surface fuels this saturation ceiling is far *above* typical loads — surface/understory fuels are usually **<1 kg/m² (≈10 Mg/ha)**, so the real problem is that fuels sit at the **low, noisy end** of the curve where soil moisture and roughness dominate. SAR therefore resolves coarse biomass gradients, not fine 1 m fuel mass.

A 2024 ALOS-2 PALSAR-2 study provides the most direct physical basis for SAR as an occlusion-filler: the **HV/HH polarization ratio was significantly *negatively* correlated with understory coverage** once canopy cover dropped below a threshold (stands ≥10 m / >2 yr), because reduced canopy cover increases microwave penetration to the understory ([GIScience & Remote Sensing 2024](https://www.tandfonline.com/doi/abs/10.1080/15481603.2024.2360771)). Conversely, for grass/rangeland, Sentinel-1 C-band alone is generally insufficient: in a rangeland AGB study (0.21–0.47 kg/m²) the VV/VH variables were **excluded from all final models** due to moisture and phenology confounding, with the best (optical-driven) RF reaching only R² = 0.517, RMSE 0.069 kg/m² ([PMC10682297](https://pmc.ncbi.nlm.nih.gov/articles/PMC10682297/)).

### Polarimetry, InSAR, and tomography

Beyond backscatter regression, three structural techniques exist:

- **Polarimetric decomposition** (e.g. Freeman-Durden surface/double-bounce/volume) can separate ground vs. canopy scattering to isolate the understory contribution.
- **PolInSAR / InSAR coherence** inverts vegetation height from volume decorrelation, but spaceborne height RMSE is **several meters** (reported PolInSAR RMSE 1.02–4.21 m; TomoSAR height RMSE 3.02 m P-band, 3.68 m L-band) — adequate for canopy, far too coarse for cm-scale grass/litter depth ([PMC11397620](https://pmc.ncbi.nlm.nih.gov/articles/PMC11397620/); [Scientific Reports TomoSAR](https://www.nature.com/articles/s41598-023-33311-y)).
- **SAR tomography (TomoSAR)** reconstructs a true 3D vertical reflectivity profile from a multi-baseline stack — conceptually the radar analog of the challenge's voxel goal — but spaceborne BIOMASS products are only 50–200 m, and meter-scale tomography requires costly airborne campaigns ([airborne TomoSAR, Nature Sci Rep](https://www.nature.com/articles/s41598-023-33311-y)).

### Current and new spaceborne missions

- **Sentinel-1** (C-band, free): GRD dual-pol VV+VH at ~10 m pixel / 20 m resolution, SLC for InSAR; 2014–present, ~6–12 day revisit ([ASF datasets](https://docs.asf.alaska.edu/datasets/using_ASF_data/)).
- **NISAR** (launched 30 July 2025, free): dual-frequency **L-band (24 cm) + S-band (9.4 cm)** polarimetric + InSAR, global ~12-day repeat; native imaging modes reach ~3–10 m (early soil-moisture products gridded at 100 m) ([eoPortal NISAR](https://www.eoportal.org/satellite-missions/nisar)). This is the most relevant *new free* source for understory fuels.
- **ESA BIOMASS** (launched April 2025): first spaceborne **P-band** (70 cm) mission with PolSAR + Pol-InSAR + TomoSAR; products AGB & height at **200 m**, disturbance at 50 m — too coarse for 1 m voxels but valuable canopy/structure context ([eoPortal BIOMASS](https://www.eoportal.org/satellite-missions/biomass)).

### Honest meter-scale limits

SAR **cannot** directly deliver 1 m voxel surface fuel load, live/dead fraction, or SAV. Its defensible roles for this challenge are: (1) a **moisture/curing proxy** (backscatter responds to vegetation water content — arguably its most unique contribution); (2) an **L-band HV understory-density proxy** where canopy is moderate; and (3) **texture/coherence heterogeneity** metrics. Output resolution is capped at several meters unless downscaled by co-registered LiDAR. The recommended architecture across the literature is therefore **SAR-as-covariate in a LiDAR/field-anchored ML model** — using SAR's frequent revisit and penetration to extrapolate between sparse LiDAR/field samples ([Wildfire fuels AI review 2025](https://www.sciencedirect.com/science/article/pii/S001282522500025X)); [Assessment of fuel availability with L-band PolSAR](https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2023EA002943)).

---

## Converting Remote Sensing to Required Fuel Properties

The challenge requires per-voxel/cell **fuel loading (kg/m²), bulk density (kg/m³), depth/height (m), live-vs-dead fraction (%), surface-area-to-volume ratio / SAVR (m²/m³)**, and ideally species and fuel moisture (%). No single sensor yields all of these directly; the practical recipe combines measured remote-sensing observables with calibration data and **SB40 parameter priors**.

### Fuel loading (kg/m²)
Derived by **calibrating a LiDAR/SAR predictor to destructively-sampled mass**. Concretely: regress per-cell LiDAR occupied-voxel fraction / understory return density / PAD (and SAR HV backscatter or VH/VV ratio as a covariate) against clip-plot dry mass. Expect achievable accuracy in the **R² 0.4–0.85** range depending on platform and canopy occlusion (TLS ~0.85 / 16% rRMSE; ALS surface strata ~0.4; LiDAR+optical 0.27–0.41). Use spatial cross-validation and report RMSE/bias against held-out plots.

### Bulk density (kg/m³)
Two routes: (1) **directly** from the TLS/ALS porosity→bulk-density calibration (SERDP 10 cm voxel porosity index scaled to QUIC-Fire/FIRETEC/FDS; [SERDP RC1064](https://depts.washington.edu/flame/docs/SERDP_RC1064_Final_Report_Aug2024.pdf)); or (2) **derived** as bulk density = loading / depth once loading and depth are estimated separately. Route (2) is simpler and more explainable for an MVP.

### Depth / height (m)
From **LiDAR near-ground height percentiles** (e.g. 95th-percentile return height in the 0–2 m stratum) or fuelbed-top minus ground. SAR PolInSAR height is too coarse (several-meter RMSE) for surface depth. Where measured depth is unavailable, fall back to **SB40 fuel-bed depth priors** (see table).

### SAVR (m²/m³)
This is the hardest to measure remotely; **neither LiDAR nor SAR delivers SAVR directly.** The defensible approach is to **assign SAVR by fuel type / species prior** from standard tables. FastFuels itself assigns SAVR uniformly (e.g. 8000 m⁻¹ litter, 9770 m⁻¹ grass; [Marcozzi et al. 2025](https://www.fs.usda.gov/rm/pubs_journals/2025/rmrs_2025_marcozzi_a001.pdf)), and the SB40 lookup tables provide SAVR by size class (1/10/100 hr + live herb/woody). Using these priors is standard practice and honest.

### Live/dead fraction (%)
Best from **field records or hyperspectral** (NEON DP3.30006 supports live/dead and species). SAR contributes a defensible **moisture/curing proxy** via seasonal VV/VH dynamics or L-band VOD, since backscatter tracks vegetation water content — arguably SAR's most unique contribution. Otherwise apply SB40 live-herb/live-woody splits, noting that SB40's herbaceous load is dynamic with curing.

### Species and fuel moisture (%)
Species from NEON vegetation-structure / hyperspectral or field plots; moisture from field samples plus the SAR seasonal proxy above. Neither is reliably retrievable at 1 m from SAR/LiDAR alone — be explicit about this.

### SB40 parameter priors (the baseline to beat)
FastFuels' default surface layer pulls LANDFIRE FBFM40 (46 SB40 codes) at 30 m and applies a **static per-class lookup**, assigning identical loading (1/10/100 hr dead, live herb/woody, kg/m²), SAVR by size class (1/m), and fuel-bed depth (m) to every pixel of a class ([FastFuels OpenAPI](https://api-v2-prod-782971006568.us-west1.run.app/openapi.json); [LANDFIRE FBFM40](https://landfire.gov/fuel/fbfm40)). These same tables are the correct **priors** to supply SAVR and to seed depth/load where measurements are missing — and the **benchmark** the proposal must beat by injecting measured spatial heterogeneity. The documented FastFuels FDS example uses Anderson FM9 (litter): **loading 0.717 kg/m², fuel-bed depth 6.1 cm** ([Marcozzi et al. 2025](https://www.fs.usda.gov/rm/pubs_journals/2025/rmrs_2025_marcozzi_a001.pdf)). The 46 SB40 codes span NB 91–99, GR 101–109, GS 121–124, SH 141–149, TU 161–165, TL 181–189, SB 201–204 ([Scott & Burgan 2005, via LANDFIRE](https://landfire.gov/fuel/fbfm40)).

### Delivery into FastFuels
Two FastFuels-compatible pathways match these outputs almost 1:1: (2) a **3D voxelized 1 m netCDF bulk-density array** (extrude loading/depth into 1 m vertical bins), or (3) a **statistical "layerset" GeoJSON** whose required columns — `fuel_type`, `fuel_loading` (kg/m²), `fuel_height` (m), `percent_cover`, `distribution` (homogeneous / uniform_random / random_clusters with `patch_size`, `patch_std_dev`), plus optional live/dead moisture — map directly to the measured patch metrics ([fastfuels-core layersets.py](https://github.com/silvxlabs/fastfuels-core/blob/main/fastfuels_core/layersets.py)).

---

## Open Datasets & Quick-Start Site

All datasets below are free and have working programmatic access as of June 2026.

| Dataset | Type / role | Access method | Resolution / detail |
|---|---|---|---|
| NEON Discrete LiDAR `DP1.30003.001` | Airborne ALS input | `neonutilities.by_tile_aop()` (Python) / `byTileAOP()` (R); NEON API | 1×1 km tiles, ~4–8 pts/m² |
| NEON CHM `DP3.30015.001` | Canopy mask / structure | `by_tile_aop()` | 1 m raster |
| NEON Herbaceous Clip Harvest `DP1.10023.001` | Destructive load truth | `load_by_product()` | 0.5×3 m clip cells → kg/m² |
| NEON Litterfall & fine woody `DP1.10033.001` | Litter truth | `load_by_product()` | plot-level |
| NEON Coarse Downed Wood `DP1.10014.001` | CWD bulk density truth | `load_by_product()` | plot-level |
| NEON Hyperspectral `DP3.30006` | Species / live-dead | AOP download | ~1 m, 426 bands |
| USGS 3DEP (via OpenTopography) | Airborne ALS input | PDAL `readers.ept` / OpenTopography REST; `OT_3DEP_Workflows` | EPT tiles, ~2–8 pts/m²; 1/10/30 m DEM |
| Sentinel-1 GRD/SLC | C-band SAR input | `asf_search` (Python), Earthdata Login; Vertex GUI | ~10 m pixel / 20 m res, VV+VH, 6–12 day |
| GEDI `GEDI02_A/B.002` | Spaceborne LiDAR cross-check | `earthaccess`, Earthdata Login | ~25 m footprints |
| LANDFIRE FBFM40 (`240FBFM40`) | Modeled baseline to beat | LFPS REST API (`lfps.usgs.gov`) | 30 m raster |
| RxCADRE 2012 TLS (`RDS-2023-0011`) | TLS input (validated pair) | USFS Research Data Archive (no login) | LAZ, mm–cm |
| RxCADRE ground fuels (`RDS-2014-0031`) | Destructive load/moisture truth | USFS Research Data Archive | clip + line-intercept, Mg/ha |
| SERDP 3D Fuels (RC19-1064) | TLS + 3D bulk-density truth | FLAME / project report | 10 cm voxel, 18 sites |

Sources: [NEON DP1.30003.001](https://data.neonscience.org/data-products/DP1.30003.001); [neonutilities Python](https://github.com/NEONScience/NEON-utilities-python); [OT_3DEP_Workflows](https://github.com/OpenTopography/OT_3DEP_Workflows); [ASF asf_search](https://docs.asf.alaska.edu/datasets/using_ASF_data/); [GEDI-Data-Resources](https://github.com/nasa/GEDI-Data-Resources); [LFPS User Guide](https://lfps.usgs.gov/LFProductsServiceUserGuide.pdf); [RxCADRE TLS RDS-2023-0011](https://www.fs.usda.gov/rds/archive/catalog/RDS-2023-0011); [RxCADRE ground fuels RDS-2014-0031](https://www.fs.usda.gov/rds/archive/Catalog/RDS-2014-0031); [SERDP RC1064](https://depts.washington.edu/flame/docs/SERDP_RC1064_Final_Report_Aug2024.pdf).

### Recommended starter site & dataset combo

There are two strong starting points, chosen by whether you weight **validation** or **scalability** more heavily.

**Strongest-validation path — RxCADRE 2012, Eglin AFB, FL (longleaf pine, grass/shrub).** The decisive advantage is that the **LiDAR input and the destructive ground truth are co-located in the same plots**, both public-domain with DOIs. The three Highly Instrumented Plots — L1G (grass), L2G (grass/shrub), L2F (forested) — have TLS LAZ clouds (`RDS-2023-0011`) and matching pre/post-fire clip-plot loading, consumption, and moisture by fuelbed component (`RDS-2014-0031`; forest units averaged 6.8 Mg/ha load, non-forest 3.0 Mg/ha).

Download steps:
1. Pull the TLS LAZ clouds from the [USFS Research Data Archive (RDS-2023-0011)](https://www.fs.usda.gov/rds/archive/catalog/RDS-2023-0011) — no login.
2. Pull the ground-fuel measurements from [RDS-2014-0031](https://www.fs.usda.gov/rds/archive/Catalog/RDS-2014-0031).
3. In PDAL: height-normalize, voxelize at 1 m (and 0.25 m for the surface layer), compute per-voxel occupied-volume / return-density.
4. Fit a simple log/linear regression mapping voxel metrics → measured loading (kg/m²); derive bulk density and depth; report R²/RMSE (target the ~16% rRMSE TLS benchmark).
5. Export a FastFuels-compatible 1 m netCDF bulk-density array; overlay LANDFIRE FBFM40 (`240FBFM40` via LFPS) to quantify divergence from the modeled baseline.

**Best-scalability path — NEON OSBS (Ordway-Swisher), FL (longleaf-pine / turkey-oak savanna).** Low structural complexity, wall-to-wall 1×1 km airborne LiDAR tiles, and co-located field plots, all one-line downloadable:

```python
import neonutilities as nu
# Airborne LiDAR input
nu.by_tile_aop(dpid='DP1.30003.001', site='OSBS', year='2023',
               easting=[...], northing=[...], savepath='./')
# Canopy height model (canopy mask)
nu.by_tile_aop(dpid='DP3.30015.001', site='OSBS', year='2023', ...)
# Destructive truth
nu.load_by_product(dpid='DP1.10023.001', site='OSBS',
                   startdate='2018-01', enddate='2023-12')  # herbaceous clip
nu.load_by_product(dpid='DP1.10033.001', site='OSBS', ...)  # litter/fine woody
nu.load_by_product(dpid='DP1.10014.001', site='OSBS', ...)  # coarse downed wood
```

Then add free Sentinel-1 GRD (VV/VH) over the same footprint via `asf_search` (Earthdata Login) as a stretch SAR-fusion/temporal-moisture layer, and benchmark against LANDFIRE FBFM40. Note the temporal offset between flight year and clip-plot dates should be documented. ([OSBS site](https://www.neonscience.org/field-sites/osbs); [neonutilities](https://github.com/NEONScience/NEON-utilities-python); [ASF](https://docs.asf.alaska.edu/datasets/using_ASF_data/).)

---

## References

- [FastFuels API v2 OpenAPI schema](https://api-v2-prod-782971006568.us-west1.run.app/openapi.json)
- [silvxlabs/fastfuels-core — layersets.py](https://github.com/silvxlabs/fastfuels-core/blob/main/fastfuels_core/layersets.py)
- [silvxlabs/fastfuels-core — voxelization.py](https://github.com/silvxlabs/fastfuels-core/blob/main/fastfuels_core/voxelization.py)
- [FastFuels SDK (Python) documentation](https://silvxlabs.github.io/fastfuels-sdk-python/)
- [Marcozzi et al. 2025 — FastFuels (RMRS open-access PDF)](https://www.fs.usda.gov/rm/pubs_journals/2025/rmrs_2025_marcozzi_a001.pdf)
- [Mell, Silva et al. 2024 — FastFuels (Environmental Modelling & Software)](https://www.sciencedirect.com/science/article/pii/S1364815224002755)
- [FastFuels (USFS Treesearch 68657)](https://research.fs.usda.gov/treesearch/68657)
- [USFS Missoula Fire Lab — FastFuels project page](https://research.fs.usda.gov/firelab/projects/fastfuels)
- [LANDFIRE — 40 Scott & Burgan Fire Behavior Fuel Models (FBFM40)](https://landfire.gov/fuel/fbfm40)
- [LANDFIRE Product Service (LFPS) User Guide](https://lfps.usgs.gov/LFProductsServiceUserGuide.pdf)
- [Hiers et al. 2009 — Wildland fuel cell concept (IJWF)](https://www.publish.csiro.au/wf/WF08084)
- [Loudermilk et al. 2012 — Linking complex forest fuel structure and fire behavior at fine scales](https://research.fs.usda.gov/treesearch/46764)
- [Rowell et al. 2021 — Non-destructive fuel volume measurements (Fire 4(3):36)](https://doi.org/10.3390/fire4030036)
- [Rowell et al. 2020 — Coupling TLS with 3D fuel biomass sampling (FEM, ScienceDirect)](https://www.sciencedirect.com/science/article/abs/pii/S0378112719320523)
- [Rowell et al. 2020 — Coupling TLS with 3D fuel biomass sampling (USFS Treesearch 59822)](https://research.fs.usda.gov/treesearch/59822)
- [Linn et al. 2020 — QUIC-Fire (Environmental Modelling & Software)](https://www.sciencedirect.com/science/article/abs/pii/S1364815219307388)
- [Vakili et al. 2016 — Spatial variability of surface fuels (IJWF)](https://www.publish.csiro.au/wf/WF15139)
- [SERDP RC19-1064 — 3D Fuels Final Report (Aug 2024)](https://depts.washington.edu/flame/docs/SERDP_RC1064_Final_Report_Aug2024.pdf)
- [FERA 3D Fuels Project](https://depts.washington.edu/fera/3dfuels/)
- [Martin-Ducup et al. 2025 — ALS bulk density / lidarforfuel (HAL)](https://hal.inrae.fr/hal-04908881v1)
- [Leite et al. 2022 — GEDI multi-layer fuel load in tropical savanna (RSE)](https://www.fs.usda.gov/rm/pubs_journals/2022/rmrs_2022_leite_r001.pdf)
- [UAV + handheld LiDAR voxel point-density proxy (Sensors, PMC12845990)](https://pmc.ncbi.nlm.nih.gov/articles/PMC12845990/)
- [Voxelizing discrete LiDAR for fire models (Remote Sensing 17(3):552)](https://www.mdpi.com/2072-4292/17/3/552)
- [L-band SAR for forest parameter estimation, 1972–2024 review (PMC11397620)](https://pmc.ncbi.nlm.nih.gov/articles/PMC11397620/)
- [Seasonal rangeland AGB with Sentinel-1 & Sentinel-2 (PMC10682297)](https://pmc.ncbi.nlm.nih.gov/articles/PMC10682297/)
- [NISAR mission (eoPortal)](https://www.eoportal.org/satellite-missions/nisar)
- [ESA BIOMASS mission (eoPortal)](https://www.eoportal.org/satellite-missions/biomass)
- [ESA BIOMASS mission description (Earth Online)](https://earth.esa.int/eogateway/missions/biomass/description)
- [Assessment of pre/post-fire fuel availability with L-band PolSAR (AGU)](https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2023EA002943)
- [Wildfire fuels mapping via AI methods review (ScienceDirect 2025)](https://www.sciencedirect.com/science/article/pii/S001282522500025X)
- [Umbra remote sensing / X-band SAR comparison](https://umbra.space/remote-sensing/)
- [Airborne SAR tomography for tropical AGB (Scientific Reports)](https://www.nature.com/articles/s41598-023-33311-y)
- [Multitemporal lidar fuel loads, Kaibab Plateau (PMC9361251)](https://pmc.ncbi.nlm.nih.gov/articles/PMC9361251/)
- [Surface fuels from lidar + Sentinel-2 in temperate forests (RSE 2023)](https://www.sciencedirect.com/science/article/abs/pii/S0034425723002626)
- [L-band SAR backscatter vs understory weed density (GIScience & RS 2024)](https://www.tandfonline.com/doi/abs/10.1080/15481603.2024.2360771)
- [Scalable GEDI + SAR fusion for global forest structure (arXiv 2510.06299)](https://arxiv.org/abs/2510.06299)
- [ForestGen3D — diffusion sub-canopy generation (arXiv 2509.16346)](https://arxiv.org/abs/2509.16346)
- [TLS metrics predict surface biomass/consumption in SE US (bioRxiv 2023)](https://www.biorxiv.org/content/10.1101/2023.01.15.524107v1.full)
- [NEON Discrete LiDAR DP1.30003.001](https://data.neonscience.org/data-products/DP1.30003.001)
- [NEON Herbaceous clip harvest DP1.10023.001](https://data.neonscience.org/data-products/DP1.10023.001)
- [NEON LiDAR data collection overview](https://www.neonscience.org/data-collection/lidar)
- [NEON OSBS field site](https://www.neonscience.org/field-sites/osbs)
- [NEON-utilities-python (neonutilities)](https://github.com/NEONScience/NEON-utilities-python)
- [ASF — using ASF data / asf_search](https://docs.asf.alaska.edu/datasets/using_ASF_data/)
- [NASA GEDI-Data-Resources (earthaccess)](https://github.com/nasa/GEDI-Data-Resources)
- [GEDI L4A Footprint AGB Density (ORNL DAAC)](https://daac.ornl.gov/GEDI/guides/GEDI_L4A_AGB_Density.html)
- [OpenTopography OT_3DEP_Workflows](https://github.com/OpenTopography/OT_3DEP_Workflows)
- [USGS 3D Elevation Program (3DEP)](https://www.usgs.gov/3d-elevation-program)
- [RxCADRE 2012 TLS point cloud (USFS RDS-2023-0011)](https://www.fs.usda.gov/rds/archive/catalog/RDS-2023-0011)
- [RxCADRE 2012 TLS (AgData Commons)](https://agdatacommons.nal.usda.gov/articles/dataset/RxCADRE_2012_Terrestrial_laser_scan_TLS_point_cloud_data_for_Eglin_Air_Force_Base/27010462)
- [RxCADRE ground fuel measurements (USFS RDS-2014-0031)](https://www.fs.usda.gov/rds/archive/Catalog/RDS-2014-0031)
- [RxCADRE fuel height models from TLS (IJWF)](https://www.publish.csiro.au/wf/wf14170)
- [Terrestrial 3D Laser Scanning for Ecosystem & Fire Effects Monitoring (USFS GTR 67919)](https://research.fs.usda.gov/treesearch/67919)
- [Non-destructive fuel volume measurements (Fire, MDPI 4(3):36)](https://www.mdpi.com/2571-6255/4/3/36)
