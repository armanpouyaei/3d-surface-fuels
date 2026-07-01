// Build the Phase-1 submission Word document (.docx) from the project results, with methodology
// diagrams, figures, and tables. Run: NODE_PATH=$(npm root -g) node scripts/build_report_docx.js
const fs = require("fs");
const path = require("path");
const { Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell, ImageRun,
        Footer, AlignmentType, LevelFormat, TableOfContents, HeadingLevel, BorderStyle,
        WidthType, ShadingType, PageNumber, PageBreak } = require("docx");

const ROOT = path.join(__dirname, "..");
const FIG = path.join(ROOT, "figures");
const PROC = path.join(ROOT, "data", "processed");
const CW = 9360; // content width (US Letter, 1" margins), DXA

// ---- helpers ----
const T = (t, o = {}) => new TextRun({ text: t, ...o });
const P = (runs, o = {}) => new Paragraph({ ...o, children: Array.isArray(runs) ? runs : [T(runs)] });
const H1 = (t) => new Paragraph({ heading: HeadingLevel.HEADING_1, children: [T(t)] });
const H2 = (t) => new Paragraph({ heading: HeadingLevel.HEADING_2, children: [T(t)] });
const bullet = (runs) => new Paragraph({ numbering: { reference: "b", level: 0 },
  children: Array.isArray(runs) ? runs : [T(runs)] });

function img(file, w, aspect, cap) {
  if (!fs.existsSync(file)) return [P(T(`[missing figure: ${path.basename(file)}]`, { italics: true, color: "AA0000" }))];
  return [
    new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 120, after: 40 },
      children: [new ImageRun({ type: "png", data: fs.readFileSync(file),
        transformation: { width: w, height: Math.round(w / aspect) },
        altText: { title: cap, description: cap, name: cap } })] }),
    new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 160 },
      children: [T(cap, { italics: true, size: 18, color: "555555" })] }),
  ];
}

function table(headers, rows, widths) {
  const total = widths.reduce((a, b) => a + b, 0);
  const bd = { style: BorderStyle.SINGLE, size: 1, color: "CCCCCC" };
  const borders = { top: bd, bottom: bd, left: bd, right: bd };
  const cell = (txt, i, head) => new TableCell({ borders, width: { size: widths[i], type: WidthType.DXA },
    shading: head ? { fill: "D5E8F0", type: ShadingType.CLEAR } : undefined,
    margins: { top: 60, bottom: 60, left: 100, right: 100 },
    children: [new Paragraph({ children: [T(String(txt), { bold: !!head, size: 18 })] })] });
  return new Table({ width: { size: total, type: WidthType.DXA }, columnWidths: widths,
    rows: [new TableRow({ tableHeader: true, children: headers.map((h, i) => cell(h, i, true)) }),
      ...rows.map((r) => new TableRow({ children: r.map((c, i) => cell(c, i, false)) }))] });
}
const spacer = () => P("", { spacing: { after: 80 } });

// ---- document body ----
const body = [];
// title block
body.push(new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 1400, after: 80 },
  children: [T("3D Surface Fuels & Vegetation Modeling", { bold: true, size: 44 })] }));
body.push(new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 200 },
  children: [T("Prize Challenge — Phase-1 Submission", { size: 30, color: "2c7fb8" })] }));
body.push(new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 120 },
  children: [T("A measured, field-calibrated 1 m surface-fuel product that beats FastFuels’ uniform layer — from free spaceborne data, anywhere.", { italics: true, size: 24 })] }));
body.push(new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 60 },
  children: [T("Phase-1 deadline 2026-07-20 · all results reproducible from open data & code", { size: 18, color: "555555" })] }));
body.push(new Paragraph({ children: [new PageBreak()] }));

// TOC
body.push(H1("Contents"));
body.push(new TableOfContents("Contents", { hyperlink: true, headingStyleRange: "1-2" }));
body.push(new Paragraph({ children: [new PageBreak()] }));

// 1. thesis
body.push(H1("1. The thesis, in one figure"));
body.push(P([T("FastFuels paints surface fuel as "), T("one number per 30 m LANDFIRE fuel-model class", { bold: true }),
  T(" — uniform within the class, by construction. We "), T("measure it at 1 m", { bold: true }),
  T(" and calibrate it to destructive ground truth. The same 510 m OSBS scene, two products:")]));
body.push(...img(path.join(FIG, "fig_vs_fastfuels.png"), 600, 1.786,
  "Figure 1. FastFuels-style uniform surface load (CV 0) vs. our measured 1 m surface load (CV 1.13) over OSBS. Same mean load; only ours carries the sub-class heterogeneity that fire models consume."));

// 2. method
body.push(H1("2. Method"));
body.push(P([T("From LiDAR we compute "), T("vertical occupancy", { bold: true }),
  T(" (the fraction of 0.15–4 m height-bins that contain returns — a density-robust understory-structure metric), voxelize it to 1 m³ bulk density, and "),
  T("field-calibrate it to surface fuel load against RxCADRE destructive clip plots", { bold: true }),
  T(" (occupancy → total load, slope 1.33 kg/m², R² 0.93). Where airborne LiDAR is absent, a "),
  T("per-pixel gradient-boosting model", { bold: true }),
  T(" predicts that same occupancy from globally-free spaceborne inputs — AlphaEarth embeddings, Sentinel-1 C-band, ALOS PALSAR L-band, and global canopy height — with conformal uncertainty and an out-of-distribution flag, then disaggregates to a FastFuels-compatible 1 m³ NetCDF. Simple, explainable, and every layer is measured or field-anchored — no hallucinated structure.")]));
body.push(...img(path.join(FIG, "fig_method.png"), 620, 2.037,
  "Figure 2. End-to-end method. LiDAR-measured occupancy trains a per-pixel model on free spaceborne inputs; RxCADRE clip plots calibrate occupancy to load; output is a 1 m³ FastFuels-compatible NetCDF."));
body.push(P([T("Differentiator vs the host team’s ForestGen3D and de Conto (2025): they generate/predict "),
  T("canopy", { italics: true }), T(" structure; we deliver an independent physical measurement of the "),
  T("surface/understory", { italics: true }), T(" layer, calibrated to destructive truth, with honest uncertainty.")]));

// 3. deliverables
body.push(H1("3. Deliverables in this submission"));
body.push(table(
  ["Deliverable", "Where"],
  [["1 m³ voxel NetCDF, FastFuels “Option C” (measured / uniform / generalized)", "build_deliverable_osbs.py → osbs_*_1m.nc"],
   ["AOI boundary polygon (WGS84 + native CRS)", "osbs_boundary.geojson"],
   ["Property maps + interactive 3D viewer", "figures/ + Streamlit dashboard"],
   ["Validation documentation", "research/RESULTS.md, VALIDATION.md"],
   ["Python ingestion/visualization tool (one command)", "scripts/read_deliverable.py"],
   ["Global “generate anywhere” demo", "dashboard/app.py"]],
  [5600, 3760]));
body.push(spacer());
body.push(P([T("Fuel properties in the NetCDF (empty cells use the challenge sentinel 1.23456):")]));
body.push(table(
  ["Variable", "Dims", "Units", "Tier"],
  [["bulk_density", "(z,y,x)", "kg/m³", "P1"],
   ["fuel_load", "(y,x)", "kg/m²", "P1"],
   ["percent_cover", "(y,x)", "%", "P1"],
   ["savr", "(z,y,x)", "1/m", "P2"],
   ["live_fraction", "(z,y,x)", "–", "P2"],
   ["dead_fuel_moisture", "(y,x)", "%", "P2"],
   ["live_fuel_moisture", "(y,x)", "%", "P2"],
   ["heat_of_combustion", "(z,y,x)", "kJ/kg", "P3"],
   ["avg patch size, <2 m heterogeneity", "attrs", "m, –", "P3"]],
  [3400, 1800, 2160, 2000]));

// 4. evidence vs criteria
body.push(H1("4. Evidence, mapped to the Phase-1 judging criteria"));

body.push(H2("(1) Utility of the methodology"));
body.push(P("Head-to-head vs the real FastFuels method (LANDFIRE FBFM40 → SB40 load lookup) on the OSBS 1 m product:"));
body.push(table(
  ["Product", "overall R²", "within-block R² (sub-30 m)", "heterogeneity CV"],
  [["FastFuels uniform", "−0.00", "0.00 (by construction)", "0.00"],
   ["Ours (end-to-end)", "0.664", "0.274", "0.73 (truth 0.89)"]],
  [3000, 1800, 3060, 1500]));
body.push(P([T("Our within-block R² of ", { size: 20 }), T("0.27", { bold: true, size: 20 }),
  T(" is exactly the sub-class heterogeneity FastFuels cannot represent. Output is drop-in FastFuels-compatible, in field-calibrated kg/m².", { size: 20 })], { spacing: { after: 120 } }));

body.push(H2("(2) Generality of approach"));
body.push(P("The identical pipeline runs across 15 ecosystems on 4 continents (US savanna, desert, conifer, deciduous, wetland; European temperate + hemiboreal; equatorial Amazon + Borneo rainforest) with no code changes. Cross-ecosystem leave-one-site-out:"));
body.push(bullet([T("Between-biome / absolute level: solved", { bold: true }), T(" — GLOBAL R² 0.49, between-biome R² 0.81.")]));
body.push(bullet([T("Within-biome fine structure, unseen forest: works", { bold: true }), T(" — median within-R² +0.11; within-Spearman ~0.3–0.6 (mean ~0.4).")]));
body.push(bullet([T("A conservative per-biome-radius OOD flag", { bold: true }), T(" marks locations unlike any training ecosystem — grounded where trained, honestly flagged elsewhere.")]));

body.push(H2("(3) Data acquisition challenges & cost"));
body.push(P("100% free / open, and no field campaign is required at inference. Inputs: USGS 3DEP LiDAR, AlphaEarth (free S3 mirror), Sentinel-1 + ALOS PALSAR + Copernicus DEM (Planetary Computer), Meta/WRI canopy height (open AWS), ESA WorldCover; truth from RxCADRE (USFS) + NEON. The global model needs only spaceborne inputs at inference — LiDAR is used to train, not to deploy."));

body.push(H2("(4) Clarity of methodology"));
body.push(P("Every step compresses to one sentence and every number is script-reproducible. We use a per-pixel gradient-boosting model — and proved it is the right choice at this scale via a faithful head-to-head vs the de Conto (2025) fully-convolutional EfficientNetV2 paradigm on the same target and spatial folds:"));
body.push(table(
  ["Model", "R² (all)", "under-canopy", "interval coverage"],
  [["Ours: per-pixel quantile GBM", "0.628", "0.375", "85%"],
   ["de Conto: fully-conv CNN (NLL)", "−0.147", "−0.380", "14%"]],
  [3760, 1800, 2000, 1800]));
body.push(...img(path.join(FIG, "deconto_headtohead.png"), 600, 2.597,
  "Figure 3. At AOI scale a per-pixel model with proper spatial CV beats the CNN paradigm decisively (the CNN needs continental training our tiled design supports)."));

body.push(H2("(5) Credibility of validation (top-weighted)"));
body.push(bullet("Spatially-blocked cross-validation everywhere; leave-one-site-out for cross-ecosystem generality."));
body.push(bullet("Calibrated uncertainty: conformal prediction intervals (85% empirical coverage) + per-cell OOD score."));
body.push(bullet([T("Field calibration to destructive truth", { bold: true }), T(": occupancy → RxCADRE total surface load, R² 0.93 across 9 Eglin burn blocks (grass 0.21 → forest 1.12 kg/m²).")]));
body.push(bullet("Honest negative results reported (cross-ecosystem within-biome is hard; NEON herb clips are the wrong stratum, R² 0.02)."));
body.push(...img(path.join(FIG, "fig_calibration.png"), 430, 1.185,
  "Figure 4. Absolute load is field-anchored: model occupancy vs RxCADRE destructive total surface load, slope 1.33 kg/m², R² 0.93."));

body.push(H2("(6) Relevance to surface & understory fuels"));
body.push(P("The target is explicitly the 0.15–4 m understory stratum (vertical occupancy → surface load), calibrated to RxCADRE surface clip plots — distinct from canopy-structure indices (de Conto’s WSCI) and from FastFuels’ canopy voxels. SAR + canopy-height features infer understory where optical saturates."));

body.push(H2("(7) Practical scalability"));
body.push(P("Global by design: free spaceborne inputs, on-demand per-AOI inference with a canonical-tile disk cache, memory-safe by construction (10 m inference, hard AOI cap, voxel-budgeted 1 m export). Coverage extends tile-by-tile — adding Amazon + Borneo moved the tropics from OOD to in-distribution."));

// 5. product + end-to-end
body.push(new Paragraph({ children: [new PageBreak()] }));
body.push(H1("5. The delivered product"));
body.push(...img(path.join(FIG, "deliverable_osbs.png"), 620, 1.662,
  "Figure 5. OSBS 1 m deliverable property maps: measured load, FastFuels-uniform baseline, generalized (spaceborne) load, fuelbed depth, bulk density, and voxel occupancy."));
body.push(...img(path.join(FIG, "pipeline_osbs.png"), 600, 1.668,
  "Figure 6. End-to-end spaceborne→ 30 m → 10 m pipeline validated against measured 3DEP truth vs a FastFuels-style uniform layer."));

// 6. limitations
body.push(H1("6. Limitations (stated plainly)"));
body.push(bullet("Absolute-load calibration is single-ecosystem (Eglin/RxCADRE, R² 0.93 over a wide range); multi-site destructive plots would refine per-biome transfer (forests read slightly low, deserts slightly high)."));
body.push(bullet("Within-biome fine structure in a brand-new biome is partially solved (forests) — the genuine sensing limit; L-band + canopy height help, GEDI is sparse."));
body.push(bullet("Fuel moisture (P2) is populated but coarse: dead FM from ERA5 (near-uniform at AOI scale), live FM a Sentinel-2 NDVI proxy. Species mix (P3) is out of scope for Phase 1."));
body.push(bullet("The OOD flag is conservative by design — it warns rather than silently extrapolating."));

// 7. reproducibility + data
body.push(H1("7. Reproducibility & data sources"));
body.push(P("pip install -r requirements.txt, then: build_deliverable_osbs.py (product), read_deliverable.py (ingest/viz), build_pipeline_osbs.py [--site] (end-to-end + generality), calibrate_load_field.py (RxCADRE calibration), compare_load_vs_fastfuels.py, global_coverage_table.py, deconto_headtohead.py, streamlit run dashboard/app.py."));
body.push(P("Data (all free/open): USGS 3DEP; AlphaEarth (Source Coop S3); Sentinel-1, ALOS PALSAR, Copernicus DEM, ESA WorldCover, Sentinel-2 (Microsoft Planetary Computer); Meta/WRI global canopy height (AWS Open Data); ERA5 (Copernicus CDS); RxCADRE (USFS Research Data Archive); NEON."));

// 8. eligibility
body.push(H1("8. Eligibility & IP"));
body.push(P("Team eligibility (18+, U.S./NATO citizens, non-federal, no federal funds) per the challenge terms. Participants retain IP; the U.S. Government and partners receive permanent access per the rules. No protective markings applied."));

// ---- assemble ----
const doc = new Document({
  creator: "3D Surface Fuels team",
  title: "3D Surface Fuels — Phase-1 Submission",
  styles: {
    default: { document: { run: { font: "Arial", size: 21 } } },
    paragraphStyles: [
      { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 30, bold: true, font: "Arial", color: "1a1a1a" },
        paragraph: { spacing: { before: 300, after: 160 }, outlineLevel: 0 } },
      { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 24, bold: true, font: "Arial", color: "2c7fb8" },
        paragraph: { spacing: { before: 200, after: 100 }, outlineLevel: 1 } },
    ],
  },
  numbering: { config: [{ reference: "b", levels: [{ level: 0, format: LevelFormat.BULLET, text: "•",
    alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 540, hanging: 260 } } } }] }] },
  sections: [{
    properties: { page: { size: { width: 12240, height: 15840 },
      margin: { top: 1440, right: 1440, bottom: 1440, left: 1440 } } },
    footers: { default: new Footer({ children: [new Paragraph({ alignment: AlignmentType.CENTER,
      children: [T("3D Surface Fuels — Phase-1 Submission · page ", { size: 16, color: "888888" }),
        new TextRun({ children: [PageNumber.CURRENT], size: 16, color: "888888" })] })] }) },
    children: body,
  }],
});

const outDir = path.join(ROOT, "submission");
fs.mkdirSync(outDir, { recursive: true });
const out = path.join(outDir, "Surface_Fuels_Phase1_Submission.docx");
Packer.toBuffer(doc).then((buf) => { fs.writeFileSync(out, buf); console.log("wrote", path.relpath ? path.relpath(ROOT, out) : out); });
