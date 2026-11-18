import json, os
from docx import Document
from docx.shared import Pt, Inches
H = os.path.dirname(os.path.abspath(__file__)); O = H + "/outputs"
R = json.load(open(O + "/results.json"))
d = Document()
st = d.styles["Normal"]; st.font.name = "Calibri"; st.font.size = Pt(11)
d.add_heading("Urban Flood Risk from Impervious Surfaces Using Satellite Data", 0)
d.add_paragraph("Project Report - Environmental Hydrology")
def sec(t, paras=(), bullets=()):
    d.add_heading(t, 1)
    for p in paras: d.add_paragraph(p)
    for b in bullets: d.add_paragraph(b, style="List Bullet")
def fig(f, cap, w=6.3):
    d.add_picture(f"{O}/{f}", width=Inches(w)); d.add_paragraph(cap).italic = True
sec("1. Abstract", ["Urbanisation replaces soil and vegetation with roofs, roads and pavement, reducing infiltration and increasing surface runoff. This project builds a reproducible workflow that derives impervious-surface fraction from multispectral satellite bands (Sentinel-2 style: green, red, NIR, SWIR), converts it to a runoff coefficient, estimates peak discharge with the rational method (Q = C i A), and combines imperviousness with elevation, slope and proximity to drainage in a GIS weighted overlay to map Low / Moderate / High flood-risk zones. The pipeline is demonstrated on a synthetic 3 km x 3 km city so it runs offline; each step maps directly onto real Sentinel-2, DEM and rainfall data."])
sec("2. Objectives", bullets=["Map impervious surfaces from satellite spectral indices (NDVI, NDBI, MNDWI).", "Quantify imperviousness per 100 m grid cell and per ward.", "Estimate runoff coefficient and peak discharge, and compare with a pre-urban baseline.", "Produce a flood-susceptibility and risk-zone map and validate it.", "Test green-infrastructure mitigation scenarios."])
sec("3. Background", ["Impervious surfaces increase runoff volume and peak discharge and shorten time of concentration, so drains are overwhelmed sooner. Satellite imagery gives consistent city-wide coverage; optical bands separate vegetation, built-up and water, while SAR (Sentinel-1) can map floodwater through cloud. Imperviousness is a major but not the only driver of flood risk: drainage capacity, topography and rainfall also matter.", "Indices: NDVI = (NIR - Red)/(NIR + Red); NDBI = (SWIR - NIR)/(SWIR + NIR); MNDWI = (Green - SWIR)/(Green + SWIR). NDBI indicates built-up areas but is not a direct measure of imperviousness."])
sec("4. Data and Study Area", ["Study area: synthetic city, 300 x 300 pixels at 10 m (3 km x 3 km), with a dense core, a secondary built-up pocket, a river and a core depression. Data layers: four reflectance bands, a DEM, a river/water mask and a design storm of 60 mm/h.", "Important: all data are synthetic and results are illustrative, not findings about a real city. For a real study, replace load_data() in flood_risk.py with Sentinel-2 (B3, B4, B8, B11 at 10/20 m), a DEM (SRTM or Cartosat), rainfall (IMD/GPM) and mapped flood locations."])
fig("fig1_indices.png", "Figure 1. False-colour composite, NDVI and NDBI.")
sec("5. Methodology", bullets=["Pre-processing: bands as surface reflectance; cloud masking and clipping would be applied to real scenes.", "Classification: k-means (4 clusters) on NDVI, NDBI, MNDWI, red and SWIR; clusters labelled by index statistics as vegetation, impervious, water or other; moderately built mixed pixels (NDBI > 0) are added to impervious.", "Quantification: impervious % on a 100 m grid and for 25 wards (5 x 5).", "Runoff: C = 0.15 + 0.80 x imperviousness (range ~0.15-0.95); Q = C i A with i = 60 mm/h; baseline uses C = 0.15.", "Flood susceptibility: weighted overlay of imperviousness density (0.35), low elevation (0.25), flat slope (0.15) and proximity to drainage (0.25). Weights are assumed, not calibrated; AHP could be used. Classes: Low < 50th, Moderate 50th-85th, High > 85th percentile.", "Validation: overall accuracy and F1 against the synthetic truth; ROC-AUC and capture rate against synthetic observed flood points."])
fig("fig2_impervious.png", "Figure 2. Land-cover classification and impervious % per 100 m cell.")
c, v = R["classification"], R["validation"]
sec("6. Results", [f"Classification: overall accuracy {c['overall_accuracy']*100:.1f}%, impervious F1 {c['f1_impervious']:.2f}; impervious area {c['impervious_area_pct']}% of the study area.", f"Flood risk: the High zone covers {v['high_zone_area_pct']}% of the area and contains {v['obs_flood_pixels_in_high_zone_pct']}% of synthetic observed flood pixels (AUC {v['AUC']})."])
fig("fig3_risk.png", "Figure 3. DEM, susceptibility score and risk zones with synthetic flood points.", 6.5)
d.add_heading("Ward-level runoff", 2)
t = d.add_table(rows=1, cols=6); t.style = "Light Grid Accent 1"
for i, h in enumerate(["Ward", "Impervious %", "C", "Area (ha)", "Q (m3/s)", "Increase vs baseline (%)"]): t.rows[0].cells[i].text = h
for w in R["wards"]:
    r = t.add_row().cells
    for i, k in enumerate(["ward", "imperv_pct", "C", "area_ha", "Q_m3s", "increase_pct"]): r[i].text = str(w[k])
d.add_paragraph()
fig("fig4_runoff.png", "Figure 4. Peak discharge by ward and imperviousness vs runoff.")
d.add_heading("Mitigation scenarios", 2)
d.add_paragraph("Reducing effective imperviousness (permeable pavement, green roofs, bioswales) lowers the mean runoff coefficient:")
for s in R["mitigation_scenarios"]: d.add_paragraph(f"{s['reduction_pct']}% reduction: mean C = {s['mean_C']}", style="List Bullet")
sec("7. Discussion", ["Wards with high imperviousness produce several times the baseline peak discharge, and the highest-risk zone coincides with built-up land in low, flat terrain near the river - consistent with the hydrological reasoning that sealing speeds and concentrates runoff.", "The near-perfect validation scores should not be over-interpreted: the synthetic flood points were generated from the same physical drivers, so the check is partly circular. It shows the pipeline is internally consistent, not that it predicts real floods."])
sec("8. Limitations", bullets=["Synthetic data; no real city results.", "10 m pixels mix roofs, roads and vegetation; k-means is unsupervised and a supervised classifier (Random Forest) with training samples is preferable.", "Rational method suits small catchments; SCS-CN or SWMM/HEC-HMS would be better for larger areas.", "Drainage network capacity, soil and antecedent moisture are not modelled; weights are assumed."])
sec("9. Future Work", bullets=["Apply to a real city (e.g. Chennai wards) using Sentinel-2, SRTM/Cartosat and IMD rainfall.", "Add Sentinel-1 SAR flood extent from a real event for validation.", "Replace weights with AHP or machine-learning calibration.", "Compare multi-year imperviousness change."])
sec("10. Conclusion", ["Satellite-derived imperviousness, combined with a simple hydrological model and GIS overlay, gives a transparent way to locate flood-prone urban pockets and to test mitigation options. The workflow is modular and ready to run on real data."])
sec("References", bullets=["USGS EROS - Impervious surface / Earthshots (urban growth and runoff).", "Yang et al. (2003), mapping large-area impervious surfaces from Landsat and high-resolution imagery (USGS).", "Xian et al. (2019), performance of WorldView-3, Sentinel-2 and Landsat-8 for impervious-surface mapping (USGS).", "Li and Bortolot (2022), Journal of Cleaner Production - land-cover change and catchment-scale urban flooding.", "Zhao et al. (2024), review of satellite SAR approaches for urban flood mapping."])
d.save(H + "/Urban_Flood_Risk_Project_Report.docx")
