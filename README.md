# Urban Flood Risk from Impervious Surfaces (Satellite Data)
Environmental Hydrology project based on the supplied presentation.

Run: `pip install numpy scipy scikit-learn matplotlib python-docx`, then `python flood_risk.py` and `python make_report.py`.

- flood_risk.py: indices, classification, impervious %, Q = C i A, weighted-overlay risk map, validation
- outputs/: figures, results.json, ward_table.csv
- Urban_Flood_Risk_Project_Report.docx: full report

Data are synthetic (offline demo). Swap load_data() for Sentinel-2 B3/B4/B8/B11 + DEM + rainfall to use real data.
