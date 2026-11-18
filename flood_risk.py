"""Urban Flood Risk from Impervious Surfaces using Satellite Data (demo pipeline).

Workflow: bands -> NDVI/NDBI -> classification -> impervious % per ward ->
rational-method runoff (Q = C i A) -> weighted-overlay flood risk -> validation.
Synthetic Sentinel-2-like data is used so the project runs offline.
To use real data, replace load_data() with rasterio reads of Sentinel-2 (B3,B4,B8,B11),
a DEM (SRTM/Cartosat) and flood-point shapefile.
"""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from scipy.ndimage import gaussian_filter, sobel
from sklearn.cluster import KMeans
from sklearn.metrics import roc_auc_score

rng = np.random.default_rng(42)
N = 300                      # 300 x 300 pixels, 10 m each -> 3 km x 3 km
PIX = 10.0
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
os.makedirs(OUT, exist_ok=True)


def load_data():
    y, x = np.mgrid[0:N, 0:N]
    # urban intensity: dense core + secondary pocket
    core = np.exp(-(((x - 150) ** 2 + (y - 150) ** 2) / (2 * 55 ** 2)))
    pocket = 0.7 * np.exp(-(((x - 70) ** 2 + (y - 230) ** 2) / (2 * 35 ** 2)))
    urban = np.clip(core + pocket + gaussian_filter(rng.normal(0, .25, (N, N)), 8), 0, 1)
    # DEM: gentle slope to river in south-east + noise, plus a depression in the core
    dem = 25 - 0.025 * (x + y) / 2 * 2 + gaussian_filter(rng.normal(0, 1, (N, N)), 15) * 4
    dem -= 3 * np.exp(-(((x - 160) ** 2 + (y - 170) ** 2) / (2 * 25 ** 2)))
    river = (np.abs((x - 20) - (y * 0.9)) < 4)
    dem[river] -= 3
    water = river | (dem < np.percentile(dem, 1.2))
    # reflectances (green, red, nir, swir)
    veg = np.clip(1 - urban, 0, 1)
    red = 0.05 + 0.20 * urban + rng.normal(0, .01, (N, N))
    nir = 0.45 * veg + 0.22 * urban + rng.normal(0, .015, (N, N))
    swir = 0.12 + 0.25 * urban + rng.normal(0, .015, (N, N))
    green = 0.07 + 0.12 * urban + rng.normal(0, .01, (N, N))
    for b in (red, nir, swir, green):
        b[water] = [0.03, 0.02, 0.01, 0.06][[id(red), id(nir), id(swir), id(green)].index(id(b))] if False else b[water]
    red[water], nir[water], swir[water], green[water] = 0.04, 0.02, 0.01, 0.08
    rain_mm_hr = 60 + 0.0 * x                               # design storm intensity
    return dict(green=green, red=red, nir=nir, swir=swir, dem=dem, water=water, truth_urban=urban), rain_mm_hr


def idx(a, b):
    return (a - b) / (a + b + 1e-9)


def main():
    d, rain = load_data()
    ndvi = idx(d["nir"], d["red"])
    ndbi = idx(d["swir"], d["nir"])
    mndwi = idx(d["green"], d["swir"])

    # ---- classification (k-means on indices + bands, then label clusters) ----
    feats = np.stack([ndvi, ndbi, mndwi, d["red"], d["swir"]], -1).reshape(-1, 5)
    km = KMeans(n_clusters=4, n_init=5, random_state=0).fit(feats)
    lab = km.labels_.reshape(N, N)
    stats = {k: (ndvi[lab == k].mean(), ndbi[lab == k].mean(), mndwi[lab == k].mean()) for k in range(4)}
    water_k = max(stats, key=lambda k: stats[k][2])
    rest = [k for k in stats if k != water_k]
    veg_k = max(rest, key=lambda k: stats[k][0])
    rest = [k for k in rest if k != veg_k]
    imp_k = max(rest, key=lambda k: stats[k][1])
    other_k = [k for k in rest if k != imp_k][0]
    cls = np.zeros((N, N), int)   # 0 other,1 veg,2 impervious,3 water
    cls[lab == veg_k] = 1; cls[lab == imp_k] = 2; cls[lab == water_k] = 3
    # mixed pixels (moderate NDBI) counted as impervious too (semi-built)
    cls[(lab == other_k) & (ndbi > 0.0)] = 2
    impervious = (cls == 2)

    # accuracy vs hidden synthetic truth
    truth = d["truth_urban"] > 0.5
    acc = (impervious == truth).mean()
    tp = (impervious & truth).sum(); fp = (impervious & ~truth).sum(); fn = (~impervious & truth).sum()
    f1 = 2 * tp / (2 * tp + fp + fn)

    # ---- imperviousness fraction on 100 m grid (10x10 px) & wards (5x5 grid) ----
    frac = impervious.reshape(N // 10, 10, N // 10, 10).mean((1, 3))
    ward_n = 5
    wsz = N // ward_n
    ward_id = (np.arange(N)[:, None] // wsz) * ward_n + (np.arange(N)[None, :] // wsz)
    ward = []
    for w in range(ward_n ** 2):
        m = ward_id == w
        ip = impervious[m].mean()
        C = 0.15 + 0.80 * ip                      # runoff coefficient (ASCE-style 0.15-0.95)
        A = m.sum() * PIX ** 2                    # m2
        i = 60 / 1000 / 3600                      # m/s
        Q = C * i * A                             # m3/s
        ward.append(dict(ward=w + 1, imperv_pct=round(100 * ip, 1), C=round(C, 2),
                         area_ha=round(A / 1e4, 1), Q_m3s=round(Q, 2)))
    # pre-urban baseline (C = 0.15) for comparison
    for r in ward:
        r["Q_baseline_m3s"] = round(0.15 * (60 / 1000 / 3600) * r["area_ha"] * 1e4, 2)
        r["increase_pct"] = round(100 * (r["Q_m3s"] / r["Q_baseline_m3s"] - 1), 0)

    # ---- flood susceptibility: weighted overlay ----
    def norm(a): return (a - a.min()) / (a.max() - a.min() + 1e-9)
    imp_density = gaussian_filter(impervious.astype(float), 6)
    elev_low = 1 - norm(gaussian_filter(d["dem"], 2))
    slope = np.hypot(sobel(d["dem"], 0), sobel(d["dem"], 1)) / (8 * PIX)
    flat = 1 - norm(np.clip(slope, 0, np.percentile(slope, 99)))
    dist_water = __import__("scipy.ndimage", fromlist=["x"]).distance_transform_edt(~d["water"]) * PIX
    near = 1 - norm(np.clip(dist_water, 0, 600))
    W = dict(imperviousness=.35, low_elevation=.25, flat_slope=.15, near_drainage=.25)
    score = (W["imperviousness"] * norm(imp_density) + W["low_elevation"] * elev_low +
             W["flat_slope"] * flat + W["near_drainage"] * near)
    score[d["water"]] = np.nan
    q1, q2 = np.nanpercentile(score, [50, 85])
    risk = np.where(score < q1, 1, np.where(score < q2, 2, 3)).astype(float)
    risk[d["water"]] = np.nan

    # ---- validation with synthetic observed flood points ----
    # "observed" floods generated from hydrologic logic + noise (independent of the weights)
    p = norm(gaussian_filter(impervious.astype(float), 10)) * .5 + elev_low * .5
    p = norm(p + rng.normal(0, .08, (N, N)))
    obs = (p > np.nanpercentile(p, 90)) & ~d["water"]
    ok = ~d["water"]
    auc = roc_auc_score(obs[ok].ravel(), np.nan_to_num(score[ok]).ravel())
    hi = risk == 3
    capture = (obs & hi).sum() / obs.sum()
    hi_frac = np.nansum(hi) / ok.sum()

    # ---- sensitivity: impervious reduction scenario (green infrastructure) ----
    scen = []
    for red in (0, 10, 20, 30):
        tot = sum(0.15 + 0.80 * (r["imperv_pct"] / 100 * (1 - red / 100)) for r in ward) / len(ward)
        scen.append(dict(reduction_pct=red, mean_C=round(tot, 3)))

    # ---- figures ----
    def save(name): plt.tight_layout(); plt.savefig(f"{OUT}/{name}", dpi=170); plt.close()
    ext = [0, N * PIX / 1000, 0, N * PIX / 1000]
    rgb = np.clip(np.stack([d["swir"] * 3, d["nir"] * 1.6, d["red"] * 3], -1), 0, 1)
    fig, ax = plt.subplots(1, 3, figsize=(13, 4.3))
    ax[0].imshow(rgb, extent=ext); ax[0].set_title("Synthetic Sentinel-2 false colour")
    im = ax[1].imshow(ndvi, cmap="RdYlGn", extent=ext); ax[1].set_title("NDVI"); plt.colorbar(im, ax=ax[1], fraction=.046)
    im = ax[2].imshow(ndbi, cmap="magma", extent=ext); ax[2].set_title("NDBI"); plt.colorbar(im, ax=ax[2], fraction=.046)
    for a in ax: a.set_xlabel("km"); a.set_ylabel("km")
    save("fig1_indices.png")

    cmap = ListedColormap(["#d9c9a3", "#3c9a3c", "#b0252a", "#2f6fb0"])
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.6))
    ax[0].imshow(cls, cmap=cmap, vmin=0, vmax=3, extent=ext); ax[0].set_title("Land-cover classification")
    from matplotlib.patches import Patch
    ax[0].legend(handles=[Patch(color=c, label=l) for c, l in zip(["#d9c9a3", "#3c9a3c", "#b0252a", "#2f6fb0"], ["Other/bare", "Vegetation", "Impervious", "Water"])],
                 loc="lower left", fontsize=7)
    im = ax[1].imshow(100 * frac, cmap="Reds", extent=ext, vmin=0, vmax=100); ax[1].set_title("Impervious % (100 m grid)")
    plt.colorbar(im, ax=ax[1], fraction=.046)
    for a in ax: a.set_xlabel("km")
    save("fig2_impervious.png")

    fig, ax = plt.subplots(1, 3, figsize=(13, 4.3))
    im = ax[0].imshow(d["dem"], cmap="terrain", extent=ext); ax[0].set_title("DEM (m)"); plt.colorbar(im, ax=ax[0], fraction=.046)
    im = ax[1].imshow(score, cmap="YlOrRd", extent=ext); ax[1].set_title("Flood susceptibility score"); plt.colorbar(im, ax=ax[1], fraction=.046)
    ax[2].imshow(risk, cmap=ListedColormap(["#4caf50", "#ffc107", "#d32f2f"]), vmin=1, vmax=3, extent=ext)
    oy, ox = np.where(obs)
    ax[2].scatter(ox[::25] * PIX / 1000, (N - oy[::25]) * PIX / 1000, s=2, c="k", label="Observed flood pts (synthetic)")
    ax[2].set_title("Risk zones (Low / Mod / High)"); ax[2].legend(fontsize=7, loc="lower left")
    for a in ax: a.set_xlabel("km")
    save("fig3_risk.png")

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    wi = [r["ward"] for r in ward]
    ax[0].bar(wi, [r["Q_baseline_m3s"] for r in ward], label="Pre-urban (C=0.15)", color="#7fb77e")
    ax[0].bar(wi, [r["Q_m3s"] - r["Q_baseline_m3s"] for r in ward], bottom=[r["Q_baseline_m3s"] for r in ward],
              label="Added by imperviousness", color="#c0392b")
    ax[0].set_xlabel("Ward"); ax[0].set_ylabel("Peak Q (m³/s)"); ax[0].set_title("Rational-method peak discharge, 60 mm/h"); ax[0].legend(fontsize=8)
    ax[1].scatter([r["imperv_pct"] for r in ward], [r["Q_m3s"] for r in ward], c="#c0392b")
    ax[1].set_xlabel("Impervious %"); ax[1].set_ylabel("Q (m³/s)"); ax[1].set_title("Imperviousness vs runoff")
    save("fig4_runoff.png")

    results = dict(classification=dict(overall_accuracy=round(acc, 3), f1_impervious=round(f1, 3),
                   impervious_area_pct=round(100 * impervious.mean(), 1)),
                   weights=W, validation=dict(AUC=round(auc, 3), obs_flood_pixels_in_high_zone_pct=round(100 * capture, 1),
                   high_zone_area_pct=round(100 * hi_frac, 1)), wards=ward, mitigation_scenarios=scen)
    json.dump(results, open(f"{OUT}/results.json", "w"), indent=2)
    import csv
    with open(f"{OUT}/ward_table.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=ward[0].keys()); w.writeheader(); w.writerows(ward)
    print(json.dumps({k: results[k] for k in ("classification", "validation", "mitigation_scenarios")}, indent=1))


if __name__ == "__main__":
    main()
