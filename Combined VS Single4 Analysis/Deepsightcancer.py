import os, json, tempfile, subprocess

# === CONFIGURATION ===
LABEL_FILE = "/raid/data01/deephealth/labels/dh_dh0new_labels.csv"
INPUT_DIR = "/raid/data01/deephealth/dh_dh0new/"
OUTPUT_DIR = "/raid/mpsych/OMAMA/2025/PLAYGROUND/DeepSight_CancerOnly/"
DEEPSIGHT_SCRIPT = "/home2/deephealth/scripts/deepsight2.sh"
IGNORE_CHECKS = ["SAC-30", "SAC-40", "SAC-50", "SAC-60", "FAC-200"]

os.makedirs(OUTPUT_DIR, exist_ok=True)

# === STEP 1: Find studies with IndexCancer ===
cancer_studies = set()
with open(LABEL_FILE) as f:
    for line in f:
        if "IndexCancer" in line:
            uid = line.strip().split(",")[0]
            cancer_studies.add(uid)

# === STEP 2: Filter valid studies with >=4 DXm images ===
study_ids = sorted([
    s for s in cancer_studies
    if os.path.isdir(os.path.join(INPUT_DIR, s)) and
    len([f for f in os.listdir(os.path.join(INPUT_DIR, s)) if f.startswith("DXm")]) >= 4
])

# === FUNCTIONS ===
def run_deepsight(case_list_path, input_path, output_path, ignore=False):
    os.makedirs(output_path, exist_ok=True)
    cmd = [DEEPSIGHT_SCRIPT, "-i", input_path, "-o", output_path, "-cl", case_list_path]
    if ignore:
        cmd += ["--additional_params", "--checks_to_ignore"] + IGNORE_CHECKS
    subprocess.run(cmd)

def extract_scores(json_path):
    if not os.path.exists(json_path):
        return None, None, None
    with open(json_path) as f:
        data = json.load(f)
    dicoms = data.get("results_raw", {}).get("dicom_results", {})
    total, L, R = 0.0, 0.0, 0.0
    for dicom in dicoms.values():
        for obj in dicom.get("none", []):
            score = obj.get("score", 0.0)
            lat = obj.get("laterality")
            if lat == "L": L += score
            elif lat == "R": R += score
            else: total += score
    return round(total, 4), round(L, 4) if L else None, round(R, 4) if R else None

# === MAIN LOOP ===
results = []
for idx, study in enumerate(study_ids, 1):
    study_path = os.path.join(INPUT_DIR, study)
    dicoms = sorted([f for f in os.listdir(study_path) if f.startswith("DXm")])[:4]
    dicom_paths = [os.path.join(study_path, f) for f in dicoms]

    print(f"\n📂 Running DeepSight on {study} ({len(dicom_paths)} images)")

    # --- Combined Run ---
    with tempfile.NamedTemporaryFile("w", delete=False) as tmp:
        for path in dicom_paths:
            tmp.write(path + "\n")
    out_comb = os.path.join(OUTPUT_DIR, f"{study}_combined")
    run_deepsight(tmp.name, study_path, out_comb)
    os.remove(tmp.name)

    try:
        comb_sub = sorted(os.listdir(out_comb))[0]
        comb_json = os.path.join(out_comb, comb_sub, "results_full.json")
        comb_total, comb_L, comb_R = extract_scores(comb_json)
    except Exception:
        comb_total, comb_L, comb_R = None, None, None
    results.append(f"{study},combined,{comb_total},{comb_L},{comb_R},-")

    # --- Single Runs (4) ---
    agg_total, agg_L, agg_R = 0.0, 0.0, 0.0
    for i, dicom in enumerate(dicom_paths):
        with tempfile.NamedTemporaryFile("w", delete=False) as tmp:
            tmp.write(dicom + "\n")
        out_single = os.path.join(OUTPUT_DIR, f"{study}_img{i}")
        run_deepsight(tmp.name, os.path.dirname(dicom), out_single, ignore=True)
        os.remove(tmp.name)

        try:
            single_sub = sorted(os.listdir(out_single))[0]
            single_json = os.path.join(out_single, single_sub, "results_full.json")
            t, l, r = extract_scores(single_json)
        except Exception:
            t, l, r = None, None, None

        if t: agg_total += t
        if l: agg_L += l
        if r: agg_R += r
        results.append(f"{study},img{i},{t},{l},{r},{'match' if t else 'no detection'}")

    match_status = "✅ match" if round(agg_L, 4) == comb_L and round(agg_R, 4) == comb_R else "❌ mismatch"
    results.append(f"{study},aggregate,{agg_total},{agg_L},{agg_R},{match_status}")
    print(f"✅ Done {idx}/{len(study_ids)}")

# === SAVE OUTPUT CSV ===
out_csv = os.path.join(OUTPUT_DIR, "combined_vs_single_score_comparison.csv")
with open(out_csv, "w") as f:
    f.write("study_id,mode,total_score,L_score,R_score,match_status\n")
    for row in results:
        f.write(row + "\n")
print(f"\n📁 CSV written to: {out_csv}")
