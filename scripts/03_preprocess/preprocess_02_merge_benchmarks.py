#!/usr/bin/env python3
"""
Step 2: Merge benchmark results with triage labels.

Reads:  Wei CSVs from WEI_DIR
        AE xlsx from AE_XLSX
        INTERMEDIATE_DIR/combined_genetic_2d.csv
        INTERMEDIATE_DIR/combined_cellular_2d.csv
Writes: INTERMEDIATE_DIR/wei_genetic_merged_2d.csv
        INTERMEDIATE_DIR/wei_genetic_5k_merged_2d.csv
        INTERMEDIATE_DIR/wei_cellular_iid_merged_2d.csv
        INTERMEDIATE_DIR/wei_cellular_ood_merged_2d.csv
        INTERMEDIATE_DIR/ae_panel_a_merged_2d.csv
        INTERMEDIATE_DIR/ae_panel_c_merged_2d.csv
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from config import *
import pandas as pd

os.makedirs(str(INTERMEDIATE_DIR), exist_ok=True)

# ── Load triage tables ────────────────────────────────────────────
triage_gen = pd.read_csv(intermediate_path('combined_genetic_2d.csv'))
triage_cel_path = intermediate_path('combined_cellular_2d.csv')
triage_cel = pd.read_csv(triage_cel_path) if os.path.exists(triage_cel_path) else pd.DataFrame()

print(f"Loaded triage: genetic={len(triage_gen)} rows, cellular={len(triage_cel)} rows")


# ---- Wei et al. benchmark ----
def merge_wei_with_triage(wei_csv_path, triage_df, triage_key_cols, label):
    """
    Merge a Wei performance CSV with triage table.

    Wei CSVs have columns: method, cor, Rank_pcc, ..., DataSet, op
    Triage CSVs have: dataset, perturbation, ..., triage_2d

    Merge key: (DataSet == dataset) and (op stripped of _\\d+ == perturbation)
    """
    if not os.path.exists(wei_csv_path):
        print(f"  ⚠ Not found: {wei_csv_path}")
        return None

    raw = pd.read_csv(wei_csv_path, index_col=0)
    print(f"\n  [{label}] Raw: {len(raw)} rows, {raw['DataSet'].nunique()} datasets")

    # Standardize columns
    raw = raw.rename(columns={'cor': 'pcc', 'DataSet': 'dataset'})
    raw['perturbation'] = raw['op'].apply(extract_perturbation_from_wei_op)

    # Triage columns to merge
    triage_cols = ['dataset', 'perturbation'] + triage_key_cols
    triage_slim = triage_df[triage_cols].drop_duplicates(subset=['dataset', 'perturbation'])

    # Merge
    merged = raw.merge(triage_slim, on=['dataset', 'perturbation'], how='inner')
    print(f"  Merged: {len(merged)} rows ({len(raw) - len(merged)} dropped, "
          f"{len(merged)/len(raw)*100:.0f}% retained)")

    # Derived columns
    merged['reliability'] = merged['sb_from_median_all_genes']
    merged['ceiling']     = merged['ceiling_from_median_all_genes']
    # A zero ceiling means the target carries no reliable signal; leave CE undefined rather than infinite.
    merged['CE']          = merged['pcc'] / merged['ceiling'].replace(0, float('nan'))

    return merged


# Genetic perturbation (top 100 DEGs)
gen_triage_cols = ['sb_from_median_all_genes', 'ceiling_from_median_all_genes',
                   'cos_sim', 'triage_2d']

print("=== Wei Genetic (top 100) ===")
wei_gen = merge_wei_with_triage(
    str(WEI_DIR / 'Genetic_perturbation' / 'genetic_single_performance_top100.csv'),
    triage_gen, gen_triage_cols, 'genetic_top100')
if wei_gen is not None:
    out = intermediate_path('wei_genetic_merged_2d.csv')
    wei_gen.to_csv(out, index=False)
    print(f"  ✓ Saved {out}")

# Genetic perturbation (top 5000)
print("=== Wei Genetic (top 5000) ===")
wei_gen5k = merge_wei_with_triage(
    str(WEI_DIR / 'Genetic_perturbation' / 'genetic_single_performance_top5000.csv'),
    triage_gen, gen_triage_cols, 'genetic_top5000')
if wei_gen5k is not None:
    out = intermediate_path('wei_genetic_5k_merged_2d.csv')
    wei_gen5k.to_csv(out, index=False)
    print(f"  ✓ Saved {out}")

# Genetic combo (doubles): Norman, Replogle_exp6, Schmidt, Wessels
print("=== Wei Genetic COMBO (top 100) ===")
wei_gen_combo = merge_wei_with_triage(
    str(WEI_DIR / 'Genetic_perturbation' / 'genetic_combo_performance_top100.csv'),
    triage_gen, gen_triage_cols, 'genetic_combo_top100')
if wei_gen_combo is not None:
    out = intermediate_path('wei_genetic_combo_merged_2d.csv')
    wei_gen_combo.to_csv(out, index=False)
    print(f"  ✓ Saved {out}")

# Chemical single: sciplex3 × 3 cell lines
print("=== Wei Chemical SINGLE (top 100) ===")
wei_chem_single = merge_wei_with_triage(
    str(WEI_DIR / 'Chemical_perturbation' / 'chemical_single_performance_top100.csv'),
    triage_gen, gen_triage_cols, 'chemical_single_top100')
if wei_chem_single is not None:
    out = intermediate_path('wei_chemical_single_merged_2d.csv')
    wei_chem_single.to_csv(out, index=False)
    print(f"  ✓ Saved {out}")

# Chemical combo: sciplex3_comb (drug combinations)
print("=== Wei Chemical COMBO (top 100) ===")
wei_chem_combo = merge_wei_with_triage(
    str(WEI_DIR / 'Chemical_perturbation' / 'chemical_combo_performance_top100.csv'),
    triage_gen, gen_triage_cols, 'chemical_combo_top100')
if wei_chem_combo is not None:
    out = intermediate_path('wei_chemical_combo_merged_2d.csv')
    wei_chem_combo.to_csv(out, index=False)
    print(f"  ✓ Saved {out}")

# Cellular context IID
if len(triage_cel) > 0:
    cel_triage_cols = ['sb_from_median_all_genes', 'ceiling_from_median_all_genes',
                       'cos_sim', 'triage_2d']

    def _normalize(s):
        """Normalize string for fuzzy matching: lowercase, strip special chars."""
        return str(s).replace(' ', '_').replace('.', '_').replace('-', '').lower()

    def merge_wei_cellular(wei_csv_path, triage_df, label):
        """
        Merge Wei cellular performance with triage.
        
        Key challenge: Wei 'op' encodes perturbation_context (e.g. 'Infected_Pericentral')
        while triage 'condition' = context|perturbation (e.g. 'Pericentral|Infected').
        We match via normalization.
        """
        if not os.path.exists(wei_csv_path):
            print(f"  ⚠ Not found: {wei_csv_path}")
            return None
        raw = pd.read_csv(wei_csv_path, index_col=0)
        print(f"\n  [{label}] Raw: {len(raw)} rows, {raw['DataSet'].nunique()} datasets")
        raw = raw.rename(columns={'cor': 'pcc', 'DataSet': 'dataset'})

        # Build normalized condition → original condition lookup per dataset
        # triage condition = "context|perturbation" → normalized "perturbation_context"
        cond_lookup = {}  # (dataset, normalized_op) → condition
        for _, row in triage_df[['dataset', 'condition', 'perturbation', 'context']].drop_duplicates().iterrows():
            normed = _normalize(f"{row['perturbation']}_{row['context']}")
            cond_lookup[(row['dataset'], normed)] = row['condition']

        # Map each Wei row to its triage condition
        def match_op(r):
            normed_op = _normalize(r['op'])
            key = (r['dataset'], normed_op)
            if key in cond_lookup:
                return cond_lookup[key]
            # Fallback: strip common prefixes (e.g. Wei "Pat101" vs triage "101")
            normed_stripped = normed_op.replace('pat', '')
            key2 = (r['dataset'], normed_stripped)
            if key2 in cond_lookup:
                return cond_lookup[key2]
            return None

        raw['condition'] = raw.apply(match_op, axis=1)

        matched = raw.dropna(subset=['condition'])
        print(f"  Condition-matched: {len(matched)} / {len(raw)} rows "
              f"({len(matched)/len(raw)*100:.0f}%)")

        if len(matched) == 0:
            return None

        # Merge with triage on (dataset, condition)
        slim = triage_df[['dataset', 'condition'] + cel_triage_cols].drop_duplicates(
            subset=['dataset', 'condition'])
        merged = matched.merge(slim, on=['dataset', 'condition'], how='inner')
        print(f"  Merged: {len(merged)} rows ({len(merged)/len(raw)*100:.0f}% retained)")

        if len(merged) > 0:
            merged['reliability'] = merged['sb_from_median_all_genes']
            merged['ceiling']     = merged['ceiling_from_median_all_genes']
            merged['CE']          = merged['pcc'] / merged['ceiling'].replace(0, float('nan'))
        return merged

    print("=== Wei Cellular IID ===")
    wei_cel_iid = merge_wei_cellular(
        str(WEI_DIR / 'Cellular_context_iid' / 'cellular_iid_performance_top100.csv'),
        triage_cel, 'cellular_iid')
    if wei_cel_iid is not None and len(wei_cel_iid) > 0:
        out = intermediate_path('wei_cellular_iid_merged_2d.csv')
        wei_cel_iid.to_csv(out, index=False)
        print(f"  ✓ Saved {out}")

    print("=== Wei Cellular OOD ===")
    wei_cel_ood = merge_wei_cellular(
        str(WEI_DIR / 'Cellular_context_ood' / 'cellular_ood_performance_top100.csv'),
        triage_cel, 'cellular_ood')
    if wei_cel_ood is not None and len(wei_cel_ood) > 0:
        out = intermediate_path('wei_cellular_ood_merged_2d.csv')
        wei_cel_ood.to_csv(out, index=False)
        print(f"  ✓ Saved {out}")


# ---- Ahlmann-Eltze benchmark ----
def merge_ae_panel(sheet_name, label):
    """Merge an AE panel with genetic triage."""
    if not os.path.exists(str(AE_XLSX)):
        print(f"  ⚠ AE xlsx not found: {AE_XLSX}")
        return None

    raw = pd.read_excel(str(AE_XLSX), sheet_name)
    print(f"\n  [{label}] Raw: {len(raw)} rows, {raw['dataset_name'].nunique()} datasets")

    # Clean perturbation: strip +ctrl
    raw['perturbation'] = raw['perturbation'].apply(clean_ae_perturbation)

    # Map dataset names to match triage
    raw['dataset'] = raw['dataset_name'].map(AE_DATASET_MAP)
    unmapped = raw[raw['dataset'].isna()]['dataset_name'].unique()
    if len(unmapped) > 0:
        print(f"  ⚠ Unmapped datasets: {unmapped}")
    raw = raw.dropna(subset=['dataset'])

    # Triage columns
    triage_slim = triage_gen[['dataset', 'perturbation'] + gen_triage_cols].drop_duplicates(
        subset=['dataset', 'perturbation'])

    merged = raw.merge(triage_slim, on=['dataset', 'perturbation'], how='inner')
    print(f"  Merged: {len(merged)} rows ({len(merged)/len(raw)*100:.0f}% retained)")

    # Derived columns
    merged['reliability'] = merged['sb_from_median_all_genes']
    merged['ceiling']     = merged['ceiling_from_median_all_genes']
    # 'r2_delta' is a Pearson correlation, so its attenuation bound is sqrt(rho) = 'ceiling'.
    merged['CE_r2delta']  = merged['r2_delta'] / merged['ceiling'].replace(0, float('nan'))

    return merged

print("\n=== Ahlmann-Eltze Panel A ===")
ae_a = merge_ae_panel('Panel A', 'AE Panel A')
if ae_a is not None:
    out = intermediate_path('ae_panel_a_merged_2d.csv')
    ae_a.to_csv(out, index=False)
    print(f"  ✓ Saved {out}")

print("\n=== Ahlmann-Eltze Panel C ===")
ae_c = merge_ae_panel('Panel C', 'AE Panel C')
if ae_c is not None:
    out = intermediate_path('ae_panel_c_merged_2d.csv')
    ae_c.to_csv(out, index=False)
    print(f"  ✓ Saved {out}")

# ── AE doubles (Norman from perturbation_prediction.xlsx, Panel A) ─────────
def merge_ae_doubles(label):
    """Same merge logic as merge_ae_panel but reads doubles xlsx (Panel A)."""
    if not os.path.exists(str(AE_XLSX_DOUBLES)):
        print(f"  ⚠ AE doubles xlsx not found: {AE_XLSX_DOUBLES}")
        return None
    raw = pd.read_excel(str(AE_XLSX_DOUBLES), 'Panel A')
    print(f"\n  [{label}] Raw: {len(raw)} rows, "
          f"{raw['dataset_name'].nunique()} datasets")
    raw['perturbation'] = raw['perturbation'].apply(clean_ae_perturbation)
    raw['dataset'] = raw['dataset_name'].map(AE_DATASET_MAP)
    unmapped = raw[raw['dataset'].isna()]['dataset_name'].unique()
    if len(unmapped) > 0:
        print(f"  ⚠ Unmapped datasets: {unmapped}")
    raw = raw.dropna(subset=['dataset'])
    triage_slim = triage_gen[['dataset', 'perturbation'] + gen_triage_cols].drop_duplicates(
        subset=['dataset', 'perturbation'])
    # A double perturbation is an unordered gene pair, but the two tables do not agree on the order
    # ("A+B" vs "B+A"), so match on a canonically sorted key.
    def _pair_key(p):
        return '+'.join(sorted(str(p).split('+')))
    raw['_pair'] = raw['perturbation'].apply(_pair_key)
    triage_slim = triage_slim.assign(_pair=triage_slim['perturbation'].apply(_pair_key))
    triage_slim = triage_slim.drop(columns=['perturbation']).drop_duplicates(subset=['dataset', '_pair'])
    merged = raw.merge(triage_slim, on=['dataset', '_pair'], how='inner').drop(columns=['_pair'])
    print(f"  Merged: {len(merged)} rows ({len(merged)/len(raw)*100:.0f}% retained)")
    merged['reliability'] = merged['sb_from_median_all_genes']
    merged['ceiling']     = merged['ceiling_from_median_all_genes']
    # 'r2_delta' is a Pearson correlation, so its attenuation bound is sqrt(rho) = 'ceiling'.
    merged['CE_r2delta']  = merged['r2_delta'] / merged['ceiling'].replace(0, float('nan'))
    return merged

print("\n=== Ahlmann-Eltze Doubles (Norman) ===")
ae_dbl = merge_ae_doubles('AE Doubles Panel A')
if ae_dbl is not None:
    out = intermediate_path('ae_doubles_merged_2d.csv')
    ae_dbl.to_csv(out, index=False)
    print(f"  ✓ Saved {out}")

print("\n✓ Step 2 complete.")
