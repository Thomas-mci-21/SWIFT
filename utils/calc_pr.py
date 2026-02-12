#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import pandas as pd
import glob

def calc_metrics(csv_path):
    df = pd.read_csv(csv_path)
    y_true = df['label'].apply(lambda x: 1 if str(x).lower() in ['true', '1'] else 0)
    y_pred = df['prediction'].apply(lambda x: 1 if str(x).lower() in ['true', '1'] else 0)

    tp = ((y_true == 1) & (y_pred == 1)).sum()
    tn = ((y_true == 0) & (y_pred == 0)).sum()
    fp = ((y_true == 0) & (y_pred == 1)).sum()
    fn = ((y_true == 1) & (y_pred == 0)).sum()

    p_true = tp / (tp + fp) if (tp + fp) > 0 else 0
    r_true = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_true = 2 * p_true * r_true / (p_true + r_true) if (p_true + r_true) > 0 else 0

    p_false = tn / (tn + fn) if (tn + fn) > 0 else 0
    r_false = tn / (tn + fp) if (tn + fp) > 0 else 0
    f1_false = 2 * p_false * r_false / (p_false + r_false) if (p_false + r_false) > 0 else 0

    acc = (tp + tn) / len(df)
    steps = df['total_steps'].mean() if 'total_steps' in df.columns else 0

    return p_true, r_true, f1_true, p_false, r_false, f1_false, acc, steps

patterns = {
    'FCB': {
        'No-Search': 'fcb_no_search*.csv',
        'Fixed-1': 'fcb_fixed_1*.csv',
        'Fixed-3': 'fcb_fixed_3*.csv',
        'Fixed-5': 'fcb_fixed_5*.csv',
        'FIRE': 'fcb_fire*.csv',
        'NLI Baseline': 'fcb_nli_baseline*.csv',
        'LLM-Critic': 'fcb_llm_critic*.csv',
        'SWIFT': 'fcb_swift*.csv',
    },
    'FacTool': {
        'No-Search': 'factool_no_search*.csv',
        'Fixed-1': 'factool_fixed_1*.csv',
        'Fixed-3': 'factool_fixed_3*.csv',
        'Fixed-5': 'factool_fixed_5*.csv',
        'FIRE': 'factool_fire*.csv',
        'NLI Baseline': 'factool_nli_baseline*.csv',
        'LLM-Critic': 'factool_llm_critic*.csv',
        'SWIFT': 'factool_swift*.csv',
    },
    'FELM': {
        'No-Search': 'felm_no_search*.csv',
        'Fixed-1': 'felm_fixed_1*.csv',
        'Fixed-3': 'felm_fixed_3*.csv',
        'Fixed-5': 'felm_fixed_5*.csv',
        'FIRE': 'felm_fire*.csv',
        'NLI Baseline': 'felm_nli_baseline*.csv',
        'LLM-Critic': 'felm_llm_critic*.csv',
        'SWIFT': 'felm_swift*.csv',
    },
    'BingCheck': {
        'No-Search': 'bingcheck_no_search*.csv',
        'Fixed-1': 'bingcheck_fixed_1*.csv',
        'Fixed-3': 'bingcheck_fixed_3*.csv',
        'Fixed-5': 'bingcheck_fixed_5*.csv',
        'FIRE': 'bingcheck_fire*.csv',
        'NLI Baseline': 'bingcheck_nli_baseline*.csv',
        'LLM-Critic': 'bingcheck_llm_critic*.csv',
        'SWIFT': 'bingcheck_swift*.csv',
    },
    'HoVer': {
        'No-Search': 'hover_no_search*.csv',
        'Fixed-1': 'hover_fixed_1*.csv',
        'Fixed-3': 'hover_fixed_3*.csv',
        'Fixed-5': 'hover_fixed_5*.csv',
        'NLI Baseline': 'hover_nli_baseline*.csv',
        'LLM-Critic': 'hover_llm_critic*.csv',
        'SWIFT': 'hover_swift*.csv',
    }
}

for dataset, methods in patterns.items():
    print('\n### ' + dataset)
    for method, pattern in methods.items():
        matches = glob.glob(os.path.join('predictions', pattern))
        if matches:
            pt, rt, f1t, pf, rf, f1f, acc, steps = calc_metrics(matches[0])
            print("| {} | {:.3f} | {:.3f} | {:.3f} | {:.3f} | {:.3f} | {:.3f} | {:.1%} | {:.2f} |".format(
                method, pt, rt, f1t, pf, rf, f1f, acc, steps))
        else:
            if not (method == 'FIRE' and dataset == 'HoVer'):
                print("# {} NOT FOUND".format(method))
