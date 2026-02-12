#!/bin/bash
cd ~/swift
python3 << 'PYEOF'
import os
import glob
import pandas as pd

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
    'FCB': [
        ('No-Search', 'fcb_no_search'),
        ('Fixed-1', 'fcb_fixed_1'),
        ('Fixed-3', 'fcb_fixed_3'),
        ('Fixed-5', 'fcb_fixed_5'),
        ('FIRE', 'fcb_fire'),
        ('NLI Baseline', 'fcb_nli_baseline'),
        ('LLM-Critic', 'fcb_llm_critic'),
        ('SWIFT', 'fcb_swift'),
    ],
    'FacTool': [
        ('No-Search', 'factool_no_search'),
        ('Fixed-1', 'factool_fixed_1'),
        ('Fixed-3', 'factool_fixed_3'),
        ('Fixed-5', 'factool_fixed_5'),
        ('FIRE', 'factool_fire'),
        ('NLI Baseline', 'factool_nli_baseline'),
        ('LLM-Critic', 'factool_llm_critic'),
        ('SWIFT', 'factool_swift'),
    ],
    'FELM': [
        ('No-Search', 'felm_no_search'),
        ('Fixed-1', 'felm_fixed_1'),
        ('Fixed-3', 'felm_fixed_3'),
        ('Fixed-5', 'felm_fixed_5'),
        ('FIRE', 'felm_fire'),
        ('NLI Baseline', 'felm_nli_baseline'),
        ('LLM-Critic', 'felm_llm_critic'),
        ('SWIFT', 'felm_swift'),
    ],
    'BingCheck': [
        ('No-Search', 'bingcheck_no_search'),
        ('Fixed-1', 'bingcheck_fixed_1'),
        ('Fixed-3', 'bingcheck_fixed_3'),
        ('Fixed-5', 'bingcheck_fixed_5'),
        ('FIRE', 'bingcheck_fire'),
        ('NLI Baseline', 'bingcheck_nli_baseline'),
        ('LLM-Critic', 'bingcheck_llm_critic'),
        ('SWIFT', 'bingcheck_swift'),
    ],
    'HoVer': [
        ('No-Search', 'hover_no_search'),
        ('Fixed-1', 'hover_fixed_1'),
        ('Fixed-3', 'hover_fixed_3'),
        ('Fixed-5', 'hover_fixed_5'),
        ('NLI Baseline', 'hover_nli_baseline'),
        ('LLM-Critic', 'hover_llm_critic'),
        ('SWIFT', 'hover_swift'),
    ]
}

for dataset, methods in patterns.items():
    print('')
    print('### ' + dataset)
    for method, prefix in methods:
        matches = glob.glob('predictions/' + prefix + '*.csv')
        if matches:
            pt, rt, f1t, pf, rf, f1f, acc, steps = calc_metrics(matches[0])
            print("| {} | {:.3f} | {:.3f} | {:.3f} | {:.3f} | {:.3f} | {:.3f} | {:.1%} | {:.2f} |".format(
                method, pt, rt, f1t, pf, rf, f1f, acc, steps))
        else:
            if not (method == 'FIRE' and dataset == 'HoVer'):
                print("# {} NOT FOUND".format(method))
PYEOF
