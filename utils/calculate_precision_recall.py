#!/usr/bin/env python3
"""
从 prediction CSV 计算 Precision/Recall/F1，用于补充 Table 2
支持 True/False 二分类的完整指标
"""

import pandas as pd
import os
import sys
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from common.utils import calculate_metrics


def get_dataset_and_method(filename):
    """从文件名提取数据集和方法名"""
    # 格式: {dataset}_{method}_predictions.csv
    # 例如: fcb_swift_v5_predictions.csv
    parts = filename.replace('_predictions.csv', '').split('_')

    # 映射数据集
    dataset_map = {
        'fcb': 'FCB',
        'factool': 'FacTool',
        'felm': 'FELM',
        'bingcheck': 'BingCheck',
        'hover': 'HoVer'
    }

    # 提取数据集 (第一个部分)
    dataset_key = parts[0]
    dataset = dataset_map.get(dataset_key, dataset_key.upper())

    # 提取方法名 (剩余部分)
    method_parts = parts[1:]

    # 处理特殊方法名
    method_name = '_'.join(method_parts)

    # 标准化方法名映射
    method_mapping = {
        'nosearch': 'No-Search',
        'fixed1': 'Fixed-1',
        'fixed3': 'Fixed-3',
        'fixed5': 'Fixed-5',
        'nli': 'NLI Baseline',
        'llmcritic': 'LLM-Critic',
        'swift': 'SWIFT',
        'swift': 'SWIFT',
        'swift': 'SWIFT',  # t=0.5
        'fire': 'FIRE'
    }

    # 处理 threshold 变体 (swift_t01, t02, 等)
    if method_name.startswith('swift_t') and len(method_name) > 7:
        threshold = method_name[6:8]  # 提取 "01", "02", etc.
        method = f'SWIFT (t=0.{threshold[1]})'
    elif method_name.startswith('swift'):
        method = 'SWIFT'
    elif method_name in method_mapping:
        method = method_mapping[method_name]
    else:
        method = method_name.upper()

    return dataset, method


def calculate_detailed_metrics(df):
    """计算详细的 True/False 分类指标"""
    # 处理布尔值和字符串两种情况
    # 确保 label 和 prediction 是布尔值
    df['label_bool'] = df['label'].apply(lambda x: True if x in [True, 'True', 'true', 1, '1'] else False)
    df['prediction_bool'] = df['prediction'].apply(lambda x: True if x in [True, 'True', 'true', 1, '1'] else False)

    # 分离 True 和 False 类
    true_df = df[df['label_bool'] == True]
    false_df = df[df['label_bool'] == False]

    # True 类指标
    true_tp = len(true_df[true_df['prediction_bool'] == True])
    true_fp = len(false_df[false_df['prediction_bool'] == True])
    true_fn = len(true_df[true_df['prediction_bool'] == False])

    true_precision = true_tp / (true_tp + true_fp) if (true_tp + true_fp) > 0 else 0.0
    true_recall = true_tp / (true_tp + true_fn) if (true_tp + true_fn) > 0 else 0.0
    true_f1 = 2 * true_precision * true_recall / (true_precision + true_recall) if (true_precision + true_recall) > 0 else 0.0

    # False 类指标
    false_tp = len(false_df[false_df['prediction_bool'] == False])
    false_fp = len(true_df[true_df['prediction_bool'] == False])
    false_fn = len(false_df[false_df['prediction_bool'] == True])

    false_precision = false_tp / (false_tp + false_fp) if (false_tp + false_fp) > 0 else 0.0
    false_recall = false_tp / (false_tp + false_fn) if (false_tp + false_fn) > 0 else 0.0
    false_f1 = 2 * false_precision * false_recall / (false_precision + false_recall) if (false_precision + false_recall) > 0 else 0.0

    # 总体准确率
    accuracy = df['correct'].mean()

    # 平均步数
    avg_steps = df['total_steps'].mean()

    return {
        'true_precision': true_precision,
        'true_recall': true_recall,
        'true_f1': true_f1,
        'false_precision': false_precision,
        'false_recall': false_recall,
        'false_f1': false_f1,
        'accuracy': accuracy,
        'avg_steps': avg_steps,
        'num_claims': len(df),
        'num_true': len(true_df),
        'num_false': len(false_df)
    }


def process_csv(csv_path):
    """处理单个 CSV 文件"""
    try:
        df = pd.read_csv(csv_path)

        # 检查必需列
        required_cols = ['label', 'prediction', 'correct', 'total_steps']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            print(f"  Warning: Missing columns {missing_cols}, skipping...")
            return None

        metrics = calculate_detailed_metrics(df)
        return metrics

    except Exception as e:
        print(f"  Error processing {csv_path}: {e}")
        return None


def main():
    # 定义需要处理的文件 (只处理主要方法，不包括 threshold 变体)
    datasets = ['fcb', 'factool', 'felm', 'bingcheck', 'hover']
    methods = ['nosearch', 'fixed1', 'fixed3', 'fixed5', 'nli', 'llmcritic', 'swift_v5']

    # FIRE baseline 使用不同的文件名格式，从 FIRE 目录读取
    fire_datasets = ['fcb', 'factool', 'felm', 'bingcheck']  # hover 没有 FIRE

    # 结果存储
    results = {}

    print("=" * 80)
    print("计算所有方法的 Precision/Recall/F1")
    print("=" * 80)

    # 处理 SWIFT 各方法
    for dataset in datasets:
        results[dataset.upper()] = {}

        for method in methods:
            filename = f"{dataset}_{method}_predictions.csv"

            # 检查文件是否存在
            csv_path = f"/home1/pzm/swift/predictions/{filename}"
            if not os.path.exists(csv_path):
                print(f"Warning: {filename} not found, skipping...")
                continue

            print(f"\n[{dataset.upper()}] Processing: {method}")

            # 提取方法名用于显示
            method_display = method.replace('_', ' ').title().replace('V5', 'v5').replace('Nli', 'NLI').replace('Llmcritic', 'LLM-Critic').replace('Fixed1', 'Fixed-1').replace('Fixed3', 'Fixed-3').replace('Fixed5', 'Fixed-5').replace('Nosearch', 'No-Search')

            metrics = process_csv(csv_path)
            if metrics:
                results[dataset.upper()][method_display] = metrics

                # 打印结果
                print(f"  True:  P={metrics['true_precision']:.3f}, R={metrics['true_recall']:.3f}, F1={metrics['true_f1']:.3f}")
                print(f"  False: P={metrics['false_precision']:.3f}, R={metrics['false_recall']:.3f}, F1={metrics['false_f1']:.3f}")
                print(f"  Acc: {metrics['accuracy']:.3f}, Steps: {metrics['avg_steps']:.2f}")

    # 处理 FIRE baseline (从 ~/fire/predictions/ 读取)
    print("\n" + "=" * 80)
    print("Processing FIRE baseline...")
    print("=" * 80)

    for dataset in fire_datasets:
        filename = f"{dataset}_fire_ddg_predictions.csv"
        csv_path = f"/home1/pzm/fire/predictions/{filename}"

        if not os.path.exists(csv_path):
            print(f"Warning: FIRE {filename} not found, skipping...")
            continue

        print(f"\n[{dataset.upper()}] Processing: FIRE (DDG)")

        metrics = process_csv(csv_path)
        if metrics:
            method_display = 'FIRE (DDG)'
            if dataset.upper() not in results:
                results[dataset.upper()] = {}
            results[dataset.upper()][method_display] = metrics

            # 打印结果
            print(f"  True:  P={metrics['true_precision']:.3f}, R={metrics['true_recall']:.3f}, F1={metrics['true_f1']:.3f}")
            print(f"  False: P={metrics['false_precision']:.3f}, R={metrics['false_recall']:.3f}, F1={metrics['false_f1']:.3f}")
            print(f"  Acc: {metrics['accuracy']:.3f}, Steps: {metrics['avg_steps']:.2f}")

    # 生成 Markdown 表格
    print("\n" + "=" * 80)
    print("生成 Markdown 表格")
    print("=" * 80)

    # 方法定义顺序
    method_order = ['No-Search', 'Fixed-1', 'Fixed-3', 'Fixed-5', 'FIRE (Ddg)', 'NLI Baseline', 'LLM-Critic', 'Swift V5']

    for dataset in ['FCB', 'Factool', 'FELM', 'Bingcheck', 'Hover']:
        if dataset not in results:
            continue

        print(f"\n### {dataset}\n")
        print("| Method | True P | True R | True F1 | False P | False R | False F1 | Acc | Steps |")
        print("|--------|:------:|:------:|:-------:|:-------:|:-------:|:--------:|:---:|:-----:|")

        # 按顺序输出方法
        for target_method in ['No-Search', 'Fixed-1', 'Fixed-3', 'Fixed-5', 'FIRE (DDG)', 'NLI Baseline', 'LLM-Critic', 'SWIFT']:
            # 查找匹配的方法名
            method_key = None
            for key in results[dataset].keys():
                normalized_key = key.replace(' ', '').replace('-', '').lower()
                normalized_target = target_method.replace(' ', '').replace('-', '').lower()
                if normalized_key == normalized_target or normalized_key.replace('v5', '') == normalized_target.lower() or normalized_key.replace('(ddg)', '') == normalized_target.lower():
                    method_key = key
                    break

            if method_key is None:
                continue

            m = results[dataset][method_key]
            method_display = method_key.replace('V5', 'v5').replace('(DDG)', '(DDG)').replace('(Ddg)', '(DDG)')

            print(f"| {method_display} | {m['true_precision']:.3f} | {m['true_recall']:.3f} | {m['true_f1']:.3f} | {m['false_precision']:.3f} | {m['false_recall']:.3f} | {m['false_f1']:.3f} | {m['accuracy']:.1%} | {m['avg_steps']:.2f} |")


if __name__ == '__main__':
    main()
