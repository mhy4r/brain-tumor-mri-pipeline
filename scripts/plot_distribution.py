#!/usr/bin/env python3
import json
import re
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# ── dark theme defaults ──
plt.rcParams.update({
    'text.color': '#E2E8F0',
    'axes.labelcolor': '#E2E8F0',
    'axes.titlecolor': '#FFFFFF',
    'xtick.color': '#E2E8F0',
    'ytick.color': '#E2E8F0',
    'legend.labelcolor': '#E2E8F0',
    'legend.edgecolor': '#4A5568',
    'axes.edgecolor': '#4A5568',
    'axes.facecolor': '#1A202C',
    'figure.facecolor': '#1A202C',
    'savefig.facecolor': '#1A202C',
})

DATA_JSON = os.path.join(os.path.dirname(__file__), '..', '..', 'dataset', 'DATA.json')
OUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'reports', 'figures')
os.makedirs(OUT_DIR, exist_ok=True)

WEIGHTINGS = ['T1', 'T2', 'T1C+']


def parse_class(class_str):
    """Split 'Glioma T1C+' -> ('Glioma', 'T1C+')"""
    for w in WEIGHTINGS:
        if class_str.endswith(w):
            tumor = class_str[:-len(w)].strip()
            return tumor, w
    return class_str.strip(), 'Unknown'


def extract_patient_id(filename):
    """Extract patient id from image filename."""
    base = os.path.basename(filename)
    m = re.search(r'(\d+)\.(jpg|png)$', base, re.IGNORECASE)
    return m.group(1) if m else base


def load_data():
    with open(DATA_JSON) as f:
        data = json.load(f)

    # {tumor -> count}
    tumor_img = {}
    tumor_patients = {}
    # {weighting -> count}
    weight_img = {}
    weight_patients = {}
    # {(tumor, weighting) -> count}
    combo_img = {}
    combo_patients = {}

    for filename, info in data.items():
        class_str = info.get('class', 'Unknown')
        tumor, weight = parse_class(class_str)
        pid = extract_patient_id(filename)

        # image counts
        tumor_img[tumor] = tumor_img.get(tumor, 0) + 1
        weight_img[weight] = weight_img.get(weight, 0) + 1
        key = (tumor, weight)
        combo_img[key] = combo_img.get(key, 0) + 1

        # patient-unique counts
        if tumor not in tumor_patients:
            tumor_patients[tumor] = set()
        tumor_patients[tumor].add(pid)

        if weight not in weight_patients:
            weight_patients[weight] = set()
        weight_patients[weight].add(pid)

        if key not in combo_patients:
            combo_patients[key] = set()
        combo_patients[key].add(pid)

    return tumor_img, {t: len(s) for t, s in tumor_patients.items()}, \
           weight_img, {w: len(s) for w, s in weight_patients.items()}, \
           combo_img, {k: len(s) for k, s in combo_patients.items()}


def plot_barh(ax, labels, values, title, xlabel, color='#319795', value_label=True):
    sorted_pairs = sorted(zip(values, labels))
    values, labels = zip(*sorted_pairs)
    bars = ax.barh(range(len(labels)), values, color=color, edgecolor='#2D3748', height=0.6)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=10)
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_title(title, fontsize=12, fontweight='bold')
    if value_label:
        for bar, v in zip(bars, values):
            ax.text(bar.get_width() + max(values) * 0.005,
                    bar.get_y() + bar.get_height() / 2,
                    str(v), va='center', fontsize=9, color='#E2E8F0')


def plot_combo_matrix(ax, tumors, weights, combo_dict, title, label):
    matrix = np.zeros((len(tumors), len(weights)), dtype=int)
    for i, t in enumerate(tumors):
        for j, w in enumerate(weights):
            matrix[i, j] = combo_dict.get((t, w), 0)
    im = ax.imshow(matrix, cmap='YlGnBu', aspect='auto')
    ax.set_xticks(range(len(weights)))
    ax.set_xticklabels(weights, fontsize=11)
    ax.set_yticks(range(len(tumors)))
    ax.set_yticklabels(tumors, fontsize=9)
    ax.set_title(title, fontsize=12, fontweight='bold')
    for i in range(len(tumors)):
        for j in range(len(weights)):
            val = matrix[i, j]
            if val > 0:
                color = 'white' if val > matrix.max() / 2 else '#1A202C'
                ax.text(j, i, str(val), ha='center', va='center', fontsize=9, color=color)
    plt.colorbar(im, ax=ax, label=label)


def main():
    tumor_img, tumor_pat, weight_img, weight_pat, combo_img, combo_pat = load_data()

    # ── Panel A: Tumor × Weighting image heatmap ──
    fig, ax = plt.subplots(figsize=(12, 7))
    tumors = sorted(tumor_img.keys())
    plot_combo_matrix(ax, tumors, WEIGHTINGS, combo_img,
                      'Image Count per Tumor Type × MRI Weighting', 'Images')
    plt.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, 'tumor_class_distribution.png'), dpi=150, facecolor='#1A202C')
    plt.close()
    print('Saved: tumor_class_distribution.png')

    # ── Panel B: Tumor × Weighting patient heatmap ──
    fig, ax = plt.subplots(figsize=(12, 7))
    plot_combo_matrix(ax, tumors, WEIGHTINGS, combo_pat,
                      'Unique Patient Count per Tumor Type × MRI Weighting', 'Patients')
    plt.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, 'tumor_class_distribution_patients.png'), dpi=150, facecolor='#1A202C')
    plt.close()
    print('Saved: tumor_class_distribution_patients.png')

    # ── Panel C: Grouped bar — images vs patients per tumor ──
    tumors_sorted = sorted(tumor_img.keys(), key=lambda t: tumor_img[t], reverse=True)
    x = np.arange(len(tumors_sorted))
    width = 0.35
    fig, ax = plt.subplots(figsize=(14, 7))
    bars1 = ax.bar(x - width/2, [tumor_img[t] for t in tumors_sorted], width,
                   label='Images', color='#319795', edgecolor='#2D3748')
    bars2 = ax.bar(x + width/2, [tumor_pat[t] for t in tumors_sorted], width,
                   label='Patients', color='#7C3AED', edgecolor='#2D3748')
    ax.set_xticks(x)
    ax.set_xticklabels(tumors_sorted, rotation=45, ha='right', fontsize=11)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title('Image Count vs Patient Count per Tumor Type', fontsize=14, fontweight='bold')
    ax.legend(fontsize=12)
    for bar in bars1:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + max(tumor_img.values()) * 0.005,
                str(int(h)), ha='center', va='bottom', fontsize=9, color='#E2E8F0')
    for bar in bars2:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + max(tumor_img.values()) * 0.005,
                str(int(h)), ha='center', va='bottom', fontsize=9, color='#E2E8F0')
    plt.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, 'tumor_img_vs_patients.png'), dpi=150, facecolor='#1A202C')
    plt.close()
    print('Saved: tumor_img_vs_patients.png')

    # ── Panel D: Weighting totals (bar) ──
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    colors = ['#00D2FF', '#00E676', '#FF9F43']
    w_names = [w for w in WEIGHTINGS if w in weight_img]
    w_vals = [weight_img[w] for w in w_names]
    bars = ax1.bar(w_names, w_vals, color=colors[:len(w_names)], edgecolor='#2D3748', width=0.5)
    ax1.set_title('Images per MRI Weighting', fontweight='bold')
    ax1.set_ylabel('Image Count')
    for bar, v in zip(bars, w_vals):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(w_vals) * 0.01,
                 str(v), ha='center', fontsize=10, fontweight='bold', color='#E2E8F0')

    w_pat = [weight_pat[w] for w in w_names]
    bars = ax2.bar(w_names, w_pat, color=colors[:len(w_names)], edgecolor='#2D3748', width=0.5)
    ax2.set_title('Patients per MRI Weighting', fontweight='bold')
    ax2.set_ylabel('Patient Count')
    for bar, v in zip(bars, w_pat):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(w_pat) * 0.01,
                 str(v), ha='center', fontsize=10, fontweight='bold', color='#E2E8F0')
    plt.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, 'mri_weighting_distribution.png'), dpi=150, facecolor='#1A202C')
    plt.close()
    print('Saved: mri_weighting_distribution.png')

    # ── print summary ──
    print(f'\nTotal images: {sum(tumor_img.values())}')
    print(f'Total unique patients: {sum(tumor_pat.values())}')
    print(f'\nTumor classes: {len(tumor_img)}')
    for t in sorted(tumor_img, key=lambda x: tumor_img[x], reverse=True):
        print(f'  {t:30s} images={tumor_img[t]:4d}  patients={tumor_pat[t]:4d}')
    print('\nWeightings:')
    for w in w_names:
        print(f'  {w:6s}  images={weight_img[w]:5d}  patients={weight_pat[w]:5d}')
    print('\nDone.')


if __name__ == '__main__':
    main()