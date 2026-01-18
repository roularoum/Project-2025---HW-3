#!/usr/bin/env python3
"""Create Recall@N vs QPS trade-off plots from grid search CSV files."""

import csv
import sys
from pathlib import Path

try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    import numpy as np
except ImportError:
    print("Error: matplotlib and numpy required. Install with: pip install matplotlib numpy")
    sys.exit(1)


def load_csv(csv_path: Path) -> list[dict]:
    """Load grid search CSV and return list of dicts."""
    rows = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({
                'recall': float(row['avg_recall']),
                'qps': float(row['qps']),
                'method': row['method'],
            })
    return rows


def create_combined_plot(output_path: Path) -> None:
    """Create combined Recall@50 vs QPS plot for all methods."""
    csv_files = {
        'Euclidean LSH': Path('grid_lsh.csv'),
        'Hypercube': Path('grid_hypercube.csv'),
        'IVF-Flat': Path('grid_ivfflat.csv'),
        'IVF-PQ': Path('grid_ivfpq.csv'),
        'Neural LSH': Path('grid_neural.csv'),
    }
    
    fig, ax = plt.subplots(figsize=(10, 7))
    
    colors = {
        'Euclidean LSH': '#1f77b4',
        'Hypercube': '#ff7f0e',
        'IVF-Flat': '#2ca02c',
        'IVF-PQ': '#d62728',
        'Neural LSH': '#9467bd',
    }
    
    markers = {
        'Euclidean LSH': 'o',
        'Hypercube': 's',
        'IVF-Flat': '^',
        'IVF-PQ': 'D',
        'Neural LSH': 'v',
    }
    
    for method_name, csv_file in csv_files.items():
        csv_path = Path(__file__).parent / csv_file
        if not csv_path.exists():
            print(f"Warning: {csv_file} not found, skipping")
            continue
        
        rows = load_csv(csv_path)
        if not rows:
            continue
        
        recalls = [r['recall'] for r in rows]
        qps_list = [r['qps'] for r in rows]
        
        ax.scatter(
            qps_list,
            recalls,
            label=method_name,
            color=colors.get(method_name, '#000000'),
            marker=markers.get(method_name, 'o'),
            s=80,
            alpha=0.7,
            edgecolors='black',
            linewidths=0.5,
        )
    
    ax.set_xlabel('QPS (Queries Per Second)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Recall@50 (vs BLAST Top-50)', fontsize=12, fontweight='bold')
    ax.set_title('Trade-off: Recall@50 vs QPS across ANN Methods', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=10, framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_xscale('log')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Created plot: {output_path}")
    plt.close()


def main():
    script_dir = Path(__file__).parent
    output_path = script_dir / 'recall_vs_qps_plot.png'
    create_combined_plot(output_path)


if __name__ == '__main__':
    main()
