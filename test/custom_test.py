import os
import csv

import numpy as np

import dla_common as common
import matplotlib.pyplot as plt

RESULTS_DIR = common.RESULTS_DIR
LABELS = {1: "Cfg 1\nrandom 10k", 2: "Cfg 2\nmulti 15k",
          3: "Cfg 3\nradial 10k", 4: "Cfg 4\nradial 25k"}
COLORS = {1: "#1f77b4", 2: "#d62728", 3: "#2ca02c", 4: "#9467bd"}


def comparison_growth(bundles):
    fig = plt.figure(figsize=(9, 6))
    for cid, b in bundles.items():
        r = b["results"]
        th = np.asarray(r["time-history"])
        gh = np.asarray(r["growth-history"])
        nw = b["metrics"]["n_walkers"]
        plt.plot(th, gh / nw * 100.0, color=COLORS[cid], linewidth=2.2,
                 label=LABELS[cid].replace("\n", " "))
    plt.title("Growth comparison (completion vs iteration)",
              fontsize=14, fontweight="bold")
    plt.xlabel("Iteration", fontsize=12)
    plt.ylabel("Completion (% of target walkers)", fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(fontsize=11)
    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, "comparison_growth.png")
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def comparison_metrics(bundles):
    cids = sorted(bundles)
    labels = [LABELS[c] for c in cids]
    colors = [COLORS[c] for c in cids]
    comp = [bundles[c]["metrics"]["completion_rate"] * 100 for c in cids]
    thr = [bundles[c]["metrics"]["throughput_pps"] for c in cids]
    eff = [bundles[c]["metrics"]["efficiency_ppi"] for c in cids]
    wall = [bundles[c]["metrics"]["wall_time_s"] for c in cids]

    panels = [
        ("Completion rate (%)", comp, "%.0f"),
        ("Speed: throughput (particles/s)", thr, "%.0f"),
        ("Efficiency (particles/iteration)", eff, "%.2f"),
        ("Wall-clock time (s)", wall, "%.1f"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    fig.suptitle("Cross-configuration comparison", fontsize=15, fontweight="bold")
    for ax, (title, vals, fmt) in zip(axes.ravel(), panels):
        bars = ax.bar(labels, vals, color=colors, alpha=0.85)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.grid(True, axis="y", linestyle="--", alpha=0.5)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    fmt % v, ha="center", va="bottom", fontsize=10)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    path = os.path.join(RESULTS_DIR, "comparison_metrics.png")
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def write_csv(bundles):
    fields = ["name", "N", "n_seeds", "n_walkers", "injection", "inj_radius",
              "n_stuck", "aggregated", "completion_rate", "n_aggregate",
              "wall_time_s", "iters_sampled", "throughput_pps", "efficiency_ppi",
              "iters_to_25pct", "iters_to_50pct", "iters_to_75pct", "iters_to_90pct",
              "fractal_dimension"]
    path = os.path.join(RESULTS_DIR, "metrics_summary.csv")
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for cid in sorted(bundles):
            w.writerow({k: bundles[cid]["metrics"].get(k) for k in fields})


def write_analysis(bundles):
    m = {cid: bundles[cid]["metrics"] for cid in sorted(bundles)}

    def row(cid):
        x = m[cid]
        df = f"{x['fractal_dimension']:.3f}" if x["fractal_dimension"] is not None else "not measured"
        return (f"| {cid} | {x['injection']}"
                + (f" (R={x['inj_radius']})" if x['injection'] == 'radial' else "")
                + f" | {x['n_walkers']} | {x['completion_rate']*100:.1f}% "
                f"| {x['n_aggregate']} | {x['wall_time_s']:.1f} "
                f"| {x['throughput_pps']:.0f} | {x['efficiency_ppi']:.3f} | {df} |")

    fastest = max(m, key=lambda c: m[c]["throughput_pps"])
    most_eff = max(m, key=lambda c: m[c]["efficiency_ppi"])
    best_comp = max(m, key=lambda c: m[c]["completion_rate"])

    lines = [
        "# DLA test results - measured analysis",
        "",
        f"All runs: lattice N=512, seed=42 (reproducible). Numbers below are "
        f"measured by `test/custom_test.py`, not hand-written.",
        "",
        "## Summary table",
        "",
        "| Cfg | Injection | Walkers | Completion | Clusters | Time (s) | "
        "Throughput (part/s) | Efficiency (part/iter) | D_f |",
        "|-----|-----------|---------|-----------|----------|----------|"
        "--------------------|------------------------|-----|",
    ]
    lines += [row(c) for c in sorted(m)]
    lines += [
        "",
        "## Growth",
        "",
        "`comparison_growth.png` overlays completion (% of target walkers) vs "
        "iteration. Iterations to reach growth milestones:",
        "",
        "| Cfg | 25% | 50% | 75% | 90% |",
        "|-----|-----|-----|-----|-----|",
    ]
    for c in sorted(m):
        x = m[c]
        lines.append(f"| {c} | {common._fmt(x['iters_to_25pct'])} | "
                     f"{common._fmt(x['iters_to_50pct'])} | "
                     f"{common._fmt(x['iters_to_75pct'])} | "
                     f"{common._fmt(x['iters_to_90pct'])} |")
    lines += [
        "",
        "Random injection (Cfg 1, 2) spawns walkers in a bounding box hugging "
        "the cluster, so they stick within very few iterations; the curves rise "
        "steeply. Radial injection (Cfg 3, 4) launches walkers from a circle of "
        "radius 180, so each walker must diffuse inward before sticking and the "
        "curves rise more gradually.",
        "",
        "## Completion",
        "",
        f"Highest completion within budget: **Config {best_comp}** "
        f"({m[best_comp]['completion_rate']*100:.1f}%). "
        "Completion rate is the fraction of injected walkers that joined the "
        "aggregate before the step budget ran out. The multi-seed run (Config 2) "
        f"produced {m[2]['n_aggregate']} separate clusters via flood-fill, which "
        "is why a single centre-based fractal dimension is not reported for it.",
        "",
        "## Speed",
        "",
        f"Fastest throughput: **Config {fastest}** "
        f"({m[fastest]['throughput_pps']:.0f} particles/s). "
        f"Most iteration-efficient (particles stuck per iteration): "
        f"**Config {most_eff}** ({m[most_eff]['efficiency_ppi']:.3f}). "
        "Throughput is particles stuck per wall-clock second; efficiency is "
        "particles stuck per simulation iteration (independent of CPU speed). "
        "Radial configurations have lower efficiency because many iterations are "
        "spent diffusing rather than sticking.",
        "",
        "## Figures (in results/)",
        "",
        "- `configN_structure.png` - aggregate at ~10% / ~50% / 100% growth",
        "- `configN_growth.png` - growth curve for that configuration",
        "- `configN_mass_radius.png` - log-log mass-radius fit (Configs 1, 3, 4)",
        "- `comparison_growth.png`, `comparison_metrics.png` - cross-config plots",
        "",
    ]
    path = os.path.join(RESULTS_DIR, "ANALYSIS.md")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))


def main():
    bundles = {}
    for cid in (1, 2, 3, 4):
        print(f"\n===== Running configuration {cid} =====")
        bundles[cid] = common.run_and_save(cid)

    print("\n===== Building comparison artefacts =====")
    comparison_growth(bundles)
    comparison_metrics(bundles)
    write_csv(bundles)
    write_analysis(bundles)
    print(f"All outputs written to: {RESULTS_DIR}")
    for f in sorted(os.listdir(RESULTS_DIR)):
        print("  -", f)


if __name__ == "__main__":
    main()
