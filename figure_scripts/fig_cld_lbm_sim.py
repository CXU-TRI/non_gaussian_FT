"""
Generate CLD (Compact Letter Display) violin plots for simulation experiments.

Settings covered:
  A, A', B_DP, B_FP, B_FP_noise, B_DP_1_10_lr, explore1_DP

Paper figures produced:
  Fig 2  — setting_A_RI_avg.png, setting_A_FT_avg.png
  Fig 3  — setting_B_DP_FT_avg.png
  Fig 9  — setting_B_DP_1_10_lr_FT_avg.png
  Fig 11 — setting_B_DP_FT_avg_3_seen.png, setting_B_DP_FT_avg_5_unseen.png
  Fig 12 — setting_B_FP_FT_avg.png
  Fig 13 — setting_B_FP_noise_FT_avg.png
  Fig 14 — setting_Aprime_FT_avg.png

Usage:
    python fig_cld_sim.py                       # process all
    python fig_cld_sim.py --settings setting_A  # one setting
"""

import argparse
import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from sequentialized_barnard_tests import Decision, Hypothesis
from sequentialized_barnard_tests.lai import MirroredLaiTest
from sequentialized_barnard_tests.step import MirroredStepTest

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PAPER_DIR = "/home/chenxu/Downloads/anzu_personal/Videos/New_prior_paper/New_prior_paper"  # website copy: data still from the paper
sys.path.insert(0, os.path.join(PAPER_DIR, "scripts"))  # for sibling imports (fig_introspection)
FIG_DIR = os.path.join(SCRIPT_DIR, "_build")  # website copy: write here, not into the paper
JSON_DIR = os.path.expanduser("~/Downloads/anzu_personal/Videos/Z_New_prior_final")

# ---------------------------------------------------------------------------
# Display-name mapping
# ---------------------------------------------------------------------------

DISPLAY_NAMES = {
    "Z":               "$\\mathcal{Z}$",
    "A_LBM":           "$A_{\\mathrm{pre}}$",
    "A_LBMplus025Z":   "$A_{\\mathrm{pre}}{+}0.25\\mathcal{Z}$",
    "A_LBMplus05Z":    "$A_{\\mathrm{pre}}{+}0.5\\mathcal{Z}$",
    "A_LBMplusZ":      "$A_{\\mathrm{pre}}{+}\\sigma_{A}\\mathcal{Z}$",
    "O_LBMplus025Z":   "$E_{\\mathrm{pre}}{+}\\sigma_{E}\\mathcal{Z}$",
    "O_LBMplus05Z":    "$E_{\\mathrm{pre}}{+}0.5\\mathcal{Z}$",
    "O_LBMplusZ":      "$E_{\\mathrm{pre}}{+}\\mathcal{Z}$",
    "Cocos":           "Cocos",
    "BRIDGER":         "BRIDGER",
    "Retrieval":       "Retrieval",
    "LORA":            "LORA",
    "DRIFT":           "DRIFT",
    "FULL":            "FULL",
}


# Map from output-file stem to custom title (for avg plots only)
TITLE_OVERRIDES = {
    # website copy: no title on these figures (empty -> set_title skipped)
    "setting_A_RI_avg":              "",
    "setting_A_FT_avg":              "",
    "setting_B_DP_FT_avg":           "",
    "setting_B_DP_1_10_lr_FT_avg":   "",
    "setting_B_DP_FT_avg_3_seen":    r"Setting $B_{\mathrm{sim}}$, 3 Easier Tasks",
    "setting_B_DP_FT_avg_5_unseen":  r"Setting $B_{\mathrm{sim}}$, 5 Harder Tasks",
    "setting_B_FP_FT_avg":           "Setting $B_{FP}$ FM Objective",
    "setting_B_FP_noise_FT_avg":     "Setting $B_{FP}$ Noise Ablation",
    "setting_Aprime_FT_avg":         "Setting A$'$",
    "setting_explore1_DP_FT_avg":    "Setting B (Exploration)",
}


def _display_name(method_key):
    """Return the display name for a method key, handling _1_10 variants."""
    if method_key.endswith("_1_10"):
        base = method_key[:-5]  # strip '_1_10'
        base_disp = DISPLAY_NAMES.get(base, base)
        return f"{base_disp} (w/ 1/10)"
    return DISPLAY_NAMES.get(method_key, method_key)

# ---------------------------------------------------------------------------
# CLD computation
# ---------------------------------------------------------------------------

def compact_letter_display(significant_pair_list, sorted_model_list):
    num_models = len(sorted_model_list)
    model_to_index = {m: i for i, m in enumerate(sorted_model_list)}
    significant_index_pairs = [
        (model_to_index[m1], model_to_index[m2])
        for m1, m2 in significant_pair_list
    ]

    def remove_redundant_columns(matrix):
        changed = True
        while changed:
            changed = False
            for i in range(len(matrix)):
                for j in range(len(matrix)):
                    if i != j:
                        si = {idx for idx, c in enumerate(matrix[i]) if c}
                        sj = {idx for idx, c in enumerate(matrix[j]) if c}
                        if si.issubset(sj):
                            matrix.pop(i)
                            changed = True
                            break
                if changed:
                    break
        return matrix

    letter_matrix = [["a"] * num_models]
    for m1, m2 in significant_index_pairs:
        while any(col[m1] and col[m2] for col in letter_matrix):
            for ci, col in enumerate(letter_matrix):
                if col[m1] and col[m2]:
                    new_col = col.copy()
                    new_col[m1] = ""
                    col[m2] = ""
                    letter_matrix[ci] = col
                    letter_matrix.append(new_col)
                    letter_matrix = remove_redundant_columns(letter_matrix)
                    break

    def _col_sort_key(col):
        first = next((i for i, c in enumerate(col) if c), len(col))
        last = next((i for i, c in enumerate(reversed(col)) if c), -1)
        last = len(col) - 1 - last if last >= 0 else len(col)
        return (first, last)

    letter_matrix.sort(key=_col_sort_key)
    for idx, col in enumerate(letter_matrix):
        r = chr(ord("a") + idx)
        letter_matrix[idx] = [r if c else "" for c in col]

    result = []
    for mi in range(num_models):
        letters = "".join(
            letter_matrix[ci][mi] for ci in range(len(letter_matrix))
            if letter_matrix[ci][mi]
        )
        result.append(letters)
    return result


def compare_success_and_get_cld(
    model_name_list, success_array_list,
    global_confidence_level=0.95, max_sample_size_per_model=200,
    rng=None, shuffle=False, test_method_name="step",
):
    if rng is None:
        rng = np.random.default_rng(42)
    num_models = len(model_name_list)
    global_alpha = 1 - global_confidence_level
    num_comparisons = max(1, num_models * (num_models - 1) // 2)
    individual_alpha = global_alpha / num_comparisons

    if test_method_name == "step":
        TestClass = MirroredStepTest
    else:
        TestClass = MirroredLaiTest

    test = TestClass(
        alternative=Hypothesis.P0LessThanP1,
        alpha=individual_alpha,
        n_max=max_sample_size_per_model,
    )
    test.reset()

    arrays = {}
    for name, arr in zip(model_name_list, success_array_list):
        a = np.array(arr, dtype=bool).copy()
        if shuffle:
            rng.shuffle(a)
        arrays[name] = a

    comparisons = {}
    for ia in range(num_models):
        for ib in range(ia + 1, num_models):
            ma, mb = model_name_list[ia], model_name_list[ib]
            aa, ab_ = arrays[ma], arrays[mb]
            n = min(len(aa), len(ab_))
            comparisons[(ma, mb)] = test.run_on_sequence(aa[:n], ab_[:n]).decision

    sig_pairs = [k for k, v in comparisons.items() if v != Decision.FailToDecide]
    sorted_models = [
        m for m, _ in sorted(
            arrays.items(), key=lambda kv: np.mean(kv[1]) if len(kv[1]) else 0.0,
            reverse=True,
        )
    ]
    letters = compact_letter_display(sig_pairs, sorted_models)
    return {m: l for m, l in zip(sorted_models, letters)}


def draw_samples_from_beta_posterior(success_array, rng, num_samples=10000):
    n = len(success_array)
    s = int(np.sum(success_array))
    return stats.beta(1 + s, 1 + n - s).rvs(num_samples, random_state=rng)

# ---------------------------------------------------------------------------
# Color map
# ---------------------------------------------------------------------------

METHOD_COLORS = {
    "Z": "#2ca02c",
    "A_LBM": "#e377c2",
    "A_LBMplus025Z": "#1f77b4",
    "A_LBMplus05Z": "#6baed6",
    "A_LBMplusZ": "#08519c",
    "O_LBMplus025Z": "#ff7f0e",
    "O_LBMplus05Z": "#ffbb78",
    "O_LBMplusZ": "#d62728",
    "Cocos": "#17becf",
    "BRIDGER": "#bcbd22",
    "Retrieval": "#555555",
    "LORA": "#9467bd",
    "DRIFT": "#8c564b",
    "FULL": "#2ca02c",
}


def _lighten_hex(hex_color, factor=0.5):
    """Blend *hex_color* toward white by *factor* (0 = unchanged, 1 = white)."""
    h = hex_color.lstrip("#")
    r, g, b = int(h[:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    r = int(r + (255 - r) * factor)
    g = int(g + (255 - g) * factor)
    b = int(b + (255 - b) * factor)
    return f"#{r:02x}{g:02x}{b:02x}"


# Auto-generate lighter _1_10 variants for every base method
for _base, _col in list(METHOD_COLORS.items()):
    METHOD_COLORS[f"{_base}_1_10"] = _lighten_hex(_col)

# Task subgroups for B_DP sub-averages
B_DP_SEEN_TASKS = {"PlaceAvocado", "PutMug", "PutSpatula"}
B_DP_UNSEEN_TASKS = {"DumpVeg", "PutContainer", "PutFruit", "SeparateFruit", "TurnContainer"}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def extract_bools(entry):
    if entry is None or entry.get("SR") == "MISSING":
        return None
    return np.array(entry["Success_Failure"], dtype=bool)


def _save_fig(fig, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    base, ext = os.path.splitext(path)
    pdf_path = base + ".pdf"
    fig.savefig(pdf_path, dpi=300, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    print(f"  Saved: {os.path.relpath(pdf_path)}")

# ---------------------------------------------------------------------------
# Violin CLD plot (non-fractional)
# ---------------------------------------------------------------------------

def plot_violin_cld(method_names, success_arrays, output_path, title="", rng=None, ylim=None):
    if rng is None:
        rng = np.random.default_rng(42)

    valid = [
        (m, a) for m, a in zip(method_names, success_arrays)
        if a is not None and len(a) > 0
    ]
    if len(valid) < 2:
        print(f"  Skipping {title}: fewer than 2 valid methods")
        return

    names = [m for m, _ in valid]
    arrays = [a for _, a in valid]
    num = len(names)

    n_max = max(len(a) for a in arrays)
    test_method = "lai" if n_max > global_nmax else "step"
    print(f"    Computing CLD ({test_method}, n_max={n_max}, {num} methods)...")
    cld_dict = compare_success_and_get_cld(
        names, arrays,
        global_confidence_level=0.95,
        max_sample_size_per_model=n_max,
        rng=rng, shuffle=False,
        test_method_name=test_method,
    )
    print(f"    CLD results: { {m: cld_dict.get(m, '?') for m in names} }")

    print(f"    Drawing posteriors...")
    posteriors = [draw_samples_from_beta_posterior(a, rng) for a in arrays]
    means = [np.mean(p) for p in posteriors]
    colors = [METHOD_COLORS.get(m, "gray") for m in names]
    display_labels = [_display_name(m) for m in names]

    fig, ax = plt.subplots(figsize=(max(7, 1.8 * num), 5.5))
    parts = ax.violinplot(
        posteriors, positions=np.arange(num),
        showmeans=True, showmedians=False, showextrema=False, widths=0.75,
    )
    for pc, c in zip(parts["bodies"], colors):
        pc.set_facecolor(c)
        pc.set_alpha(0.6)
    parts["cmeans"].set_color("black")
    parts["cmeans"].set_linewidth(0.8)

    vw = 0.75  # matches widths= in ax.violinplot above
    for i, (m, arr, mu) in enumerate(zip(names, arrays, means)):
        letter = cld_dict.get(m, "?")
        n_s, n_t = int(np.sum(arr)), len(arr)
        empirical_mean = n_s / n_t
        # Colored dot at empirical mean
        ax.plot(i, empirical_mean, "o", color=colors[i],
                markersize=5, zorder=4)
        # CLD letter: to the right of the violin, slightly above the mean line
        ax.text(i + vw / 2 + 0.04, mu + 0.01, letter,
                fontsize=16, fontweight="bold",
                ha="left", va="bottom",
                color=METHOD_COLORS.get(m, "gray"), zorder=5)
        # Sample count stays at the top of the axes
        ax.text(i, 0.97, f"{n_s}/{n_t}", fontsize=11, ha="center", va="top",
                transform=ax.get_xaxis_transform())

    ax.set_xticks(np.arange(num))
    ax.set_xticklabels(display_labels, rotation=25, ha="right", fontsize=14)
    ax.set_ylabel("Success Rate", fontsize=15)
    if title:  # website copy: no title when empty
        ax.set_title(f"{title} (↑)", fontsize=15)
    ax.tick_params(axis="y", labelsize=14)
    ax.set_xlim(-0.5, num - 1 + vw / 2 + 0.55)
    if ylim is not None:
        ax.set_ylim(ylim)
    ax.grid(True, axis="y", linestyle="--", alpha=0.3)
    fig.tight_layout()
    _save_fig(fig, output_path)

# ---------------------------------------------------------------------------
# Scaling + violin plot (fractional -- B_DP)
# ---------------------------------------------------------------------------

def plot_scaling_with_violins(
    fractions, methods, data_by_frac, output_path, title="", rng=None,
    figsize=(9, 7), font_scale=1.0, cld_fontsize=8, legend_ncol=None,
    violin_width=0.22, frac_step=3.6, cluster_gap=0.8,
    highlight_frac_idx=None, highlight_label="",
):
    """Vertical violins grouped by fraction (like per-task plot but fractions as groups)."""
    if rng is None:
        rng = np.random.default_rng(42)

    frac_labels = [f.replace("_percent", "%") for f in fractions]
    n_frac = len(fractions)
    n_meth = len(methods)

    VIOLIN_WIDTH = violin_width
    FRAC_STEP = frac_step
    CLUSTER_GAP = cluster_gap
    total_cluster_width = FRAC_STEP - CLUSTER_GAP
    step = total_cluster_width / n_meth
    method_offsets = np.linspace(
        -total_cluster_width / 2 + step / 2,
         total_cluster_width / 2 - step / 2,
        n_meth,
    )
    frac_centres = np.arange(n_frac) * FRAC_STEP

    fig, ax = plt.subplots(figsize=figsize)

    print(f"    Drawing vertical posteriors for {n_meth} methods x {n_frac} fractions...")

    # CLD per fraction
    cld_per_frac = {}
    for fi, frac in enumerate(fractions):
        frac_names, frac_arrs = [], []
        for method in methods:
            arr = data_by_frac.get(frac, {}).get(method)
            if arr is not None and len(arr) > 0:
                frac_names.append(method)
                frac_arrs.append(arr)
        if len(frac_names) >= 2:
            n_max = max(len(a) for a in frac_arrs)
            test_method = "lai" if n_max > global_nmax else "step"
            print(f"      {frac}: {len(frac_names)} methods, {test_method}, n_max={n_max}")
            cld_per_frac[fi] = compare_success_and_get_cld(
                frac_names, frac_arrs,
                global_confidence_level=0.95,
                max_sample_size_per_model=n_max,
                rng=rng, shuffle=False,
                test_method_name=test_method,
            )
        else:
            cld_per_frac[fi] = {}

    # Draw violins
    for fi, frac in enumerate(fractions):
        for mi, method in enumerate(methods):
            color = METHOD_COLORS.get(method, "gray")
            arr = data_by_frac.get(frac, {}).get(method)
            if arr is None or len(arr) == 0:
                continue
            posterior = draw_samples_from_beta_posterior(arr, rng)
            mu = np.mean(posterior)
            empirical_mean = np.sum(arr) / len(arr)

            xc = frac_centres[fi] + method_offsets[mi]

            parts = ax.violinplot(
                [posterior], positions=[xc],
                showmeans=True, showmedians=False, showextrema=False,
                widths=VIOLIN_WIDTH,
            )
            for pc in parts["bodies"]:
                pc.set_facecolor(color)
                pc.set_alpha(0.55)
                pc.set_edgecolor(color)
                pc.set_linewidth(0.6)
            parts["cmeans"].set_color("black")
            parts["cmeans"].set_linewidth(2.0)
            # Empirical mean dot (small so it doesn't obscure the black line)
            ax.plot(xc, empirical_mean, "o", color=color, markersize=2, zorder=4)

            # CLD letter above violin tip (stored for rank-based placement below)
            cld_dict = cld_per_frac[fi]
            letter = cld_dict.get(method, "")
            if letter:
                if not hasattr(ax, '_cld_data'):
                    ax._cld_data = []
                ax._cld_data.append((fi, xc, mu, letter, color))

        # Legend entries (only on first fraction)
        if fi == 0:
            for mi, method in enumerate(methods):
                color = METHOD_COLORS.get(method, "gray")
                ax.plot([], [], color=color, marker="o", markersize=5,
                        linewidth=0, label=_display_name(method))

    # CLD letters: centered above violin tip, same distance from tip for all
    if hasattr(ax, '_cld_data') and ax._cld_data:
        LETTER_OFFSET = 0.07
        for (fi, xc, mu, letter, color) in ax._cld_data:
            ax.text(xc, mu + LETTER_OFFSET, letter,
                    fontsize=cld_fontsize, fontweight="bold",
                    ha="center", va="bottom",
                    color=color, zorder=6)

    # Website copy: rounded highlight box around one fraction group.
    # x: from just inside the left axis edge to just before the first dashed
    #    divider (both derived from frac_centres / FRAC_STEP, no pixels).
    # y: from a bit below the lowest violin body to a bit above the tallest
    #    CLD letter of that group; label sits centred inside the top edge.
    if highlight_frac_idx is not None:
        from matplotlib.patches import FancyBboxPatch
        hi = highlight_frac_idx
        x_left = frac_centres[0] - FRAC_STEP * 0.55 + FRAC_STEP * 0.04
        x_sep = (frac_centres[hi] + frac_centres[hi + 1]) / 2
        x_right = x_sep - FRAC_STEP * 0.04
        ys_lo, ys_hi = [], []
        for method in methods:
            arr = data_by_frac.get(fractions[hi], {}).get(method)
            if arr is None or len(arr) == 0:
                continue
            post = draw_samples_from_beta_posterior(arr, np.random.default_rng(0))
            ys_lo.append(np.percentile(post, 0.1))
            ys_hi.append(np.mean(post))
        letter_top = max(ys_hi) + LETTER_OFFSET + 0.06  # letter height ~0.05 in data units
        y_bot = max(0.0, min(ys_lo) - 0.03)
        label_h = 0.11 if "\n" in highlight_label else 0.06
        y_top = letter_top + 0.02 + label_h + 0.02
        teal = "#1a6b5c"
        ax.add_patch(FancyBboxPatch(
            (x_left, y_bot), x_right - x_left, y_top - y_bot,
            boxstyle="round,pad=0,rounding_size=0.25",
            mutation_aspect=(y_top - y_bot) / (x_right - x_left) * 0.15,
            fill=False, edgecolor=teal, linewidth=1.5, zorder=5,
        ))
        ax.text((x_left + x_right) / 2, y_top - 0.02, highlight_label,
                fontsize=cld_fontsize - 1, fontweight="bold", color=teal,
                ha="center", va="top", linespacing=1.1, zorder=6)
        # Make sure the box top stays inside the axes
        ax._hl_ytop = y_top

    # Dashed gray separators between fraction groups
    for fi in range(n_frac - 1):
        sep_x = (frac_centres[fi] + frac_centres[fi + 1]) / 2
        ax.axvline(sep_x, color="#999999", linestyle="--", linewidth=0.8, alpha=0.6, zorder=1)

    # X-axis: fraction labels at group centres
    ax.set_xticks(frac_centres)
    ax.set_xticklabels(frac_labels, fontsize=int(round(14 * font_scale)))
    ax.set_xlim(frac_centres[0] - FRAC_STEP * 0.55,
                frac_centres[-1] + FRAC_STEP * 0.55)
    ax.set_ylabel("Success Rate", fontsize=int(round(15 * font_scale)))
    ax.set_xlabel("Fraction of demonstrations", fontsize=int(round(14 * font_scale)))
    # Auto y-limit: start at 0, cap above the highest CLD letter position
    all_means = [np.sum(data_by_frac.get(f, {}).get(m, [])) / max(len(data_by_frac.get(f, {}).get(m, [1])), 1)
                 for f in fractions for m in methods
                 if data_by_frac.get(f, {}).get(m) is not None and len(data_by_frac.get(f, {}).get(m, [])) > 0]
    y_max = min(1.0, max(all_means) + 0.20) if all_means else 1.0
    if hasattr(ax, "_hl_ytop"):
        y_max = min(1.0, max(y_max, ax._hl_ytop + 0.03))
    ax.set_ylim(0, y_max)
    if title:  # website copy: no title when empty
        ax.set_title(f"{title} (↑)", fontsize=int(round(15 * font_scale)))
    ax.tick_params(axis="y", labelsize=int(round(13 * font_scale)))

    # Legend below figure
    fig.legend(
        *ax.get_legend_handles_labels(),
        loc="lower center", bbox_to_anchor=(0.5, -0.03),
        ncol=(legend_ncol if legend_ncol is not None else min(4, n_meth)),
        fontsize=int(round(13 * font_scale)),
    )
    fig.tight_layout(rect=(0, 0.09, 1, 1))
    _save_fig(fig, output_path)

# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def process_non_fractional(results, output_dir, setting_name):
    rng = np.random.default_rng(42)

    # Per-task plots (no shared ylim needed)
    for group in results:
        group_data = results[group]
        tasks = list(group_data.keys())
        label = f"{setting_name}_{group}"
        task_dir = os.path.join(output_dir, label)

        for ti, task in enumerate(tasks):
            print(f"  [{label}] Task {ti+1}/{len(tasks)}: {task}")
            mnames = list(group_data[task].keys())
            arrays = [extract_bools(group_data[task][m]) for m in mnames]
            plot_violin_cld(
                mnames, arrays,
                os.path.join(task_dir, f"{task}.png"),
                title=f"{label} -- {task}", rng=rng,
            )

    # Collect avg data for all groups first
    avg_data = {}  # group -> {method: concat_array}
    for group in results:
        group_data = results[group]
        tasks = list(group_data.keys())
        methods = list(group_data[tasks[0]].keys())
        label = f"{setting_name}_{group}"

        print(f"  [{label}] Computing average across {len(tasks)} tasks...")
        concat = {}
        for m in methods:
            parts = [extract_bools(group_data[t].get(m)) for t in tasks]
            parts = [p for p in parts if p is not None]
            if parts:
                combined = np.concatenate(parts)
                if len(parts) > 1:
                    rng.shuffle(combined)
                concat[m] = combined
        avg_data[group] = concat

    # Determine shared ylim if there are exactly 2 groups (Setting A: RI + FT)
    shared_ylim = None
    if len(avg_data) == 2:
        all_posteriors = []
        rng_tmp = np.random.default_rng(99)
        for group, concat in avg_data.items():
            for m, arr in concat.items():
                post = draw_samples_from_beta_posterior(arr, rng_tmp)
                all_posteriors.append(post)
        all_vals = np.concatenate(all_posteriors)
        pad = (all_vals.max() - all_vals.min()) * 0.15
        shared_ylim = (max(0, all_vals.min() - pad), min(1, all_vals.max() + pad))

    # Now plot each group's avg
    for group, concat in avg_data.items():
        label = f"{setting_name}_{group}"
        filestem = f"{label}_avg"
        title = TITLE_OVERRIDES.get(filestem, f"{label} (Average, {len(list(results[group].keys()))} tasks)")
        plot_violin_cld(
            list(concat.keys()), list(concat.values()),
            os.path.join(output_dir, f"{label}_avg.png"),
            title=title, rng=rng, ylim=shared_ylim,
        )


def process_fractional(results, output_dir, setting_name):
    rng = np.random.default_rng(42)
    for group in results:
        group_data = results[group]
        fractions = list(group_data.keys())
        tasks = list(group_data[fractions[0]].keys())
        methods = list(group_data[fractions[0]][tasks[0]].keys())
        label = f"{setting_name}_{group}"
        task_dir = os.path.join(output_dir, label)

        # Per-task scaling plot
        for ti, task in enumerate(tasks):
            print(f"  [{label}] Task {ti+1}/{len(tasks)}: {task}")
            data_by_frac = {}
            for frac in fractions:
                data_by_frac[frac] = {}
                for m in methods:
                    arr = extract_bools(group_data[frac].get(task, {}).get(m))
                    if arr is not None:
                        data_by_frac[frac][m] = arr
            plot_scaling_with_violins(
                fractions, methods, data_by_frac,
                os.path.join(task_dir, f"{task}.png"),
                title=f"{label} -- {task}", rng=rng,
            )

        # Averages (concatenate across tasks, shuffle to remove task ordering bias)
        # Build sub-averages for B_DP: avg_all, avg_3 (seen), avg_5 (unseen)
        avg_groups = [("avg", tasks)]
        if B_DP_SEEN_TASKS.intersection(tasks) and B_DP_UNSEEN_TASKS.intersection(tasks):
            seen = [t for t in tasks if t in B_DP_SEEN_TASKS]
            unseen = [t for t in tasks if t in B_DP_UNSEEN_TASKS]
            avg_groups.append((f"avg_{len(seen)}_seen", seen))
            avg_groups.append((f"avg_{len(unseen)}_unseen", unseen))

        for avg_name, avg_tasks in avg_groups:
            print(f"  [{label}] Computing {avg_name} across {len(avg_tasks)} tasks...")
            avg_by_frac = {}
            for frac in fractions:
                avg_by_frac[frac] = {}
                for m in methods:
                    parts = []
                    for task in avg_tasks:
                        arr = extract_bools(group_data[frac].get(task, {}).get(m))
                        if arr is not None:
                            parts.append(arr)
                    if parts:
                        combined = np.concatenate(parts)
                        if len(parts) > 1:
                            rng.shuffle(combined)
                        avg_by_frac[frac][m] = combined
            filestem = f"{label}_{avg_name}"
            title = TITLE_OVERRIDES.get(filestem, f"{label} ({avg_name}, {len(avg_tasks)} tasks)")
            # Double-width, shorter, larger fonts for the main Setting B_sim figure
            extra = {}
            if filestem in (
                "setting_B_DP_FT_avg",
                "setting_B_DP_FT_avg_3_seen",
                "setting_B_DP_FT_avg_5_unseen",
            ):
                # Single-column width, taller violins, larger CLD letters,
                # wider per-violin spacing so letters don't overlap neighbors
                extra = dict(
                    figsize=(10, 7),
                    cld_fontsize=12,
                    violin_width=0.32,
                    frac_step=10.5,
                    cluster_gap=0.8,
                )
                if filestem == "setting_B_DP_FT_avg":  # website copy: 5% highlight box
                    extra.update(
                        highlight_frac_idx=fractions.index("5_percent") if "5_percent" in fractions else 0,
                        highlight_label="5% \u2248 10\u201320 demos\nper task",
                    )
            plot_scaling_with_violins(
                fractions, methods, avg_by_frac,
                os.path.join(output_dir, f"{label}_{avg_name}.png"),
                title=title, rng=rng, **extra,
            )


def _filter_methods(data, methods_filter):
    """Recursively filter a results dict to only keep specified methods."""
    if isinstance(data, dict):
        if "SR" in data:
            return data
        filtered = {}
        for k, v in data.items():
            result = _filter_methods(v, methods_filter)
            if isinstance(result, dict) and "SR" not in result and not result:
                continue
            filtered[k] = result
        if methods_filter and all(isinstance(v, dict) and "SR" in v for v in filtered.values()):
            filtered = {k: v for k, v in filtered.items() if k in methods_filter}
        return filtered
    return data


ALL_SETTINGS = {
    "setting_A":              (os.path.join(JSON_DIR, "setting_A_results.json"), False),
    "setting_Aprime":         (os.path.join(JSON_DIR, "setting_Aprime_results.json"), False),
    "setting_B_FP":           (os.path.join(JSON_DIR, "setting_B_FP_pretrained_results.json"), False),
    "setting_B_FP_noise":     (os.path.join(JSON_DIR, "setting_B_FP_pretrained_noise_ablation_results.json"), False),
    "setting_B_DP":           (os.path.join(JSON_DIR, "setting_B_DP_pretrained_results.json"), True),
    "setting_B_DP_1_10_lr":   (os.path.join(JSON_DIR, "setting_B_DP_pretrained_1_10_lr_results.json"), False),
}


def main():
    parser = argparse.ArgumentParser(description="CLD violin plots for sim experiments")
    parser.add_argument("--methods", nargs="+", default=None,
                        help="Only include these methods (e.g. --methods Z A_LBM O_LBMplus025Z)")
    parser.add_argument("--settings", nargs="+", default=None,
                        choices=list(ALL_SETTINGS.keys()),
                        help="Only run these settings (default: all)")
    args = parser.parse_args()

    os.makedirs(FIG_DIR, exist_ok=True)

    print("NOTE: First CLD computation may be slow (STEP policy synthesis, one-time cost).")
    if args.methods:
        print(f"  Filtering to methods: {args.methods}")
    print()

    active = args.settings or list(ALL_SETTINGS.keys())
    for sname in active:
        fpath, is_fractional = ALL_SETTINGS[sname]
        if not os.path.exists(fpath):
            print(f"Skipping {sname} ({fpath} not found)")
            continue
        print(f"\n{'='*50}\n  {sname}\n{'='*50}")
        data = json.load(open(fpath))
        if args.methods:
            data = _filter_methods(data, set(args.methods))
        if is_fractional:
            process_fractional(data, FIG_DIR, sname)
        else:
            process_non_fractional(data, FIG_DIR, sname)


if __name__ == "__main__":
    global_nmax = 600
    main()
