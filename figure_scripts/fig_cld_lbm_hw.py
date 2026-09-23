#!/usr/bin/env python3
"""
Generate CLD violin plots for hardware experiments (real-robot evaluations).

Figures produced:
  Fig 4  – setting_B_real_ab_test.png / .pdf
           5 methods aggregated across 5 tasks (Kitchen_50, Cookies_50, Bike_50,
           BusBin_50, FoodBank_50).
  Fig 10 – setting_B_real_1_10_lr.png / .pdf
           2 methods (LBM_FT_no_1_10, LBM_FT_paper) from formal_latest --find_all.
  Fig 15 – setting_B_real_prior_selection.png / .pdf
           3 methods (A_DP, AplusCZ_DP, O025Z_DP) from the 25-rollout pilot
           across 3 tasks (Kitchen_paper, Bike_paper, FoodBank_paper).

Data is read from pre-computed *_completed.yaml + *_downloaded_jsons/ in
~/Downloads/anzu_personal/Videos/Hardware_new_prior/.

Usage:
    python fig_cld_hw.py
"""

import json
import os
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from glob import glob
from scipy import stats

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PAPER_DIR = "/home/chenxu/Downloads/anzu_personal/Videos/New_prior_paper/New_prior_paper"  # website copy: data still from the paper
sys.path.insert(0, os.path.join(PAPER_DIR, "scripts"))  # for sibling imports (fig_introspection)
FIG_DIR = os.path.join(SCRIPT_DIR, "_build")  # website copy: write here, not into the paper

HW_ROOT = os.path.join(
    os.path.expanduser("~"),
    "Downloads", "anzu_personal", "Videos", "Hardware_new_prior",
)

# ---------------------------------------------------------------------------
# Display-name and colour maps
# ---------------------------------------------------------------------------
DISPLAY_NAME_MAP = {
    "Gaussian_DP":    r"$\mathcal{Z}$",
    "O025Z_DP":       r"$E_{\mathrm{pre}}{+}\sigma_{E}\mathcal{Z}$",
    "Cocos_DP":       "Cocos",
    "BRIDGER_DP":     "BRIDGER",
    "Retrieval_DP":   "Retrieval",
    "LBM_FT_no_1_10": r"$\mathcal{Z}$",
    "LBM_FT_paper":   r"$\mathcal{Z}$ (w/ 1/10)",
    "A_DP":           r"$A_{\mathrm{pre}}$",
    "AplusCZ_DP":     r"$A_{\mathrm{pre}}{+}\mathcal{Z}$",
}

COLOR_MAP = {
    "Gaussian_DP":     "#2ca02c",
    "O025Z_DP":        "#ff7f0e",
    "Cocos_DP":        "#17becf",
    "BRIDGER_DP":      "#bcbd22",
    "Retrieval_DP":    "#555555",
    "LBM_FT_no_1_10":  "#2ca02c",
    "LBM_FT_paper":    "#b9eab3",
    "A_DP":            "#e377c2",
    "AplusCZ_DP":      "#1f77b4",
}

# ---------------------------------------------------------------------------
# CLD + data-loading helpers (replicated from process_with_CLD.py)
# ---------------------------------------------------------------------------

def extract_timestamp(save_dir):
    """Extract the full timestamp from a save_dir path."""
    match = re.search(r"\d{4}-\d{2}-\d{2}T\d{2}-\d{2}-\d{2}-\d{2}-\d{2}", save_dir)
    return match.group(0) if match else None


def _find_task_key(data, question_to_use, ignore_part=False):
    for k in data:
        if not str(k).startswith(question_to_use):
            continue
        if ignore_part and ("part" in str(k).lower()):
            continue
        return k
    return None


def compact_letter_display(significant_pair_list, sorted_model_list):
    """Piepho (2004) CLD algorithm."""
    num_models = len(sorted_model_list)
    model_to_index = {m: i for i, m in enumerate(sorted_model_list)}
    sig_idx = [(model_to_index[a], model_to_index[b]) for a, b in significant_pair_list]

    def remove_redundant(matrix):
        changed = True
        while changed:
            changed = False
            for i in range(len(matrix)):
                for j in range(len(matrix)):
                    if i != j:
                        si = {k for k, c in enumerate(matrix[i]) if c}
                        sj = {k for k, c in enumerate(matrix[j]) if c}
                        if si.issubset(sj):
                            matrix.pop(i)
                            changed = True
                            break
                if changed:
                    break
        return matrix

    letter_matrix = [["a"] * num_models]
    for i1, i2 in sig_idx:
        while any(col[i1] and col[i2] for col in letter_matrix):
            for ci, col in enumerate(letter_matrix):
                if col[i1] and col[i2]:
                    new_col = col.copy()
                    new_col[i1] = ""
                    col[i2] = ""
                    letter_matrix[ci] = col
                    letter_matrix.append(new_col)
                    letter_matrix = remove_redundant(letter_matrix)
                    break

    def _col_sort_key(col):
        first = next((i for i, c in enumerate(col) if c), len(col))
        last = next((i for i, c in enumerate(reversed(col)) if c), -1)
        last = len(col) - 1 - last if last >= 0 else len(col)
        return (first, last)

    letter_matrix.sort(key=_col_sort_key)
    for idx, col in enumerate(letter_matrix):
        repl = chr(ord("a") + idx)
        letter_matrix[idx] = [repl if c else "" for c in col]

    result = []
    for mi in range(num_models):
        letters = "".join(
            letter_matrix[ci][mi] for ci in range(len(letter_matrix)) if letter_matrix[ci][mi]
        )
        result.append(letters)
    return result


def compare_success_and_get_cld(
    model_name_list,
    success_array_list,
    global_confidence_level=0.95,
    rng=None,
    shuffle=False,
):
    """Run sequential A/B tests and return CLD dict."""
    from sequentialized_barnard_tests import Decision, Hypothesis
    from sequentialized_barnard_tests.step import MirroredStepTest
    from sequentialized_barnard_tests.lai import MirroredLaiTest

    num_models = len(model_name_list)
    global_alpha = 1 - global_confidence_level
    num_comparisons = num_models * (num_models - 1) // 2
    individual_alpha = global_alpha / max(num_comparisons, 1)
    n_max = max(len(arr) for arr in success_array_list)

    # Choose test method based on sample size
    try:
        test = MirroredStepTest(
            alternative=Hypothesis.P0LessThanP1,
            alpha=individual_alpha,
            n_max=n_max,
        )
        test_ok = True
    except Exception:
        test_ok = False

    if not test_ok:
        test = MirroredLaiTest(
            alternative=Hypothesis.P0LessThanP1,
            alpha=individual_alpha,
            n_max=n_max,
        )
    test.reset()

    success_dict = {}
    for idx in range(num_models):
        model = model_name_list[idx]
        arr = success_array_list[idx].copy()
        if shuffle and rng is not None:
            rng.shuffle(arr)
        success_dict[model] = arr

    comparisons = {}
    for ia in range(num_models):
        for ib in range(ia + 1, num_models):
            ma, mb = model_name_list[ia], model_name_list[ib]
            aa, ab = success_dict[ma], success_dict[mb]
            n_common = min(len(aa), len(ab))
            result = test.run_on_sequence(aa[:n_common], ab[:n_common])
            comparisons[(ma, mb)] = result.decision

    sig_pairs = [k for k, v in comparisons.items() if v != Decision.FailToDecide]
    models_sorted = sorted(
        success_dict.keys(),
        key=lambda m: np.mean(success_dict[m]) if len(success_dict[m]) else 0.0,
        reverse=True,
    )
    letters = compact_letter_display(sig_pairs, models_sorted)
    return dict(zip(models_sorted, letters))


def welch_cld(method_names, arrays, alpha=0.05, higher_is_better=True):
    """Welch's t-test with Bonferroni correction -> CLD letters."""
    n = len(method_names)
    num_comp = n * (n - 1) // 2
    individual_alpha = alpha / max(num_comp, 1)

    sig_pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            _, p = stats.ttest_ind(arrays[i], arrays[j], equal_var=False)
            if p <= individual_alpha:
                sig_pairs.append((method_names[i], method_names[j]))

    sorted_methods = sorted(
        method_names,
        key=lambda m: np.mean(arrays[method_names.index(m)]),
        reverse=higher_is_better,
    )
    cld = compact_letter_display(sig_pairs, sorted_methods)
    return dict(zip(sorted_methods, cld))


def draw_samples_from_beta_posterior(success_array, rng, n_samples=10000):
    n_trials = len(success_array)
    n_succ = int(np.sum(success_array))
    n_fail = n_trials - n_succ
    posterior = stats.beta(1 + n_succ, 1 + n_fail)
    return posterior.rvs(n_samples, random_state=rng)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _resolve_task_sources(task_name, find_all):
    """Return list of {name, yaml_path, downloaded_dir}."""
    if not find_all:
        base = task_name
        return [{
            "name": base,
            "yaml_path": os.path.join(HW_ROOT, f"{base}_completed.yaml"),
            "downloaded_dir": os.path.join(HW_ROOT, f"{base}_downloaded_jsons"),
        }]
    pattern = os.path.join(HW_ROOT, f"*{task_name}*_completed.yaml")
    sources = []
    for yp in sorted(glob(pattern)):
        fn = os.path.basename(yp)
        if not fn.endswith("_completed.yaml"):
            continue
        base = fn[:-len("_completed.yaml")]
        sources.append({
            "name": base,
            "yaml_path": yp,
            "downloaded_dir": os.path.join(HW_ROOT, f"{base}_downloaded_jsons"),
        })
    return sources


def load_hw_results(
    task_sources,
    compare_checkpoints=None,
    question_to_use="Task succeeded",
    questions_to_ignore=(),
    completion_prefixes=None,
    compute_partials=True,
    ignore_part_keys=False,
):
    """
    Load hardware results from YAML + JSON files.

    Returns:
        results: dict[ckpt] -> {success, total, bools, task_completion}
        any_partial: bool – whether any partial-credit questions were found
    """
    import yaml

    compare_set = set(compare_checkpoints or [])
    results = {}
    any_partial = False

    for src in task_sources:
        yaml_path = src["yaml_path"]
        downloaded_dir = src["downloaded_dir"]
        if not os.path.exists(yaml_path):
            print(f"Warning: YAML not found, skipping: {yaml_path}")
            continue
        with open(yaml_path, "r") as f:
            entries = yaml.safe_load(f) or []

        for entry in entries:
            ckpt = entry["checkpoint_name"]
            if compare_set and ckpt not in compare_set:
                continue
            if ckpt not in results:
                results[ckpt] = {
                    "success": 0,
                    "total": 0,
                    "bools": [],
                    "task_completion": [],
                }

            save_dir = entry.get("save_dir")
            if not save_dir:
                continue
            timestamp = extract_timestamp(save_dir)
            if not timestamp:
                continue

            json_path = os.path.join(downloaded_dir, f"{timestamp}.json")
            if not os.path.exists(json_path):
                continue
            if os.path.getsize(json_path) == 0:
                continue
            try:
                with open(json_path, "r") as f:
                    data = json.load(f)
            except Exception:
                continue

            task_key = _find_task_key(data, question_to_use, ignore_part=ignore_part_keys)
            if not task_key:
                continue
            if any(task_key.startswith(pfx) for pfx in questions_to_ignore):
                continue

            succeeded = str(data[task_key]).strip().upper() == "Y"
            results[ckpt]["total"] += 1
            if succeeded:
                results[ckpt]["success"] += 1
            results[ckpt]["bools"].append(succeeded)

            if compute_partials:
                keys = list(data.keys())
                if completion_prefixes:
                    yn_keys = [
                        k for k in keys
                        if k != task_key
                        and str(data[k]).strip().upper() in ("Y", "N")
                        and not any(k.startswith(pfx) for pfx in questions_to_ignore)
                        and any(k.startswith(pfx) for pfx in completion_prefixes)
                    ]
                else:
                    task_idx = keys.index(task_key)
                    yn_keys = [
                        k for k in keys[:task_idx]
                        if str(data[k]).strip().upper() in ("Y", "N")
                        and not any(k.startswith(pfx) for pfx in questions_to_ignore)
                    ]
                num_y = sum(1 for k in yn_keys if str(data[k]).strip().upper() == "Y")
                num_total = len(yn_keys)
                if num_total > 0:
                    any_partial = True
                partial_credit = 1.0 if succeeded else (num_y / num_total if num_total > 0 else 0.0)
                results[ckpt]["task_completion"].append(partial_credit)

    return results, any_partial


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_cld_violins(
    model_names,
    success_arrays,
    cld_letters,
    avg_partials,
    output_path,
    rng,
    figsize=None,
    show_partial_labels=True,
    title="",
):
    """
    Draw CLD violin plots with x-tick labels (no legend).

    Args:
        model_names: list of internal checkpoint names (used for colour/display lookup)
        success_arrays: list of bool arrays
        cld_letters: list of CLD letter strings
        avg_partials: list of floats (average partial-completion fraction)
        output_path: save path (without extension; both .png and .pdf are saved)
        rng: numpy Generator
        figsize: optional (w, h)
        show_partial_labels: whether to show "Part. X%" above violins
    """
    num = len(model_names)
    if figsize is None:
        figsize = (max(7, 2 * num), 5)

    display_names = [DISPLAY_NAME_MAP.get(n, n) for n in model_names]
    colors = [COLOR_MAP.get(n, "gray") for n in model_names]

    posterior_samples = []
    means = []
    for arr in success_arrays:
        samples = draw_samples_from_beta_posterior(arr, rng)
        posterior_samples.append(samples)
        means.append(np.mean(samples))

    fig, ax = plt.subplots(figsize=figsize)

    parts = ax.violinplot(
        posterior_samples,
        positions=np.arange(num),
        showmeans=True,
        showmedians=False,
        showextrema=False,
        widths=0.8,
    )
    for pc, color in zip(parts["bodies"], colors):
        pc.set_facecolor(color)
        pc.set_alpha(0.6)
    parts["cmeans"].set_color("black")
    parts["cmeans"].set_linewidth(0.8)

    abs_success = [int(np.sum(a)) for a in success_arrays]
    abs_total = [len(a) for a in success_arrays]

    # Empirical mean dots (Haruki convention: dot = empirical, horizontal line = posterior)
    for i, (s, n) in enumerate(zip(abs_success, abs_total)):
        ax.plot(i, s / n, "o", color=colors[i], markersize=5, zorder=4)

    vw = 0.8  # matches widths= in ax.violinplot above
    for i, (x, y, letters) in enumerate(
        zip(np.arange(num), means, cld_letters)
    ):
        # CLD letter: to the right of the violin, slightly above the mean line
        ax.text(
            x + vw / 2 + 0.04, y + 0.01, letters,
            fontsize=16, fontweight="bold",
            color=colors[i], ha="left", va="bottom", zorder=5,
        )

    ax.set_xticks(np.arange(num))
    # x-tick labels: method name; counts inside axes at top
    ax.set_xticklabels(display_names, fontsize=13, rotation=0, ha="center")
    for i, (s, t) in enumerate(zip(abs_success, abs_total)):
        ax.text(i, 0.97, f"{s}/{t}", fontsize=10, ha="center", va="top",
                transform=ax.get_xaxis_transform())
    ax.set_xlim(-0.5, num - 1 + vw / 2 + 0.55)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Success Rate", fontsize=14)
    if title:  # website copy: no title when empty
        ax.set_title(f"{title} (↑)", fontsize=14)

    plt.tight_layout()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    base, _ = os.path.splitext(output_path)
    for ext in (".pdf",):
        out = base + ext
        fig.savefig(out, dpi=300, bbox_inches="tight", pad_inches=0.08)
        print(f"Saved: {out}")
    plt.close(fig)


def plot_stacked_violins(
    model_names,
    success_arrays,
    cld_letters_top,
    partial_arrays,
    cld_letters_bottom,
    avg_partials,
    output_path,
    rng,
    figsize=None,
    show_partial_labels=True,
    title="",
    horizontal=False,
):
    """
    Draw a two-row stacked figure:
      Top: Success Rate (Beta-posterior violins with STEP/Lai CLD letters)
      Bottom: Partial Task Progress (continuous violins with Welch CLD letters)
    """
    num = len(model_names)
    if figsize is None:
        figsize = (max(7, 2 * num), 9)

    display_names = [DISPLAY_NAME_MAP.get(n, n) for n in model_names]
    colors = [COLOR_MAP.get(n, "gray") for n in model_names]

    # Beta-posterior samples for top panel
    posterior_samples = []
    means_top = []
    for arr in success_arrays:
        samples = draw_samples_from_beta_posterior(arr, rng)
        posterior_samples.append(samples)
        means_top.append(np.mean(samples))

    if horizontal:
        fig, (ax_top, ax_bot) = plt.subplots(1, 2, figsize=figsize or (12, 3.5))
    else:
        fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=figsize)

    # ── Top panel: Success Rate ──────────────────────────────────────────
    parts_top = ax_top.violinplot(
        posterior_samples,
        positions=np.arange(num),
        showmeans=True,
        showmedians=False,
        showextrema=False,
        widths=0.8,
    )
    for pc, color in zip(parts_top["bodies"], colors):
        pc.set_facecolor(color)
        pc.set_alpha(0.6)
    parts_top["cmeans"].set_color("black")
    parts_top["cmeans"].set_linewidth(0.8)

    abs_success = [int(np.sum(a)) for a in success_arrays]
    abs_total = [len(a) for a in success_arrays]

    # Empirical mean dots on top panel
    for i, (s, n) in enumerate(zip(abs_success, abs_total)):
        ax_top.plot(i, s / n, "o", color=colors[i], markersize=5, zorder=4)

    vw = 0.8
    for i, (x, y, letters) in enumerate(
        zip(np.arange(num), means_top, cld_letters_top)
    ):
        ax_top.text(
            x + vw / 2 + 0.04, y + 0.01, letters,
            fontsize=16, fontweight="bold",
            color=colors[i], ha="left", va="bottom", zorder=5,
        )

    ax_top.set_xticks(np.arange(num))
    # SR panel: method name as x-tick, count inside axes at top
    ax_top.set_xticklabels(display_names, fontsize=13, rotation=0, ha="center")
    for i, (s, t) in enumerate(zip(abs_success, abs_total)):
        ax_top.text(i, 0.97, f"{s}/{t}", fontsize=10, ha="center", va="top",
                    transform=ax_top.get_xaxis_transform())
    ax_top.set_xlim(-0.5, num - 1 + vw / 2 + 0.55)
    ax_top.set_ylabel("Success Rate", fontsize=14)
    if title:  # website copy: no title when empty (bottom panel keeps its own)
        ax_top.set_title(f"{title} (↑)", fontsize=14)

    # ── Bottom panel: Partial Task Progress ──────────────────────────────
    means_bot = [np.mean(pa) for pa in partial_arrays]

    parts_bot = ax_bot.violinplot(
        partial_arrays,
        positions=np.arange(num),
        showmeans=False,
        showmedians=False,
        showextrema=False,
        widths=0.8,
    )
    for pc, color in zip(parts_bot["bodies"], colors):
        pc.set_facecolor(color)
        pc.set_alpha(0.6)
    # Empirical mean dots on bottom panel (no horizontal line — empirical distribution)
    for i, pa in enumerate(partial_arrays):
        ax_bot.plot(i, np.mean(pa), "o", color=colors[i], markersize=5, zorder=4)

    for i, (x, y, letters) in enumerate(
        zip(np.arange(num), means_bot, cld_letters_bottom)
    ):
        ax_bot.text(
            x + vw / 2 + 0.04, y + 0.01, letters,
            fontsize=16, fontweight="bold",
            color=colors[i], ha="left", va="bottom", zorder=5,
        )

    ax_bot.set_xticks(np.arange(num))
    xtick_labels_bot = []
    for name in display_names:
        xtick_labels_bot.append(name)
    ax_bot.set_xticklabels(xtick_labels_bot, fontsize=13, rotation=0, ha="center")
    ax_bot.set_xlim(-0.5, num - 1 + vw / 2 + 0.55)
    ax_bot.set_ylim(-0.05, 1.05)
    ax_bot.set_ylabel("Partial Task Progress", fontsize=14)
    ax_bot.set_title("Partial Task Progress (↑)", fontsize=14)

    fig.tight_layout()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    base, _ = os.path.splitext(output_path)
    for ext in (".pdf",):
        out = base + ext
        fig.savefig(out, dpi=300, pad_inches=0.05)
        print(f"Saved: {out}")
    plt.close(fig)


def _build_and_plot(
    label,
    task_sources,
    method_order,
    output_name,
    rng,
    figsize=None,
    shuffle=True,
    title="",
    horizontal=False,
):
    """Load data, compute CLD, and plot for one figure."""
    print(f"\n{'=' * 60}")
    print(f"  {label}")
    print(f"{'=' * 60}")
    print(f"Task sources ({len(task_sources)}):")
    for src in task_sources:
        print(f"  - {os.path.basename(src['yaml_path'])}")

    results, any_partial = load_hw_results(
        task_sources=task_sources,
        compare_checkpoints=method_order,
        compute_partials=True,
        ignore_part_keys=True,  # find_all behaviour
    )

    # Preserve the requested method order
    ordered = [m for m in method_order if m in results and len(results[m]["bools"]) > 0]
    if len(ordered) < 2:
        print(f"WARNING: Only {len(ordered)} methods with data; skipping {label}.")
        return

    model_names = ordered
    success_arrays = [np.array(results[m]["bools"], dtype=bool) for m in model_names]
    avg_partials = []
    for m in model_names:
        tc = results[m].get("task_completion", [])
        total = results[m]["total"]
        avg_partials.append(sum(tc) / total if total > 0 else 0.0)

    # Print summary
    for m, arr, ap in zip(model_names, success_arrays, avg_partials):
        s = int(arr.sum())
        n = len(arr)
        print(f"  {m:25s}  {s}/{n} = {s / n:.1%}   partial={ap:.1%}")

    # CLD
    n_max = max(len(a) for a in success_arrays)
    cld_dict = compare_success_and_get_cld(
        model_names, success_arrays,
        global_confidence_level=0.95,
        rng=rng, shuffle=shuffle,
    )
    cld_list = [cld_dict[m] for m in model_names]
    print(f"  CLD: { {m: l for m, l in zip(model_names, cld_list)} }")

    output_path = os.path.join(FIG_DIR, output_name)

    # If partial-credit data is available, produce a stacked two-panel figure
    # (top: SR violins, bottom: partial-progress violins with Welch CLD).
    partial_arrays = [np.array(results[m].get("task_completion", [])) for m in model_names]
    has_partial = any_partial and all(len(pa) > 0 for pa in partial_arrays)

    if has_partial:
        from fig_introspection import welch_cld as _welch_cld
        partial_cld_dict = _welch_cld(
            model_names, partial_arrays, alpha=0.05, higher_is_better=True
        )
        partial_cld_list = [partial_cld_dict[m] for m in model_names]
        print(f"  Partial-progress CLD: { {m: l for m, l in zip(model_names, partial_cld_list)} }")
        plot_stacked_violins(
            model_names, success_arrays, cld_list,
            partial_arrays, partial_cld_list, avg_partials,
            output_path, rng,
            figsize=figsize if horizontal else ((figsize[0], 9) if figsize else None),
            show_partial_labels=any_partial,
            title=title,
            horizontal=horizontal,
        )
    else:
        plot_cld_violins(
            model_names, success_arrays, cld_list, avg_partials,
            output_path, rng,
            figsize=figsize,
            show_partial_labels=any_partial,
            title=title,
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    os.makedirs(FIG_DIR, exist_ok=True)
    rng = np.random.default_rng(123)

    # ── Fig 4: 5 methods x 5 tasks (_50) ─────────────────────────────────
    fig4_sources = _resolve_task_sources("_50", find_all=True)
    fig4_methods = ["Gaussian_DP", "O025Z_DP", "Cocos_DP", "BRIDGER_DP", "Retrieval_DP"]
    _build_and_plot(
        label="Fig 4 – Setting B real (5 methods x 5 tasks)",
        task_sources=fig4_sources,
        method_order=fig4_methods,
        output_name="setting_B_real_ab_test.png",
        rng=rng,
        figsize=(16, 5.5),  # website copy: 1x2 side-by-side (paper: stacked 10x9)
        title="",  # website copy: no title
        horizontal=True,
    )

    # ── Fig 10: 2 methods from formal_latest ─────────────────────────────
    fig10_sources = _resolve_task_sources("formal_latest", find_all=True)
    fig10_methods = ["LBM_FT_no_1_10", "LBM_FT_paper"]
    _build_and_plot(
        label="Fig 10 – Setting B real 1/10 learning rate",
        task_sources=fig10_sources,
        method_order=fig10_methods,
        output_name="setting_B_real_1_10_lr.png",
        rng=rng,
        figsize=(10, 3.5),
        title="",  # website copy: no title
        horizontal=True,
    )

    # ── Fig 15: 3 methods from pilot (25 rollouts, *_paper) ──────────────
    fig15_sources = _resolve_task_sources("paper", find_all=True)
    # Filter to only the 25-rollout pilot YAML files
    # (Kitchen_paper, Bike_paper, FoodBank_paper — exclude *_with_paper)
    fig15_sources = [
        s for s in fig15_sources
        if s["name"] in ("Kitchen_paper", "Bike_paper", "FoodBank_paper")
    ]
    fig15_methods = ["A_DP", "AplusCZ_DP", "O025Z_DP"]
    _build_and_plot(
        label="Fig 15 – Setting B real prior selection (pilot)",
        task_sources=fig15_sources,
        method_order=fig15_methods,
        output_name="setting_B_real_prior_selection.png",
        rng=rng,
        figsize=(10, 3.5),
        title=r"Setting $B_{\mathrm{real}}$ Prior Selection",
        horizontal=True,
    )

    print("\nDone.")


if __name__ == "__main__":
    main()
