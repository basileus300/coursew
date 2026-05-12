import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Literal

import matplotlib.pyplot as plt
import numpy as np

try:
    import pymzml
except ImportError:
    pymzml = None


MONO_MASS = {
    "C": 12.0000000,
    "H": 1.00782503223,
    "N": 14.00307400443,
    "O": 15.99491461957,
    "S": 31.9720711744,
    "F": 18.99840316273,
    "Cl": 34.968852682,
    "Br": 78.9183376,
    "I": 126.904468,
}


@dataclass
class Fragment:
    mz: float
    intensity: float
    formula: str
    series: str = ""


def parse_formula(formula: str) -> Dict[str, int]:
    tokens = re.findall(r"([A-Z][a-z]?)(\d*)", formula.replace(" ", ""))
    comp: Dict[str, int] = {}
    for elem, cnt in tokens:
        n = int(cnt) if cnt else 1
        comp[elem] = comp.get(elem, 0) + n
    return comp


def comp_to_formula(comp: Dict[str, int]) -> str:
    order = []
    if "C" in comp:
        order.append("C")
    if "H" in comp:
        order.append("H")
    for e in sorted(comp.keys()):
        if e not in ("C", "H"):
            order.append(e)

    parts = []
    for e in order:
        n = comp.get(e, 0)
        if n > 0:
            parts.append(f"{e}{n if n != 1 else ''}")
    return "".join(parts)


def shift_formula_by_h(formula: str, h_shift: int) -> str:
    comp = parse_formula(formula)
    comp["H"] = max(0, comp.get("H", 0) + int(h_shift))
    return comp_to_formula(comp)


def formula_mass_u(comp: Dict[str, int], mass_table: Dict[str, float] = MONO_MASS) -> float:
    return sum(mass_table[e] * n for e, n in comp.items())


def multiply_comp(comp: Dict[str, int], k: int) -> Dict[str, int]:
    return {e: int(n * k) for e, n in comp.items()}


def parse_polymer_expression(expr: str) -> Tuple[str, Optional[int]]:
    """
    Accepts:
        (C2H4)k
        (C6H11NO)k
        (C12H22N2O2)k
        C2H4
    Returns repeat-unit formula and numeric k if provided.
    """
    s = expr.strip().replace(" ", "")
    m = re.match(r"^\(([^)]+)\)(\d+|k|n)?$", s, flags=re.IGNORECASE)
    if m:
        rep = m.group(1)
        k_raw = m.group(2)
        if k_raw is None or k_raw.lower() in ("k", "n"):
            return rep, None
        return rep, int(k_raw)
    return s, None


def h_saturation_limit(c: int, n_n: int, n_x: int = 0) -> int:
    return 2 * c + n_n + 2 - n_x


def infer_polymer_family(polymer_expr: str) -> str:
    repeat_formula, _ = parse_polymer_expression(polymer_expr)
    rep = parse_formula(repeat_formula)
    has_n = rep.get("N", 0) > 0
    has_o = rep.get("O", 0) > 0
    hetero = any(e not in ("C", "H", "N", "O") for e in rep)

    if not has_n and not has_o and not hetero:
        return "PE"
    if has_n and has_o:
        return "PA"
    return "GENERIC"


def normalize_to_max(y: np.ndarray, target: float = 100.0) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    m = float(np.max(y)) if y.size else 0.0
    if m <= 0:
        return y.copy()
    return y / m * target


def moving_average(y: np.ndarray, win: int) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    win = int(max(3, win))

    if win >= y.size:
        win = y.size if (y.size % 2 == 1) else max(3, y.size - 1)
    if win % 2 == 0:
        win -= 1
        if win < 3:
            win = 3

    k = np.ones(win, dtype=float) / win
    return np.convolve(y, k, mode="same")


def gaussian_smooth(y: np.ndarray, sigma_pts: float) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    if sigma_pts <= 0:
        return y.copy()

    radius = int(max(3, np.ceil(4 * sigma_pts)))
    x = np.arange(-radius, radius + 1)
    k = np.exp(-0.5 * (x / sigma_pts) ** 2)
    k /= k.sum()
    return np.convolve(y, k, mode="same")


def denoise_experimental(
    y: np.ndarray,
    baseline_win_pts: int = 101,
    smooth_sigma_pts: float = 2.0,
) -> np.ndarray:
    baseline = moving_average(y, win=baseline_win_pts)
    y2 = np.asarray(y, dtype=float) - baseline
    y2[y2 < 0] = 0.0
    return gaussian_smooth(y2, sigma_pts=float(smooth_sigma_pts))


def pick_local_maxima_peaks(
    mz_axis: np.ndarray,
    y: np.ndarray,
    min_rel_height: float = 0.03,
    min_distance_pts: int = 2,
) -> Tuple[np.ndarray, np.ndarray]:
    mz_axis = np.asarray(mz_axis, dtype=float)
    y = np.asarray(y, dtype=float)

    if y.size < 3:
        return np.array([]), np.array([])

    y_norm = normalize_to_max(y, 100.0)
    thr = float(min_rel_height * np.max(y_norm))

    idx = np.where(
        (y_norm[1:-1] > y_norm[:-2])
        & (y_norm[1:-1] >= y_norm[2:])
        & (y_norm[1:-1] >= thr)
    )[0] + 1

    if idx.size == 0:
        return np.array([]), np.array([])

    idx_sorted = idx[np.argsort(y_norm[idx])[::-1]]
    chosen = []
    blocked = np.zeros_like(y_norm, dtype=bool)

    for i in idx_sorted:
        if blocked[i]:
            continue
        chosen.append(i)
        lo = max(0, i - min_distance_pts)
        hi = min(y_norm.size, i + min_distance_pts + 1)
        blocked[lo:hi] = True

    chosen = np.array(sorted(chosen), dtype=int)
    return mz_axis[chosen], y_norm[chosen]


def calculate_simulated_peaks(
    polymer_expr: str,
    polymer_family: Literal["auto", "PE", "PA", "GENERIC"] = "auto",
    mz_min: float = 20.0,
    mz_max: float = 350.0,
    max_peaks: int = 256,
    use_integer_mz: bool = True,
    max_c: int = 128,
    max_n: Optional[int] = None,
    max_o: Optional[int] = None,
    apply_budget_if_k_leq: int = 10,
    add_m_plus_one: bool = True,
    neutron_mass_u: float = 1.00866491595,
    min_m_plus_one_rel_intensity: float = 0.01,
) -> List[Fragment]:
    """
    Unified peak calculation for PE-like and PA-like polymers from the gross repeat-unit formula.

    The function performs in one place:
    1. repeat-unit parsing,
    2. automatic polymer-family selection if polymer_family='auto',
    3. candidate fragment enumeration,
    4. theoretical m/z calculation,
    5. EI-like intensity scoring,
    6. fragment-series classification,
    7. base-peak normalization,
    8. optional addition of M+1 isotope companion peaks.

    Examples:
        polymer_expr='(C2H4)k'       -> PE-like hydrocarbon series
        polymer_expr='(C6H11NO)k'    -> PA6-like N/O-containing series
        polymer_expr='(C12H22N2O2)k' -> PA66-like N/O-containing series
    """
    repeat_formula, k = parse_polymer_expression(polymer_expr)
    rep = parse_formula(repeat_formula)

    if polymer_family == "auto":
        family = infer_polymer_family(polymer_expr)
    else:
        family = polymer_family.upper()

    total = None
    if isinstance(k, int) and k <= apply_budget_if_k_leq:
        total = multiply_comp(rep, k)

    c_budget = total.get("C", 10**9) if total else 10**9
    h_budget = total.get("H", 10**9) if total else 10**9
    n_budget = total.get("N", 10**9) if total else 10**9
    o_budget = total.get("O", 10**9) if total else 10**9

    c_max_enum = int(min(max_c, c_budget))

    if max_n is None:
        if family == "PA":
            max_n_enum = 2 if rep.get("N", 0) >= 2 else 1
        else:
            max_n_enum = rep.get("N", 0)
    else:
        max_n_enum = int(max_n)

    if max_o is None:
        if family == "PA":
            max_o_enum = 2 if rep.get("O", 0) >= 2 else 1
        else:
            max_o_enum = rep.get("O", 0)
    else:
        max_o_enum = int(max_o)

    max_n_enum = int(min(max_n_enum, n_budget))
    max_o_enum = int(min(max_o_enum, o_budget))

    candidates: List[Tuple[Dict[str, int], float, str]] = []

    for c in range(1, c_max_enum + 1):
        family_compositions: List[Dict[str, int]] = []

        if family == "PE":
            # Hydrocarbon EI-like PE fragments:
            # CnH2n+1+, CnH2n-1+, CnH2n-3+
            for h in (2 * c + 1, 2 * c - 1, 2 * c - 3):
                if h >= 0:
                    family_compositions.append({"C": c, "H": h})

        elif family == "PA":
            # Weak hydrocarbon background.
            for h in (2 * c + 1, 2 * c - 1, 2 * c - 3):
                if h >= 0:
                    family_compositions.append({"C": c, "H": h})

            # N-containing fragments: nitrile / amine / iminium-like families.
            if max_n_enum >= 1:
                for h in (2 * c - 1, 2 * c + 1, 2 * c - 3):
                    if h >= 0:
                        family_compositions.append({"C": c, "H": h, "N": 1})

            # Amide / lactam-type fragments.
            if max_n_enum >= 1 and max_o_enum >= 1:
                for h in (2 * c + 1, 2 * c - 1, 2 * c - 3, 2 * c - 5):
                    if h >= 0:
                        family_compositions.append({"C": c, "H": h, "N": 1, "O": 1})

            # Heavier diamide fragments, useful for PA66-type repeat units.
            if max_n_enum >= 2 and max_o_enum >= 2:
                for h in (2 * c + 2, 2 * c, 2 * c - 2):
                    if h >= 0:
                        family_compositions.append({"C": c, "H": h, "N": 2, "O": 2})

        else:
            # Generic fallback: enumerate a wider unsaturation range.
            for n_n in range(0, max_n_enum + 1):
                for n_o in range(0, max_o_enum + 1):
                    h_sat = h_saturation_limit(c, n_n, n_x=0)
                    for uns in range(0, 8):
                        h = h_sat - 2 * uns
                        if h >= 0:
                            comp = {"C": c, "H": h}
                            if n_n:
                                comp["N"] = n_n
                            if n_o:
                                comp["O"] = n_o
                            family_compositions.append(comp)
                        if h - 1 >= 0:
                            comp = {"C": c, "H": h - 1}
                            if n_n:
                                comp["N"] = n_n
                            if n_o:
                                comp["O"] = n_o
                            family_compositions.append(comp)

        for comp in family_compositions:
            h = comp.get("H", 0)
            n_n = comp.get("N", 0)
            n_o = comp.get("O", 0)

            if c > c_budget or h > h_budget or n_n > n_budget or n_o > o_budget:
                continue

            mz = formula_mass_u(comp)
            if mz < mz_min or mz > mz_max:
                continue

            nominal = int(round(mz))

            # -------------------------------
            # PE-like scoring
            # -------------------------------
            if family == "PE":
                tau = 150.0
                base = np.exp(-mz / tau)

                if h == 2 * c + 1:
                    series = "alkyl"
                    boost = 2.2
                elif h == 2 * c - 1:
                    series = "alkenyl"
                    boost = 2.0
                elif h == 2 * c - 3:
                    series = "dienyl"
                    boost = 1.35
                else:
                    series = "other"
                    boost = 0.15

                preferred = {
                    41, 43, 55, 57, 69, 71, 83, 85,
                    97, 99, 111, 113, 125, 127, 141, 155, 169
                }
                if nominal in preferred:
                    boost *= 1.8

                center_c = 10.0
                sigma_c = 5.0
                boost *= np.exp(-0.5 * ((c - center_c) / sigma_c) ** 2) * 1.2

                score = float(base * boost)

            # -------------------------------
            # PA-like scoring
            # -------------------------------
            elif family == "PA":
                base = np.exp(-mz / 82.0)
                score = base

                if n_n == 0 and n_o == 0:
                    series = "hydrocarbon"
                    hydro = 0.12
                    if h == 2 * c + 1:
                        series = "alkyl"
                        hydro *= 2.0
                    elif h == 2 * c - 1:
                        series = "alkenyl"
                        hydro *= 1.8
                    elif h == 2 * c - 3:
                        series = "dienyl"
                        hydro *= 1.2
                    if nominal in {41, 43, 55, 57, 67, 69, 71, 83, 85}:
                        hydro *= 1.5
                    score *= hydro

                elif n_n >= 1 and n_o == 0:
                    series = "N-fragment"
                    n_score = 1.0 + 0.55 * n_n
                    if h == 2 * c - 1:
                        series = "nitrile"
                        n_score *= 2.7
                    elif h == 2 * c + 1:
                        series = "amine/iminium"
                        n_score *= 1.8
                    elif h == 2 * c - 3:
                        n_score *= 1.5
                    if nominal in {41, 42, 55, 56, 69, 70, 83, 84, 97, 98, 111}:
                        n_score *= 1.6
                    score *= n_score

                elif n_n >= 1 and n_o >= 1:
                    series = "amide-hetero"
                    amide = 2.1 + 0.45 * n_n + 0.25 * n_o
                    if h == 2 * c + 1:
                        series = "amide"
                        amide *= 2.1
                    elif h == 2 * c - 1:
                        series = "lactam/amide"
                        amide *= 2.5
                    elif h == 2 * c - 3:
                        series = "amide-unsat"
                        amide *= 1.8
                    elif h == 2 * c - 5:
                        amide *= 1.25

                    if nominal in {30, 42, 43, 55, 56, 57, 69, 70, 84, 85, 98, 99, 111, 113, 126, 127}:
                        amide *= 1.8

                    # PA6 / caprolactam-like signature.
                    if rep.get("C", 0) == 6 and rep.get("N", 0) == 1 and rep.get("O", 0) == 1 and nominal == 113:
                        amide *= 2.4

                    score *= amide

                else:
                    series = "other"

                center_c = 5.0 if rep.get("C", 0) <= 6 else 6.0
                sigma_c = 2.2
                score *= np.exp(-0.5 * ((c - center_c) / sigma_c) ** 2) * 1.8

                if n_n + n_o > 2:
                    score *= 0.75 ** (n_n + n_o - 2)

                score = float(score)

            # -------------------------------
            # Generic scoring
            # -------------------------------
            else:
                series = "hetero" if (n_n + n_o) > 0 else "hydrocarbon"
                tau = 120.0
                base = np.exp(-mz / tau)
                hetero_boost = 1.0 + 0.6 * n_n + 0.35 * n_o
                score = float(base * hetero_boost)

            if score > 0:
                candidates.append((comp, score, series))

    if not candidates:
        return []

    # Merge candidates with identical nominal or exact m/z.
    merged: Dict[float, Tuple[Dict[str, int], float, str]] = {}
    for comp, score, series in candidates:
        mz = formula_mass_u(comp)
        mz_key = float(int(round(mz))) if use_integer_mz else float(mz)
        if mz_key not in merged or score > merged[mz_key][1]:
            merged[mz_key] = (comp, score, series)

    fragments = [
        Fragment(
            mz=mz_key,
            intensity=score,
            formula=comp_to_formula(comp),
            series=series,
        )
        for mz_key, (comp, score, series) in merged.items()
    ]

    fragments.sort(key=lambda f: f.intensity, reverse=True)
    fragments = fragments[:max_peaks]

    # Normalize base simulated intensities.
    max_int = max(f.intensity for f in fragments) if fragments else 1.0
    for f in fragments:
        f.intensity = 100.0 * f.intensity / max_int

    fragments.sort(key=lambda f: f.mz)

    if not add_m_plus_one:
        return fragments

    # Add M+1 isotope companion peaks.
    augmented: List[Fragment] = []
    for frag in fragments:
        augmented.append(frag)

        comp = parse_formula(frag.formula)
        rel = (
            0.0107 * comp.get("C", 0)
            + 0.000115 * comp.get("H", 0)
            + 0.00364 * comp.get("N", 0)
            + 0.00038 * comp.get("O", 0)
            + 0.0076 * comp.get("S", 0)
        )

        if rel < float(min_m_plus_one_rel_intensity):
            continue

        mz_new = float(frag.mz + neutron_mass_u)
        if mz_new < mz_min or mz_new > mz_max:
            continue

        augmented.append(
            Fragment(
                mz=mz_new,
                intensity=float(frag.intensity * rel),
                formula=shift_formula_by_h(frag.formula, +1),
                series=f"{frag.series}_M+1" if frag.series else "M+1",
            )
        )

    augmented.sort(key=lambda f: f.mz)
    return augmented


def _pymzml_peaks_arrays(spec) -> Tuple[np.ndarray, np.ndarray]:
    last_err = None
    for peak_type in ("centroided", "raw"):
        try:
            arr = np.array(list(spec.peaks(peak_type)), dtype=float)
            if arr.size == 0:
                return np.array([]), np.array([])
            return arr[:, 0], arr[:, 1]
        except Exception as e:
            last_err = e
    raise RuntimeError(f"Cannot extract peaks from spectrum. Last error: {last_err}")


def read_best_scan_peaks(
    mzml_path: str,
    mz_min: float,
    mz_max: float,
    mode: str = "max_tic",
) -> Tuple[np.ndarray, np.ndarray, Optional[int], Optional[str]]:
    if pymzml is None:
        raise ImportError("pymzml is not installed. Install with: pip install pymzml")
    if not Path(mzml_path).exists():
        raise FileNotFoundError(f"mzML not found: {mzml_path}")

    run = pymzml.run.Reader(str(mzml_path))

    if mode == "sum_all":
        mz_all, i_all = [], []
        last_scan_idx = None
        last_scan_id = None

        for scan_idx, spec in enumerate(run):
            mzs, ints = _pymzml_peaks_arrays(spec)
            if mzs.size == 0:
                continue
            mask = (mzs >= mz_min) & (mzs <= mz_max)
            mz_all.append(mzs[mask])
            i_all.append(ints[mask])
            last_scan_idx = scan_idx
            last_scan_id = str(getattr(spec, "ID", scan_idx))

        if not mz_all:
            return np.array([]), np.array([]), None, None

        return np.concatenate(mz_all), np.concatenate(i_all), last_scan_idx, last_scan_id

    best_mz, best_int, best_tic = None, None, -1.0
    best_scan_idx = None
    best_scan_id = None

    for scan_idx, spec in enumerate(run):
        mzs, ints = _pymzml_peaks_arrays(spec)
        if mzs.size == 0:
            continue

        tic = float(np.sum(ints))
        if tic > best_tic:
            best_tic = tic
            best_mz, best_int = mzs, ints
            best_scan_idx = scan_idx
            best_scan_id = str(getattr(spec, "ID", scan_idx))

    if best_mz is None:
        return np.array([]), np.array([]), None, None

    mask = (best_mz >= mz_min) & (best_mz <= mz_max)
    return best_mz[mask], best_int[mask], best_scan_idx, best_scan_id


def read_experimental_spectrum_binned(
    mzml_path: str,
    mz_min: float,
    mz_max: float,
    bin_step: float = 0.1,
    mode: str = "max_tic",
) -> Tuple[np.ndarray, np.ndarray, Optional[int], Optional[str]]:
    mz_axis = np.arange(mz_min, mz_max + bin_step, bin_step, dtype=float)
    binned = np.zeros_like(mz_axis, dtype=float)

    mzs, ints, scan_idx, scan_id = read_best_scan_peaks(
        mzml_path, mz_min, mz_max, mode=mode
    )
    if mzs.size == 0:
        return mz_axis, binned, scan_idx, scan_id

    idx = np.rint((mzs - mz_min) / bin_step).astype(int)

    valid = (idx >= 0) & (idx < binned.size)
    if np.any(valid):
        np.add.at(binned, idx[valid], ints[valid])

    return mz_axis, binned, scan_idx, scan_id


def match_peaks_nearest(
    sim_mz: np.ndarray,
    exp_mz: np.ndarray,
    tol_mz: float = 0.1,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    sim_mz = np.asarray(sim_mz, dtype=float)
    exp_mz = np.asarray(exp_mz, dtype=float)

    nearest = np.full(sim_mz.shape, np.nan, dtype=float)
    nearest_idx = np.full(sim_mz.shape, -1, dtype=int)
    matched = np.zeros(sim_mz.shape, dtype=bool)

    if sim_mz.size == 0 or exp_mz.size == 0:
        return matched, nearest, nearest_idx

    order = np.argsort(exp_mz)
    exp_sorted = exp_mz[order]

    for i, m in enumerate(sim_mz):
        j = int(np.searchsorted(exp_sorted, m))
        candidate_positions = []

        if 0 <= j < exp_sorted.size:
            candidate_positions.append(j)
        if 0 <= j - 1 < exp_sorted.size:
            candidate_positions.append(j - 1)

        if not candidate_positions:
            continue

        candidates = exp_sorted[candidate_positions]
        k = int(np.argmin(np.abs(candidates - m)))

        if abs(candidates[k] - m) <= tol_mz:
            matched[i] = True
            nearest[i] = candidates[k]
            nearest_idx[i] = int(order[candidate_positions[k]])

    return matched, nearest, nearest_idx


def compute_staggered_label_offsets(
    x_vals: np.ndarray,
    cluster_gap_mz: float = 4.0,
    base_offset_pts: float = 6.0,
    step_offset_pts: float = 12.0,
    max_levels: int = 5,
) -> np.ndarray:
    x_vals = np.asarray(x_vals, dtype=float)
    if x_vals.size == 0:
        return np.array([], dtype=float)

    offsets = np.full(x_vals.shape, float(base_offset_pts), dtype=float)
    order = np.argsort(x_vals)

    prev_x = None
    level = 0
    for idx in order:
        x = float(x_vals[idx])
        if prev_x is None or (x - prev_x) > float(cluster_gap_mz):
            level = 0
        else:
            level = (level + 1) % int(max_levels)
        offsets[idx] = float(base_offset_pts + level * step_offset_pts)
        prev_x = x

    return offsets


def plot_overlay_exp_vs_sim(
    mz_axis: np.ndarray,
    exp_y: np.ndarray,
    fragments: List[Fragment],
    label_fragments: Optional[List[Fragment]] = None,
    exp_sticks: Optional[List[Tuple[float, float]]] = None,
    sim_scale: float = 0.70,
    sim_x_offset: float = 0.15,
    tol_mz: float = 0.1,
    label_top_n_matched: Optional[int] = None,
    label_all_matched: bool = True,
    show_match_delta: bool = False,
    label_stagger: bool = True,
    label_cluster_gap_mz: float = 4.0,
    label_base_offset_pts: float = 6.0,
    label_step_offset_pts: float = 12.0,
    label_max_levels: int = 5,
    exp_min_rel_height: float = 0.03,
    exp_min_distance_pts: int = 2,
    show_unmatched: bool = True,
    title: str = "GC-MS EI overlay: Experimental vs Simulated",
) -> None:
    if label_fragments is None:
        label_fragments = fragments

    if exp_sticks is not None and len(exp_sticks) > 0:
        exp_mz_sticks = np.array([m for m, _ in exp_sticks], dtype=float)
        exp_i_sticks = np.array([i for _, i in exp_sticks], dtype=float)
    else:
        exp_mz_sticks, exp_i_sticks = pick_local_maxima_peaks(
            mz_axis=mz_axis,
            y=exp_y,
            min_rel_height=exp_min_rel_height,
            min_distance_pts=exp_min_distance_pts,
        )

    sim_mz = np.array([f.mz for f in fragments], dtype=float)
    sim_i = np.array([f.intensity for f in fragments], dtype=float)
    sim_i = normalize_to_max(sim_i, 100.0) * float(sim_scale)

    matched_mask, nearest_exp, _ = match_peaks_nearest(
        sim_mz, exp_mz_sticks, tol_mz=tol_mz
    )

    sim_mz_mat = sim_mz[matched_mask] + sim_x_offset
    sim_i_mat = sim_i[matched_mask]

    sim_mz_unm = sim_mz[~matched_mask] + sim_x_offset
    sim_i_unm = sim_i[~matched_mask]

    plt.figure(figsize=(12, 5))

    plt.vlines(
        exp_mz_sticks,
        0,
        exp_i_sticks,
        color="black",
        linewidth=1.4,
        alpha=0.95,
        label="Experimental",
        zorder=2,
    )

    if show_unmatched and sim_mz_unm.size:
        plt.vlines(
            sim_mz_unm,
            0,
            sim_i_unm,
            color="red",
            linewidth=1.0,
            alpha=0.22,
            label="Simulated (unmatched)",
            zorder=3,
        )

    if sim_mz_mat.size:
        plt.vlines(
            sim_mz_mat,
            0,
            sim_i_mat,
            color="red",
            linewidth=1.8,
            alpha=0.90,
            label="Simulated (matched)",
            zorder=4,
        )

    label_mz = np.array([f.mz for f in label_fragments], dtype=float)
    label_i = np.array([f.intensity for f in label_fragments], dtype=float)
    label_i = normalize_to_max(label_i, 100.0) * float(sim_scale)

    label_matched_mask, label_nearest_exp, _ = match_peaks_nearest(
        label_mz, exp_mz_sticks, tol_mz=tol_mz
    )

    if np.any(label_matched_mask):
        matched_idx_global = np.where(label_matched_mask)[0]

        if label_all_matched:
            label_indices = matched_idx_global
        elif label_top_n_matched is not None and label_top_n_matched > 0:
            matched_int = label_i[matched_idx_global]
            top_local = np.argsort(matched_int)[::-1][: min(label_top_n_matched, matched_idx_global.size)]
            label_indices = matched_idx_global[top_local]
        else:
            label_indices = np.array([], dtype=int)

        label_offsets_pts = np.full(label_indices.shape, 6.0, dtype=float)
        if label_stagger and label_indices.size > 0:
            label_x_positions = label_mz[label_indices] + sim_x_offset
            label_offsets_pts = compute_staggered_label_offsets(
                label_x_positions,
                cluster_gap_mz=label_cluster_gap_mz,
                base_offset_pts=label_base_offset_pts,
                step_offset_pts=label_step_offset_pts,
                max_levels=label_max_levels,
            )

        for local_i, gi in enumerate(label_indices):
            f = label_fragments[gi]
            x = float(f.mz + sim_x_offset)
            y = float(label_i[gi] + 2.0)
            exp_match = float(label_nearest_exp[gi])
            delta = float(f.mz - exp_match)

            label = (
                f"{f.formula}\n{f.mz:.1f}"
                if not show_match_delta
                else f"{f.formula}\n{f.mz:.1f}\nΔ={delta:+.1f}"
            )

            plt.annotate(
                label,
                xy=(x, y),
                xytext=(0, float(label_offsets_pts[local_i])),
                textcoords="offset points",
                rotation=90,
                fontsize=6,
                ha="center",
                va="bottom",
                color="red",
            )

    plt.xlabel("m/z")
    plt.ylabel("Intensity (normalized)")
    plt.title(title)
    plt.ylim(0, 110)
    plt.tight_layout()
    plt.legend()
    plt.show()


def plot_overlay_raw_exp_vs_sim(
    exp_mz_raw: np.ndarray,
    exp_i_raw: np.ndarray,
    sim_fragments: List[Fragment],
    label_fragments: Optional[List[Fragment]] = None,
    sim_scale: float = 0.75,
    tol_mz: float = 0.1,
    x_min: float = 20.0,
    x_max: float = 200.0,
    title: str = "GC-MS overlay without binning",
    label_all_matched: bool = True,
    label_top_n_matched: Optional[int] = None,
    label_stagger: bool = True,
    label_cluster_gap_mz: float = 4.0,
    label_base_offset_pts: float = 6.0,
    label_step_offset_pts: float = 12.0,
    label_max_levels: int = 5,
):
    if label_fragments is None:
        label_fragments = sim_fragments

    exp_mz_raw = np.asarray(exp_mz_raw, dtype=float)
    exp_i_raw = np.asarray(exp_i_raw, dtype=float)

    sim_mz = np.array([f.mz for f in sim_fragments], dtype=float)
    sim_i = np.array([f.intensity for f in sim_fragments], dtype=float)

    exp_i_plot = normalize_to_max(exp_i_raw, 100.0)
    sim_i_plot = normalize_to_max(sim_i, 100.0) * sim_scale

    matched_mask, _, _ = match_peaks_nearest(sim_mz, exp_mz_raw, tol_mz=tol_mz)

    plt.figure(figsize=(12, 5))

    plt.vlines(
        exp_mz_raw,
        0,
        exp_i_plot,
        color="black",
        linewidth=1.0,
        alpha=0.85,
        label="Experimental (no binning)"
    )

    if np.any(~matched_mask):
        plt.vlines(
            sim_mz[~matched_mask],
            0,
            sim_i_plot[~matched_mask],
            color="red",
            linewidth=1.0,
            alpha=0.25,
            label="Simulated (unmatched)"
        )

    if np.any(matched_mask):
        plt.vlines(
            sim_mz[matched_mask],
            0,
            sim_i_plot[matched_mask],
            color="red",
            linewidth=1.6,
            alpha=0.9,
            label="Simulated (matched)"
        )

    lbl_mz = np.array([f.mz for f in label_fragments], dtype=float)
    lbl_i = np.array([f.intensity for f in label_fragments], dtype=float)
    lbl_i = normalize_to_max(lbl_i, 100.0) * sim_scale

    lbl_matched, _, _ = match_peaks_nearest(lbl_mz, exp_mz_raw, tol_mz=tol_mz)
    matched_idx = np.where(lbl_matched)[0]

    if label_all_matched:
        label_indices = matched_idx
    elif label_top_n_matched is not None and label_top_n_matched > 0:
        matched_int = lbl_i[matched_idx]
        top_local = np.argsort(matched_int)[::-1][:min(label_top_n_matched, matched_idx.size)]
        label_indices = matched_idx[top_local]
    else:
        label_indices = np.array([], dtype=int)

    label_offsets_pts = np.full(label_indices.shape, 6.0, dtype=float)
    if label_stagger and label_indices.size > 0:
        label_x_positions = lbl_mz[label_indices]
        label_offsets_pts = compute_staggered_label_offsets(
            label_x_positions,
            cluster_gap_mz=label_cluster_gap_mz,
            base_offset_pts=label_base_offset_pts,
            step_offset_pts=label_step_offset_pts,
            max_levels=label_max_levels,
        )

    for local_i, gi in enumerate(label_indices):
        frag = label_fragments[gi]
        x = float(frag.mz)
        y = float(lbl_i[gi] + 2.0)

        plt.annotate(
            f"{frag.formula}\n{frag.mz:.1f}",
            xy=(x, y),
            xytext=(0, float(label_offsets_pts[local_i])),
            textcoords="offset points",
            rotation=90,
            fontsize=6,
            ha="center",
            va="bottom",
            color="red",
        )

    plt.xlabel("m/z")
    plt.ylabel("Intensity (normalized)")
    plt.xlim(x_min, x_max)
    plt.ylim(0, 110)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_experimental_only(
    exp_mz_raw: np.ndarray,
    exp_i_raw: np.ndarray,
    exp_mz_binned: Optional[np.ndarray] = None,
    exp_i_binned: Optional[np.ndarray] = None,
    use_binned: bool = False,
    x_min: float = 20.0,
    x_max: float = 200.0,
    title: str = "Experimental spectrum only",
):
    plt.figure(figsize=(12, 5))

    if use_binned and exp_mz_binned is not None and exp_i_binned is not None:
        y_plot = normalize_to_max(exp_i_binned, 100.0)
        plt.plot(
            exp_mz_binned,
            y_plot,
            color="black",
            linewidth=1.2,
            label="Experimental (binned)"
        )
    else:
        y_plot = normalize_to_max(exp_i_raw, 100.0)
        plt.vlines(
            exp_mz_raw,
            0,
            y_plot,
            color="black",
            linewidth=1.0,
            alpha=0.9,
            label="Experimental (no binning)"
        )

    plt.xlabel("m/z")
    plt.ylabel("Intensity (normalized)")
    plt.xlim(x_min, x_max)
    plt.ylim(0, 110)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # ====== MAIN INPUTS ======
    # Examples:
    #   PE: POLYMER_EXPR = "(C2H4)k"
    #   PA6: POLYMER_EXPR = "(C6H11NO)k"
    #   PA66: POLYMER_EXPR = "(C12H22N2O2)k"
    POLYMER_EXPR = "(C2H4)k"
    POLYMER_FAMILY = "auto"  # "auto", "PE", "PA", or "GENERIC"

    EXP_MZML_PATH = "PE_2.mzML"

    MZ_MIN = 20.0
    MZ_MAX = 350.0
    BIN_STEP = 0.02

    EXP_MODE = "max_tic"

    BASELINE_WIN_PTS = 101
    SMOOTH_SIGMA_PTS = 2.0

    MAX_C = 128
    MAX_N = None  # None = infer from formula and polymer family
    MAX_O = None  # None = infer from formula and polymer family
    MAX_PEAKS = 256
    USE_INTEGER_MZ = True

    SIM_SCALE = 0.75

    USE_SIM_PLUS_ONE_NEUTRON = True
    NEUTRON_MASS_U = 1.00866491595

    MATCH_TOL_MZ = 0.1

    LABEL_STAGGER = True
    LABEL_CLUSTER_GAP_MZ = 4.0
    LABEL_BASE_OFFSET_PTS = 6.0
    LABEL_STEP_OFFSET_PTS = 12.0
    LABEL_MAX_LEVELS = 5

    inferred_family = infer_polymer_family(POLYMER_EXPR) if POLYMER_FAMILY == "auto" else POLYMER_FAMILY
    print(f"Polymer input: {POLYMER_EXPR}")
    print(f"Polymer family used for scoring: {inferred_family}")

    # ---- read experimental spectrum: binned ----
    mz_axis, exp_binned, selected_scan_idx, selected_scan_id = read_experimental_spectrum_binned(
        EXP_MZML_PATH,
        MZ_MIN,
        MZ_MAX,
        BIN_STEP,
        mode=EXP_MODE,
    )

    exp_binned = denoise_experimental(
        exp_binned,
        BASELINE_WIN_PTS,
        SMOOTH_SIGMA_PTS,
    )

    # ---- read experimental spectrum: raw / no binning ----
    exp_mz_raw, exp_i_raw, _, _ = read_best_scan_peaks(
        EXP_MZML_PATH,
        MZ_MIN,
        MZ_MAX,
        mode=EXP_MODE,
    )

    # ---- simulate from polymer gross formula ----
    plot_fragments = calculate_simulated_peaks(
        polymer_expr=POLYMER_EXPR,
        polymer_family=POLYMER_FAMILY,
        mz_min=MZ_MIN,
        mz_max=MZ_MAX,
        max_peaks=MAX_PEAKS,
        use_integer_mz=USE_INTEGER_MZ,
        max_c=MAX_C,
        max_n=MAX_N,
        max_o=MAX_O,
        add_m_plus_one=USE_SIM_PLUS_ONE_NEUTRON,
        neutron_mass_u=NEUTRON_MASS_U,
    )

    print("Top simulated fragments used for plotting:")
    for f in sorted(plot_fragments, key=lambda x: x.intensity, reverse=True)[:30]:
        print(f"m/z {f.mz:>8.3f} | I {f.intensity:6.2f} | {f.formula:>18s} | {f.series}")

    scan_info = f"scan idx={selected_scan_idx}, scan ID={selected_scan_id}"
    print(f"Selected experimental scan: {scan_info}")

    # =========================================================
    # 1) WITH BINNING
    # =========================================================
    plot_overlay_exp_vs_sim(
        mz_axis=mz_axis,
        exp_y=exp_binned,
        fragments=plot_fragments,
        label_fragments=plot_fragments,
        exp_sticks=None,
        sim_scale=SIM_SCALE,
        sim_x_offset=0,
        tol_mz=MATCH_TOL_MZ,
        label_all_matched=True,
        label_top_n_matched=None,
        show_match_delta=False,
        label_stagger=LABEL_STAGGER,
        label_cluster_gap_mz=LABEL_CLUSTER_GAP_MZ,
        label_base_offset_pts=LABEL_BASE_OFFSET_PTS,
        label_step_offset_pts=LABEL_STEP_OFFSET_PTS,
        label_max_levels=LABEL_MAX_LEVELS,
        exp_min_rel_height=0.00,
        exp_min_distance_pts=1,
        show_unmatched=True,
        title=f"1) With binning ({BIN_STEP} m/z): EXP({Path(EXP_MZML_PATH).name}, {scan_info}) vs SIM({POLYMER_EXPR}, {inferred_family})",
    )

    # =========================================================
    # 2) WITHOUT BINNING
    # =========================================================
    plot_overlay_raw_exp_vs_sim(
        exp_mz_raw=exp_mz_raw,
        exp_i_raw=exp_i_raw,
        sim_fragments=plot_fragments,
        label_fragments=plot_fragments,
        sim_scale=SIM_SCALE,
        tol_mz=MATCH_TOL_MZ,
        x_min=MZ_MIN,
        x_max=MZ_MAX,
        title=f"2) Without binning: EXP({Path(EXP_MZML_PATH).name}, {scan_info}) vs SIM({POLYMER_EXPR}, {inferred_family})",
        label_all_matched=True,
        label_top_n_matched=None,
        label_stagger=LABEL_STAGGER,
        label_cluster_gap_mz=LABEL_CLUSTER_GAP_MZ,
        label_base_offset_pts=LABEL_BASE_OFFSET_PTS,
        label_step_offset_pts=LABEL_STEP_OFFSET_PTS,
        label_max_levels=LABEL_MAX_LEVELS,
    )

    # =========================================================
    # 3) WITHOUT SIMULATED PEAKS
    # =========================================================
    plot_experimental_only(
        exp_mz_raw=exp_mz_raw,
        exp_i_raw=exp_i_raw,
        exp_mz_binned=mz_axis,
        exp_i_binned=exp_binned,
        use_binned=False,
        x_min=MZ_MIN,
        x_max=MZ_MAX,
        title=f"3) Experimental spectrum only: EXP({Path(EXP_MZML_PATH).name}, {scan_info})",
    )
