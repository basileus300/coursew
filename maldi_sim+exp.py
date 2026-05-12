import numpy as np
import matplotlib.pyplot as plt
import csv
import re
from pathlib import Path
from typing import Optional, Tuple

# ========= 1) Optional dependency: pymzml =========
try:
    import pymzml
except ImportError:
    pymzml = None


# ========= 2) Masses=========
def resolve_existing_path(*candidates: str) -> str:
    for p in candidates:
        if Path(p).exists():
            return p
    return candidates[0]


def load_atomic_masses(path: str, column: str = "avg mass") -> dict:
    """
    Reads a tab-delimited txt like:
    element    code    mono mass    avg mass
    Hydrogen   H       ...          ...
    """
    masses = {}
    with open(path, encoding="utf-8") as f:
        lines = [line for line in f if line.strip()]  # skip empty lines
    reader = csv.DictReader(lines, delimiter="\t")
    for row in reader:
        code = row["code"].strip()
        if not code:
            continue
        masses[code] = float(row[column])
    return masses


def parse_formula(formula: str) -> dict:
    tokens = re.findall(r"([A-Z][a-z]?)(\d*)", formula)
    comp = {}
    for elem, count in tokens:
        n = int(count) if count else 1
        comp[elem] = comp.get(elem, 0) + n
    return comp


def formula_mass_u(formula: str, masses_u: dict) -> float:
    comp = parse_formula(formula)
    m = 0.0
    for elem, n in comp.items():
        if elem not in masses_u:
            raise KeyError(f"Element {elem} not found in masses table")
        m += n * masses_u[elem]
    return m


# ========= 3) PS simulation (MALDI-TOF) =========
def simulate_ps_oligomers(
    n_min: int,
    n_max: int,
    monomer_mass_u: float,
    end_group_mass_u: float,
    adduct_mass_u: float,
    intensity_model: str = "gaussian",  # "gaussian" | "flat"
) -> Tuple[np.ndarray, np.ndarray]:
    """
    m/z(n) = end_group + adduct + monomer * n
    """
    n = np.arange(n_min, n_max + 1)
    mz = end_group_mass_u + adduct_mass_u + monomer_mass_u * n

    if intensity_model == "flat":
        inten = np.ones_like(mz, dtype=float)
    else:
        # gaussian envelope
        center = (n_min + n_max) / 2
        sigma = max((n_max - n_min) / 4, 1.0)
        inten = np.exp(-0.5 * ((n - center) / sigma) ** 2)

    inten = inten / inten.max() * 100.0
    return mz, inten


# ========= 4) mzML reading (MALDI) =========
def _pymzml_peaks_arrays(spec) -> Tuple[np.ndarray, np.ndarray]:
    """
    pymzML may require peak_type. Try centroided first (often MALDI exports centroid),
    then raw.
    """
    last_err = None
    for peak_type in ("centroided", "raw"):
        try:
            peaks = spec.peaks(peak_type)
            arr = np.array(list(peaks), dtype=float)
            if arr.size == 0:
                return np.array([]), np.array([])
            return arr[:, 0], arr[:, 1]
        except Exception as e:
            last_err = e
    raise RuntimeError(f"Cannot read peaks from spectrum. Last error: {last_err}")


def read_mzml_spectrum(
    mzml_path: str,
    ms_level: int = 1,
    mode: str = "max_tic",   # "max_tic" | "sum_all"
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns (m/z, intensity) for one representative experimental spectrum.
    - max_tic: choose the scan with maximal TIC (good default)
    - sum_all: sum all spectra of given ms_level (can add noise if many scans)
    """
    if pymzml is None:
        raise ImportError("pymzml is not installed. Install: pip install pymzml")
    if not Path(mzml_path).exists():
        raise FileNotFoundError(f"mzML not found: {mzml_path}")

    run = pymzml.run.Reader(mzml_path)

    if mode == "sum_all":
        all_mz = []
        all_int = []
        for spec in run:
            if getattr(spec, "ms_level", None) != ms_level:
                continue
            mzs, ints = _pymzml_peaks_arrays(spec)
            if mzs.size == 0:
                continue
            all_mz.append(mzs)
            all_int.append(ints)
        if not all_mz:
            raise ValueError("No spectra found in mzML.")
        # concatenate and return (we'll bin later)
        return np.concatenate(all_mz), np.concatenate(all_int)

    # mode == "max_tic"
    best_mz, best_int = None, None
    best_tic = -1.0
    for spec in run:
        if getattr(spec, "ms_level", None) != ms_level:
            continue
        mzs, ints = _pymzml_peaks_arrays(spec)
        if mzs.size == 0:
            continue
        tic = float(np.sum(ints))
        if tic > best_tic:
            best_tic = tic
            best_mz, best_int = mzs, ints

    if best_mz is None:
        raise ValueError("No spectra found in mzML.")
    return best_mz, best_int


def bin_spectrum(
    mz: np.ndarray,
    inten: np.ndarray,
    mz_min: float,
    mz_max: float,
    step: float = 0.1
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Fast binning to a regular axis for plotting.
    """
    axis = np.arange(mz_min, mz_max + step, step)
    binned = np.zeros_like(axis, dtype=float)

    idx = np.floor((mz - mz_min) / step).astype(int)
    valid = (idx >= 0) & (idx < binned.size)
    if np.any(valid):
        np.add.at(binned, idx[valid], inten[valid])

    return axis, binned


def normalize_to_max(y: np.ndarray, target: float = 100.0) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    m = float(np.max(y)) if y.size else 0.0
    if m <= 0:
        return y
    return y / m * target


# ========= 5) Plot overlay =========
def plot_overlay_exp_and_sim(
    mz_axis: np.ndarray,
    exp_y: np.ndarray,
    sim_mz: np.ndarray,
    sim_int: np.ndarray,
    title: str,
    xlim: Optional[Tuple[float, float]] = None,
):
    plt.figure(figsize=(12, 5))

    # Experimental: black line + light gray fill
    plt.plot(mz_axis, exp_y, label="Experimental", linewidth=1.2, color="black", zorder=1)
    plt.fill_between(mz_axis, 0, exp_y, alpha=0.12, color="black", zorder=0)

    # Simulated: red sticks
    plt.vlines(sim_mz, 0, sim_int, label="Simulated", linewidth=1.2, alpha=0.9, color="red", zorder=3)

    plt.xlabel("m/z")
    plt.ylabel("Intensity (norm.)")
    plt.title(title)
    if xlim is not None:
        plt.xlim(*xlim)

    plt.legend()
    plt.ylim(0, 110)
    plt.tight_layout()
    plt.show()


# ========= 6) Main =========
if __name__ == "__main__":
    # ---- USER INPUTS ----
    EXP_MZML = "PS-500+DIT.mzML"  #
    EXP_MODE = "max_tic"          # "max_tic" или "sum_all"
    EXP_BIN_STEP = 0.1            # MALDI-TOF обычно 0.01–0.2, зависит от данных

    MZ_CUTOFF_MAX = 1000.0
    # PS model parameters (MALDI-TOF)

    masses_file = resolve_existing_path("avg_masses.txt", "avg masses.txt")
   # masses_u = load_atomic_masses(masses_file, column="avg mass")

    masses_u = load_atomic_masses(masses_file, column="mono mass")
    PS_MONOMER = formula_mass_u("C8H8", masses_u)      # ~104.15 u
    PS_ENDGRP  = formula_mass_u("C4H10", masses_u)                                  # оставляем как в твоей модели (можешь поменять)

    ADDUCT = 108.904755

    SIM_MAX = 30.0

    N_MIN = 1
    N_MAX = int((MZ_CUTOFF_MAX - (PS_ENDGRP + ADDUCT)) / PS_MONOMER)                     # <-- под MALDI обычно нужно больше, подними если спектр уходит выше
    INT_MODEL = "gaussian"

    N_MAX = 7           # № of monomers

    # ---- read experimental ----
    mz_exp, i_exp = read_mzml_spectrum(EXP_MZML, ms_level=1, mode=EXP_MODE)

    # ---- choose plotting range from experiment ----
    mz_min = float(np.percentile(mz_exp, 0.5))
    mz_max = MZ_CUTOFF_MAX
    # чуть расширим
    mz_min = max(0.0, mz_min - 50)
    mz_max = mz_max + 50

    mz_axis, exp_binned = bin_spectrum(mz_exp, i_exp, mz_min=mz_min, mz_max=mz_max, step=EXP_BIN_STEP)
    exp_binned = normalize_to_max(exp_binned, 100.0)

    # ---- simulate ----
    sim_mz, sim_int = simulate_ps_oligomers(
        n_min=N_MIN,
        n_max=N_MAX,
        monomer_mass_u=PS_MONOMER,
        end_group_mass_u=PS_ENDGRP,
        adduct_mass_u=ADDUCT,
        intensity_model=INT_MODEL
    )

    sim_int = normalize_to_max(sim_int, SIM_MAX)
    mask = (sim_mz >= mz_min) & (sim_mz <= mz_max)
    sim_mz_plot = sim_mz[mask]
    sim_int_plot = sim_int[mask]

    # ---- overlay ----
    plot_overlay_exp_and_sim(
        mz_axis=mz_axis,
        exp_y=exp_binned,
        sim_mz=sim_mz_plot,
        sim_int=sim_int_plot,
        title=f"MALDI-TOF overlay: PS model vs {Path(EXP_MZML).name} (mode={EXP_MODE})",
        xlim=(mz_min, mz_max)
    )
