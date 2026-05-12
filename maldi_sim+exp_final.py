import csv
import re
from pathlib import Path
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

# ========= 1) Optional dependency: pymzml =========
try:
    import pymzml
except ImportError:
    pymzml = None


# ========= 2) Mass handling =========
def resolve_existing_path(*candidates: str) -> str:
    """Return the first existing path from the provided candidates."""
    for path_str in candidates:
        if Path(path_str).exists():
            return path_str
    return candidates[0]



def load_atomic_masses(path: str, column: str = "avg mass") -> dict:
    """
    Read atomic masses from a tab-delimited file.

    Expected columns include at least:
    element    code    mono mass    avg mass
    """
    masses = {}
    with open(path, encoding="utf-8") as f:
        lines = [line for line in f if line.strip()]

    reader = csv.DictReader(lines, delimiter="\t")
    for row in reader:
        code = row["code"].strip()
        if not code:
            continue
        masses[code] = float(row[column])
    return masses



def parse_formula(formula: str) -> dict:
    """Parse a molecular formula such as C8H8 or C4H10O2 into a composition dictionary."""
    formula = formula.strip()
    if not formula:
        return {}

    tokens = re.findall(r"([A-Z][a-z]?)(\d*)", formula)
    composition = {}
    for element, count in tokens:
        n = int(count) if count else 1
        composition[element] = composition.get(element, 0) + n
    return composition



def formula_mass_u(formula: str, masses_u: dict) -> float:
    """Calculate the mass of a molecular formula from the supplied atomic-mass table."""
    composition = parse_formula(formula)
    mass = 0.0
    for element, count in composition.items():
        if element not in masses_u:
            raise KeyError(f"Element {element} not found in masses table")
        mass += count * masses_u[element]
    return mass



def resolve_component_mass(
    formula: Optional[str],
    masses_u: dict,
    mass_override_u: Optional[float] = None,
) -> float:
    """
    Resolve the mass of one spectrum component.

    Priority:
    1) explicit mass override
    2) molecular formula
    3) zero mass if neither is provided
    """
    if mass_override_u is not None:
        return float(mass_override_u)
    if formula:
        return formula_mass_u(formula, masses_u)
    return 0.0


# ========= 3) Universal MALDI-TOF simulation =========
def simulate_oligomer_peaks_from_formula(
    repeat_unit_formula: str,
    masses_u: dict,
    n_min: int,
    n_max: int,
    end_group_formula: Optional[str] = None,
    adduct_formula: Optional[str] = None,
    end_group_mass_u: Optional[float] = None,
    adduct_mass_u: Optional[float] = None,
    intensity_model: str = "gaussian",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simulate a MALDI-TOF oligomer series from molecular formulas.

    The peak positions are calculated as:
        m/z(n) = M(end groups) + n * M(repeat unit) + M(adduct)

    When intensity_model == "gaussian", the relative intensities are assigned as:
        I_n = exp(-0.5 * ((n - center) / sigma)^2)

    Parameters
    ----------
    repeat_unit_formula : str
        Molecular formula of one repeat unit.
    masses_u : dict
        Atomic masses table.
    n_min, n_max : int
        Minimum and maximum oligomer number.
    end_group_formula : str, optional
        Molecular formula of the end groups.
    adduct_formula : str, optional
        Molecular formula of the adduct.
    end_group_mass_u : float, optional
        Explicit end-group mass override.
    adduct_mass_u : float, optional
        Explicit adduct mass override.
    intensity_model : str
        "gaussian" or "flat".

    Returns
    -------
    mz : np.ndarray
        Simulated m/z values.
    intensity : np.ndarray
        Relative intensities normalized to 100.
    """
    if n_min < 1:
        raise ValueError("n_min must be >= 1")
    if n_max < n_min:
        raise ValueError("n_max must be >= n_min")

    repeat_unit_mass_u = resolve_component_mass(repeat_unit_formula, masses_u)
    end_group_mass_total_u = resolve_component_mass(end_group_formula, masses_u, end_group_mass_u)
    adduct_mass_total_u = resolve_component_mass(adduct_formula, masses_u, adduct_mass_u)

    n_values = np.arange(n_min, n_max + 1, dtype=float)
    mz = end_group_mass_total_u + adduct_mass_total_u + repeat_unit_mass_u * n_values

    if intensity_model == "flat":
        intensity = np.ones_like(mz, dtype=float)
    elif intensity_model == "gaussian":
        center = (n_min + n_max) / 2
        sigma = max((n_max - n_min) / 4, 1.0)
        intensity = np.exp(-0.5 * ((n_values - center) / sigma) ** 2)
    else:
        raise ValueError("intensity_model must be 'gaussian' or 'flat'")

    intensity = intensity / intensity.max() * 100.0
    return mz, intensity


# ========= 4) mzML reading =========
def _pymzml_peaks_arrays(spec) -> Tuple[np.ndarray, np.ndarray]:
    """
    Read peak arrays from one spectrum.

    The function tries centroided peaks first and raw peaks second.
    """
    last_err = None
    for peak_type in ("centroided", "raw"):
        try:
            peaks = spec.peaks(peak_type)
            arr = np.array(list(peaks), dtype=float)
            if arr.size == 0:
                return np.array([]), np.array([])
            return arr[:, 0], arr[:, 1]
        except Exception as exc:
            last_err = exc
    raise RuntimeError(f"Cannot read peaks from spectrum. Last error: {last_err}")



def read_mzml_spectrum(
    mzml_path: str,
    ms_level: int = 1,
    mode: str = "max_tic",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return one representative experimental spectrum from an mzML file.

    Modes
    -----
    max_tic : choose the scan with the highest total ion current
    sum_all : sum all scans of the selected MS level
    """
    if pymzml is None:
        raise ImportError("pymzml is not installed. Install it with: pip install pymzml")
    if not Path(mzml_path).exists():
        raise FileNotFoundError(f"mzML not found: {mzml_path}")

    run = pymzml.run.Reader(mzml_path)

    if mode == "sum_all":
        all_mz = []
        all_int = []
        for spec in run:
            if getattr(spec, "ms_level", None) != ms_level:
                continue
            mzs, intensities = _pymzml_peaks_arrays(spec)
            if mzs.size == 0:
                continue
            all_mz.append(mzs)
            all_int.append(intensities)

        if not all_mz:
            raise ValueError("No spectra found in mzML.")
        return np.concatenate(all_mz), np.concatenate(all_int)

    if mode != "max_tic":
        raise ValueError("mode must be 'max_tic' or 'sum_all'")

    best_mz = None
    best_int = None
    best_tic = -1.0

    for spec in run:
        if getattr(spec, "ms_level", None) != ms_level:
            continue
        mzs, intensities = _pymzml_peaks_arrays(spec)
        if mzs.size == 0:
            continue
        tic = float(np.sum(intensities))
        if tic > best_tic:
            best_tic = tic
            best_mz = mzs
            best_int = intensities

    if best_mz is None:
        raise ValueError("No spectra found in mzML.")

    return best_mz, best_int



def bin_spectrum(
    mz: np.ndarray,
    intensity: np.ndarray,
    mz_min: float,
    mz_max: float,
    step: float = 0.1,
) -> Tuple[np.ndarray, np.ndarray]:
    """Bin a spectrum to a regular m/z axis for plotting."""
    axis = np.arange(mz_min, mz_max + step, step)
    binned = np.zeros_like(axis, dtype=float)

    idx = np.floor((mz - mz_min) / step).astype(int)
    valid = (idx >= 0) & (idx < binned.size)
    if np.any(valid):
        np.add.at(binned, idx[valid], intensity[valid])

    return axis, binned



def normalize_to_max(y: np.ndarray, target: float = 100.0) -> np.ndarray:
    """Scale an array so that its maximum becomes the target value."""
    y = np.asarray(y, dtype=float)
    y_max = float(np.max(y)) if y.size else 0.0
    if y_max <= 0:
        return y
    return y / y_max * target


# ========= 5) Plot overlay =========
def plot_overlay_exp_and_sim(
    mz_axis: np.ndarray,
    exp_y: np.ndarray,
    sim_mz: np.ndarray,
    sim_int: np.ndarray,
    title: str,
    xlim: Optional[Tuple[float, float]] = None,
) -> None:
    """Plot the experimental spectrum and simulated stick spectrum on one figure."""
    plt.figure(figsize=(12, 5))

    plt.plot(mz_axis, exp_y, label="Experimental", linewidth=1.2, color="black", zorder=1)
    plt.fill_between(mz_axis, 0, exp_y, alpha=0.12, color="black", zorder=0)
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
    # ---- User inputs ----
    EXP_MZML = "PS-500+DIT.mzML"
    EXP_MODE = "max_tic"
    EXP_BIN_STEP = 0.1
    MZ_CUTOFF_MAX = 1000.0

    # Polymer definition by molecular formulas
    REPEAT_UNIT_FORMULA = "C8H8"
    END_GROUP_FORMULA = "C4H10"

    # Use a fixed isotopic adduct mass for silver.
    # Set ADDUCT_FORMULA = "Ag" and ADDUCT_MASS_U = None if you want the mass table value instead.
    ADDUCT_FORMULA = None
    ADDUCT_MASS_U = 108.904755

    SIM_MAX = 30.0
    N_MIN = 1
    N_MAX = 7
    INTENSITY_MODEL = "gaussian"

    masses_file = resolve_existing_path("avg_masses.txt", "avg masses.txt")
    masses_u = load_atomic_masses(masses_file, column="mono mass")

    # ---- Read experimental spectrum ----
    mz_exp, i_exp = read_mzml_spectrum(EXP_MZML, ms_level=1, mode=EXP_MODE)

    # ---- Choose plotting range ----
    mz_min = float(np.percentile(mz_exp, 0.5))
    mz_max = MZ_CUTOFF_MAX
    mz_min = max(0.0, mz_min - 50)
    mz_max = mz_max + 50

    mz_axis, exp_binned = bin_spectrum(
        mz_exp,
        i_exp,
        mz_min=mz_min,
        mz_max=mz_max,
        step=EXP_BIN_STEP,
    )
    exp_binned = normalize_to_max(exp_binned, 100.0)

    # ---- Simulate oligomer peaks from formulas ----
    sim_mz, sim_int = simulate_oligomer_peaks_from_formula(
        repeat_unit_formula=REPEAT_UNIT_FORMULA,
        masses_u=masses_u,
        n_min=N_MIN,
        n_max=N_MAX,
        end_group_formula=END_GROUP_FORMULA,
        adduct_formula=ADDUCT_FORMULA,
        adduct_mass_u=ADDUCT_MASS_U,
        intensity_model=INTENSITY_MODEL,
    )

    sim_int = normalize_to_max(sim_int, SIM_MAX)
    mask = (sim_mz >= mz_min) & (sim_mz <= mz_max)
    sim_mz_plot = sim_mz[mask]
    sim_int_plot = sim_int[mask]

    # ---- Plot overlay ----
    plot_overlay_exp_and_sim(
        mz_axis=mz_axis,
        exp_y=exp_binned,
        sim_mz=sim_mz_plot,
        sim_int=sim_int_plot,
        title=(
            f"MALDI-TOF overlay: repeat unit {REPEAT_UNIT_FORMULA} vs "
            f"{Path(EXP_MZML).name} (mode={EXP_MODE})"
        ),
        xlim=(mz_min, mz_max),
    )
