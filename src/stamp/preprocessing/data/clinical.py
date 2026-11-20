"""Loading the clinical table into patients and their slides.

Patients are keyed by `group_key = hospital + "_" + patient_id`, the same key the
split pipeline uses, so patients whose local IDs coincide at different hospitals
stay separate.
"""

import logging
import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from stamp.preprocessing.pipeline.config import ClinicalColumns

_logger = logging.getLogger("stamp")

WSI_EXTENSIONS = frozenset(
    {".svs", ".tif", ".tiff", ".ndpi", ".mrxs", ".scn", ".vms", ".vmu",
     ".svslide", ".bif", ".qptiff", ".czi"}
)  # fmt: skip

LABEL_VALUES = {"Y": 1, "N": 0, "1": 1, "0": 0}
"""Accepted label spellings (case-insensitive) and the integer label they map to."""

ID_COLUMNS = ("patient_id", "slide_id", "hospital")


@dataclass(frozen=True)
class Slide:
    slide_id: str
    path: Path | None
    """None if no WSI with this slide_id was found in wsi_dir."""

    @property
    def file_stem(self) -> str:
        return safe_file_stem(self.slide_id)


@dataclass(frozen=True)
class Patient:
    group_key: str
    patient_id: str
    hospital: str
    label: int
    slides: tuple[Slide, ...]
    """Sorted by slide_id. This order defines slide_idx in the patient's bag."""

    @property
    def file_stem(self) -> str:
        return safe_file_stem(self.group_key)


def safe_file_stem(name: str) -> str:
    """Replace characters that are unsafe in file names with underscores."""
    return re.sub(r"[^\w.\-]+", "_", name)


def load_patients(
    clinical_table: Path, columns: ClinicalColumns, wsi_dir: Path
) -> list[Patient]:
    """Patients sorted by group_key, each with its slides sorted by slide_id."""
    table = load_clinical_table(clinical_table, columns)
    wsis = find_wsis(wsi_dir)
    _log_unmatched_slides(table, wsis, wsi_dir)

    patients = []
    for group_key, rows in table.groupby("group_key", sort=True):
        first = rows.iloc[0]
        patients.append(
            Patient(
                group_key=str(group_key),
                patient_id=str(first["patient_id"]),
                hospital=str(first["hospital"]),
                label=int(first["label"]),
                slides=tuple(
                    Slide(slide_id, wsis.get(slide_id))
                    for slide_id in sorted(rows["slide_id"])
                ),
            )
        )
    return patients


def load_clinical_table(path: Path, columns: ClinicalColumns) -> pd.DataFrame:
    """Read and validate the clinical CSV.

    Returns one row per slide with columns
    hospital, patient_id, group_key, slide_id, label (int).
    """
    table = pd.read_csv(path, dtype=str)
    table = _rename_columns(table, columns)
    table = _clean_ids(table)
    table["group_key"] = table["hospital"] + "_" + table["patient_id"]
    table = _parse_labels(table)
    _check_consistency(table)
    return table[
        ["hospital", "patient_id", "group_key", "slide_id", "label"]
    ].reset_index(drop=True)


def find_wsis(wsi_dir: Path) -> dict[str, Path]:
    """Map file stem -> path for every WSI under wsi_dir (searched recursively)."""
    found: dict[str, Path] = {}
    duplicates: list[str] = []
    for path in sorted(wsi_dir.rglob("*")):
        if path.suffix.lower() not in WSI_EXTENSIONS or not path.is_file():
            continue
        if path.stem in found:
            duplicates.append(path.stem)
        found[path.stem] = path
    if duplicates:
        raise ValueError(
            f"Several WSIs share a file stem in {wsi_dir}: {duplicates[:20]}"
        )
    return found


def _rename_columns(table: pd.DataFrame, columns: ClinicalColumns) -> pd.DataFrame:
    for target, candidates in columns.model_dump().items():
        if target in table.columns:
            continue
        source = next((c for c in candidates if c in table.columns), None)
        if source is None:
            raise ValueError(
                f"Clinical CSV is missing {target!r}. Expected one of {candidates}. "
                f"Available columns: {table.columns.tolist()}"
            )
        table = table.rename(columns={source: target})
    return table


def _clean_ids(table: pd.DataFrame) -> pd.DataFrame:
    table = table.copy()
    for column in ID_COLUMNS:
        table[column] = table[column].fillna("").str.strip()
        if table[column].eq("").any():
            raise ValueError(f"Clinical CSV has missing or empty values in {column!r}")
    return table


def _parse_labels(table: pd.DataFrame) -> pd.DataFrame:
    """Map labels to 0/1 and drop rows whose label isn't recognised."""
    raw = table["label"]
    labels = raw.fillna("").str.strip().str.upper().map(LABEL_VALUES)
    invalid = labels.isna()
    if invalid.any():
        _logger.warning(
            f"Clinical CSV: dropping {int(invalid.sum())} rows with invalid labels "
            f"{raw[invalid].unique().tolist()}"
        )
    table = table.loc[~invalid].copy()
    table["label"] = labels[~invalid].astype(int)
    if table.empty:
        raise ValueError("Clinical CSV has no rows with a valid label")
    return table


def _check_consistency(table: pd.DataFrame) -> None:
    duplicated = table["slide_id"].duplicated(keep=False)
    if duplicated.any():
        raise ValueError(
            "Duplicate slide IDs in clinical CSV: "
            f"{table.loc[duplicated, 'slide_id'].tolist()[:20]}"
        )

    labels_per_patient = table.groupby("group_key")["label"].nunique()
    if (inconsistent := labels_per_patient[labels_per_patient > 1]).any():
        raise ValueError(
            f"Patients with more than one label: {inconsistent.index.tolist()[:20]}"
        )

    if table["group_key"].drop_duplicates().map(safe_file_stem).duplicated().any():
        raise ValueError("Two different patients would get the same file name")


def _log_unmatched_slides(
    table: pd.DataFrame, wsis: dict[str, Path], wsi_dir: Path
) -> None:
    listed = set(table["slide_id"])
    if missing := sorted(listed - set(wsis)):
        _logger.warning(
            f"{len(missing)} slides in the clinical CSV have no WSI in {wsi_dir}: "
            f"{missing[:20]}"
        )
    if unused := len(set(wsis) - listed):
        _logger.info(
            f"{unused} WSIs in {wsi_dir} are not in the clinical CSV (ignored)"
        )
