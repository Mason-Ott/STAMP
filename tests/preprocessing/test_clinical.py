from pathlib import Path

import pandas as pd
import pytest

from stamp.preprocessing.data.clinical import load_clinical_table, load_patients
from stamp.preprocessing.pipeline.config import ClinicalColumns

from .conftest import LABELS


def write_csv(tmp_path: Path, rows: dict) -> Path:
    path = tmp_path / "clinical.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def test_patients_are_keyed_by_hospital_and_id(dataset):
    patients = load_patients(dataset.clinical_table, ClinicalColumns(), dataset.wsi_dir)
    assert {p.group_key: p.label for p in patients} == LABELS
    p1 = next(p for p in patients if p.group_key == "H1_P1")
    assert [s.slide_id for s in p1.slides] == ["S1", "S2"]
    assert all(s.path is not None and s.path.is_file() for s in p1.slides)


def test_canonical_column_names(tmp_path):
    path = write_csv(
        tmp_path,
        {"patient_id": ["a"], "slide_id": ["s"], "label": ["1"], "hospital": ["h"]},
    )
    table = load_clinical_table(path, ClinicalColumns())
    assert table.to_dict("records") == [
        {
            "hospital": "h",
            "patient_id": "a",
            "group_key": "h_a",
            "slide_id": "s",
            "label": 1,
        }
    ]


def test_rows_with_invalid_labels_are_dropped(tmp_path):
    path = write_csv(
        tmp_path,
        {
            "patient_id": ["a", "b"],
            "slide_id": ["s1", "s2"],
            "label": ["y", "maybe"],
            "hospital": ["h", "h"],
        },
    )
    assert load_clinical_table(path, ClinicalColumns())["slide_id"].tolist() == ["s1"]


def test_missing_wsi_gives_slide_without_path(tmp_path):
    path = write_csv(
        tmp_path,
        {
            "patient_id": ["a"],
            "slide_id": ["nowhere"],
            "label": ["N"],
            "hospital": ["h"],
        },
    )
    (wsi_dir := tmp_path / "wsis").mkdir()
    (patient,) = load_patients(path, ClinicalColumns(), wsi_dir)
    assert patient.slides[0].path is None


@pytest.mark.parametrize(
    "rows, message",
    [
        (
            {"patient_id": ["a", "b"], "slide_id": ["s", "s"], "label": ["1", "0"]},
            "Duplicate slide IDs",
        ),
        (
            {"patient_id": ["a", "a"], "slide_id": ["s1", "s2"], "label": ["1", "0"]},
            "more than one label",
        ),
        (
            {"patient_id": ["a", " "], "slide_id": ["s1", "s2"], "label": ["1", "0"]},
            "empty values in 'patient_id'",
        ),
    ],
)
def test_inconsistent_tables_are_rejected(tmp_path, rows, message):
    path = write_csv(tmp_path, {**rows, "hospital": ["h", "h"]})
    with pytest.raises(ValueError, match=message):
        load_clinical_table(path, ClinicalColumns())


def test_missing_column_is_reported(tmp_path):
    path = write_csv(tmp_path, {"patient_id": ["a"], "slide_id": ["s"], "label": ["1"]})
    with pytest.raises(ValueError, match="missing 'hospital'"):
        load_clinical_table(path, ClinicalColumns())
