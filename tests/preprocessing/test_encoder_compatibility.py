"""The pipeline's table of which extractor each encoder accepts must match the
encoders themselves. The encoders only declare it in `__init__`, which loads
weights, so this reads their source instead."""

import ast
from pathlib import Path

import pytest

import stamp.encoding.encoder as encoder_package
from stamp.preprocessing.steps.models import _ENCODER_CLASSES, ENCODER_EXTRACTORS


def declared_extractors(module_file: Path) -> set[str]:
    tree = ast.parse(module_file.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.keyword) and node.arg == "required_extractors":
            return {
                element.attr
                for element in ast.walk(node.value)
                if isinstance(element, ast.Attribute)
            }
    raise AssertionError(f"{module_file} declares no required_extractors")


@pytest.mark.parametrize("encoder", list(_ENCODER_CLASSES))
def test_table_matches_the_encoder(encoder):
    module_name, _ = _ENCODER_CLASSES[encoder]
    module_file = Path(encoder_package.__file__).with_name(f"{module_name}.py")
    declared = declared_extractors(module_file)
    assert {e.name for e in ENCODER_EXTRACTORS[encoder]} == declared


def test_every_supported_encoder_is_in_the_table():
    assert set(ENCODER_EXTRACTORS) == set(_ENCODER_CLASSES)
