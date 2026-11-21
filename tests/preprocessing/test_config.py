import pytest

from stamp.preprocessing.pipeline.config import PipelineConfig, StepName, check_steps


def config(**settings) -> PipelineConfig:
    return PipelineConfig.model_validate(
        {"experiment": "e", "clinical_table": "c.csv", **settings}
    )


@pytest.mark.parametrize(
    "settings, expected",
    [
        ({}, ["tile", "combine", "sample", "extract"]),
        ({"encoder": "titan"}, ["tile", "combine", "sample", "extract", "encode"]),
        ({"sampling": {"enabled": False}}, ["tile", "combine", "extract"]),
        (
            {"order": "extract_then_sample"},
            ["tile", "extract_slides", "combine", "sample"],
        ),
        (
            {"order": "extract_then_sample", "slide_feat_dir": "feats"},
            ["import_slides", "combine", "sample"],
        ),
    ],
)
def test_default_steps(settings, expected):
    assert [str(s) for s in config(**settings).default_steps()] == expected


@pytest.mark.parametrize(
    "settings, steps, message",
    [
        ({}, [StepName.EXTRACT_SLIDES], "don't apply to order"),
        ({"order": "extract_then_sample"}, [StepName.EXTRACT], "don't apply to order"),
        ({}, [StepName.ENCODE], "needs pipeline.encoder"),
        ({"sampling": {"enabled": False}}, [StepName.SAMPLE], "sampling.enabled"),
    ],
)
def test_steps_that_dont_fit_the_config_are_rejected(settings, steps, message):
    with pytest.raises(ValueError, match=message):
        check_steps(config(**settings), steps)


def test_slide_feat_dir_needs_extract_then_sample():
    with pytest.raises(ValueError, match="slide_feat_dir"):
        config(slide_feat_dir="feats")


@pytest.mark.parametrize(
    "sampling, tag",
    [
        ({}, "first_n-8192"),
        ({"enabled": False}, "full"),
        ({"strategy": "random", "seed": 3}, "random-8192"),
        ({"pad_mode": "zero", "redistribute": False}, "first_n-8192-noredist-pad_zero"),
    ],
)
def test_sampling_tag_names_the_settings(sampling, tag):
    assert config(sampling=sampling).sampling.tag == tag
