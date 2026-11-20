"""Small in-memory test data for the step unit tests (no slides needed)."""

from stamp.preprocessing.data.bags import PatientInfo, SlideInfo

PATIENT = PatientInfo(group_key="H_P", patient_id="P", hospital="H", label=1)


def slide_info(slide_id: str = "S", n_tiles: int = 3) -> SlideInfo:
    return SlideInfo(
        slide_id=slide_id,
        slide_path=f"/wsis/{slide_id}.svs",
        mpp=0.5,
        width_px=1000,
        height_px=1000,
        cache_file=f"/cache/{slide_id}.zip",
        n_tiles=n_tiles,
    )
