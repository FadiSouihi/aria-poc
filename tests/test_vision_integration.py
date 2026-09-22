"""Integration test: the real YOLO+tracker path on the bundled fixture.

Skips cleanly when the vision stack or weights are unavailable, so the
hermetic suite stays green on machines without them. Uses the SAME code path
the running service uses (``PersonVisionService._infer``).
"""
import pytest


def _make_service():
    ultralytics = pytest.importorskip("ultralytics")
    pytest.importorskip("cv2")
    from aria.core.config import Config
    from aria.perception.vision import PersonVisionService, resolve_repo_path
    from helpers import make_bus

    svc = PersonVisionService()
    svc.attach(make_bus(), Config({}, "<t>"))
    svc._tracker_yaml = resolve_repo_path("configs/tracker_bytetrack.yaml")
    svc.device_str = "cpu"
    for candidate in (resolve_repo_path(f"weights/yolo26n.pt"), resolve_repo_path("yolo26n.pt")):
        if candidate.exists():
            svc.model = ultralytics.YOLO(str(candidate))
            break
    else:
        try:
            svc.model = ultralytics.YOLO("yolo26n.pt")  # one-time auto-download
        except Exception as exc:
            pytest.skip(f"yolo26n weights unavailable: {exc}")
    return svc


def _fixture_image():
    import cv2
    from aria.perception.vision import resolve_repo_path

    path = resolve_repo_path("fixtures/bus.jpg")
    if not path.exists():
        pytest.skip("bus.jpg fixture not present")
    img = cv2.imread(str(path))
    assert img is not None
    return img


def test_person_detection_on_fixture():
    svc = _make_service()
    img = _fixture_image()
    tracks = svc._infer(img)
    assert len(tracks) >= 1, "bus.jpg contains people; expect at least one track"
    for tid, bbox, conf in tracks:
        assert isinstance(tid, int) and tid >= 0
        assert len(bbox) == 4 and bbox[2] > bbox[0] and bbox[3] > bbox[1]
        assert conf >= 0.2


def test_track_ids_persist_across_frames():
    svc = _make_service()
    img = _fixture_image()
    first = {t[0] for t in svc._infer(img)}
    second = {t[0] for t in svc._infer(img)}
    assert first & second, "persist=True must keep identity stable across cycles"
