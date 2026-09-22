"""Contract tests: FaceGallery + identity voting (FUNC-06 / NFR-07 fixes).

Pure-logic tests need no models. The integration test exercises the real
YuNet+SFace path on the bundled fixture when the weights are present.
"""
import numpy as np
import pytest

from aria.perception.face import DST5, FaceGallery, align_crop, consistent_vote


# -- gallery ------------------------------------------------------------------
def test_gallery_match_best_and_runner(tmp_path):
    gal = FaceGallery()
    a = np.array([1.0, 0.0], dtype=np.float32)
    b = np.array([0.0, 1.0], dtype=np.float32)
    gal.centroids = {"alice": a, "bob": b}
    name, sim, runner = gal.match(np.array([1.0, 0.05], dtype=np.float32))
    assert name == "alice" and sim > 0.99 and runner < 0.1


def test_gallery_load_from_npz(tmp_path):
    d = tmp_path / "faces"
    d.mkdir()
    np.savez(d / "alice.npz", emb=np.array([1.0, 0.0], dtype=np.float32), count=3)
    np.savez(d / "bob.npz", emb=np.array([0.0, 1.0], dtype=np.float32), count=5)
    gal = FaceGallery().load(d)
    assert set(gal.centroids) == {"alice", "bob"}


# -- voting rule ----------------------------------------------------------------
def test_consistent_vote_needs_full_consecutive_window():
    votes = [("alice", 0.9), ("alice", 0.88)]
    assert consistent_vote(votes, vote_frames=3) is None  # not enough votes yet
    votes.append(("alice", 0.91))
    assert consistent_vote(votes, 3) == ("alice", 0.91)


def test_consistent_vote_rejects_mixed_and_none():
    assert consistent_vote([("alice", 0.9), ("bob", 0.9), ("alice", 0.9)], 3) is None
    assert consistent_vote([(None, -1.0), (None, -1.0), (None, -1.0)], 3) is None


def test_consistent_vote_recovers_after_flip():
    votes = [("alice", 0.9)] * 3 + [("bob", 0.9), ("alice", 0.9)]
    assert consistent_vote(votes, 3) is None  # window [bob, alice, alice] — mixed
    votes.append(("alice", 0.91))
    assert consistent_vote(votes, 3) is None  # bob still inside the 3-frame window
    votes.append(("alice", 0.92))
    assert consistent_vote(votes, 3) == ("alice", 0.92)  # window unanimous again


# -- integration (real models, bundled fixture) -----------------------------------
def test_align_embed_match_on_fixture():
    cv2 = pytest.importorskip("cv2")
    pytest.importorskip("onnxruntime")
    from aria.perception.face import SFaceEmbedder
    from aria.perception.vision import resolve_repo_path

    yunet = resolve_repo_path("weights/face_detection_yunet_2023mar.onnx")
    sface = resolve_repo_path("weights/face_recognition_sface_2021dec.onnx")
    bus = resolve_repo_path("fixtures/bus.jpg")
    if not (yunet.exists() and sface.exists() and bus.exists()):
        pytest.skip("face models/fixture not present")

    img = cv2.imread(str(bus))
    detector = cv2.FaceDetectorYN.create(str(yunet), "", (img.shape[1], img.shape[0]),
                                         score_threshold=0.6, top_k=10)
    _, faces = detector.detect(img)
    assert faces is not None and len(faces) >= 1, "fixture should contain a detectable face"

    embedder = SFaceEmbedder(str(sface))
    face = max(faces, key=lambda f: f[2] * f[3])
    aligned = align_crop(img, np.array(face[4:14], dtype=np.float32))
    emb = embedder(aligned)
    assert emb.shape == (128,)
    assert abs(np.linalg.norm(emb) - 1.0) < 1e-3

    # same crop must self-match above the SFace same-person threshold
    gal = FaceGallery()
    gal.centroids = {"me": emb}
    name, sim, _ = gal.match(embedder(aligned))
    assert name == "me" and sim >= 0.363
