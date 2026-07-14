import numpy as np

from app.services.object_memory_service import ObjectMemoryService
from app.services.tracking_service import TrackedObject


def test_object_memory_merges_raw_id_reset_into_one_card():
    memory = ObjectMemoryService(max_gap_frames=90)
    frame = np.full((720, 1280, 3), 180, dtype=np.uint8)

    first = memory.update([
        TrackedObject(3, [100, 100, 220, 180], 0.92, "car")
    ], frame)
    second = memory.update([
        TrackedObject(41, [112, 108, 234, 190], 0.86, "truck")
    ], frame)

    assert first[0].track_id == second[0].track_id
    summary = memory.card_summary(first[0].track_id)
    assert summary is not None
    assert summary["observations"] == 2
    assert set(summary["merged_track_ids"]) == {3, 41}
    assert memory.last_stats["merged_tracks"] == 1


def test_object_memory_keeps_separate_same_frame_objects():
    memory = ObjectMemoryService()
    frame = np.full((720, 1280, 3), 180, dtype=np.uint8)

    tracks = memory.update([
        TrackedObject(1, [100, 100, 220, 180], 0.91, "car"),
        TrackedObject(2, [800, 100, 920, 180], 0.89, "truck"),
    ], frame)

    assert len(tracks) == 2
    assert {track.track_id for track in tracks} == {1, 2}
    assert memory.last_stats["same_frame_duplicates"] == 0


def test_object_memory_suppresses_same_frame_duplicate_boxes():
    memory = ObjectMemoryService()
    frame = np.full((720, 1280, 3), 180, dtype=np.uint8)

    tracks = memory.update([
        TrackedObject(1, [100, 100, 220, 180], 0.91, "car"),
        TrackedObject(2, [108, 104, 218, 178], 0.88, "car"),
    ], frame)

    assert len(tracks) == 1
    assert tracks[0].track_id == 1
    assert memory.last_stats["same_frame_duplicates"] == 1


def test_object_memory_card_can_store_embedding_vector():
    memory = ObjectMemoryService(
        embedding_enabled=True,
        embedding_model="test_embedding_v1",
        store_embedding_vector=True,
    )
    track = TrackedObject(
        7,
        [100, 100, 220, 180],
        0.91,
        "car",
        appearance_signature=[0.5, 0.5, 0.5, 0.5],
    )

    tracks = memory.update([track])
    summary = memory.card_summary(tracks[0].track_id)

    assert summary is not None
    assert summary["embedding"]["enabled"] is True
    assert summary["embedding"]["model"] == "test_embedding_v1"
    assert summary["embedding"]["dim"] == 4
    assert summary["embedding"]["hash"]
    assert summary["embedding"]["vector"] == [0.5, 0.5, 0.5, 0.5]
