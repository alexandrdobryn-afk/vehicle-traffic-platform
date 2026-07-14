import numpy as np

from app.services.kalman_prediction_service import KalmanPredictionService
from app.services.tracking_service import TrackedObject


def _track(x1: int, track_id: int = 7) -> TrackedObject:
    return TrackedObject(track_id, [x1, 20, x1 + 20, 40], 0.9, "person")


def test_kalman_bridges_short_observation_gap():
    service = KalmanPredictionService(max_prediction_frames=2, smoothing=True)
    service.update([_track(10)], (100, 200, 3))
    observed = service.update([_track(14)], (100, 200, 3))
    predicted = service.update([], (100, 200, 3))

    assert observed[0].predicted is False
    assert predicted[0].track_id == 7
    assert predicted[0].predicted is True
    assert predicted[0].confidence < 0.9


def test_kalman_expires_prediction_after_configured_gap():
    service = KalmanPredictionService(max_prediction_frames=1)
    service.update([_track(10)], (100, 200, 3))
    assert len(service.update([], (100, 200, 3))) == 1
    assert service.update([], (100, 200, 3)) == []
