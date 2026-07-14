from app.defaults import default_source_pipeline_config


def test_aerial_source_defaults_enable_small_object_pipeline():
    config = default_source_pipeline_config()

    assert config["object_confidence_threshold"] == 0.55
    assert config["target_classes"] == []
    assert config["min_track_frames_for_event"] == 4
    assert config["min_box_width"] == 14
    assert config["max_box_aspect_ratio"] == 6.0
    assert config["aerial"]["enabled"] is True
    assert config["aerial"]["tile_size"] == 1024
    assert config["aerial"]["tile_overlap"] == 0.2
    assert config["aerial"]["nms_iou"] == 0.35
    assert config["kalman_prediction"]["enabled"] is True
    assert config["classification"]["enabled"] is True
    assert config["segmentation"]["enabled"] is False
