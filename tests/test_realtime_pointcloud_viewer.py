import numpy as np

from agentic_memory_nav.visualization.realtime_pointcloud_viewer import RealtimePointCloudViewer


def test_viewer_accepts_empty_pointcloud_without_crashing() -> None:
    viewer = RealtimePointCloudViewer(port=0)
    try:
        viewer.update(
            np.array([], dtype=np.float32),
            frame_id="frame_0000",
            robot=(0.0, 0.0, 0.0),
            yaw=0.0,
            rgb=np.zeros((4, 4, 3), dtype=np.uint8),
        )
        with viewer._lock:
            assert viewer._payload["count"] == 0
            assert viewer._payload["points"] == []
    finally:
        viewer.close()
