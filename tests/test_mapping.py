import numpy as np

from agentic_memory_nav.common.types import FrameObservation
from agentic_memory_nav.mapping.mock_mapper import MockMapper


def test_mock_mapper_stream_and_state(tmp_path):
    mapper = MockMapper(keyframe_interval=2)
    mapper.start()
    for index in range(3):
        update = mapper.update(
            FrameObservation(str(index), float(index), np.zeros((16, 16, 3), np.uint8))
        )
    assert update.map_version == 3
    assert len(mapper.buffer.keyframes) == 2
    mapper.save_state(tmp_path)
    restored = MockMapper()
    restored.load_state(tmp_path)
    assert restored.get_global_pointcloud().shape[1] == 3
