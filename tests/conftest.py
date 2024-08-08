import numpy as np
import pytest


@pytest.fixture
def random_video_generator():
    def _generate_test_video(n_frames=3):
        """
        Generate a test video with n_frames.
        """
        return np.random.randint(0, 255, (n_frames, 300, 300, 3), dtype=np.uint8)

    return _generate_test_video
