# TextVideoClassifier

```python
from vision_agent_tools.tools.text_video_classifier import TextVideoClassifier
from decord import VideoReader
from decord import cpu

# Path to your video
video_path = "path/to/your/video.mp4"

# Load the video
vr = VideoReader(video_path, ctx=cpu(0))

# Subsample frames
frame_idxs = range(0, len(vr) - 1, 20)
frames = vr.get_batch(frame_idxs).asnumpy()

# Calculate video timestamps
video_time = len(vr) / vr.get_avg_fps()

vid_classifier = TextVideoClassifier()

# The results will be a probability distribution over the target text for each frame.
# In the example below, you will get an Nx2 array where the first entry is a tuple
# with the probabilty of 'not dog' and the probability of 'dog'
results = vid_classifier(video=frames, target_text=["not dog", "dog"])
```

::: vision_agent_tools.tools.text_video_classifier
