# TextVideoTemporalClassifier

```python
from vision_agent_tools.tools.text_video_temporal_classifier import TextVideoTemporalClassifier
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

video_classifier = TextVideoTemporalClassifier()

# The result will be a list of binary values representing if 'dog' was found in 
# each chunk of frames or not. 'chunk_size' allows you to change how many frames are
# in each chunk. For example if your video is 15 frames long, and you set 'chunk_size'
# to 5, and let's say the first 10 frames there are no dogs and the last 5 frames there
# is a dog, you will get [0, 0, 1], meaning a dog was found in the last 5 frames.
results = video_classifier("dog", video=frames)
```

:::vision_agent_tools.tools.text_video_temporal_classifier
