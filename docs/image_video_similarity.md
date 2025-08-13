# ImageVideoSimilarity

```python
from vision_agent_tools.tools.image_video_similarity import ImageVideoSimilarity
from decord import VideoReader
from decord import cpu
from PIL import Image

# Path to your target image
image_path = "path/to/your/image.jpg"

# Path to your video
video_path = "path/to/your/video.mp4"

# Load the image
target_image = Image.open(image_path)

# Load the video
vr = VideoReader(video_path, ctx=cpu(0))

# Subsample frames
frame_idxs = range(0, len(vr) - 1, 20)
frames = vr.get_batch(frame_idxs).asnumpy()

# Calculate video timestamps
video_time = len(vr) / vr.get_avg_fps()

# Create the CLIPMediaSim instance
image_sim = ImageVideoSimilarity()

# Run video similarity against the target image
# The results should be a list of similarity scores between each frame and the target iamge
results = image_sim(video=frames, target_image=target_image)
```

::: vision_agent_tools.tools.image_video_similarity
