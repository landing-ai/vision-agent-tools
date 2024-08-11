# CLIPMediaSim

## Video similarity

```python
from vision_agent_tools.tools.clip_media_sim import CLIPMediaSim
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
clip_media_sim = CLIPMediaSim()

# Run video similarity against the target image
results = clip_media_sim(video=frames, target_image=target_image)

# The results should be a list of [index_of_frame, confidence_score] where the
# video is similar to the target image.

# To find the time at which a given frame happens, you can do the following

time_per_frame = video_time / len(frames)

timestamp = results[0][0] * time_per_frame

print("Similarity detection complete!")

```

You can also run similarity against a target text doing the following:

```python
results = clip_media_sim(video=frames, target_text="a turtle holding the earth")
```

::: vision_agent_tools.tools.clip_media_sim
