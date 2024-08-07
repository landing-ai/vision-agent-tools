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
vr = VideoReader('va-generation.mp4', ctx=cpu(0))

# Subsample frames
frame_idxs = range(0, len(vr) - 1, 20)
frames = vr.get_batch(frame_idxs).asnumpy()

# Calculate video timestamps
video_time = len(vr) / vr.get_avg_fps()
time_per_frame = video_time / len(frames)

timestamps = [frame_idx * time_per_frame for frame_idx in frame_idxs]


# Create the CLIPMediaSim instance
clip_media_sim = CLIPMediaSim()

# Run video similarity against the target image
results = clip_media_sim(video=frames, timestamps=timestamps, target_image=target_image)

# The results should be a list of [timestamp, confidence_score] where the
# video is similar to the target image.


print("Similarity detection complete!")

```

You can also run similarity against a target text doing the following:

```python
...
results = clip_media_sim(video=frames, timestamps=timestamps, target_text="a turtle holding the earth")
```


::: vision_agent_tools.tools.controlnet_aux