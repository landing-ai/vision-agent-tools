# CLIPMediaSim

## Video similarity

```python
import cv2
from PIL import Image

from vision_agent_tools.models.clip_media_sim import CLIPMediaSim

# Path to your target image
image_path = "path/to/your/image.jpg"

# Path to your video
video_path = "path/to/your/video.mp4"

# Load the image
target_image = Image.open(image_path)

# Load the video into frames
cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
frames = []
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    frames.append(frame)
cap.release()

# Calculate video timestamps
video_time = len(frames) / fps

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

::: vision_agent_tools.models.clip_media_sim
