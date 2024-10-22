# Florence2Sam2

This tool uses Florence2 and the SAM-2 model to do text to instance segmentation on image or video inputs.

```python
import cv2

from vision_agent_tools.models.florence2_sam2 import Florence2SAM2


# Path to your video
video_path = "path/to/your/video.mp4"

# Load the video into frames
cap = cv2.VideoCapture(video_path)
frames = []
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    frames.append(frame)
cap.release()

# Create the Florence2SAM2 instance
florence2_sam2 = Florence2SAM2()

# segment all the instances of the prompt "ball" for all video frames
results = florence2_sam2(prompt="ball", video=frames)

# Returns a list of list where the first list represents the frames and the inner
# list contains all the predictions per frame. The annotation ID can be used
# to track the same object across different frames. For example:
[
    [
        {
            "id": 0
            "mask": rle
            "label": "ball"
            "bbox": [x_min, y_min, x_max, y_max]
        }
    ],
    [
        {
            "id": 0
            "mask": rle
            "label": "ball"
            "bbox": [x_min, y_min, x_max, y_max]
        }
    ]
]

print("Instance segmentation complete!")

```

You can also run similarity against an image and get additionally bounding boxes doing the following:

```python
results = florence2_sam2(image=image, prompts=["ball"])
```

::: vision_agent_tools.models.florence2_sam2
