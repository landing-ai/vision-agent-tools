# Qwen2-VL

This example demonstrates how to use the Qwen2-VL model to   to answer questions about images or videos.

__NOTE__: The Qwen2-VL model should be used in GPU environments.

```python
import cv2
import numpy as np
from vision_agent_tools.models.qwen2_vl import Qwen2VL

# (replace this path with your own!)
video_path = "path/to/your/my_video.mp4"

# Load the video into frames
cap = cv2.VideoCapture(video_path)
frames = []
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frames.append(frame)
cap.release()
frames = np.stack(frames, axis=0)

# Initialize the Qwen2VL model
run_inference = Qwen2VL()
prompt = "Here are some frames of a video. Describe this video in detail"
# Time to put Qwen2VL to work!
answer = run_inference(video=frames, prompt=prompt)

# Print the output answer
print(answer)
```

::: vision_agent_tools.models.qwen2_vl
