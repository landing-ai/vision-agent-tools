# InternLM-XComposer-2.5

This example demonstrates how to use the InternLM-XComposer-2.5 tool to   to answer questions about images or videos.

__NOTE__: The InternLM-XComposer-2.5 model should be used in GPU environments.

```python
import cv2

from vision_agent_tools.models.internlm_xcomposer2 import InternLMXComposer2

# (replace this path with your own!)
video_path = "path/to/your/my_video.mp4"

# Load the video into frames
cap = cv2.VideoCapture(video_path)
frames = []
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    frames.append(frame)
cap.release()

# Initialize the InternLMXComposer2 model
run_inference = InternLMXComposer2()
prompt = "Here are some frames of a video. Describe this video in detail"
# Time to put InternLMXComposer2 to work!
answer = run_inference(video=p_video, prompt=prompt)

# Print the output answer
print(answer)
```

::: vision_agent_tools.models.internlm_xcomposer2
