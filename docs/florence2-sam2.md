# Florence2Sam2

This tool uses FlorenceV2 and the SAM-2 model to do text to instance segmentation on image or video inputs.

```python
from vision_agent_tools.tools.florence2_sam2 import Florence2SAM2
from decord import VideoReader
from decord import cpu


# Path to your video
video_path = "path/to/your/video.mp4"

# Load the video
vr = VideoReader(video_path, ctx=cpu(0))

# Subsample frames
frame_idxs = range(0, len(vr) - 1, 20)
frames = vr.get_batch(frame_idxs).asnumpy()

# Create the Florence2SAM2 instance
florence2_sam2 = Florence2SAM2()

# segment all the instances of the prompt "ball" for all video frames
results = florence2_sam2(video=frames, prompts=["ball"])

# Returns a dictionary where the first key is the frame index then an annotation
# ID, then an object with the mask, label and possibly bbox (for images) for each
# annotation ID. For example:
# {
#     0:
#         {
#             0: ImageBboxMaskLabel({"mask": np.ndarray, "label": "car"}),
#             1: ImageBboxMaskLabel({"mask", np.ndarray, "label": "person"})
#         },
#     1: ...
# }

print("Instance segmentation complete!")

```

You can also run similarity against an image and get additionally bounding boxes doing the following:

```python
results = florence2_sam2(image=image, prompts=["ball"])
```

::: vision_agent_tools.tools.florence2_sam2
