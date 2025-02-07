from vision_agent_tools.tools.internlm_xcomposer2 import InternLMXComposer2
from vision_agent_tools.types import VideoNumpy
from vision_agent_tools.tools.shared_types import BaseTool


class TextVideoTemporalClassifier(BaseTool):
    """
    Takes in a video and a prompt and identifies which parts of the video contain that
    prompt.
    """
    def __init__(self, model: str = "ixc") -> None:
        """
        Initializes the TextVideoTemporalClassifier
        """
        if model != "ixc":
            raise ValueError("Only accepts 'ixc' model")

        self.model = InternLMXComposer2()

    def __call__(self, prompt: str, video: VideoNumpy, chunk_size: int = 1) -> list[int]:
        """
        Takes in a prompt, video and a chunk size and identifies which 'chunks' of the
        video contain that prompt. Returns a binary string indicating for each 'chunk'
        whether or not the prompt is contained in that 'chunk'.

        Args:
            prompt (str): The prompt to test.
            video (VideoNumpy): The input video to be processed.
            chunk_size (int): How many frames you want each chunk to be composed of.
        """

        prompt = f"Is {prompt} in these frames? Answer with Yes or No."
        output = []
        for i in range(0, video.shape[0], chunk_size):
            chunk = video[i : i + chunk_size, :, :, :]
            response = self.model(prompt, video=chunk)
            output.append(1 if "yes" in response.lower() else 0)
        return output
