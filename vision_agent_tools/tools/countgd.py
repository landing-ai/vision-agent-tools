import os
import os.path as osp
import torch

from PIL import Image
from .utils import download, CHECKPOINT_DIR
from typing import Optional, Any
from torch import nn
from pydantic import BaseModel
from vision_agent_tools.shared_types import BaseTool

from countgd.datasets_inference import transforms as cgd_transforms
from countgd.util.slconfig import SLConfig
from countgd.util.misc import nested_tensor_from_tensor_list
from count_gd.groundingdino.util.box_ops import box_cxcywh_to_xyxy


TEXT_MODEL_NAME = "bert-base-uncased"
CONFIG_NAME = "../helpers/cfg_countgd.py"
DEFAULT_CONFIDENCE = 0.23


class CountGDInferenceData(BaseModel):
    """
    Represents an inference result from the CountGD model.

    Attributes:
        label (str): The predicted label for the detected object.
        score (float): The confidence score associated with the prediction (between 0 and 1).
        bbox (list[float]): A list of four floats representing the bounding box coordinates (xmin, ymin, xmax, ymax)
                          of the detected object in the image.
    """

    label: str
    score: float
    bbox: list[float]


class CountGDCounting:
    def __init__(self, device) -> None:
        CHECKPOINT = (
            "https://drive.google.com/file/d/1JpfsZtcGLUM0j05CpTN4kCOrmrLf_vLz/view?usp=sharing",
            "count_gd.pt",
        )
        BERT_CHECKPOINT = (
            "bert-base-uncased",
            os.path.join(MODEL_ZOO, "bert-base-uncased"),
        )
        _SEED = 49
        CFG_FILE = self.config_file = osp.join(
            CURRENT_DIR,
            "configs",
            "cfg_fsc147_test.py",
        )

        # download required checkpoints
        self.model_checkpoint_path = download(
            url=CHECKPOINT[0],
            path=os.path.join(MODEL_ZOO, CHECKPOINT[1]),
        )

        if not os.path.exists(os.path.join(MODEL_ZOO, BERT_CHECKPOINT[1])):
            config = BertConfig.from_pretrained(BERT_CHECKPOINT[0])
            model = BertModel.from_pretrained(
                BERT_CHECKPOINT[0], add_pooling_layer=False, config=config
            )
            tokenizer = AutoTokenizer.from_pretrained(BERT_CHECKPOINT[0])

            config.save_pretrained(BERT_CHECKPOINT[1])
            model.save_pretrained(BERT_CHECKPOINT[1])
            tokenizer.save_pretrained(BERT_CHECKPOINT[1])

        # setup seed and device
        torch.manual_seed(_SEED)
        np.random.seed(_SEED)
        self.device = device

        # build model
        print("building model ... ...")
        cfg = SLConfig.fromfile(CFG_FILE)
        model = self.build_model_main(cfg, BERT_CHECKPOINT[1])
        checkpoint = torch.load(self.model_checkpoint_path, map_location="cpu")["model"]
        model.load_state_dict(checkpoint, strict=False)
        self.model = model.eval().to(device)
        print("build model, done.")

        # build pre-processing transforms
        normalize = cgd_transforms.Compose(
            [
                cgd_transforms.ToTensor(),
                cgd_transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        self.data_transform = cgd_transforms.Compose(
            [
                cgd_transforms.RandomResize([800], max_size=1333),
                normalize,
            ]
        )

    def build_model_main(self, cfg, text_encoder):
        # we use register to maintain models from catdet6 on.
        from .count_gd.models_inference.registry import MODULE_BUILD_FUNCS

        assert cfg.modelname in MODULE_BUILD_FUNCS._module_dict
        # Add required args to cfg
        cfg.device = self.device
        cfg.text_encoder_type = text_encoder
        build_func = MODULE_BUILD_FUNCS.get(cfg.modelname)
        model, criterion, postprocessors = build_func(cfg)
        return model

    @torch.no_grad()
    def __call__(
        self,
        image: Union[str, Image.Image],
        text: str,
        visual_prompts: List[List[float]],
        threshold: float,
    ):
        assert text != "" or len(
            visual_prompts
        ), "Either text or prompts should be provided"
        if visual_prompts:
            assert len(visual_prompts) < 4, "Only max 3 visual prompts are supported"

        prompts = {"image": image, "points": visual_prompts}
        input_image, _ = self.data_transform(image, {"exemplars": torch.tensor([])})
        input_image = input_image.unsqueeze(0).to(self.device)
        exemplars = prompts["points"]

        input_image_exemplars, exemplars = self.data_transform(
            prompts["image"], {"exemplars": torch.tensor(exemplars)}
        )
        input_image_exemplars = input_image_exemplars.unsqueeze(0).to(self.device)
        exemplars = [exemplars["exemplars"].to(self.device)]

        model_output = self.model(
            nested_tensor_from_tensor_list(input_image),
            nested_tensor_from_tensor_list(input_image_exemplars),
            exemplars,
            [torch.tensor([0]).to(self.device) for _ in range(len(input_image))],
            captions=[text + " ."] * len(input_image),
        )

        ind_to_filter = list(range(len(model_output["token"][0].word_ids)))
        logits = model_output["pred_logits"].sigmoid()[0][:, ind_to_filter]
        boxes = model_output["pred_boxes"][0]

        # Filter out low confidence detections
        box_mask = logits.max(dim=-1).values > threshold
        logits = logits[box_mask, :].cpu().numpy()
        boxes = box_cxcywh_to_xyxy(boxes[box_mask, :])
        boxes = boxes.cpu().numpy()

        # create output structure required by va
        assert len(boxes) == len(logits)
        result = []
        labels = text.split(" .") if text.strip() != "" else ["object"]

        for i in range(len(boxes)):
            if len(labels) == 1:
                lbl = labels[0]
            else:
                lbl_token_ids = [self.model.tokenizer(x)["input_ids"] for x in labels]
                pred_token_id = model_output["token"][0].ids[int(logits[i].argmax())]
                for i, lbl_token_id in enumerate(lbl_token_ids):
                    if pred_token_id in lbl_token_id:
                        lbl = labels[i]
                        break
            result.append(
                {
                    "bbox": boxes[i].tolist(),
                    "score": float(logits[i].max()),
                    "label": lbl,
                }
            )

        out_label = "Detected instances predicted with"
        if len(text.strip()) > 0:
            out_label += " text"
            if exemplars[0].size()[0] == 1:
                out_label += " and " + str(exemplars[0].size()[0]) + " visual exemplar."
            elif exemplars[0].size()[0] > 1:
                out_label += (
                    " and " + str(exemplars[0].size()[0]) + " visual exemplars."
                )
            else:
                out_label += "."
        elif exemplars[0].size()[0] > 0:
            if exemplars[0].size()[0] == 1:
                out_label += " " + str(exemplars[0].size()[0]) + " visual exemplar."
            else:
                out_label += " " + str(exemplars[0].size()[0]) + " visual exemplars."
        else:
            out_label = "Nothing specified to detect."

        print(out_label)
        # save viz
        # output_img = overlay_bounding_boxes(np.array(image), result)
        # output_img = Image.fromarray(output_img)
        return result


class NShotCounting(BaseTool):
    """
    Tool for object counting using the zeroshot and n-shot versions of the LOCA model from the paper
    [A Low-Shot Object Counting Network With Iterative Prototype Adaptation ](https://github.com/djukicn/loca).

    """

    _CHECKPOINT_DIR = CHECKPOINT_DIR

    def __init__(self, zero_shot=True, img_size=512) -> None:
        """
        Initializes the LOCA model.

        Args:
            img_size (int): Size of the input image.

        """
        if not osp.exists(self._CHECKPOINT_DIR):
            os.makedirs(self._CHECKPOINT_DIR)

        ZSHOT_CHECKPOINT = (
            "https://drive.google.com/file/d/11-gkybBmBhQF2KZyo-c2-4IGUmor_JMu/view?usp=sharing",
            "count_zero_shot.pt",
        )
        FSHOT_CHECKPOINT = (
            "https://drive.google.com/file/d/1rTG7AjGmasfOYFm-ZzSbVQH9daYgOoIS/view?usp=sharing",
            "count_few_shot.pt",
        )

        # init model
        self._model = LOCA(
            image_size=img_size,
            num_encoder_layers=3,
            num_ope_iterative_steps=3,
            num_objects=3 if zero_shot else 1,
            zero_shot=zero_shot,
            emb_dim=256,
            num_heads=8,
            kernel_dim=3,
            backbone_name="resnet50",
            swav_backbone=True,
            train_backbone=False,
            reduction=8,
            dropout=0.1,
            layer_norm_eps=1e-5,
            mlp_factor=8,
            norm_first=True,
            activation=nn.GELU,
            norm=True,
        )

        self.device = (
            "cuda"
            if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available() else "cpu"
        )

        if zero_shot:
            self.model_checkpoint_path = download(
                url=ZSHOT_CHECKPOINT[0],
                path=os.path.join(self._CHECKPOINT_DIR, ZSHOT_CHECKPOINT[1]),
            )
        else:
            self.model_checkpoint_path = download(
                url=FSHOT_CHECKPOINT[0],
                path=os.path.join(self._CHECKPOINT_DIR, FSHOT_CHECKPOINT[1]),
            )

        state_dict = torch.load(
            self.model_checkpoint_path, map_location=torch.device(self.device)
        )["model"]
        self._model.load_state_dict(state_dict)
        self._model.to(self.device)
        self._model.eval()
        self.img_size = img_size

    @torch.inference_mode()
    def __call__(
        self,
        image: Image.Image,
        bbox: Optional[list[int]] = None,
    ) -> CountingDetection:
        """
        LOCA injects shape and appearance information into object queries
        to precisely count objects of various sizes in densely and sparsely populated scenarios.
        It also extends to a zeroshot scenario and achieves excellent localization and count errors
        across the entire low-shot spectrum.

        Args:
            image (Image.Image): The input image for object detection.
            bbox (list[int]): A list of four ints representing the bounding box coordinates (xmin, ymin, xmax, ymax)
                        of the detected query in the image.

        Returns:
            CountingDetection: An object type containing:
                - The count of the objects found similar to the bbox query.
                - A list of numpy arrays representing the masks of the objects found.
        """
        if bbox:
            assert len(bbox) == 4, "Bounding box should be in format [x1, y1, x2, y2]"
        image = image.convert("RGB")
        w, h = image.size
        img_t = T.Compose(
            [
                T.ToTensor(),
                T.Resize((self.img_size, self.img_size)),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )(image).to(self.device)
        if bbox:
            bbox = (torch.tensor(bbox) / torch.tensor([w, h, w, h]) * self.img_size).to(
                self.device
            )
        else:
            bbox = torch.ones(2, device=self.device)

        out, _ = self._model(img_t[None], bbox[None].unsqueeze(0))

        n_objects = out.flatten(1).sum(dim=1).cpu().numpy().item()

        dmap = (out - torch.min(out)) / (torch.max(out) - torch.min(out)) * 255
        density_map = dmap.squeeze().cpu().numpy().astype("uint8")
        return CountingDetection(count=round(n_objects), heat_map=[density_map])
