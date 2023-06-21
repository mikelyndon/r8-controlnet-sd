# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md
import os
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler
from diffusers.utils import load_image
import torch
import tempfile

from cog import BasePredictor, Input, Path


class Predictor(BasePredictor):
    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""
        # self.model = torch.load("./weights.pth")
        self.controlnet = [
            ControlNetModel.from_pretrained("lllyasviel/control_v11p_sd15_canny", cache_dir="models", local_files_only=True, torch_dtype=torch.float16),
            ControlNetModel.from_pretrained("lllyasviel/control_v11p_sd15_openpose", cache_dir="models", local_files_only=True, torch_dtype=torch.float16),
            ControlNetModel.from_pretrained("lllyasviel/control_v11f1p_sd15_depth", cache_dir="models", local_files_only=True, torch_dtype=torch.float16),
            ControlNetModel.from_pretrained("lllyasviel/control_v11p_sd15_mlsd", cache_dir="models", local_files_only=True, torch_dtype=torch.float16),
            ControlNetModel.from_pretrained("lllyasviel/control_v11p_sd15_seg", cache_dir="models", local_files_only=True, torch_dtype=torch.float16),
            ControlNetModel.from_pretrained("lllyasviel/control_v11p_sd15_softedge", cache_dir="models", local_files_only=True, torch_dtype=torch.float16),
            ]
        self.pipe = StableDiffusionControlNetPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5", cache_dir="models", local_files_only=True, controlnet=self.controlnet, torch_dtype=torch.float16,
            # safety_checker=None
        )
        self.pipe.scheduler = UniPCMultistepScheduler.from_config(self.pipe.scheduler.config)

    def predict(
        self,
        prompt: str = Input(description="Input prompt", default="a magpie wearing a tophat"),
        negative_prompt: str = Input(description="Specify things to not see in the output", default="monochrome, lowres, bad anatomy, worst quality, low quality"),
        seed: int = Input(description="Random seed. Leave blank to randomize the seed", default=None),
        canny: Path = Input(description="Canny input image" ),
        pose: Path = Input(description="Pose input image" ),
        depth: Path = Input(description="Depth input image"),
        mlsd: Path = Input(description="MLSD input image"),
        seg: Path = Input(description="SEG input image"),
        softedge: Path = Input(description="Soft Edge (hed) input image"),
        canny_weight: float = Input(description="A value between 0 and 1 to emphasize the conditioning by the canny edges.", ge=0.0, le=1.0, default=1.0),
        pose_weight: float = Input(description="A value between 0 and 1 to emphasize the conditioning by the open pose.", ge=0.0, le=1.0, default=0.0),
        depth_weight: float = Input(description="A value between 0 and 1 to emphasize the conditioning by the depth.", ge=0.0, le=1.0, default=0.0),
        mlsd_weight: float = Input(description="A value between 0 and 1 to emphasize the conditioning by the mlsd.", ge=0.0, le=1.0, default=0.0),
        seg_weight: float = Input(description="A value between 0 and 1 to emphasize the conditioning by the seg.", ge=0.0, le=1.0, default=0.0),
        softedge_weight: float = Input(description="A value between 0 and 1 to emphasize the conditioning by the soft edge (hed).", ge=0.0, le=1.0, default=0.0),
    ) -> Path:
        """Run a single prediction on the model"""
        # self.pipe.enable_xformers_memory_efficient_attention()
        self.pipe.enable_model_cpu_offload()

        if seed is None:
            seed = int.from_bytes(os.urandom(2), "big")
        generator = torch.Generator(device="cpu").manual_seed(seed)
        # images = []
        # scales = []
        canny = load_image(str(canny))
        pose = load_image(str(pose))
        depth = load_image(str(depth))
        mlsd = load_image(str(mlsd))
        seg = load_image(str(seg))
        softedge = load_image(str(softedge))
        # if canny:
        #     canny = load_image(str(canny))
        #     images.append(canny)
        #     scales.append(canny_weight)
        # if pose:
        #     pose = load_image(str(pose))
        #     images.append(pose)
        #     scales.append(pose_weight)
        # if depth:
        #     depth = load_image(str(depth))
        #     images.append(depth)
        #     scales.append(depth_weight)
        # if mlsd:
        #     mlsd = load_image(str(mlsd))
        #     images.append(mlsd)
        #     scales.append(mlsd_weight)
        # if seg:
        #     seg = load_image(str(seg))
        #     images.append(seg)
        #     scales.append(seg_weight)
        # if softedge:
        #     softedge = load_image(str(softedge))
        #     images.append(softedge)
        #     scales.append(softedge_weight)
        images = [canny, pose, depth, mlsd, seg, softedge]

        output = self.pipe(
            prompt,
            images,
            num_inference_steps=20,
            generator=generator,
            negative_prompt=negative_prompt,
            # controlnet_conditioning_scale=scales,
            controlnet_conditioning_scale=[canny_weight, pose_weight, depth_weight, mlsd_weight, seg_weight, softedge_weight],
        ).images[0]

        out_path = Path(tempfile.mkdtemp()) / "out.png"
        print("out_path", out_path)
        output.save(str(out_path))
        return out_path


