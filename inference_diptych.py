import torch
from diffusers.utils import load_image, check_min_version
from controlnet_flux import FluxControlNetModel
from transformer_flux import FluxTransformer2DModel
from attn_processor import FluxAttnProcessor2_0__dipytch, FluxSingleAttnProcessor2_0__dipytch
from diffusers.models.attention_processor import FluxAttnProcessor2_0, FluxSingleAttnProcessor2_0
from pipeline_flux_controlnet_inpaint import FluxControlNetInpaintingPipeline
from PIL import Image
import cv2
import numpy as np
import matplotlib.pyplot as plt
import argparse
from torchvision.ops import box_convert

from segment_anything import SamPredictor, sam_model_registry

from groundingdino.util.inference import load_model, load_image, predict, annotate

check_min_version("0.30.2")

parser = argparse.ArgumentParser()
parser.add_argument("--image_path", type=str, default='./images/bear.jpg')
parser.add_argument("--prompt", type=str, default="A diptych with two side-by-side images of same toy. On the top, a photo of toy. At the bottom, replicate this toy but as \"the toy is playing guitar\".")
parser.add_argument("--subject", type=str, default="toy")
parser.add_argument("--box_treshold", type=float, default=0.35)
parser.add_argument("--text_treshold", type=float, default=0.25)
args = parser.parse_args()

# Groundingdino
dino_model = load_model("./GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py", 
                        "./GroundingDINO/weights/groundingdino_swint_ogc.pth")
IMAGE_PATH = args.image_path
TEXT_PROMPT = f"{args.subject} ."
BOX_TRESHOLD = args.box_treshold
TEXT_TRESHOLD = args.text_treshold

image_source, image = load_image(IMAGE_PATH)

boxes, logits, phrases = predict(
    model=dino_model,
    image=image,
    caption=TEXT_PROMPT,
    box_threshold=BOX_TRESHOLD,
    text_threshold=TEXT_TRESHOLD
)

annotated_frame = annotate(image_source=image_source, boxes=boxes, logits=logits, phrases=phrases)
cv2.imwrite("./output/annotated_image.jpg", annotated_frame)

# SAM
sam = sam_model_registry["default"](checkpoint="./SAM_checkpoints/sam_vit_h_4b8939.pth")
predictor = SamPredictor(sam)
predictor.set_image(image_source)
h, w, _ = image_source.shape
boxes = boxes * torch.Tensor([w, h, w, h])
boxes = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy").numpy()
masks, _, _ = predictor.predict(
    point_coords=None,
    point_labels=None,
    box=boxes[0][None, :],
    multimask_output=False,
)

mask = masks[0]
mask = np.tile(mask[..., np.newaxis], (1, 1, 3))
masked_image = image_source * mask
mask_pil = Image.fromarray((mask * 255).astype('uint8'), 'RGB')
# mask_pil.save("./output/mask.png")
inverse_mask = Image.eval(mask_pil, lambda x: 255 - x)
result = masked_image + np.array(inverse_mask)
result_pil = Image.fromarray(result.astype('uint8'))
# result_pil.save("./output/samed_image.png")

# Paste patches (different from paper)
mask_cp = np.ones_like(image_source) * 255
mask_zero = np.ones_like(image_source) * 0
mask_diptych = np.concat([mask_zero, mask_cp], axis=0)
image_diptych = np.concat([result, mask_zero], axis=0)

# Build pipeline
del sam, predictor, dino_model
controlnet = FluxControlNetModel.from_pretrained("alimama-creative/FLUX.1-dev-Controlnet-Inpainting-Beta", 
                                                torch_dtype=torch.bfloat16)
transformer = FluxTransformer2DModel.from_pretrained(
        "black-forest-labs/FLUX.1-dev", subfolder='transformer', torch_dtype=torch.bfloat16
    )

# Setting attn processor
attn_procs = {}
transformer_sd = transformer.state_dict()
for name in transformer.attn_processors.keys():
    if name.startswith("single"):
        attn_procs[name] = FluxSingleAttnProcessor2_0__dipytch()  # actually, there is no difference
    else:
        attn_procs[name] = FluxAttnProcessor2_0__dipytch()
    
transformer.set_attn_processor(attn_procs)
del attn_procs, transformer_sd

# Loading pipeline
pipe = FluxControlNetInpaintingPipeline.from_pretrained(
    "/apdcephfs/share_302508626/yijicheng/model/flux-dev",
    controlnet=controlnet,
    transformer=transformer,
    torch_dtype=torch.bfloat16
).to("cuda")
pipe.transformer.to(torch.bfloat16)
pipe.controlnet.to(torch.bfloat16)

# Load image and mask
size = (768, 768)
image = Image.fromarray(image_diptych)
mask = Image.fromarray(mask_diptych)
image = image.resize((768, 768*2))
mask = mask.resize((768, 768*2))
image.save('./output/image_diptych.png')
mask.save('./output/mask_diptych.png')
generator = torch.Generator(device="cuda").manual_seed(24)

# Runing
result = pipe(
    prompt=args.prompt,
    height=size[1] * 2,
    width=size[0],
    control_image=image,
    control_mask=mask,
    num_inference_steps=30,
    generator=generator,
    controlnet_conditioning_scale=0.95,
    guidance_scale=3.5,
    negative_prompt="",
    true_guidance_scale=1.0 # default: 3.5 for alpha and 1.0 for beta
).images[0]

result.save(f'./output/flux_inpaint_{args.subject}.png')
