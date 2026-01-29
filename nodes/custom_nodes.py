"""
FitDiT ComfyUI Nodes
Virtual Try-on nodes for ComfyUI based on FitDiT
"""

import os
import math
import torch
import numpy as np
import random
from PIL import Image
from transformers import CLIPVisionModelWithProjection
import folder_paths

from preprocess.humanparsing.run_parsing import Parsing
from preprocess.dwpose import DWposeDetector
from src.pose_guider import PoseGuider
from src.utils_mask import get_mask_location
from src.pipeline_stable_diffusion_3_tryon import StableDiffusion3TryOnPipeline
from src.transformer_sd3_garm import SD3Transformer2DModel as SD3Transformer2DModel_Garm
from src.transformer_sd3_vton import SD3Transformer2DModel as SD3Transformer2DModel_Vton

fitdit_models_relative_path = "FitDiT_models"
fitdit_models_path = os.path.join(folder_paths.models_dir, fitdit_models_relative_path)
os.makedirs(fitdit_models_path, exist_ok=True)


def tensor2pil(image):
    return Image.fromarray((image.squeeze(0).cpu().clamp(0, 1) * 255).byte().numpy())

def pil2tensor(image):
    return torch.from_numpy(np.asarray(image, dtype=np.float32) / 255.0).unsqueeze(0)

def resize_image(img, target_size=768):
    width, height = img.size
    
    if width < height:
        scale = target_size / width
    else:
        scale = target_size / height
    
    new_width = int(round(width * scale))
    new_height = int(round(height * scale))
    
    resized_img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
    
    return resized_img

def pad_and_resize(im, new_width=768, new_height=1024, pad_color=(255, 255, 255), mode=Image.Resampling.LANCZOS):
    old_width, old_height = im.size
    
    ratio_w = new_width / old_width
    ratio_h = new_height / old_height
    if ratio_w < ratio_h:
        new_size = (new_width, round(old_height * ratio_w))
    else:
        new_size = (round(old_width * ratio_h), new_height)
    
    im_resized = im.resize(new_size, mode)

    pad_w = math.ceil((new_width - im_resized.width) / 2)
    pad_h = math.ceil((new_height - im_resized.height) / 2)

    new_im = Image.new('RGB', (new_width, new_height), pad_color)
    new_im.paste(im_resized, (pad_w, pad_h))

    return new_im, pad_w, pad_h

def unpad_and_resize(padded_im, pad_w, pad_h, original_width, original_height):
    """Unpad and resize image back to original dimensions"""
    width, height = padded_im.size
    
    left = pad_w
    top = pad_h
    right = width - pad_w
    bottom = height - pad_h
    
    cropped_im = padded_im.crop((left, top, right, bottom))
    resized_im = cropped_im.resize((original_width, original_height), Image.Resampling.LANCZOS)

    return resized_im


class FitDiT_LoadModel:
    """Load FitDiT Model"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "device": (["cuda", "cpu", "cuda:0", "cuda:1"], {"default": "cuda"}),
                "dtype": (["bf16", "fp16"], {"default": "bf16"}),
                "offload": ("BOOLEAN", {"default": False}),
                "aggressive_offload": ("BOOLEAN", {"default": False}),
            }
        }
    
    RETURN_TYPES = ("FITDIT_MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load_model"
    CATEGORY = "FitDiT"
    
    def load_model(self, device, dtype, offload, aggressive_offload):
        model_root = fitdit_models_path
        
        if not os.path.exists(model_root):
            raise ValueError(f"Model root path does not exist: {model_root}")
        
        # Verify necessary subdirectories
        required_subdirs = ["transformer_garm", "transformer_vton", "pose_guider"]
        for subdir in required_subdirs:
            subdir_path = os.path.join(model_root, subdir)
            if not os.path.exists(subdir_path):
                raise ValueError(
                    f"Required subdirectory not found: {subdir_path}\n"
                    f"Please ensure the following structure:\n"
                    f"  {model_root}/\n"
                    f"    ├── transformer_garm/\n"
                    f"    ├── transformer_vton/\n"
                    f"    ├── pose_guider/\n"
                    f"    ├── dwpose/\n"
                    f"    └── parsing_*/"
                )
        
        weight_dtype = torch.float16 if dtype == "fp16" else torch.bfloat16
        
        # Load transformers (model_root subdirectories)
        transformer_garm = SD3Transformer2DModel_Garm.from_pretrained(
            os.path.join(model_root, "transformer_garm"), 
            torch_dtype=weight_dtype,
            local_files_only=True,
            low_cpu_mem_usage=True
        )
        transformer_vton = SD3Transformer2DModel_Vton.from_pretrained(
            os.path.join(model_root, "transformer_vton"), 
            torch_dtype=weight_dtype,
            local_files_only=True,
            low_cpu_mem_usage=True
        )
        
        # Load pose guider
        pose_guider = PoseGuider(
            conditioning_embedding_channels=1536, 
            conditioning_channels=3, 
            block_out_channels=(32, 64, 256, 512)
        )
        pose_guider.load_state_dict(
            torch.load(os.path.join(model_root, "pose_guider", "diffusion_pytorch_model.bin"), 
                      weights_only=True)
        )
        
        clip_large_path = os.path.join(model_root, "image_encoder")
        if not os.path.exists(clip_large_path):
            clip_large_path = os.path.join(folder_paths.models_dir, "clip", "clip-vit-large-patch14")
        
        if os.path.exists(clip_large_path):
            try:
                image_encoder_large = CLIPVisionModelWithProjection.from_pretrained(
                    clip_large_path, 
                    dtype=weight_dtype,
                    local_files_only=True,
                    use_safetensors=True,
                    low_cpu_mem_usage=True
                )
            except (OSError, ValueError):
                image_encoder_large = CLIPVisionModelWithProjection.from_pretrained(
                    clip_large_path, 
                    dtype=weight_dtype,
                    local_files_only=True,
                    low_cpu_mem_usage=True
                )
        else:
            raise ValueError(
                f"CLIP model 'clip-vit-large-patch14' not found.\n"
                f"Please place it in:\n"
                f"  {model_root}/image_encoder/  OR\n"
                f"  {folder_paths.models_dir}/clip/clip-vit-large-patch14/"
            )
        
        clip_bigG_path = os.path.join(model_root, "image_encoder_bigG")
        if not os.path.exists(clip_bigG_path):
            clip_bigG_path = os.path.join(folder_paths.models_dir, "clip", "CLIP-ViT-bigG-14-laion2B-39B-b160k")
        
        if os.path.exists(clip_bigG_path):
            try:
                image_encoder_bigG = CLIPVisionModelWithProjection.from_pretrained(
                    clip_bigG_path, 
                    dtype=weight_dtype,
                    local_files_only=True,
                    use_safetensors=True,
                    low_cpu_mem_usage=True
                )
            except (OSError, ValueError):
                image_encoder_bigG = CLIPVisionModelWithProjection.from_pretrained(
                    clip_bigG_path, 
                    dtype=weight_dtype,
                    local_files_only=True,
                    low_cpu_mem_usage=True
                )
        else:
            raise ValueError(
                f"CLIP model 'CLIP-ViT-bigG-14-laion2B-39B-b160k' not found.\n"
                f"Please place it in:\n"
                f"  {model_root}/image_encoder_bigG/  OR\n"
                f"  {folder_paths.models_dir}/clip/CLIP-ViT-bigG-14-laion2B-39B-b160k/"
            )
        
        with torch.no_grad():
            pose_guider.to(device=device, dtype=weight_dtype)
            image_encoder_large.to(device=device, dtype=weight_dtype)
            image_encoder_bigG.to(device=device, dtype=weight_dtype)
        
        pipeline = StableDiffusion3TryOnPipeline.from_pretrained(
            model_root, 
            torch_dtype=weight_dtype, 
            transformer_garm=transformer_garm, 
            transformer_vton=transformer_vton, 
            pose_guider=pose_guider, 
            image_encoder_large=image_encoder_large, 
            image_encoder_bigG=image_encoder_bigG,
            local_files_only=True
        )
        
        if offload:
            pipeline.enable_model_cpu_offload()
            dwprocessor = DWposeDetector(model_root=model_root, device='cpu')
            parsing_model = Parsing(model_root=model_root, device='cpu')
        elif aggressive_offload:
            pipeline.enable_sequential_cpu_offload()
            dwprocessor = DWposeDetector(model_root=model_root, device='cpu')
            parsing_model = Parsing(model_root=model_root, device='cpu')
        else:
            pipeline.to(device)
            dwprocessor = DWposeDetector(model_root=model_root, device=device)
            parsing_model = Parsing(model_root=model_root, device=device)
        
        model_data = {
            "pipeline": pipeline,
            "dwprocessor": dwprocessor,
            "parsing_model": parsing_model,
            "device": device,
            "dtype": weight_dtype
        }
        
        return (model_data,)


class FitDiT_GenerateMask:
    """Generate mask and pose from model image"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("FITDIT_MODEL",),
                "model_image": ("IMAGE",),
                "category": (["Upper-body", "Lower-body", "Dresses"], {"default": "Upper-body"}),
                "offset_top": ("INT", {"default": 0, "min": -200, "max": 200, "step": 1}),
                "offset_bottom": ("INT", {"default": 0, "min": -200, "max": 200, "step": 1}),
                "offset_left": ("INT", {"default": 0, "min": -200, "max": 200, "step": 1}),
                "offset_right": ("INT", {"default": 0, "min": -200, "max": 200, "step": 1}),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "MASK", "IMAGE")
    RETURN_NAMES = ("masked_image", "mask", "pose_image")
    FUNCTION = "generate_mask"
    CATEGORY = "FitDiT"
    
    def generate_mask(self, model, model_image, category, offset_top, offset_bottom, offset_left, offset_right):
        dwprocessor = model["dwprocessor"]
        parsing_model = model["parsing_model"]
        
        # Convert tensor to PIL
        vton_img = tensor2pil(model_image)
        vton_img_det = resize_image(vton_img)
        
        # Detect pose
        with torch.inference_mode():
            pose_image, keypoints, _, candidate = dwprocessor(np.array(vton_img_det)[:,:,::-1])
            candidate[candidate < 0] = 0
            candidate = candidate[0]
            
            candidate[:, 0] *= vton_img_det.width
            candidate[:, 1] *= vton_img_det.height
            
            pose_image = pose_image[:,:,::-1]  # RGB
            pose_image = Image.fromarray(pose_image)
            
            # Parse model
            model_parse, _ = parsing_model(vton_img_det)
            
            # Generate mask
            data =get_mask_location(
                category, model_parse, 
                candidate, model_parse.width, model_parse.height, 
                offset_top, offset_bottom, offset_left, offset_right
            )

            if not data:
                raise ValueError("Mask generation failed. Please check the input image and parameters.")

            mask = data[0]
            mask_gray = data[1]
            mask = mask.resize(vton_img.size)
            mask_gray = mask_gray.resize(vton_img.size)
            mask = mask.convert("L")
            mask_gray = mask_gray.convert("L")
            
            masked_vton_img = Image.composite(mask_gray, vton_img, mask)
        
        # Convert to tensors
        masked_image_tensor = pil2tensor(masked_vton_img)
        mask_tensor = pil2tensor(mask)
        pose_tensor = pil2tensor(pose_image)
        
        return (masked_image_tensor, mask_tensor, pose_tensor)


class FitDiT_TryOn:
    """FitDiT Virtual Try-on"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("FITDIT_MODEL",),
                "model_image": ("IMAGE",),
                "garment_image": ("IMAGE",),
                "mask": ("MASK",),
                "pose_image": ("IMAGE",),
                "steps": ("INT", {"default": 20, "min": 15, "max": 50, "step": 1}),
                "guidance_scale": ("FLOAT", {"default": 2.0, "min": 1.0, "max": 5.0, "step": 0.1}),
                "seed": ("INT", {"default": -1, "min": -1, "max": 2147483647, "step": 1}),
                "resolution": (["768x1024", "1152x1536", "1536x2048"], {"default": "1152x1536"}),
            },
            "optional": {
                "num_images": ("INT", {"default": 1, "min": 1, "max": 4, "step": 1}),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("output",)
    FUNCTION = "try_on"
    CATEGORY = "FitDiT"
    
    def try_on(self, model, model_image, garment_image, mask, pose_image, 
               steps, guidance_scale, seed, resolution, num_images=1):
        pipeline = model["pipeline"]
        
        # Parse resolution
        new_width, new_height = map(int, resolution.split("x"))
        
        vton_img = tensor2pil(model_image)
        garm_img = tensor2pil(garment_image)
        mask_img = tensor2pil(mask)
        pose_img = tensor2pil(pose_image)
        
        model_image_size = vton_img.size
        
        # Pad and resize
        garm_img, _, _ = pad_and_resize(garm_img, new_width=new_width, new_height=new_height)
        vton_img, pad_w, pad_h = pad_and_resize(vton_img, new_width=new_width, new_height=new_height)
        mask_img, _, _ = pad_and_resize(mask_img, new_width=new_width, new_height=new_height, pad_color=(0,0,0))
        mask_img = mask_img.convert("L")
        pose_img, _, _ = pad_and_resize(pose_img, new_width=new_width, new_height=new_height, pad_color=(0,0,0))
        
        # Set seed
        if seed == -1:
            seed = random.randint(0, 2147483647)
        
        # Run inference
        with torch.inference_mode():
            res = pipeline(
                height=new_height,
                width=new_width,
                guidance_scale=guidance_scale,
                num_inference_steps=steps,
                generator=torch.Generator("cpu").manual_seed(seed),
                cloth_image=garm_img,
                model_image=vton_img,
                mask=mask_img,
                pose_image=pose_img,
                num_images_per_prompt=num_images
            ).images
        
        # Unpad and resize back
        for idx in range(len(res)):
            res[idx] = unpad_and_resize(res[idx], pad_w, pad_h, model_image_size[0], model_image_size[1])
        
        # Convert results to tensor
        output_tensors = []
        for img in res:
            output_tensors.append(pil2tensor(img))
        
        if len(output_tensors) > 1:
            output = torch.cat(output_tensors, dim=0)
        else:
            output = output_tensors[0]
        
        return (output,)


NODE_CLASS_MAPPINGS = {
    "FitDiT_LoadModel": FitDiT_LoadModel,
    "FitDiT_GenerateMask": FitDiT_GenerateMask,
    "FitDiT_TryOn": FitDiT_TryOn,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FitDiT_LoadModel": "FitDiT Load Model",
    "FitDiT_GenerateMask": "FitDiT Generate Mask",
    "FitDiT_TryOn": "FitDiT Virtual Try-on",
}