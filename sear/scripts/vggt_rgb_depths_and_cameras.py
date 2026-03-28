"""
Extracts camera poses and depths of rgb images representing a scene and saves them.
It also saves thermal images.
"""

import gc
import json
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import tyro
from dataclasses_reverse_cli.reverse_cli import ReverseCli
from PIL import Image
from torchvision.transforms import ToPILImage
from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.pose_enc import pose_encoding_to_extri_intri

from sear.data_processing.frame_info import FrameInfo
from sear.visualization.point_cloud_from_dataset import (
    PointCloudFromDatasetCreator,
)


@dataclass
class VGGTParameters(ReverseCli):
    model_path: Path
    """Directory containing input model"""
    scene_dir: Path = Path("input")
    """Directory containing input images"""
    rgb_images_name: str = "images"
    """Directory containing rgb images"""
    thermal_images_name: str = "thermal"
    """Directory containing thermal images"""
    output_dir: Path = Path("./outputs")
    """Directory to save output files"""
    convert_to_rgb: bool = True
    """Whether to remove the alpha channel presented in images"""
    use_thermal: bool = False
    """Whether to use thermals instead of rgb to create predictions"""


def _remove_rgba_and_preprocess(image_names: list[Path]) -> torch.Tensor:
    """
    Reads images from `image_names`, converts them to RGB (important if they are RGBA)
    and preprocesseses.

    :return: tensor of processed images.
    """
    with tempfile.TemporaryDirectory() as temporary_directory:
        image_names_result: list[Path] = []
        for image_path in image_names:
            image = Image.open(image_path)
            image = image.convert("RGB")
            save_path = Path(temporary_directory) / image_path.name
            image.save(save_path)
            image_names_result.append(save_path)
        images = load_and_preprocess_images(image_path_list=image_names_result)
    return images


def create_vggt_dataset(args: VGGTParameters) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load model
    model = VGGT()
    state_dict = torch.load(args.model_path, map_location=device)
    model.load_state_dict(state_dict)
    del model.point_head
    model.point_head = None
    del model.track_head
    model.track_head = None
    gc.collect()
    torch.cuda.empty_cache()

    model = model.to(device)
    model.eval()

    # read images
    images_dir = args.scene_dir / args.rgb_images_name
    image_names = sorted([f for f in images_dir.iterdir()])
    if args.convert_to_rgb:
        images_list = _remove_rgba_and_preprocess(image_names)
    else:
        images_list = load_and_preprocess_images(image_path_list=image_names)

    thermals_dir = args.scene_dir / args.thermal_images_name
    thermals_names = sorted([f for f in thermals_dir.iterdir()])
    thermals_list = load_and_preprocess_images(image_path_list=thermals_names)

    # run vggt
    with torch.inference_mode():
        with torch.amp.autocast(str(device), dtype=torch.float16):
            input_frames = images_list
            if params.use_thermal:
                input_frames = thermals_list
            input_frames = input_frames.to(device)
            model_prediction = model.forward(images=input_frames)
            pose_enc_list = model_prediction["pose_enc_list"]
            pred_depth = model_prediction["depth"]

        extrinsics_pred_world2cam, intrinsics = pose_encoding_to_extri_intri(
            pose_enc_list[-1].to(torch.float32), images_list.shape[-2:]
        )

        pred_depth = pred_depth[0]
        extrinsics_pred_world2cam = extrinsics_pred_world2cam[0]
        intrinsics = intrinsics[0]

    # save the results
    if args.output_dir.exists():
        shutil.rmtree(args.output_dir)
    args.output_dir.mkdir()
    output_images_folder_name = "images"
    output_depths_folder_name = "depths"
    output_thermal_folder_name = "thermal"
    for output_folder_name in [
        output_images_folder_name,
        output_depths_folder_name,
        output_thermal_folder_name,
    ]:
        (args.output_dir / output_folder_name).mkdir()

    to_pil = ToPILImage()

    transforms: dict[
        str, str | float | list[dict[str, str | float | list[list[float]]]]
    ] = {}
    frame_type = list[dict[str, str | float | list[list[float]]]]
    transforms["frames"]: list[dict[str, frame_type]] = []  # type: ignore
    transforms["type"] = "ThermoScenes"

    for i in range(len(images_list)):
        # save images
        image = to_pil(images_list[i])
        file_path = Path(output_images_folder_name) / image_names[i].name
        image.save(args.output_dir / file_path)

        depth_file_path = Path(output_depths_folder_name) / (
            image_names[i].stem + ".npy"
        )
        np.save(args.output_dir / depth_file_path, pred_depth[i].cpu().numpy())

        thermal = to_pil(thermals_list[i])
        thermal_file_path = Path(output_thermal_folder_name) / thermals_names[i].name
        thermal.save(args.output_dir / thermal_file_path)

        # create poses
        frame_rgb = FrameInfo(
            extrinsic_matrix_world2cam=extrinsics_pred_world2cam[i].cpu(),
            intrinsic_matrix=intrinsics[i].cpu(),
            width=image.size[0],
            height=image.size[1],
            image_path=file_path,
            depth_path=depth_file_path,
        ).to_dict()
        frame_rgb["type"] = "rgb"

        frame_thermal = FrameInfo(
            extrinsic_matrix_world2cam=extrinsics_pred_world2cam[i].cpu(),
            intrinsic_matrix=intrinsics[i].cpu(),
            width=thermal.size[0],
            height=thermal.size[1],
            image_path=thermal_file_path,
            depth_path=depth_file_path,
        ).to_dict()
        frame_rgb["type"] = "thermal"

        transforms["frames"].append({"rgb": frame_rgb, "thermal": frame_thermal})

    with open(args.output_dir / "transforms.json", "w") as f:
        json.dump(transforms, f, indent=4)

    # create point cloud
    point_cloud_creator = PointCloudFromDatasetCreator(scene_path=args.output_dir)
    point_cloud_creator.create_point_clouds(output_folder=args.output_dir)


if __name__ == "__main__":
    params = tyro.cli(VGGTParameters)
    create_vggt_dataset(params)
