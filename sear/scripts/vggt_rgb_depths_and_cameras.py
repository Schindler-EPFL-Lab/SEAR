"""
Extracts camera poses and depths of rgb images representing a scene and saves them.
It also saves thermal images.
"""

import json
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import tyro
from dataclasses_reverse_cli.reverse_cli import ReverseCli
from general_thermal.models.thermal_vggt import ThermalVGGT
from PIL import Image
from torchvision.transforms import ToPILImage
from vggt.utils.load_fn import load_and_preprocess_images

from sear.data_processing.frame_info import FrameInfo


@dataclass
class VGGTParameters(ReverseCli):
    model_path: Path = Path("model")
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

    reconstruction = ThermalVGGT(
        model_path=args.model_path,
        image_dir=args.scene_dir,
        output_dir=args.output_dir,
        conf_treshold=50.0,
        stride=1,
        save_processed_images=False,
        use_half_dataset=False,
        shuffle_output=False,
    )

    # read images
    images_dir = args.scene_dir / args.rgb_images_name
    image_names = sorted([f for f in images_dir.iterdir()])
    if args.convert_to_rgb:
        images_list = _remove_rgba_and_preprocess(image_names)
    else:
        images_list = load_and_preprocess_images(image_path_list=image_names)
    images_list = images_list.to(device)

    thermals_dir = args.scene_dir / args.thermal_images_name
    thermals_names = sorted([f for f in thermals_dir.iterdir()])
    thermals_list = load_and_preprocess_images(image_path_list=thermals_names)

    # run vggt
    extrinsic_world2cam, intrinsic, depth_map, _ = reconstruction.run_VGGT(
        images=images_list,
    )

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
        np.save(args.output_dir / depth_file_path, depth_map[i])

        thermal = to_pil(thermals_list[i])
        thermal_file_path = Path(output_thermal_folder_name) / thermals_names[i].name
        thermal.save(args.output_dir / thermal_file_path)

        # create poses
        frame_rgb = FrameInfo(
            extrinsic_matrix_world2cam=extrinsic_world2cam[i],
            intrinsic_matrix=intrinsic[i],
            width=image.size[0],
            height=image.size[1],
            image_path=file_path,
            depth_path=depth_file_path,
        ).to_dict()
        frame_rgb["type"] = "rgb"

        frame_thermal = FrameInfo(
            extrinsic_matrix_world2cam=extrinsic_world2cam[i],
            intrinsic_matrix=intrinsic[i],
            width=thermal.size[0],
            height=thermal.size[1],
            image_path=thermal_file_path,
            depth_path=depth_file_path,
        ).to_dict()
        frame_rgb["type"] = "thermal"

        transforms["frames"].append({"rgb": frame_rgb, "thermal": frame_thermal})

    with open(args.output_dir / "transforms.json", "w") as f:
        json.dump(transforms, f, indent=4)


if __name__ == "__main__":
    params = tyro.cli(VGGTParameters)
    create_vggt_dataset(params)
