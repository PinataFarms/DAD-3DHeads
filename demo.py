from functools import partial
from collections import namedtuple
import os
from fire import Fire
from pytorch_toolbelt.utils import read_rgb_image

from predictor import FaceMeshPredictor
from demo_utils import (
    draw_landmarks,
    draw_3d_landmarks,
    draw_mesh,
    draw_pose,
    get_uv_texture,
    get_mesh,
    get_flame_params,
    get_output_path,
    MeshSaver,
    ImageSaver,
    JsonSaver,
)

DemoFuncs = namedtuple(
    "DemoFuncs",
    ["processor", "saver"],
)

demo_funcs = {
    "68_landmarks": DemoFuncs(draw_landmarks, ImageSaver),
    "191_landmarks": DemoFuncs(partial(draw_3d_landmarks, subset="191"), ImageSaver),
    "445_landmarks": DemoFuncs(partial(draw_3d_landmarks, subset="445"), ImageSaver),
    "head_mesh": DemoFuncs(partial(draw_mesh, subset="head"), ImageSaver),
    "face_mesh": DemoFuncs(partial(draw_mesh, subset="face"), ImageSaver),
    "pose": DemoFuncs(draw_pose, ImageSaver),
    "uv_texture": DemoFuncs(get_uv_texture, ImageSaver),
    "3d_mesh": DemoFuncs(get_mesh, MeshSaver),
    "flame_params": DemoFuncs(get_flame_params, JsonSaver)
}


def demo(
    input_image_path: str = 'images/demo_heads/1.jpeg',
    outputs_folder: str = "outputs",
    type_of_output: str = "68_landmarks",
) -> None:

    os.makedirs(outputs_folder, exist_ok=True)

    # Preprocess and get predictions.
    image = read_rgb_image(input_image_path)
    predictor = FaceMeshPredictor.dad_3dnet()
    predictions = predictor(image)

    # Get the resulting output.
    result = demo_funcs[type_of_output].processor(predictions, image)
    # Save the demo output.
    saver = demo_funcs[type_of_output].saver()  # instantiate the Saver
    output_path = get_output_path(input_image_path, outputs_folder, type_of_output, saver.extension)
    saver(result, output_path)


if __name__ == "__main__":
    Fire(demo)
