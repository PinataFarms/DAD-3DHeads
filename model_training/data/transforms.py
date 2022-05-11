import typing
from typing import Union, Tuple, Optional, Sequence, Iterable
import albumentations as A
import cv2


def get_resize_fn(
    new_size: Union[int, Tuple[int, int]], mode: str = "resize"
) -> Union[A.BasicTransform, A.Compose]:
    if mode == "resize":
        if isinstance(new_size, Sequence):
            new_height, new_width = typing.cast(Iterable, new_size)
        else:
            new_height, new_width = new_size, new_size
        return A.Resize(new_height, new_width)
    elif mode == "longest_max_size":
        return A.Compose(
            [
                A.LongestMaxSize(new_size, always_apply=True),
                A.PadIfNeeded(new_size, new_size, border_mode=cv2.BORDER_CONSTANT, always_apply=True),
            ]
        )
    else:
        raise KeyError(mode)


def get_normalize_fn(mode: str = "imagenet") -> Optional[A.Normalize]:
    if mode == "imagenet":
        return A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    elif mode == "mean":
        return A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    else:
        return None