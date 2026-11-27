"""Load the shared Albumentations recipes from JSON."""

import json
import re
from pathlib import Path

import albumentations as A
import cv2


DEFAULT_PROFILE_FILE = Path(__file__).resolve().parents[1] / 'configs' / 'augmentation_profiles.json'
IMAGE_EXTENSIONS = frozenset({'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'})

PHOTOMETRIC_TRANSFORMS = {
    transform.__name__: transform for transform in (
        A.CLAHE, A.ColorJitter, A.GaussianBlur, A.GaussNoise, A.HueSaturationValue,
        A.ImageCompression, A.ISONoise, A.MotionBlur, A.NoOp,
        A.RandomBrightnessContrast, A.RandomGamma,
    )
}
TRANSFORMS = PHOTOMETRIC_TRANSFORMS | {
    transform.__name__: transform for transform in (
        A.Affine, A.CoarseDropout, A.CropAndPad, A.HorizontalFlip, A.LongestMaxSize,
        A.PadIfNeeded, A.Perspective, A.RandomRotate90, A.Rotate, A.SafeRotate,
        A.VerticalFlip,
    )
}
CV2_PARAMETERS = {
    'border_mode': {
        'BORDER_CONSTANT': cv2.BORDER_CONSTANT,
        'BORDER_REFLECT': cv2.BORDER_REFLECT,
        'BORDER_REFLECT_101': cv2.BORDER_REFLECT_101,
        'BORDER_REPLICATE': cv2.BORDER_REPLICATE,
    },
    'interpolation': {
        'INTER_NEAREST': cv2.INTER_NEAREST,
        'INTER_LINEAR': cv2.INTER_LINEAR,
        'INTER_CUBIC': cv2.INTER_CUBIC,
        'INTER_AREA': cv2.INTER_AREA,
        'INTER_LANCZOS4': cv2.INTER_LANCZOS4,
    },
}


def build_transforms(specs: list[dict], *, photometric_only: bool = False) -> list:
    """Materialize a fresh transform list so pipelines have independent state."""
    if not isinstance(specs, list):
        raise ValueError('擴增配方必須是陣列')
    allowed = PHOTOMETRIC_TRANSFORMS if photometric_only else TRANSFORMS
    transforms = []
    for spec in specs:
        if not isinstance(spec, dict) or not isinstance(spec.get('transform'), str):
            raise ValueError('每個擴增步驟必須是包含 transform 名稱的物件')
        params = spec.copy()
        name = params.pop('transform')
        probability = params.get('p', 0.5)
        if type(probability) not in (int, float) or not 0 <= probability <= 1:
            raise ValueError(f'{name} 的 p 必須是 0 到 1 的數字')
        if name == 'OneOf':
            if params.keys() - {'transforms', 'p'} or not params.get('transforms'):
                raise ValueError('OneOf 需要非空的 transforms 陣列，並可指定 p')
            children = build_transforms(params['transforms'], photometric_only=photometric_only)
            if not any(child.p > 0 for child in children):
                raise ValueError('OneOf 至少需要一個 p 大於 0 的子步驟')
            transforms.append(A.OneOf(children, p=probability))
            continue
        if name not in allowed:
            raise ValueError(f'此模式不支援擴增步驟 {name!r}')
        if 'strict' in params:
            raise ValueError(f'{name} 不接受 strict 參數')
        for parameter, values in CV2_PARAMETERS.items():
            value = params.get(parameter)
            if isinstance(value, str):
                if value not in values:
                    raise ValueError(f'{name} 的 {parameter} 不支援 {value!r}')
                params[parameter] = values[value]
        transforms.append(allowed[name](**params, strict=True))
    return transforms


def load_profiles(config_path: Path | None = None) -> dict[str, dict[str, list[dict]]]:
    """Read defaults, then replace named recipes provided by a project file."""
    profiles = json.loads(DEFAULT_PROFILE_FILE.read_text(encoding='utf-8'))
    if config_path is not None:
        overrides = json.loads(config_path.read_text(encoding='utf-8'))
        if not isinstance(overrides, dict) or set(overrides) - profiles.keys():
            raise ValueError('設定檔須使用 shared、annotated、image_only 或 obb 區塊')
        for mode, recipes in overrides.items():
            if not isinstance(recipes, dict):
                raise ValueError(f'{mode} 必須是以配方名稱為鍵的物件')
            profiles[mode].update(recipes)

    shared = profiles.pop('shared')
    reserved_obb = {'original', 'horizontal_flip', 'vertical_flip', 'rotation'}
    if reserved_obb & (profiles['obb'].keys() | shared.keys()):
        raise ValueError('obb 配方名稱不可覆寫內建的補邊、翻轉或旋轉')
    for mode, recipes in profiles.items():
        if shared.keys() & recipes.keys():
            raise ValueError(f'{mode} 的配方名稱不可與 shared 重複')

    for mode, recipes in (*profiles.items(), ('shared', shared)):
        for name, specs in recipes.items():
            if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]*', name):
                raise ValueError(f'無效配方名稱 {mode}.{name}')
            try:
                build_transforms(specs, photometric_only=mode in ('shared', 'obb'))
            except (ValueError, TypeError) as error:
                raise ValueError(f'{mode}.{name}: {error}') from error

    for recipes in profiles.values():
        recipes.update(shared)
    return profiles
