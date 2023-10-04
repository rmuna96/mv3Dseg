from monai.transforms import (
    Compose,
    Spacingd,
    Orientationd,
    ScaleIntensityd,
    SpatialPadd,
    AddChanneld,
    CropForegroundd,
    LoadImaged,
    RandGaussianNoised,
    RandCropByPosNegLabeld,
    EnsureTyped,
    RandRotated,
    RandAxisFlipd,
    Rand3DElasticd,
    EnsureType,
    AsDiscrete,
    KeepLargestConnectedComponent
)
from monai.transforms.transform import RandomizableTransform, MapTransform
from monai.config import KeysCollection
from typing import Any, Optional

import numpy as np


class RandSwapAxisd(RandomizableTransform, MapTransform):
    """
        Custom transform implementation based on monai's implementation.
        This transform randomly swaps axis of the image. It may be useful
        to make the network generalizing on input images with different
        orientations.
    """

    def __init__(
            self,
            keys: KeysCollection,
            prob: float = 0.1,
            allow_missing_keys: bool = False,
    ) -> None:
        MapTransform.__init__(self, keys, allow_missing_keys)
        RandomizableTransform.__init__(self, prob)

    def randomize(self, data: Optional[Any] = None) -> None:
        super().randomize(None)
        if not self._do_transform:
            return None

    def __call__(self, data, randomize: bool = True):
        d = dict(data)
        if randomize:
            self.randomize()

        for key in self.keys:
            if self._do_transform:
                a, b = np.random.choice(np.arange(1, 4), 2, replace=False)
                d[key] = np.swapaxes(d[key], -a, -b)
        return d


def preprocessing(cf):
    """
        Prepocessing transforms for the training and
        validation dataloader.
    """

    train_transforms = [

        LoadImaged(keys=["image", "label"]),
        AddChanneld(keys=["image", "label"]),

        # change to execute transforms with Tensor data
        EnsureTyped(keys=["image", "label"]),
        Spacingd(["image", "label"], (0.5, 0.5, 0.5), diagonal=True, mode=('bilinear', 'nearest')),
        Orientationd(keys=["image", "label"], axcodes='RAS'),

        #intensity transform
        ScaleIntensityd("image"),
        RandGaussianNoised(keys=['image'],
                           prob=0.4,
                           mean=0.1,
                           std=0.3),
        #spatial transform
        CropForegroundd(keys=["image", "label"], source_key="image"),
        SpatialPadd(keys=["image", "label"], spatial_size=cf.transform_params.crop_size),
        RandRotated(keys=['image', 'label'],
                    prob=0.4,
                    range_x=(-3.14, 3.14),
                    range_y=(-3.14, 3.14),
                    range_z=(-3.14, 3.14),
                    mode=['bilinear', 'nearest']),
        RandAxisFlipd(keys=['image', 'label'],
                      prob=0.4,
                      ),
        Rand3DElasticd(keys=['image', 'label'],
                       sigma_range=(1, 3),
                       magnitude_range=(10, 30),
                       prob=0.4,
                       padding_mode=('zeros'),
                       mode=['bilinear', 'nearest']),
        RandCropByPosNegLabeld(         #this transform make the images a list of dicts even if I set 1 sample
            keys=["image", "label"],
            label_key="label",
            spatial_size=cf.transform_params.crop_size, #!todo 2/3 each dimensions according to the mean dimension of the dataset
            pos=0.8,
            neg=0.2,
            num_samples=1,
            allow_smaller=True,
        ),
        EnsureTyped(keys=["image", "label"])
    ]

    val_transforms = [
        LoadImaged(keys=["image", "label"]),
        AddChanneld(keys=["image", "label"]),
        Spacingd(["image", "label"], (0.5, 0.5, 0.5), diagonal=True, mode=('bilinear', 'nearest')),
        ScaleIntensityd("image"),
        EnsureTyped(keys=["image", "label"]),
    ]

    return Compose(train_transforms), Compose(val_transforms)


def postprocessing(cf, test=False):
    """
        Postprocessing transforms for the predictions.
    """
    if test:
        l = [i for i in range(1, cf.model_params.out_channels)]
        post_label = Compose([
            EnsureType(),
            AsDiscrete(to_onehot=cf.model_params.out_channels),
            KeepLargestConnectedComponent(applied_labels=l)])

    else:
        post_label = Compose([
            EnsureType(),
            AsDiscrete(to_onehot=cf.model_params.out_channels)])

    post_pred = Compose([
        EnsureType(),
        AsDiscrete(argmax=True, to_onehot=cf.model_params.out_channels)])

    return post_label, post_pred
