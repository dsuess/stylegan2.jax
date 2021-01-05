"""
[1] T. Karras, S. Laine, M. Aittala, J. Hellsten, J. Lehtinen, and T. Aila,
    “Analyzing and Improving the Image Quality of StyleGAN,” arXiv:1912.04958
    [cs, eess, stat], Mar. 2020
"""

import functools as ft
from math import log2, sqrt
from typing import List, Tuple, Union

import haiku as hk
import jax
import jax.nn as jnn
import jax.numpy as jnp

from stylegan2._typing import ActivationFunction

from .layers import ConvDownsample2D
from .utils import ChannelOrder, _init  # pylint: disable=unused-import


def minibatch_stddev_layer(
    x: jnp.ndarray,
    group_size: int = None,
    num_new_features: int = 1,
    data_format: ChannelOrder = ChannelOrder.channels_last,
) -> jnp.ndarray:
    """Minibatch standard deviation layer. Adds the standard deviation of
    subsets of size `group_size` taken over the batch dimension as features
    to x.

    Args:
        x ([type]): [description]
        group_size (int, optional): [description]. Defaults to None.
        num_new_features (int, optional): [description]. Defaults to 1.
        data_format (str, optional): [description]. Defaults to "channels_last".

    Returns:
        [type]: [description]

    >>> x = jnp.zeros((4, 23, 26, 3))
    >>> y = minibatch_stddev_layer(x, group_size=2, data_format=ChannelOrder.channels_last)
    >>> y.shape
    (4, 23, 26, 4)
    >>> x = jnp.zeros((4, 8, 23, 26))
    >>> y = minibatch_stddev_layer(x, num_new_features=4, data_format=ChannelOrder.channels_first)
    >>> y.shape
    (4, 12, 23, 26)

    FIXME Rewrite using allreduce ops like psum to allow non-batched definition
          of networks
    """
    # pylint: disable=invalid-name
    if data_format == ChannelOrder.channels_last:
        N, H, W, C = x.shape
    else:
        N, C, H, W = x.shape

    group_size = min(group_size, N) if group_size is not None else N
    C_ = C // num_new_features

    if data_format == ChannelOrder.channels_last:
        y = jnp.reshape(x, (group_size, -1, H, W, num_new_features, C_))
    else:
        y = jnp.reshape(x, (group_size, -1, num_new_features, C_, H, W))

    y_centered = y - jnp.mean(y, axis=0, keepdims=True)
    y_std = jnp.sqrt(jnp.mean(y_centered * y_centered, axis=0) + 1e-8)

    if data_format == ChannelOrder.channels_last:
        y_std = jnp.mean(y_std, axis=(1, 2, 4))
        y_std = y_std.reshape((-1, 1, 1, num_new_features))
        y_std = jnp.tile(y_std, (group_size, H, W, 1))
    else:
        y_std = jnp.mean(y_std, axis=(2, 3, 4))
        y_std = y_std.reshape((-1, num_new_features, 1, 1))
        y_std = jnp.tile(y_std, (group_size, 1, H, W))

    return jnp.concatenate(
        (x, y_std), axis=3 if data_format == ChannelOrder.channels_last else 1
    )


class ResidualDiscriminatorBlock(hk.Module):
    """

    >>> module = _init(ResidualDiscriminatorBlock, in_features=4, out_features=8)
    >>> x = jnp.zeros((1, 64, 64, 4))
    >>> params = module.init(jax.random.PRNGKey(0), x)
    >>> y = module.apply(params, None, x)
    >>> y.shape
    (1, 32, 32, 8)

    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        activation_function: ActivationFunction = jnn.leaky_relu,
        data_format: ChannelOrder = ChannelOrder.channels_last,
        resample_kernel: jnp.ndarray = None,
        name: str = None,
    ):
        super().__init__(name=name)
        self.in_features = in_features
        self.out_features = out_features
        self.activation_function = activation_function
        self.data_format = data_format
        self.resample_kernel = (
            resample_kernel if resample_kernel is not None else jnp.array([1, 3, 3, 1])
        )

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        y = hk.Conv2D(
            self.in_features,
            kernel_shape=3,
            padding="SAME",
            data_format=self.data_format.name,
            name="conv0",
        )(x)
        y = self.activation_function(y)
        y = ConvDownsample2D(
            self.out_features,
            kernel_shape=3,
            resample_kernel=self.resample_kernel,
            downsample_factor=2,
            data_format=self.data_format,
        )(y)
        y = self.activation_function(y)

        residual = ConvDownsample2D(
            self.out_features,
            kernel_shape=1,
            resample_kernel=self.resample_kernel,
            downsample_factor=2,
        )(x)
        return (y + residual) / sqrt(2)


def _get_num_features(
    base_features: int, image_size: Tuple[int, int], max_hidded_feature_size: int
) -> List[int]:
    """
    Gets number of features for the blocks. Each block includes a downsampling
    step by a factor of two and at the end, we want the resolution to be
    down to 4x4 (for square images)

    >>> features = _get_num_features(64, (512, 512), 1024)
    >>> 512 // 2**(len(features) - 1)
    4
    >>> features[3]
    512
    """
    for size in image_size:
        assert (
            2 ** int(log2(size)) == size
        ), f"Image size must be a power of 2, got {image_size}"
    # determine the number of layers based on smaller side length
    shortest_side = min(*image_size)
    num_blocks = int(log2(shortest_side)) - 1
    num_features = (base_features * (2 ** i) for i in range(num_blocks))
    # we want to bring it down to 4x4 at the end of the last block
    return [min(n, max_hidded_feature_size) for n in num_features]


class ResidualDiscriminator(hk.Module):
    # pylint: disable=line-too-long
    """Residular discriminator architecture, see [1], Fig. 7c for details.

    Args:
        image_size (Union[int, Tuple[int, int]]): Size of the image. If
            only a single integer is passed, we assume square images.
            Each value must be a power of two.
        base_features(int): Number of features for the first convolutional layer.
            The i-th layer of the network is of size `base_features * 2**i`
        max_hidden_feature_size (int, optional): Maximum number of channels
            for intermediate convolutions. Defaults to 512.
        name (str, optional): Name of Haiku module. Defaults to None.

    >>> module = _init(ResidualDiscriminator, image_size=64, max_hidden_feature_size=16)
    >>> x = jnp.zeros((2, 64, 64, 3))
    >>> params = module.init(jax.random.PRNGKey(0), x)
    >>> y = module.apply(params, None, x)
    >>> y.shape
    (2, 1)

    TODO:
        - add Attention similar to https://github.com/lucidrains/stylegan2-pytorch/blob/54c79f430d0da3b02f570c6e1ef74d09190cd311/stylegan2_pytorch/stylegan2_pytorch.py#L557
        - add adaptive dropout: https://github.com/NVlabs/stylegan2-ada/blob/main/training/networks.py#L526
    """

    def __init__(
        self,
        image_size: Union[int, Tuple[int, int]],
        base_features: int = 32,
        max_hidden_feature_size: int = 512,
        activation_function: ActivationFunction = jnn.leaky_relu,
        mbstd_group_size: int = None,
        mbstd_num_features: int = 1,
        data_format: ChannelOrder = ChannelOrder.channels_last,
        name: str = "residual_discriminator",
    ):
        super().__init__(name=name)

        self.conv_args = {"data_format": data_format.name}
        size_t: Tuple[int, int] = image_size if isinstance(image_size, tuple) else (
            image_size,
            image_size,
        )
        self.base_features = base_features
        self.num_features = _get_num_features(
            2 * base_features, size_t, max_hidden_feature_size
        )
        self.activation_function = activation_function
        self.data_format = data_format
        self.stddev_layer = ft.partial(
            minibatch_stddev_layer,
            group_size=mbstd_group_size,
            num_new_features=mbstd_num_features,
            data_format=data_format,
        )

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:

        y = hk.Conv2D(self.base_features, kernel_shape=1, **self.conv_args)(x)
        y = self.activation_function(y)

        for idx, (n_in, n_out) in enumerate(
            zip(self.num_features[1:], self.num_features)
        ):
            y = ResidualDiscriminatorBlock(
                n_in,
                n_out,
                activation_function=self.activation_function,
                data_format=self.data_format,
                name=f"block_{idx}",
            )(y)

        # final block running on 4x4 feature maps
        if self.data_format == ChannelOrder.channels_last:
            assert min(y.shape[1:3]) == 4
        else:
            assert min(y.shape[2:4]) == 4

        y = self.stddev_layer(y)
        y = hk.Conv2D(
            self.num_features[-2], kernel_shape=3, padding="VALID", **self.conv_args
        )(y)
        y = self.activation_function(y)
        y = jnp.reshape(y, (y.shape[0], -1))
        y = hk.Linear(self.num_features[-1])(y)
        y = self.activation_function(y)

        # Prediction head
        y = hk.Linear(1, name="logits")(y)
        return y
