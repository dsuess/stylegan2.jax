import functools as ft
from typing import Tuple, Union

import haiku as hk
import jax
from jax import nn as jnn
from jax import numpy as jnp

from stylegan2._typing import ActivationFunction

from .discriminator import _get_num_features
from .layers import ModulatedConv2D, Upsample2D, UpsampleConv2D
from .utils import ChannelOrder, _init  # pylint: disable=unused-import


class SkipGenerator(hk.Module):
    # pylint: disable=line-too-long
    """Residular discriminator architecture, see [1], Fig. 7c for details.

    Args:
        image_size (Union[int, Tuple[int, int]]): Size of the image. If
            only a single integer is passed, we assume square images.
            Each value must be a power of two.
        base_features(int): Number of features for the last convolutional layer.
            The i-th layer of the network is of size `base_features * 2**i`
        max_hidden_feature_size (int, optional): Maximum number of channels
            for intermediate convolutions. Defaults to 512.
        name (str, optional): Name of Haiku module. Defaults to None.

    >>> module = _init(SkipGenerator, image_size=64, max_hidden_feature_size=16)
    >>> latents = jnp.zeros((1, 10, 512))
    >>> params = module.init(jax.random.PRNGKey(0), latents)
    >>> y = module.apply(params, jax.random.PRNGKey(0), latents)
    >>> y.shape
    (1, 64, 64, 3)
    """

    def __init__(
        self,
        image_size: Union[int, Tuple[int, int]],
        output_channels: int = 3,
        base_features: int = 32,
        max_hidden_feature_size: int = 512,
        activation_function: ActivationFunction = jnn.leaky_relu,
        data_format: ChannelOrder = ChannelOrder.channels_last,
        name: str = "skip_discriminator",
    ):
        super().__init__(name=name)

        self.conv_args = {"data_format": data_format.name}
        self.size_t: Tuple[int, int] = image_size if isinstance(
            image_size, tuple
        ) else (
            image_size,
            image_size,
        )
        self.base_features = base_features
        self.num_features = _get_num_features(
            2 * base_features, self.size_t, max_hidden_feature_size
        )
        self.output_channels = output_channels
        self.activation_function = activation_function
        self.data_format = data_format
        self.resample_kernel = jnp.array([1, 3, 3, 1])

    def get_initial_features(self) -> jnp.ndarray:
        """
        >>> func = hk.transform(lambda: SkipGenerator(
        ...     image_size=(64, 128),
        ...     max_hidden_feature_size=16).get_initial_features())
        >>> params = func.init(jax.random.PRNGKey(0))
        >>> x = func.apply(params, jax.random.PRNGKey(0))
        >>> tuple(x.shape)
        (4, 8, 16)
        """
        # setup the initial 4x4 block
        const_size = tuple(x // 2 ** (len(self.num_features) - 1) for x in self.size_t)
        num_features = self.num_features[-1]
        if self.data_format == ChannelOrder.channels_first:
            shape = (num_features, *const_size)
        else:
            shape = (*const_size, num_features)

        return hk.get_parameter("const", shape, init=hk.initializers.RandomNormal())

    def layer(
        self,
        x: jnp.ndarray,
        latents: jnp.ndarray,
        output_channels: int,
        key: jax.random.PRNGKey,
        upsample: bool = False,
    ) -> jnp.ndarray:
        if upsample:
            conv = UpsampleConv2D(
                output_channels=output_channels,
                kernel_shape=3,
                upsample_factor=2,
                resample_kernel=self.resample_kernel,
            )
        else:
            conv = ModulatedConv2D(
                output_channels=output_channels, kernel_shape=3, padding="SAME"
            )
        y = conv(x, latents)

        if self.data_format == ChannelOrder.channels_first:
            noise_shape = (y.shape[0], 1, y.shape[2], y.shape[3])
        else:
            noise_shape = (y.shape[0], y.shape[1], y.shape[2], 1)

        noise = jax.random.normal(key, shape=noise_shape, dtype=y.dtype)
        noise_strength = hk.get_parameter(
            "noise_strength", (1, 1, 1, 1), dtype=y.dtype, init=jnp.zeros
        )
        y += noise_strength * noise
        return self.activation_function(y)

    def to_rgb(
        self, x: jnp.ndarray, latents: jnp.ndarray, rgb: jnp.ndarray = None
    ) -> jnp.ndarray:
        conv = ModulatedConv2D(
            output_channels=self.output_channels, kernel_shape=1, demodulate=False
        )
        y = self.activation_function(conv(x, latents))

        if rgb is not None:
            upsample = Upsample2D(
                upsample_factor=2, resample_kernel=self.resample_kernel
            )
            rgb = upsample(rgb)
            y = y + rgb
        return y

    def __call__(self, latents: jnp.ndarray) -> jnp.ndarray:
        const = self.get_initial_features()
        # TODO Express using vmap
        const = jnp.tile(const[None], reps=(latents.shape[0], 1, 1, 1))
        key = hk.next_rng_key()
        # 1st axis ... which block latent is used in
        # 2nd axis ... whether latent is used in layer or in ToRGB
        latents = jnp.reshape(
            latents, (latents.shape[0], 2, len(self.num_features), latents.shape[2])
        )
        y = self.layer(
            const, latents[:, 0, 0], output_channels=self.num_features[-1], key=key
        )
        rgb = self.to_rgb(y, latents[:, 0, 1])

        for layer_idx, num_features in list(enumerate(self.num_features))[1:]:
            conv_up = UpsampleConv2D(
                num_features,
                kernel_shape=3,
                upsample_factor=2,
                resample_kernel=self.resample_kernel,
            )
            y = conv_up(y, latents[:, layer_idx - 1, 1])
            conv = ModulatedConv2D(num_features, kernel_shape=3)
            y = conv(y, latents[:, layer_idx, 0])
            rgb = self.to_rgb(y, latents[:, layer_idx, 1], rgb=rgb)

        return rgb
