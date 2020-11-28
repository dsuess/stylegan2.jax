import functools as ft
from typing import Tuple, Union

import haiku as hk
import jax
from jax import numpy as jnp

from .utils import ChannelOrder, _init  # pylint: disable=unused-import


def _apply_filter_2d(
    x: jnp.ndarray,
    conv_kernel_shape: Tuple[int, int],
    filter_kernel: jnp.ndarray,
    downsample_factor: int = 1,
    data_format: ChannelOrder = ChannelOrder.channels_last,
) -> jnp.ndarray:
    """
    >>> x = jnp.zeros((1, 64, 64, 2))
    >>> conv_kernel_shape = (3, 3)
    >>> kernel = jnp.zeros((4, 4))
    >>> _apply_filter_2d(x, conv_kernel_shape, kernel, downsample_factor=2).shape
    (1, 65, 65, 2)

    >>> x = jnp.zeros((1, 2, 64, 64))
    >>> conv_kernel_shape = (3, 3)
    >>> kernel = jnp.zeros((3, 3))
    >>> _apply_filter_2d(x, conv_kernel_shape, kernel, downsample_factor=2,
    ...                  data_format=ChannelOrder.channels_first).shape
    (1, 2, 65, 65)
    """
    # pylint: disable=invalid-name
    kh, kw = filter_kernel.shape
    ch, cw = conv_kernel_shape
    if data_format == ChannelOrder.channels_first:
        dimension_numbers = ("NCHW", "OIHW", "NCHW")
        filter_kernel = filter_kernel[None, None]
        # Insert single axis so we can use vmap
        x = x[:, :, None]
        vmap_over_axis = 1
    else:
        dimension_numbers = ("NHWC", "HWOI", "NHWC")
        filter_kernel = filter_kernel[:, :, None, None]
        x = x[..., None]
        vmap_over_axis = 3

    # See https://github.com/NVlabs/stylegan2-ada/blob/main/dnnlib/tflib/ops/upfirdn_2d.py#L362
    pad_l = (kw - downsample_factor + cw) // 2
    pad_r = (kw - downsample_factor + cw - 1) // 2
    pad_t = (kh - downsample_factor + ch) // 2
    pad_b = (kh - downsample_factor + ch - 1) // 2

    conv_func = ft.partial(
        jax.lax.conv_general_dilated,
        rhs=filter_kernel,
        window_strides=(1, 1),
        padding=[(pad_t, pad_b), (pad_l, pad_r)],
        dimension_numbers=dimension_numbers,
    )
    y = jax.vmap(conv_func, in_axes=vmap_over_axis, out_axes=vmap_over_axis)(x)
    return jnp.squeeze(y, axis=vmap_over_axis + 1)


class ConvDownsample2D(hk.Module):
    """This is the `_simple_upfirdn_2d` part of
    https://github.com/NVlabs/stylegan2-ada/blob/main/dnnlib/tflib/ops/upfirdn_2d.py#L313

    >>> module = _init(
    ...     ConvDownsample2D,
    ...     output_channels=8,
    ...     kernel_shape=3,
    ...     resample_kernel=jnp.array([1, 3, 3, 1]),
    ...     downsample_factor=2)
    >>> x = jax.numpy.zeros((1, 64, 64, 4))
    >>> params = module.init(jax.random.PRNGKey(0), x)
    >>> y = module.apply(params, None, x)
    >>> tuple(y.shape)
    (1, 32, 32, 8)
    """

    def __init__(
        self,
        output_channels: int,
        kernel_shape: Union[int, Tuple[int, int]],
        resample_kernel: jnp.array,
        downsample_factor: int = 1,
        gain: float = 1.0,
        data_format: ChannelOrder = ChannelOrder.channels_last,
        name: str = None,
    ):
        super().__init__(name=name)
        if resample_kernel.ndim == 1:
            resample_kernel = resample_kernel[:, None] * resample_kernel[None, :]
        elif 0 <= resample_kernel.ndim > 2:
            raise ValueError(
                f"Resample kernel has invalid shape {resample_kernel.shape}"
            )

        self.conv = hk.Conv2D(
            output_channels,
            kernel_shape=kernel_shape,
            stride=downsample_factor,
            padding="VALID",
            data_format=data_format.name,
        )
        self.resample_kernel = jnp.array(resample_kernel) * gain / resample_kernel.sum()
        self.downsample_factor = downsample_factor
        self.data_format = data_format

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        y = _apply_filter_2d(
            x,
            self.conv.kernel_shape,
            self.resample_kernel,
            downsample_factor=self.downsample_factor,
            data_format=self.data_format,
        )
        return self.conv(y)
