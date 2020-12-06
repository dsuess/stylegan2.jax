import functools as ft
from typing import Optional, Sequence, Tuple, Union

import haiku as hk
import jax
import numpy as np
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


def mod_demod_conv(
    inputs: jnp.ndarray,
    styles: jnp.ndarray,
    orig_weight: jnp.ndarray,
    channel_index: int,
    demodulate: bool = True,
    **kwargs,
):
    assert styles.ndim == 1
    num_spatial = orig_weight.ndim - 2
    if channel_index == -1:
        new_shape = (1,) * num_spatial + (styles.size, 1)
        # Compute normalization over all axes except for output-channel
        reduce_axes = tuple(range(num_spatial + 1))
    else:
        new_shape = (styles.size, 1) + (1,) * num_spatial
        reduce_axes = (0,) + tuple(range(2, 2 + num_spatial))

    # Apply styles over input-channel dimension of weights
    weight = orig_weight * styles.reshape(new_shape)

    if demodulate:
        norm = jax.lax.square(weight).sum(axis=reduce_axes, keepdims=True)
        weight = weight * jax.lax.rsqrt(norm + 1e-8)

    inputs = jnp.expand_dims(inputs, axis=0)
    (result,) = jax.lax.conv_general_dilated(inputs, weight, **kwargs)
    return result


class ModulatedConv(hk.ConvND):
    def __init__(self, *args, demodulate: bool = True, **kwargs):

        super().__init__(*args, **kwargs)
        self.demodulate = demodulate

    def __call__(self, inputs: jnp.ndarray, latents: jnp.ndarray) -> jnp.ndarray:
        """Connects ``ConvND`` layer.
        Args:
        inputs: An array of shape ``[spatial_dims, C]`` and rank-N+1 if unbatched,
            or an array of shape ``[N, spatial_dims, C]`` and rank-N+2 if batched.
        Returns:
        An array of shape ``[spatial_dims, output_channels]`` and rank-N+1 if
            unbatched, or an array of shape ``[N, spatial_dims, output_channels]``
            and rank-N+2 if batched.
        """
        assert self.mask is None
        unbatched_rank = self.num_spatial_dims + 1
        allowed_ranks = [unbatched_rank, unbatched_rank + 1]
        if inputs.ndim not in allowed_ranks:
            raise ValueError(
                f"Input to ConvND needs to have rank in {allowed_ranks},"
                f" but input has shape {inputs.shape}."
            )

        unbatched = inputs.ndim == unbatched_rank
        if unbatched:
            inputs = jnp.expand_dims(inputs, axis=0)
            latents = jnp.expand_dims(latents, axis=0)
        assert latents.ndim == 2

        w_shape = self.kernel_shape + (
            inputs.shape[self.channel_index],
            self.output_channels,
        )

        w_init = self.w_init
        if w_init is None:
            fan_in_shape = np.prod(w_shape[:-1])
            stddev = 1.0 / np.sqrt(fan_in_shape)
            w_init = hk.initializers.TruncatedNormal(stddev=stddev)
        w = hk.get_parameter("w", w_shape, inputs.dtype, init=w_init)

        conv_fn = ft.partial(
            mod_demod_conv,
            orig_weight=w,
            channel_index=self.channel_index,
            demodulate=self.demodulate,
            window_strides=self.stride,
            padding=self.padding,
            lhs_dilation=self.lhs_dilation,
            rhs_dilation=self.kernel_dilation,
            dimension_numbers=self.dimension_numbers,
        )
        # Modulate; +1 do have default bias be == 1
        styles = hk.Linear(inputs.shape[self.channel_index])(latents) + 1
        out = jax.vmap(conv_fn)(inputs, styles)

        if self.with_bias:
            if self.channel_index == -1:
                bias_shape = (self.output_channels,)
            else:
                bias_shape = (self.output_channels,) + (1,) * self.num_spatial_dims
            b = hk.get_parameter("b", bias_shape, inputs.dtype, init=self.b_init)
            b = jnp.broadcast_to(b, out.shape)
            out = out + b

        if unbatched:
            out = jnp.squeeze(out, axis=0)
        return out


class ModulatedConv2D(ModulatedConv):
    """
    >>> module = _init(
    ...     ModulatedConv2D,
    ...     output_channels=8,
    ...     kernel_shape=3,
    ...     padding="SAME")
    >>> x = jax.numpy.zeros((3, 16, 16, 4))
    >>> latents = jax.numpy.zeros((3, 16))
    >>> params = module.init(jax.random.PRNGKey(0), x, latents)
    >>> y = module.apply(params, None, x, latents)
    >>> tuple(y.shape)
    (3, 16, 16, 8)

    Args:
        ModulatedConv ([type]): [description]
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, num_spatial_dims=2)


def mod_demod_conv_transpose(
    inputs: jnp.ndarray,
    styles: jnp.ndarray,
    orig_weight: jnp.ndarray,
    channel_index: int,
    demodulate: bool = True,
    **kwargs,
):
    assert styles.ndim == 1
    num_spatial = orig_weight.ndim - 2
    if channel_index == -1:
        new_shape = (1,) * num_spatial + (1, styles.size)
        # Compute normalization over all axes except for output-channel
        reduce_axes = tuple(range(num_spatial)) + (-1,)
    else:
        new_shape = (1, styles.size) + (1,) * num_spatial
        reduce_axes = tuple(range(1, 2 + num_spatial))

    # Apply styles over input-channel dimension of weights
    weight = orig_weight * styles.reshape(new_shape)

    if demodulate:
        norm = jax.lax.square(weight).sum(axis=reduce_axes, keepdims=True)
        weight = weight * jax.lax.rsqrt(norm + 1e-8)

    inputs = jnp.expand_dims(inputs, axis=0)
    (result,) = jax.lax.conv_transpose(inputs, weight, **kwargs)
    return result


class ModulatedConvTranpose(hk.ConvNDTranspose):
    def __init__(self, *args, demodulate=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.demodulate = demodulate

    def __call__(self, inputs: jnp.ndarray, latents: jnp.ndarray) -> jnp.ndarray:
        """Computes the transposed convolution of the input.
        Args:
        inputs: An array of shape ``[spatial_dims, C]`` and rank-N+1 if unbatched,
            or an array of shape ``[N, spatial_dims, C]`` and rank-N+2 if batched.
        Returns:
        An array of shape ``[spatial_dims, output_channels]`` and rank-N+1 if
            unbatched, or an array of shape ``[N, spatial_dims, output_channels]``
            and rank-N+2 if batched.
        """
        assert self.mask is None
        unbatched_rank = self.num_spatial_dims + 1
        allowed_ranks = [unbatched_rank, unbatched_rank + 1]
        if inputs.ndim not in allowed_ranks:
            raise ValueError(
                f"Input to ConvNDTranspose needs to have rank in "
                f"{allowed_ranks}, but input has shape {inputs.shape}."
            )

        unbatched = inputs.ndim == unbatched_rank
        if unbatched:
            inputs = jnp.expand_dims(inputs, axis=0)
            latents = jnp.expand_dims(latents, axis=0)
        assert latents.ndim == 2

        input_channels = inputs.shape[self.channel_index]
        w_shape = self.kernel_shape + (self.output_channels, input_channels)

        w_init = self.w_init
        if w_init is None:
            fan_in_shape = self.kernel_shape + (input_channels,)
            stddev = 1.0 / np.sqrt(np.prod(fan_in_shape))
            w_init = hk.initializers.TruncatedNormal(stddev=stddev)
        w = hk.get_parameter("w", w_shape, inputs.dtype, init=w_init)

        conv_fn = ft.partial(
            mod_demod_conv_transpose,
            orig_weight=w,
            channel_index=self.channel_index,
            demodulate=self.demodulate,
            strides=self.stride,
            padding=self.padding,
            dimension_numbers=self.dimension_numbers,
        )
        # Modulate; +1 do have default bias be == 1
        styles = hk.Linear(inputs.shape[self.channel_index])(latents) + 1
        out = jax.vmap(conv_fn)(inputs, styles)

        if self.with_bias:
            if self.channel_index == -1:
                bias_shape = (self.output_channels,)
            else:
                bias_shape = (self.output_channels,) + (1,) * self.num_spatial_dims
            b = hk.get_parameter("b", bias_shape, inputs.dtype, init=self.b_init)
            b = jnp.broadcast_to(b, out.shape)
            out = out + b

        if unbatched:
            out = jnp.squeeze(out, axis=0)
        return out


class ModulatedConvTranspose2D(ModulatedConvTranpose):
    """
    >>> module = _init(
    ...     ModulatedConvTranspose2D,
    ...     output_channels=8,
    ...     kernel_shape=3,
    ...     padding="SAME")
    >>> x = jax.numpy.zeros((3, 16, 16, 4))
    >>> latents = jax.numpy.zeros((3, 16))
    >>> params = module.init(jax.random.PRNGKey(0), x, latents)
    >>> y = module.apply(params, None, x, latents)
    >>> tuple(y.shape)
    (3, 16, 16, 8)

    Args:
        ModulatedConv ([type]): [description]
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, num_spatial_dims=2)
