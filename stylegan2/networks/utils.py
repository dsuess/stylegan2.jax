from enum import Enum
from typing import Any, Callable, Type

import haiku as hk
import jax
import jax.numpy as jnp


# pylint: disable=missing-class-docstring
class ChannelOrder(Enum):
    channels_last = 1
    channels_first = 2


def normalize(x: jnp.ndarray, epsilon: float = 1e-8) -> jnp.ndarray:
    """Normalizes x according to the formula:
    ..code::

       x_i = x_i / \sqrt(\sum_j x_j^2 + epsilon)

    """
    factor = jax.lax.rsqrt(jnp.sum(x * x) + epsilon)
    return x * factor


def _init(module: Type[hk.Module], *args: Any, **kwargs: Any) -> Callable:
    def run(*xs: Any) -> Any:
        instance = module(*args, **kwargs)
        return instance(*xs)

    wrapped: Callable = hk.transform(run)
    return wrapped


def _module_grad(module: hk.Module, *args: Any, **kwargs: Any):
    grad_f = jax.grad(lambda *args, **kwargs: jnp.sum(module.apply(*args)))
    return grad_f(*args, **kwargs)
