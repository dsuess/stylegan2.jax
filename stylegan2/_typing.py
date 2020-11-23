from typing import Callable

import jax.numpy as jnp

ActivationFunction = Callable[[jnp.ndarray], jnp.ndarray]
