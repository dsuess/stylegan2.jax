from typing import Callable, Optional

import jax.numpy as jnp

ActivationFunction = Callable[[jnp.ndarray], jnp.ndarray]
