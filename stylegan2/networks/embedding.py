"""
[1] T. Karras, S. Laine, and T. Aila, “A Style-Based Generator Architecture
    for Generative Adversarial Networks,” arXiv:1812.04948 [cs, stat],
    Mar. 2019

"""

import haiku as hk
import jax  # pylint: disable=unused-import
import jax.numpy as jnp
from jax import nn as jnn

from stylegan2._typing import ActivationFunction

from .utils import _init, normalize  # pylint: disable=unused-import


class StyleEmbeddingNetwork(hk.Module):
    """Network transforming raw latents `z` into disentangled "style-latents"
    `w`. See Figure 1 b) (left) in [1].

    Args:
        final_embedding_size (int, optional): Size of the output latent `w`.
            Defaults to 512.
        intermediate_latent_size (int, optional): Size of the hidden layers.
            Defaults to 512.
        depth (int, optional): Number of embedding layers, must be at least
            1. Defaults to 8.
        normalize_latents (bool, optional): Whether to apply normalization
            to input latent. Defaults to `True`.
        activation_function (ActivationFunction, optional): Activation
            function of hidden layers. Defaults to `jnn.leaky_relu`.
        name (str, optional): Name of Haiku module. Defaults to None.

    Note:
        - the original implementation had a parameter `mapping_lrmul`, which
            scales the learning rate for this Module. We'll do this on the
            optimizer level instead of including this in the definition of
            the module.

    TODO:
        - Add support for labels as part of latent by concatening, see
            https://github.com/NVlabs/stylegan2/blob/master/training/networks_stylegan2.py#L280

    >>> module = _init(
    ...     StyleEmbeddingNetwork,
    ...     final_embedding_size=4,
    ...     intermediate_latent_size=2)
    >>> x = jnp.zeros(8)
    >>> params = module.init(jax.random.PRNGKey(0), x)
    >>> y = module.apply(params, None, x)
    >>> tuple(y.shape)
    (4,)
    """

    # pylint: disable=too-many-arguments
    def __init__(
        self,
        final_embedding_size: int = 512,
        intermediate_latent_size: int = 512,
        depth: int = 8,
        normalize_latents: bool = True,
        activation_function: ActivationFunction = jnn.leaky_relu,
        name: str = None,
    ):
        assert depth > 1
        super().__init__(name=name)

        self.normalize_latents = normalize_latents
        self.activation_function = activation_function
        self.layers = [
            hk.Linear(output_size=intermediate_latent_size) for _ in range(depth - 1)
        ]
        self.layers += [hk.Linear(output_size=final_embedding_size)]

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Args:
            x (jnp.ndarray): Input latent of size `latent_size`

        Returns:
            jnp.ndarray: Output latent of size `final_embedding_size`

        """
        if self.normalize_latents:
            x = normalize(x)

        for linear in self.layers:
            x = self.activation_function(linear(x))

        return x
