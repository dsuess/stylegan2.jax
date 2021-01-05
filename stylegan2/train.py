import functools as ft
from collections import namedtuple

import click
import haiku as hk
import jax
import jax.numpy as jnp
import optax
import tensorflow_datasets as tfds

from stylegan2 import networks as nx

GAN = namedtuple("GAN", "g, d, s")
Trainer = namedtuple("Trainer", "model, optim")


def setup_models():
    models = GAN(
        hk.transform(
            lambda latents: nx.SkipGenerator(32, max_hidden_feature_size=128)(latents)
        ),
        hk.without_apply_rng(
            hk.transform(
                lambda images: nx.ResidualDiscriminator(
                    32, max_hidden_feature_size=128
                )(images)
            )
        ),
        hk.without_apply_rng(
            hk.transform(
                lambda latents: nx.style_embedding_network(
                    final_embedding_size=128, intermediate_latent_size=128
                )(latents)
            )
        ),
    )

    optimizers = GAN(
        optax.sgd(0.01, momentum=0.9),
        optax.sgd(0.01, momentum=0.9),
        optax.sgd(0.01, momentum=0.9),
    )

    return Trainer(models, optimizers)


def initialize_params(rng, trainer, batch_size):
    rngs = GAN(*jax.random.split(rng, num=3))

    latents = jnp.zeros((batch_size, 32), dtype=jnp.float32)
    params_s = trainer.model.s.init(rngs.s, latents)

    styles = trainer.model.s.apply(params_s, latents)
    styles = jnp.tile(styles[:, None, :], (1, 8, 1))
    params_g = trainer.model.g.init(rngs.g, styles)

    images = trainer.model.g.apply(params_g, rngs.g, styles)
    params_d = trainer.model.d.init(rngs.d, images)

    model_state = GAN(params_g, params_d, params_s)
    optim_state = GAN(
        *[optim.init(params) for optim, params in zip(trainer.optim, model_state)]
    )

    return Trainer(model_state, optim_state)


def generator_loss(model, rng, model_state, images):
    latents = jax.random.normal(rng, (images.shape[0], 32))
    styles = model.s.apply(model_state.s, latents)
    styles = jnp.tile(styles[:, None], (1, 8, 1))

    fake_images = model.g.apply(model_state.g, rng, styles)
    logits = model.d.apply(model_state.d, fake_images)

    # Numerical stable implementation of sparse binary cross entropy
    loss = jnp.maximum(logits, 0) - logits + jnp.log(1 + jnp.exp(-jnp.abs(logits)))
    return jnp.mean(loss)


@ft.partial(jax.jit, static_argnums=[0])
def generator_step(trainer, state, rng, images):
    loss_fn = ft.partial(generator_loss, trainer.model, rng)
    val, grads = jax.value_and_grad(loss_fn)(state.model, images)
    update_g, opt_state_g = trainer.optim.g.update(grads.g, state.optim.g)
    state_g = optax.apply_updates(state.model.g, update_g)

    update_s, opt_state_s = trainer.optim.s.update(grads.s, state.optim.s)
    state_s = optax.apply_updates(state.model.s, update_s)

    model = GAN(state_g, state.model.d, state_s)
    optim = GAN(opt_state_g, state.optim.d, opt_state_s)
    return val, Trainer(model, optim)


def discriminator_loss(model, rng, model_state, images):
    latents = jax.random.normal(rng, (images.shape[0], 32))
    styles = model.s.apply(model_state.s, latents)
    styles = jnp.tile(styles[:, None], (1, 8, 1))

    fake_images = model.g.apply(model_state.g, rng, styles)
    logits = model.d.apply(model_state.d, fake_images)
    fake_loss = jnp.maximum(logits, 0) + jnp.log(1 + jnp.exp(-jnp.abs(logits)))
    fake_loss = jnp.mean(fake_loss)

    logits = model.d.apply(model_state.d, images)
    real_loss = jnp.maximum(logits, 0) - logits + jnp.log(1 + jnp.exp(-jnp.abs(logits)))
    real_loss = jnp.mean(real_loss)

    return fake_loss + real_loss


@ft.partial(jax.jit, static_argnums=[0])
def discriminator_step(trainer, state, rng, images):
    loss_fn = ft.partial(discriminator_loss, trainer.model, rng)
    val, grads = jax.value_and_grad(loss_fn)(state.model, images)
    update_d, opt_state_d = trainer.optim.g.update(grads.d, state.optim.d)
    state_d = optax.apply_updates(state.model.d, update_d)

    model = GAN(state.model.g, state_d, state.model.s)
    optim = GAN(state.optim.g, opt_state_d, state.optim.s)
    return val, Trainer(model, optim)


@click.command()
def train():
    data = tfds.load("cifar10", split="train")
    batch_size = 64
    data = (
        data.map(lambda x: x["image"] / 255)
        .repeat()
        .take(2 ** 14)
        .shuffle(1024)
        .batch(batch_size)
    )

    rngkey, rnginit = jax.random.split(jax.random.PRNGKey(42))
    trainer = setup_models()
    state = initialize_params(rnginit, trainer, batch_size)

    for epoch in range(10):
        for images in data.as_numpy_iterator():
            rngkey, rngdisc, rnggen = jax.random.split(rngkey, num=3)
            gen_loss, state = generator_step(trainer, state, rnggen, images)
            disc_loss, state = discriminator_step(trainer, state, rngdisc, images)

            print(f"gen_loss={gen_loss}, disc_loss={disc_loss}")


if __name__ == "__main__":
    train()
