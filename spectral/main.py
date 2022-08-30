from gpflow.experimental.check_shapes import check_shapes, check_shape as cs

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf


K = 20
BASE_FREQ = .2

k = tf.cast(tf.range(K)[:, None], dtype=tf.float32)

@check_shapes("x: [N, 1]", "return: [N, K]")
def Phi(x):
    return tf.cos(
        BASE_FREQ * tf.matmul(x, cs(k, "[K, 1]"), transpose_b=True)
    )


@check_shapes("weights: [K, 1]")
def get_function(weights):
    return lambda x: tf.matmul(Phi(x), weights)


if __name__ == "__main__":
    variances = tf.exp(-.3 * k)
    weights = variances**.5 * tf.random.normal((K, 1))
    func = get_function(weights)
    xx = tf.cast(tf.linspace(0, 20, 10_000)[:, None], dtype=tf.float32)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    print(xx.dtype)
    axes[0].plot(xx, func(xx))
    axes[0].set_xlabel("$x$")
    axes[0].set_ylabel("$f(x)$")
    axes[1].bar(range(len(weights)), weights.numpy().flatten())
    axes[1].set_xlabel("$k$")
    axes[1].set_ylabel("Spectrum")
    plt.show()
