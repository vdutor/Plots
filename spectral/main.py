from gpflow.experimental.check_shapes import check_shapes, check_shape as cs

import gpflow
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf


plt.rcParams["font.size"] = 12
plt.rcParams["mathtext.fontset"] = "cm"  # Use CM for math font.
plt.rcParams["figure.autolayout"] = True  # Use tight layouts.


def tweak(
    grid=True,
    legend=None,
    legend_loc="upper right",
    spines=True,
    ticks=True,
    ax=None,
):
    """Tweak a plot.
    Args:
        grid (bool, optional): Show grid. Defaults to `True`.
        legend (bool, optional): Show legend. Automatically shows a legend if any labels
            are set.
        legend_loc (str, optional): Position of the legend. Defaults to
            "upper right".
        spines (bool, optional): Hide top and right spine. Defaults to `True`.
        ticks (bool, optional): Hide top and right ticks. Defaults to `True`.
        ax (axis, optional): Axis to tune. Defaults to `plt.gca()`.
    """
    if ax is None:
        ax = plt.gca()

    if grid:
        ax.set_axisbelow(True)  # Show grid lines below other elements.
        ax.grid(which="major", c="#c0c0c0", alpha=0.5, lw=1)

    if legend is None:
        legend = len(ax.get_legend_handles_labels()[0]) > 0

    if legend:
        leg = ax.legend(
            facecolor="#eeeeee",
            framealpha=0.7,
            loc=legend_loc,
            labelspacing=0.25,
        )
        leg.get_frame().set_linewidth(0)

    if spines:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["bottom"].set_lw(1)
        ax.spines["left"].set_lw(1)

    if ticks:
        ax.xaxis.set_ticks_position("bottom")
        ax.xaxis.set_tick_params(width=1)
        ax.yaxis.set_ticks_position("left")
        ax.yaxis.set_tick_params(width=1)



@check_shapes("x: [N, 1]", "return: [N, K]")
def Phi(x):
    return tf.cos(
        BASE_FREQ * tf.matmul(x, cs(k, "[K, 1]"), transpose_b=True)
    )


@check_shapes("x: [N, 1]", "return: [N, K]")
def PhiExp(x):
    return tf.exp(
        -(x - tf.transpose(k))**2
    )


@check_shapes("weights: [K, S]")
def get_function(weights):
    return lambda x: tf.matmul(Phi(x), weights)


def plot_function_and_spectra():
    global k
    k = tf.cast(k, dtype=tf.float32)
    variances = tf.exp(-.3 * k)
    num_samples = 4
    xx = tf.cast(tf.linspace(0, 20, 10_000)[:, None], dtype=tf.float32)
    all_weights = variances**.5 * tf.random.normal((K, num_samples))
    y_min = np.min(get_function(all_weights)(xx)) - .1
    y_max = np.max(get_function(all_weights)(xx)) + .1
    print(y_min, y_max)

    for p in range(1, 5):
        fig_funcs, ax_funcs = plt.subplots(1, 1, figsize=(3, 3))
        fig_spectra, ax_spectra = plt.subplots(1, 1, figsize=(3, 3))

        weights = all_weights[:, :p]
        func = get_function(weights)
        ax_funcs.plot(xx, func(xx))
        ax_funcs.set_xlabel("$x$")
        ax_funcs.set_ylabel("$f(x)$")
        ax_funcs.set_ylim(y_min, y_max)
        tweak(ax=ax_funcs)
        fig_funcs.savefig(f"function_{p}.{EXT}", dpi=1000)

        width = .8 / p
        for i in range(p):
            ax_spectra.bar(np.arange(len(weights)) - (i-p//2) * width, weights[:, i].numpy(), width=width)

        ax_spectra.set_xlabel("$i$")
        ax_spectra.set_ylabel(r"$\xi_i$")
        ax_spectra.set_ylim(-2, 2)
        tweak(ax=ax_spectra)
        fig_spectra.savefig(f"spectrum_{p}.{EXT}", dpi=1000)
        ax_spectra.plot(np.arange(len(weights)),variances, "k-")
        fig_spectra.savefig(f"spectrum_variances_{p}.{EXT}", dpi=1000)

        plt.show()



class DegKernel(gpflow.kernels.Kernel):
    def __init__(self, variances):
        self.variances = variances[:, 0]
        self.phi = PhiExp
        super().__init__()

    @check_shapes(
        "X: [batch..., N, D]",
        "X2: [batch2..., N2, D]",
        "return: [batch..., N, batch2..., N2] if X2 is not None",
        "return: [batch..., N, N] if X2 is None",
    )
    def K(self, X, X2 = None) -> tf.Tensor:
        Phi_x = self.phi(X)
        if X2 is not None:
            Phi_x2 = self.phi(X2)
        else:
            Phi_x2 = Phi_x

        return tf.einsum("i,ni,mi->nm", self.variances, Phi_x, Phi_x2)

    @check_shapes(
        "X: [batch..., N, D]",
        "return: [batch..., N]",
    )
    def K_diag(self, X) -> tf.Tensor:
        return tf.einsum("i,ni->n", self.variances, self.phi(X) ** 2)


def plot_mean_conf(x, mean, var, ax, color='C0'):
    ax.plot(x, mean, color, lw=2)
    ax.fill_between(
        x[:, 0],
        mean[:, 0] - 1.96 * np.sqrt(var[:, 0]),
        mean[:, 0] + 1.96 * np.sqrt(var[:, 0]),
        color=color,
        alpha=0.2,
    )

def plot_model(m, ax, x=np.linspace(0, 1, 101)[:, None], plot_data=True, color='C0'):
    if plot_data:
        X, Y = m.data
        ax.plot(X, Y, "kx", mew=1.)
    
    mean, var = m.predict_f(x)
    plot_mean_conf(x, mean, var, ax, color)


if __name__ == "__main__":
    EXT = "png"
    # kernel = DegKernel()
    N = 10
    K = 25
    BASE_FREQ = .2
    VAR = 1e-5
    k = tf.cast(tf.range(K)[:, None], dtype=gpflow.default_float()) / K * 20.

    # Also change the basis functions in get_functions
    k = tf.cast(tf.range(K)[:, None], dtype=gpflow.default_float())
    plot_function_and_spectra()
    exit(0)

    variances = tf.exp(-.3 * k * 0.0)

    xx = tf.cast(tf.linspace(0, 20, 10_000)[:, None], dtype=gpflow.default_float())
    weights = variances**.5 * tf.random.normal((K, 1), dtype=gpflow.default_float())
    f = get_function(weights)
    X = tf.random.uniform((10, 1), dtype=gpflow.default_float()) * 20
    Y = f(X) + tf.random.normal((N, 1), dtype=gpflow.default_float()) * (VAR ** .5)

    gpr = gpflow.models.GPR((X, Y), DegKernel(variances), noise_variance=VAR)

    fig, ax = plt.subplots(figsize=(4, 3))
    xx=np.linspace(-10, 30, 1001)[:, None]
    ax.plot(xx, PhiExp(xx)[:, ::4], "C0", alpha=.6)
    ax.set_xlabel("$x$")
    ax.set_ylabel(r"$\phi(x)$")
    tweak(ax)
    plt.savefig(f"basisfunctions.{EXT}", dpi=1_000)


    fig, ax = plt.subplots(figsize=(4, 3))
    tweak(ax)
    plot_model(gpr, ax, x=np.linspace(-10, 30, 1001)[:, None], plot_data=True, color='C0')
    ax.set_xlabel("$x$")
    plt.savefig(f"degenerate_kernel.{EXT}", dpi=1_000)
    plt.show()
    # plot_function_and_spectra()

