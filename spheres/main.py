# %%
import numpy as np
import gpflow
import tensorflow as tf
from gpflow.kernels import ArcCosine
import plotly.graph_objs as go

from utils import plotly_plot_spherical_function, plot_spherical_function

def _J(order, theta):
    """
    Implements the order dependent family of functions defined in equations
    4 to 7 in the reference paper.
    """
    if order == 0:
        return np.pi - theta
    elif order == 1:
        return tf.sin(theta) + (np.pi - theta) * tf.cos(theta)
    else:
        assert order == 2
        return 3.0 * tf.sin(theta) * tf.cos(theta) + (np.pi - theta) * (
            1.0 + 2.0 * tf.cos(theta) ** 2
        )

zenit = tf.reshape(tf.constant([1., 0., 0.], dtype=tf.float32), (1, 3))  # [1, 3]

def f(X):
    # X: [N, 3]
    X = tf.cast(X, dtype=zenit.dtype)
    cos_theta = tf.matmul(X, zenit, transpose_b=True)  # [N, 1]
    return tf.nn.relu(cos_theta)
    # jitter = 1e-15
    # theta = tf.acos(jitter + (1 - 2 * jitter) * cos_theta)
    # return _J(1, theta)


fig = plotly_plot_spherical_function(lambda X: f(X).numpy())

resolution = 100
x = np.linspace(-1, 1, resolution)
y = np.linspace(-1, 1, resolution)
x, y = np.meshgrid(x, y)
b = 1.05
z = b * np.ones_like(x)

def plane(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    X = tf.convert_to_tensor(X / n, dtype=zenit.dtype)
    # cos_theta = tf.matmul(X, zenit, transpose_b=True)  # [N, 1]
    return n * f(X)


grid = np.vstack([x.flatten(), y.flatten(), z.flatten()]).T
print(grid.shape)
fgrid = plane(grid).numpy().reshape(resolution, resolution)
fcolors = fgrid
C0 = "rgb(31, 119, 180)"
C1 = "rgb(255, 127, 14)"
colorscale = [[0.0, "black"], [0.5, C0], [1, C1]]
plot = go.Surface(
    contours = {
        # "x": {"show": True, "start": -1, "end": 1, "size": 0.04, "color":"red"},
        # "y": {"show": True, "start": -1, "end": 1, "size": 0.05}
    },
    x=x, y=y, z=z, surfacecolor=fcolors, colorscale=colorscale, opacity=.5)

fig.add_trace(plot)
fig.update_traces(showscale=False)



x = [0., 1.05, 0., 0.5]
y = [0.,  0., 0, .5]
z = [0.,  0., 0.0, 1.05]

pairs = [(0,1),(2, 3)]

## plot ONLY the first ball in each pair of balls
trace1 = go.Scatter3d(
    x=[x[p[0]] for p in pairs],
    y=[y[p[0]] for p in pairs],
    z=[z[p[0]] for p in pairs],
    mode='markers',
    name='markers',
    line=dict(color='black')
)

x_lines = list()
y_lines = list()
z_lines = list()

for p in pairs:
    for i in range(2):
        x_lines.append(x[p[i]])
        y_lines.append(y[p[i]])
        z_lines.append(z[p[i]])
    x_lines.append(None)
    y_lines.append(None)
    z_lines.append(None)

## set the mode to lines to plot only the lines and not the balls/markers
trace2 = go.Scatter3d(
    x=x_lines,
    y=y_lines,
    z=z_lines,
    mode='lines',
    line = dict(width = 2, color = 'rgb(0, 0,0)')
)

fig.add_trace(trace2)

# fig = go.Figure(data=[trace1, trace2])
# fig.add_scatter3d(data=trace2)

arrow_tip_ratio = 0.1
arrow_starting_ratio = 0.98

## the cone will point in the direction of vector field u, v, w 
## so we take this to be the difference between each pair 

## then hack the colorscale to force it to display the same color
## by setting the starting and ending colors to be the same

for p in pairs:
    fig.add_trace(go.Cone(
        x=[x[p[0]] + arrow_starting_ratio*(x[p[1]] - x[p[0]])],
        y=[y[p[0]] + arrow_starting_ratio*(y[p[1]] - y[p[0]])],
        z=[z[p[0]] + arrow_starting_ratio*(z[p[1]] - z[p[0]])],
        u=[arrow_tip_ratio*(x[p[1]] - x[p[0]])],
        v=[arrow_tip_ratio*(y[p[1]] - y[p[0]])],
        w=[arrow_tip_ratio*(z[p[1]] - z[p[0]])],
        showlegend=False,
        showscale=False,
        colorscale=[[0, 'rgb(0,0,0)'], [1, 'rgb(0,0,0)']]
        ))

fig.update_traces(showlegend=False)
fig.update(layout_showlegend=False)
fig.update_coloraxes(showscale=False)



fig
# %%

import matplotlib.pyplot as plt

xx = tf.linspace(-1, 1, 100)

plt.plot(xx[::-1], _J(1, tf.acos(xx[::-1])) / np.pi )
plt.xlabel(r"$x^\top x'$")
# %%