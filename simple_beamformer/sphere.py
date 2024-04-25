from numpy import pi, cos, sin, arccos, arange
import numpy as np
import mpl_toolkits.mplot3d
import matplotlib.pyplot as plt


def create_sphere(num_pts, factor=1.1, plot=False):
    tot_num_pts = factor * num_pts
    indices = arange(0, tot_num_pts, dtype=float)  # + 0.5

    #     phi = arccos(1 - 2*indices/num_pts)[:num_pts//2]
    phi = arccos(1 - factor * indices / tot_num_pts)  # [:num_pts//2]
    theta = (pi) * (1 + 5**0.5) * indices  # [:num_pts//2]

    x = cos(theta) * sin(phi)
    y = sin(theta) * sin(phi)
    z = cos(phi)
    if plot:
        n = num_pts
        fig = plt.figure()
        ax = plt.axes(projection="3d")
        ax.scatter(x[:n], y[:n], z[:n])
        ax.axis("equal")
        plt.show()
    return {"phi": theta, "theta": phi, "x": x, "y": y, "z": z}


def transform_flat(theta, phi):
    r = phi
    x = r * cos(theta)
    y = r * sin(theta)
    z = cos(phi)

    fig = go.Figure(
        data=[go.Mesh3d(z=(z), x=(x), y=(y), intensity=z, colorscale="Viridis")]
    )  # , opacity=0.5, color='rgba(244,22,100,0.6)')])
    fig.update_layout(title="Mt Bruno Elevation", autosize=False)
    fig.show()


#     fig = plt.figure()
#     ax = fig.add_subplot(projection='polar')
#     ax.scatter(theta, r)
#     plt.show()

if __name__ == "__main__":
    data = create_sphere(3000, factor=2, plot=True)
    phi = data["phi"]
    theta = data["theta"]
    transform_flat(theta, phi)
