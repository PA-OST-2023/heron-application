import numpy as np
from numpy import sin, cos, arcsin, arccos, pi, exp, arctan2
import matplotlib.pyplot as plt
from math import degrees, radians

from scipy.spatial.transform import Rotation as R


def init_array(gamma, phi_0, r_0, r_p, d_r, l_o):
    pass


def convert_to_base_system(M, phi=+radians(22.5), r=0.095):
    rot = R.from_rotvec(phi * np.array([0, 0, 1]))
    M_t = M + np.array([r, 0, -(0.01185 - 0.0056)])
    M_tilde = rot.apply(M_t)
    return M_tilde
    pass


def calculate_fancy_arm(gamma, z_o):
    m_0 = np.array([0.101157, 0.039018, z_o])
    m_1 = np.array([0.145209, 0.099498, z_o])
    m_2 = np.array([0.17107, 0.177783, z_o])
    m_3 = np.array([0.173701, 0.268701, z_o])
    M = np.vstack((m_0, m_1, m_2, m_3))
    r = R.from_rotvec(gamma * np.array([0, 1, 0]))
    M_tilde = r.apply(M)
    M_tilde = convert_to_base_system(M_tilde)
    return populate_array(M_tilde)
    pass


def populate_array(M_tilde):
    Mall = np.empty((32, 3), np.float32)
    fig, ax = plt.subplots()
    for i in range(8):
        r = R.from_rotvec(i * 2 * pi / 8 * np.array([0, 0, 1]))
        m_t = r.apply(M_tilde)
        Mall[4 * i : 4 * i + 4, :] = m_t
        ax.scatter(m_t[:, 0], m_t[:, 1])
    ax.axis("equal")
    ax.grid()
    return Mall


def make_circ_array(n_mic, r, plot=False, offset=0):
    phi = (np.arange(n_mic)) * 2 * pi / n_mic + offset
    R = np.ones_like(phi) * r
    if plot:
        fig, ax = plt.subplots(subplot_kw={"projection": "polar"}, layout="constrained")
        ax.scatter(phi, R)
        ax.set_title("Circular Array")
        ax.grid(True)
    return phi, R


def make_fancy_circ_array(n_mic, n_circ, shift, r_0, r_1, plot=False):
    circ_m = n_mic // n_circ
    dr = (r_1 - r_0) / (n_circ - 1)
    phi = np.ones(n_mic)
    r = np.zeros(n_mic)
    for i in range(n_circ):
        lb = i * circ_m
        ub = (i + 1) * circ_m
        phi[lb:ub], r[lb:ub] = make_circ_array(circ_m, r_0 + i * dr, offset=i * shift)
    if plot:
        fig, ax = plt.subplots(subplot_kw={"projection": "polar"}, layout="constrained")
        ax.scatter(phi, r)
        ax.set_title("Circular Array")
        ax.grid(True)
        plt.show()
    return phi, r


if __name__ == "__main__":
    import csv

    with open("./gt/FancyArray_TestGeometry.csv", newline="") as csvfile:
        # Create a DictReader object which interprets the first row as column titles
        csv_reader = csv.DictReader(csvfile)
        angles = []
        positions = []
        for row in csv_reader:
            angles.append(float(row["Angle"]))
            pos_tmp = []
            for i in range(4):
                pos_tmp.append(
                    np.array(
                        [
                            -float(row[f"Mic {i} Z"]),
                            -float(row[f"Mic {i} X"]),
                            float(row[f"Mic {i} Y"]),
                        ]
                    )
                )
            positions.append(np.array(pos_tmp))

    for angle, position in zip(angles, positions):
        mics = calculate_fancy_arm(radians(angle), 0.01185 - 0.0016)
        #         mics = calculate_fancy_arm(radians(angle),0.01185)
        mic_c = mics[24:28] * 100
        print(mic_c)
        print(position)
        print("-" * 20)
        plt.show()
