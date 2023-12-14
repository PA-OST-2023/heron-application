import numpy as np
from numpy import sin, cos, arcsin, arccos, pi, exp, arctan2
import matplotlib.pyplot as plt
from math import degrees, radians

from scipy.spatial.transform import Rotation as R


def init_array(gamma, phi_0, r_0, r_p, d_r, l_o):

    pass

# def convert_to_base_system(M, phi=-11.25/180 * pi, r=0.095):
def convert_to_base_system(M, phi=+22.5/180 * pi, r=0.095):
    rot = R.from_rotvec(phi * np.array([0,0,1]))
    M_t = M + np.array([r, 0, 0])
    M_tilde = rot.apply(M_t)
    return M_tilde
    pass

def calculate_fancy_arm(r_arm, phi_arm, gamma, z_o):
    m_0 = np.array([0.101157, 0.039018, z_o])
    m_1 = np.array([0.145209, 0.099498, z_o])
    m_2 = np.array([0.17107, 0.177783, z_o])
    m_3 = np.array([0.173701, 0.268701, z_o])
    M = np.vstack((m_0, m_1, m_2, m_3))
    r = R.from_rotvec(gamma * np.array([0,1,0]))
    M_tilde = r.apply(M)
    print(M)
    print(M_tilde)
    M_tilde = convert_to_base_system(M_tilde)
    return populate_array(M_tilde)
    pass

def populate_array(M_tilde):
    Mall = np.empty((32,3), np.float32)
    fig, ax = plt.subplots()
    for i in range(8):
        r = R.from_rotvec(i * 2*pi/8 * np.array([0,0,1]))
        m_t = r.apply(M_tilde)
        Mall[4*i: 4*i + 4, :] = m_t
        ax.scatter(m_t[:,0], m_t[:,1])
#     ax.scatter(Mall[:,0], Mall[:,1])
    ax.axis('equal')
    ax.grid()
    return Mall

if __name__ == "__main__":
    angles = []

    calculate_fancy_arm(0,0,0,0.0016)
    calculate_fancy_arm(0,0,radians(15),0.0016)
    calculate_fancy_arm(0,0,radians(30),0.0016)
    calculate_fancy_arm(0,0,radians(45),0.0016)
    calculate_fancy_arm(0,0,pi/2,0.0016)
    plt.show()

