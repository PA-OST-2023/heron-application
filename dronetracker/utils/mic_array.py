import numpy as np
from numpy import sin, cos, arcsin, arccos, pi, exp, arctan2
import matplotlib.pyplot as plt

from scipy.spatial.transform import Rotation as R


def init_array(gamma, phi_0, r_0, r_p, d_r, l_o):

    pass

def convert_to_base_system(M, phi=-11.25/180 * pi, r=0.095):
    rot = R.from_rotvec(phi * np.array([0,0,1]))
    M_t = M + np.array([r, 0, 0])
#     M_t = M + np.array([sin(pi/2+phi)*r, cos(pi/2+phi) * r, 0])
    M_tilde = rot.apply(M_t)
#     fig, ax = plt.subplots()
#     ax.scatter(M_tilde[:,0], M_tilde[:,1])
#     ax.set_ylim((-0.5,0.5))
#     ax.set_xlim((-0.5,0.5))
#     ax.axis('equal')
#     ax.grid()
#     plt.show()
    return M_tilde
    pass

def calculate_fancy_arm(r_arm, phi_arm, gamma, z_o):
    m_0 = np.array([0.10116, 0.03902, z_o])
    m_1 = np.array([0.14521, 0.0995, z_o])
    m_2 = np.array([0.17107, 0.17778, z_o])
    m_3 = np.array([0.1737, 0.2687, z_o])
    M = np.vstack((m_0, m_1, m_2, m_3))
    r = R.from_rotvec(gamma * np.array([0,1,0]))
    M_tilde = r.apply(M)
    print(M)
    print(M_tilde)
    M_tilde = convert_to_base_system(M_tilde)
    populate_array(M_tilde)
    pass

def populate_array(M_tilde):
    Mall = np.empty((32,3), np.float32)
    for i in range(8):
        r = R.from_rotvec(i * 2*pi/8 * np.array([0,0,1]))
        Mall[4*i: 4*i + 4, :] = r.apply(M_tilde)
    fig, ax = plt.subplots()
    ax.scatter(Mall[:,0], Mall[:,1])
    ax.axis('equal')
    ax.grid()
    import ipdb; ipdb.set_trace()

#     plt.show()


calculate_fancy_arm(0,0,0,0.0016)
calculate_fancy_arm(0,0,pi/8,0.0016)
calculate_fancy_arm(0,0,pi/4,0.0016)
calculate_fancy_arm(0,0,pi/2,0.0016)
plt.show()

