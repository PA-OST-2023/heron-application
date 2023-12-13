import numpy as np
from numpy import sin, cos, arcsin, arccos, pi, exp

from scipy.spatial.transform import Rotation as R


def init_array(gamma, phi_0, r_0, r_p, d_r, l_o):

    pass

def calculate_fancy_arm(r_arm, phi_arm, gamma, z_o):
    m_0 = np.array([0.10116, 0.03902, z_o])
    m_1 = np.array([0.14521, 0.0995, z_o])
    m_2 = np.array([0.17107, 0.17778, z_o])
    m_3 = np.array([0.1737, 0.2687, z_o])
    M = np.vstack((m_0, m_1, m_2, m_3))
    r = R.from_rotvec(gamma * np.array([0,1,0]))
    M_tilde = r.apply(M)
    import ipdb; ipdb.set_trace()
    pass

calculate_fancy_arm(0,0,pi/2,0.0016)
