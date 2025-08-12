import numpy as np
import matplotlib.pyplot as plt
import numpy.linalg as nl
import MIMO
from matplotlib import cm

M = 32
N = 32

F_M = 1 / np.sqrt(M) * MIMO.DFTmat(M)
F_N = 1 / np.sqrt(N) * MIMO.DFTmat(N)
Ptx = np.identity(M)
delta_f = 15e3
T = 1 / delta_f

X_DD = np.zeros([M, N])
X_DD[1, 1] = 1

X_TF = nl.multi_dot([F_M, X_DD, MIMO.H(F_N)])
S = nl.multi_dot([Ptx, MIMO.H(F_M), X_TF])
s = S.flatten('F')

plt.figure()
ax = plt.axes(projection='3d')
x = np.arange(M)
y = np.arange(N)
xx, yy = np.meshgrid(x, y)
X, Y = xx.ravel(), yy.ravel()
top = X_DD.flatten('F')
bottom = np.zeros_like(top)
width = depth = 1
ax.bar3d(X, Y, bottom, width, depth, top, shade=True)

plt.figure()
ax = plt.axes(projection='3d')
X, Y = np.meshgrid(np.arange(0, M), np.arange(0, N))
ax.plot_surface(X, Y, np.real(X_TF), cmap=cm.hot)
plt.xlabel('Subcarrier')
plt.ylabel('Time')
plt.suptitle('Basis function in TF-domain (Real)')
ax.view_init(-140, 60)

plt.figure()
ax = plt.axes(projection='3d')
X, Y = np.meshgrid(np.arange(0, M), np.arange(0, N))
ax.plot_surface(X, Y, np.imag(X_TF), cmap=cm.hot)
plt.xlabel('Subcarrier')
plt.ylabel('Time')
plt.suptitle('Basis function in TF-domain (Imag)')
ax.view_init(-140, 60)

plt.figure()
plt.plot(np.arange(1, len(s) + 1) * T / M, np.real(s), 'g-')
plt.grid(1, which='both')
plt.suptitle('Basis function in time-domain (Real)')
plt.xlabel('Time')

plt.figure()
plt.plot(np.arange(1, len(s) + 1) * T / M, np.imag(s), 'g-')
plt.grid(1, which='both')
plt.suptitle('Basis function in time-domain (Imag)')
plt.xlabel('Time')



# --- Save figures ---
import matplotlib.pyplot as plt
plt.figure(1); plt.savefig('/mnt/data/basis_dd_bar.png', dpi=200)
plt.figure(2); plt.savefig('/mnt/data/basis_tf_real.png', dpi=200)
plt.figure(3); plt.savefig('/mnt/data/basis_tf_imag.png', dpi=200)
plt.figure(4); plt.savefig('/mnt/data/basis_time_real.png', dpi=200)
plt.figure(5); plt.savefig('/mnt/data/basis_time_imag.png', dpi=200)
