import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.spatial import KDTree
from scipy import interpolate

def approach_Kevin():
    fig,ax = plt.subplots()
    x = np.random.rand(1)
    y = np.random.rand(1)
    r = np.random.rand(1)

    theta = np.random.rand(1)*365

    x_m = np.arange(-2,2,0.25)
    y_m = np.arange(-2,2,0.25)
    X,Y = np.meshgrid(x_m, y_m)
    P = [X.flatten(), Y.flatten()] 
    P_vector = np.zeros((len(X.flatten()), 2))

    m = 0
    for xk in range(len(X)):
        for yk in range(len(Y)):
         theta = np.random.rand(1)*365
         #plt.quiver(X[xk, yk], Y[xk, yk], r*math.cos(np.deg2rad(theta)), r*math.sin(np.deg2rad(theta)), color = 'b')
         P_vector[m][0] = X[xk, yk] + r*math.sin(np.deg2rad(theta))
         P_vector[m][1] = Y[xk, yk] + r*math.sin(np.deg2rad(theta))
         m+=1


    xi = 1
    yi = 1
    inc = 0
    theta = np.random.rand(1)*365
    idx_closest = []

    val_X_closest = []
    val_Y_closest = []

    theta = 120

    plt.quiver(xi, yi, r*math.cos(np.deg2rad(theta)), r*math.sin(np.deg2rad(theta)), angles = 'xy')

    for i in np.arange(0,2,0.1):
        x_vd = i*math.cos(np.deg2rad(theta) - (np.pi/2))
        y_vd = i*math.sin(np.deg2rad(theta) - (np.pi/2))
        x_vu = i*math.cos(np.deg2rad(theta) + (np.pi/2))
        y_vu = i*math.sin(np.deg2rad(theta) + (np.pi/2))
        plt.quiver(xi, yi, x_vd, y_vd, color = 'r', angles = 'xy')
        plt.quiver(xi, yi, x_vu, y_vu, color = 'g', angles = 'xy')
        plt.scatter(xi + x_vd, yi+y_vd, color = 'r')
        plt.scatter(xi + x_vu, yi+y_vu, color = 'g')

        kdt = KDTree(np.array(P).T)
        distd, kd = kdt.query(np.array((xi+x_vd, yi +y_vd)))
        plt.scatter(P[0][kd], P[1][kd])
        distu, ku = kdt.query(np.array((xi+x_vu, yi +y_vu)))
        plt.scatter(P[0][ku], P[1][ku])

        idx_closest.append(kd)
        idx_closest.append(ku)
        val_X_closest.append(P[0][kd])
        val_X_closest.append(P[0][ku])
        val_Y_closest.append(P[1][kd])
        val_Y_closest.append(P[1][ku])


    plt.scatter(val_X_closest, val_Y_closest)
    ax.set_aspect('equal')
    plt.show()


def approach_internet(u, v, xi, yi, span, field):
    """Generate a meshgrid and rotate it by RotRad radians."""


    field = field[xi-span:xi+span, yi-span:yi+span]
    RotRad = np.arctan(-v/u)
    # Clockwise, 2D rotation matrix
    RotMatrix = np.array([[np.cos(RotRad),  np.sin(RotRad)],
                          [-np.sin(RotRad), np.cos(RotRad)]])

    x, y = np.meshgrid(range(-span, span), range(-span, span))
    x_new, y_new =  np.einsum('ji, mni -> jmn', RotMatrix, np.dstack([x, y]))

    x_back, y_back = np.meshgrid(range(-span, span), 0)

    field_rotated = interpolate.griddata((x_new.flatten(), y_new.flatten()), field.flatten(), (x_back, y_back), method = 'linear')
    return x_back, y_back, field_rotated
