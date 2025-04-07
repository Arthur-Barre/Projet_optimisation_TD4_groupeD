import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import convolve

img_jpg = "robot_no_noise.jpg"
img_bruitee_jpg = "robot_noise.jpg"
img = plt.imread(img_jpg)
img_bruitee = plt.imread(img_bruitee_jpg)

def gradient(img):
    grad_x = np.zeros_like(img)
    grad_y = np.zeros_like(img)

    grad_x[:-1, :] = img[1:, :] - img[:-1, :]
    grad_x[-1, :] = 0

    grad_y[:, :-1] = img[:, 1:] - img[:, :-1]
    grad_y[:, -1] = 0

    return grad_x, grad_y

def divergence(grad_x, grad_y):
    div = np.zeros_like(grad_x)

    div[1:, :] += grad_x[1:, :] - grad_x[:-1, :]
    div[0, :] += grad_x[0, :]  
    div[-1, :] -= grad_x[-1, :] 

    div[:, 1:] += grad_y[:, 1:] - grad_y[:, :-1]
    div[:, 0] += grad_y[:, 0] 
    div[:, -1] -= grad_y[:, -1]

    return div

def optim_gradient_fixed_step(grad_fun, x0, l, max_iter = 100000, epsilon_grad_fun = 1e-8):
    k = 0
    xk = x0
    grad_f_xk = grad_fun(xk)
    while ((k<max_iter) and (np.linalg.norm(grad_f_xk)>epsilon_grad_fun)):
        pk = -grad_f_xk
        xk = xk + l*pk
        grad_f_xk = grad_fun(xk)
        k = k + 1
    print("Nombre d'iterations : ", k)
    return xk


grad_x, grad_y = gradient(img)
laplacien = divergence(grad_x, grad_y)

plt.figure(figsize=(6, 6))
plt.imshow(laplacien, cmap="gray")
plt.colorbar()
plt.title("Laplacien de l'image non bruitée du robot")
plt.axis("off")
plt.show()

def fonction_objectif(u, u_bruitee, lmbda):
    """
    Calcule la valeur de la fonction objectif :
    1/2 * ||u - u_bruitee||^2 + lambda * ||∇u||^2
    """
    grad_x, grad_y = gradient(u)
    norm_grad_u = np.sum(grad_x**2 + grad_y**2)
    return 0.5 * np.sum((u - u_bruitee)**2) + lmbda * norm_grad_u

def gradient_objectif(u, u_bruitee, lmbda):
    """
    Calcule le gradient de la fonction objectif.
    """
    grad_x, grad_y = gradient(u)
    div_grad = divergence(grad_x, grad_y)
    return u - u_bruitee - 2 * lmbda * div_grad

# Paramètres
lmbda = 0.2  # Poids du terme de régularisation
pas = 0.1    # Pas de descente
max_iter = 1000
epsilon = 1e-6

# Initialisation
u0 = img_bruitee  # On initialise avec l'image bruitée

# Optimisation
u_optim = optim_gradient_fixed_step(
    grad_fun=lambda u: gradient_objectif(u, img_bruitee, lmbda),
    x0=u0,
    l=pas,
    max_iter=max_iter,
    epsilon_grad_fun=epsilon
)

# Affichage de l'image restaurée
plt.figure(figsize=(6, 6))
plt.imshow(u_optim, cmap="gray")
plt.colorbar()
plt.title("Image restaurée après descente de gradient")
plt.axis("off")
plt.show()