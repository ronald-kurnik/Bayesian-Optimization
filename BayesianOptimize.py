# --------------------------------------------------------------
# Bayesian Optimization – scikit-optimize
# --------------------------------------------------------------
# pip install scikit-optimize

from skopt import gp_minimize
from skopt.plots import plot_convergence
import matplotlib.pyplot as plt
import numpy as np

# --- Black-box objective (MUST return scalar) -------------------------
def objective(x):
    x = float(x[0])  # ← extract scalar from list/tuple
    return (x**2 - 5)**2 * np.sin(x) + 0.1 * np.random.randn()

# Search space
space = [(-5.0, 10.0)]

# Run Bayesian Optimization
res = gp_minimize(
    objective,
    space,
    n_calls=30,
    random_state=0,
    noise="gaussian",   # tells GP we have noisy observations
    acq_func="EI",      # Expected Improvement
)

print(f"Best x = {res.x[0]:.3f}, f(x) = {res.fun:.3f}")
plot_convergence(res)
plt.title("Convergence Plot")
plt.show()

xv = np.linspace(-5,10,100)
yv = (xv**2 - 5)**2 * np.sin(xv)

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(xv, yv, color='blue')
ax.minorticks_on()
ax.grid(which='major',
        color='black',
        linestyle='-',
        linewidth=0.8,
        alpha=0.6)
ax.grid(which='minor',
        color='gray',
        linestyle=':',
        linewidth=0.5,
        alpha=0.6)
ax.set_title('Objective Fuction to Minimize')
ax.set_xlabel('x')
ax.set_ylabel('f(x)')
min_x = round(res.x[0],3)
min_y = round(res.fun,3)
text_to_add = (
    f"Minimum found at x = {min_x:.3f}, f(x) = {min_y:.3f}"
)
ax.text(0.5, -1500, text_to_add, 
        fontsize=10, 
        color='red', 
        bbox=dict(facecolor='white', alpha=0.5, edgecolor='red', boxstyle='round,pad=0.5'))
plt.show()