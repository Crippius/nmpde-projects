
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Spatial domain
x = np.linspace(0, 1, 200)
y = np.linspace(0, 1, 200)
X, Y = np.meshgrid(x, y)

# Parameters for spatial part
sigma = 0.1
x_centers = [(0.25, 0.25), (0.25, 0.5), (0.25, 0.75),
             (0.5, 0.25), (0.5, 0.5), (0.5, 0.75),
             (0.75, 0.25), (0.75, 0.5), (0.75, 0.75)]

# Create spatial part
spatial_part = np.zeros_like(X)
for xc, yc in x_centers:
    spatial_part += np.exp(-((X - xc)**2 + (Y - yc)**2) / sigma**2)

# Temporal part parameters
A = [1.0, 0.5, 0.25]
nu = [1, 3, 5]
phi = [0, 0, 0]

# Set up figure
fig, ax = plt.subplots(figsize=(6, 5))

def animate(frame):
    t = frame / 40  # time from 0 to 1.5 seconds
    temporal = sum(A_k * np.sin(2 * np.pi * nu_k * t + phi_k)
                   for A_k, nu_k, phi_k in zip(A, nu, phi))
    f_xt = temporal * spatial_part
    ax.clear()
    contour = ax.contourf(X, Y, f_xt, levels=40, cmap='viridis')
    ax.set_title(f"f(x, t) at t = {t:.2f}")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    return contour.collections

ani = animation.FuncAnimation(fig, animate, frames=60, interval=120, blit=False)

# Save as mp4
ani.save("f_xt_animation.mp4", writer="ffmpeg", fps=10)
print("Animation saved as f_xt_animation.mp4")
