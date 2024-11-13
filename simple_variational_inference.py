import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.colors import Normalize
from matplotlib import cm
import pickle
import sys

# Load mu_history and Sigma_history from files (if saved using pickle)
with open('mu_history.pkl', 'rb') as f:
    mu_history = pickle.load(f)

with open('Sigma_history.pkl', 'rb') as f:
    Sigma_history = pickle.load(f)

num_iterations = len(mu_history)

# Convert mu_history and Sigma_history to numpy arrays
mu_history_array = np.array(mu_history).reshape(num_iterations, -1)
Sigma_history_array = np.array(Sigma_history)  # Shape: (num_iterations, 3, 3)

# Define the 3D Gaussian function
def gaussian_3d(x, y, z, mu, Sigma):
    pos = np.stack([x, y, z], axis=-1)  # Shape (..., 3)
    diff = pos - mu  # Shape (..., 3)
    inv_Sigma = np.linalg.inv(Sigma)
    exponent = np.einsum('...i,ij,...j', diff, inv_Sigma, diff)
    denominator = np.sqrt((2 * np.pi) ** 3 * np.linalg.det(Sigma))
    density = np.exp(-0.5 * exponent) / denominator
    return density

# Define the initial grid for plotting
grid_range = np.linspace(-10, 10, 20)
X0, Y0, Z0 = np.meshgrid(grid_range, grid_range, grid_range)
X0_flat = X0.flatten()
Y0_flat = Y0.flatten()
Z0_flat = Z0.flatten()

# Create the figure and 3D axis
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
cmap = cm.plasma
norm = Normalize(vmin=0, vmax=1)

def update_plot(frame):
    ax.cla()  # Clear the axis

    # Get mu and Sigma for the current frame
    mu = mu_history_array[frame][:3]  # Use first three components
    Sigma = Sigma_history_array[frame][:3, :3]

    # Calculate the density
    density = gaussian_3d(X0_flat, Y0_flat, Z0_flat, mu, Sigma)
    density /= np.max(density)  # Normalize density to [0, 1]

    # Map the normalized density values to colors including alpha
    colors = cmap(norm(density))
    colors[:, -1] = norm(density)  # Set alpha based on normalized density

    # Scatter plot with custom colors
    sc = ax.scatter(X0_flat, Y0_flat, Z0_flat, facecolors=colors, marker='o', edgecolors='none', s=20)

    # Set axis labels and title
    ax.set_title(f'3D Gaussian Distribution at Iteration {frame + 1}')
    ax.set_xlabel('w₁')
    ax.set_ylabel('w₂')
    ax.set_zlabel('b')

    # Set axis limits
    ax.set_xlim(grid_range.min(), grid_range.max())
    ax.set_ylim(grid_range.min(), grid_range.max())
    ax.set_zlim(grid_range.min(), grid_range.max())

    # Add colorbar (update or create)
    if hasattr(update_plot, 'colorbar'):
        update_plot.colorbar.remove()
    mappable = cm.ScalarMappable(norm=norm, cmap=cmap)
    mappable.set_array([])
    update_plot.colorbar = fig.colorbar(mappable, ax=ax, fraction=0.03, pad=0.1)
    update_plot.colorbar.set_label('Density')

# Create the animation
ani = FuncAnimation(fig, update_plot, frames=num_iterations, interval=500, repeat=False)

plt.show()
sys.exit()
# Create animation
ani_real = FuncAnimation(fig, update_real, frames=max_updates, blit=False, interval=200)

# Save the animation as a GIF
ani_real.save(r"C:\Users\pasca\OneDrive\2024\2024_10\simple_variational_inference\3d_gaussian_animation_real.gif", writer='pillow', dpi=150)

# Save animation as frames
gif_path_real = r"C:\Users\pasca\OneDrive\2024\2024_10\simple_variational_inference\3d_gaussian_animation_real.gif"
output_dir_real = r"C:\Users\pasca\OneDrive\2024\2024_10\simple_variational_inference\frames_real"

# Create output directory
os.makedirs(output_dir_real, exist_ok=True)

# Open the GIF file
with Image.open(gif_path_real) as gif:
    # Loop through each frame in the GIF
    for frame_number in range(gif.n_frames):
        # Seek to the frame in the GIF
        gif.seek(frame_number)

        # Save the frame as a PNG image in output directory (frames)
        frame_path = os.path.join(output_dir_real, f"frame_{frame_number:03d}.png")
        gif.save(frame_path, format="PNG")
        print(f"Saved {frame_path}")

# Display the animation
plt.show()

# Save updates to a DataFrame after all iterations
updates_df = pd.DataFrame({
    'Iteration': np.arange(len(mu_values)),
    'Mu': mu_values,
    'Sigma': sigma_values,
    'alpha': alpha_values
})

updates_df.to_csv("parameter_updates.csv", index=False)
print("Updates saved to 'parameter_updates.csv'")








