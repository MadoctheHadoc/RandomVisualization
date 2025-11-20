import numpy as np
import matplotlib.pyplot as plt

# --- Perlin Noise Sampler (returns noise at a specific x,y) ---
class Perlin2D:
    def __init__(self, scale=40, width=300, height=300):
        self.scale = scale
        gx = width // scale + 2
        gy = height // scale + 2

        # random gradient grid
        angles = np.random.rand(gx, gy) * 2 * np.pi
        self.gradients = np.dstack((np.cos(angles), np.sin(angles)))

    def fade(self, t):
        return 6*t**5 - 15*t**4 + 10*t**3

    def dot_grid_gradient(self, ix, iy, x, y):
        grad = self.gradients[ix, iy]
        dx = x - ix
        dy = y - iy
        return dx * grad[...,0] + dy * grad[...,1]

    def sample(self, x, y):
        # scale down to grid coords
        x /= self.scale
        y /= self.scale
        
        x0 = int(np.floor(x))
        y0 = int(np.floor(y))
        x1 = x0 + 1
        y1 = y0 + 1

        sx = self.fade(x - x0)
        sy = self.fade(y - y0)

        n00 = self.dot_grid_gradient(x0, y0, x, y)
        n10 = self.dot_grid_gradient(x1, y0, x, y)
        n01 = self.dot_grid_gradient(x0, y1, x, y)
        n11 = self.dot_grid_gradient(x1, y1, x, y)

        ix0 = n00 * (1 - sx) + n10 * sx
        ix1 = n01 * (1 - sx) + n11 * sx

        v = ix0 * (1 - sy) + ix1 * sy
        return v


def manipulate_noise(x, y):
    # --- warp settings ---
    warp_frequency = 0.3      # how fast the warp changes
    warp_strength = 50        # how much the warp distorts coordinates

    # Sample warp vector
    wx = sampler.sample(x * warp_frequency, y * warp_frequency)
    wy = sampler.sample(x * warp_frequency + 100, y * warp_frequency + 100)

    # Distorted coordinates
    xw = x + wx * warp_strength
    yw = y + wy * warp_strength

    # Main noise sample (distorted)
    noise = sampler.sample(xw, yw)

    t = 0.2

    return noise



# Create sampler
w = h = 300
sampler = Perlin2D(scale=40, width=w, height=h)

# Sample pixels individually
img = np.zeros((h, w))
for y in range(h):
    for x in range(w):
        manipulated = manipulate_noise(x, y)
        img[y, x] = manipulated

# Normalize image
img = (img - img.min()) / (img.max() - img.min())

plt.figure()
plt.title("Perlin Noise Sampled Point-by-Point")
plt.imshow(img, cmap='gray')
plt.axis('off')
plt.show()
