# =====================================================
# KPOE for Grayscale Images with Full Analysis
# =====================================================

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import math
import hashlib

# =========================
# Configuration
# =========================
IMAGE_PATH = "Xray.jpg"
OUTPUT_DIR = "output_gray"
SECRET_KEY = "KPOE_2025_SECURE"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# =========================
# Utility Functions
# =========================
def derive_seed(key):
    digest = hashlib.sha256(key.encode()).digest()
    return int.from_bytes(digest[:8], 'big')

def entropy(img):
    hist = np.histogram(img.flatten(), bins=256, range=(0,256))[0]
    prob = hist / hist.sum()
    prob = prob[prob > 0]
    return -np.sum(prob * np.log2(prob))

def correlation(img):
    x = img[:, :-1].flatten()
    y = img[:, 1:].flatten()
    return np.corrcoef(x, y)[0, 1]

def mse(img1, img2):
    return np.mean((img1.astype(np.float64) - img2.astype(np.float64))**2)

def psnr(img1, img2):
    m = mse(img1, img2)
    if m == 0:
        return float("inf")
    return 10 * math.log10((255**2) / m)

def npcr(img1, img2):
    return np.sum(img1 != img2) / img1.size * 100

def uaci(img1, img2):
    return np.mean(np.abs(img1.astype(np.int16) - img2.astype(np.int16))) / 255 * 100

# =========================
# Orbit Generation
# =========================
def generate_global_orbits(h, w, seed, orbit_len=64):
    rng = np.random.default_rng(seed)
    coords = [(i, j) for i in range(h) for j in range(w)]
    rng.shuffle(coords)

    orbits = []
    for i in range(0, len(coords), orbit_len):
        orbit = coords[i:i + orbit_len]
        if len(orbit) > 1:
            orbits.append(orbit)

    return orbits

# =========================
# Encryption
# =========================
def encrypt_gray(img, key):
    h, w = img.shape
    seed = derive_seed(key)

    orbits = generate_global_orbits(h, w, seed)
    permuted = img.copy()
    rng = np.random.default_rng(seed)

    shifts = []

    # Orbit permutation
    for orbit in orbits:
        vals = [img[x, y] for x, y in orbit]
        shift = rng.integers(1, len(vals))
        shifts.append(shift)

        vals = vals[shift:] + vals[:shift]

        for (x, y), v in zip(orbit, vals):
            permuted[x, y] = v

    # Forward diffusion
    forward = permuted.copy()

    for i in range(h):
        for j in range(w):
            left = forward[i, j - 1] if j > 0 else 0
            up = forward[i - 1, j] if i > 0 else 0
            forward[i, j] ^= left ^ up

    # Reverse diffusion
    encrypted = forward.copy()

    for i in reversed(range(h)):
        for j in reversed(range(w)):
            right = encrypted[i, j + 1] if j < w - 1 else 0
            down = encrypted[i + 1, j] if i < h - 1 else 0
            encrypted[i, j] ^= right ^ down

    return permuted, encrypted, orbits, shifts

# =========================
# Decryption
# =========================
def decrypt_gray(enc, orbits, shifts):
    h, w = enc.shape

    # Undo reverse diffusion
    reverse_removed = enc.copy()

    for i in range(h):
        for j in range(w):
            right = reverse_removed[i, j + 1] if j < w - 1 else 0
            down = reverse_removed[i + 1, j] if i < h - 1 else 0
            reverse_removed[i, j] ^= right ^ down

    # Undo forward diffusion
    diff_removed = reverse_removed.copy()

    for i in reversed(range(h)):
        for j in reversed(range(w)):
            left = diff_removed[i, j - 1] if j > 0 else 0
            up = diff_removed[i - 1, j] if i > 0 else 0
            diff_removed[i, j] ^= left ^ up

    decrypted = diff_removed.copy()

    # Reverse orbit permutation
    for orbit, shift in zip(reversed(orbits), reversed(shifts)):
        vals = [decrypted[x, y] for x, y in orbit]
        vals = vals[-shift:] + vals[:-shift]

        for (x, y), v in zip(orbit, vals):
            decrypted[x, y] = v

    return decrypted

# =========================
# Main
# =========================
img = cv2.imread(IMAGE_PATH, cv2.IMREAD_GRAYSCALE)

if img is None:
    raise ValueError("Image not found")

permuted, encrypted, orbits, shifts = encrypt_gray(img, SECRET_KEY)
decrypted = decrypt_gray(encrypted, orbits, shifts)

# Save images
cv2.imwrite(f"{OUTPUT_DIR}/plain.png", img)
cv2.imwrite(f"{OUTPUT_DIR}/encrypted.png", encrypted)
cv2.imwrite(f"{OUTPUT_DIR}/decrypted.png", decrypted)

# =========================
# Metrics
# =========================
metrics = {
    "Entropy Plain": entropy(img),
    "Entropy Encrypted": entropy(encrypted),
    "Entropy Decrypted": entropy(decrypted),
    "Correlation Plain": correlation(img),
    "Correlation Encrypted": correlation(encrypted),
    "Correlation Decrypted": correlation(decrypted),
    "MSE Plain-Encrypted": mse(img, encrypted),
    "MSE Plain-Decrypted": mse(img, decrypted),
    "PSNR Plain-Encrypted": psnr(img, encrypted),
    "PSNR Plain-Decrypted": psnr(img, decrypted),
    "NPCR (%)": npcr(img, encrypted),
    "UACI (%)": uaci(img, encrypted)
}

with open(f"{OUTPUT_DIR}/metrics.txt", "w") as f:
    for k, v in metrics.items():
        f.write(f"{k}: {v:.4f}\n")

# =========================
# Figure 1: Image Comparison
# =========================
plt.figure(figsize=(12,4))

imgs = [img, encrypted, decrypted]
titles = ["Plain", "Encrypted", "Decrypted"]

for i, (im, title) in enumerate(zip(imgs, titles)):
    plt.subplot(1,3,i+1)
    plt.imshow(im, cmap='gray')
    plt.title(title)
    plt.axis('off')

plt.suptitle("Image Comparison")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/image_comparison.png")
plt.close()

# =========================
# Figure 2: Histogram Comparison
# =========================
plt.figure(figsize=(8,5))

for data, label in zip([img, encrypted, decrypted],
                       ["Plain", "Encrypted", "Decrypted"]):
    hist = np.histogram(data.flatten(), 256, (0,256))[0]
    plt.plot(hist / hist.sum(), label=label)

plt.title("Normalized Histogram Comparison")
plt.xlabel("Pixel Intensity")
plt.ylabel("Probability")
plt.legend()
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/histogram_comparison.png")
plt.close()

# =========================
# Figure 3: Correlation Comparison
# =========================
plt.figure(figsize=(12,4))

for i, (im, title) in enumerate(zip(imgs, titles)):
    plt.subplot(1,3,i+1)
    plt.scatter(im[:,:-1], im[:,1:], s=1)
    plt.title(title)
    plt.axis('off')

plt.suptitle("Horizontal Pixel Correlation Comparison")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/correlation_comparison.png")
plt.close()

# =========================
# Figure 4: Metric Comparison
# =========================
labels = ["Entropy", "Correlation"]
plain_vals = [metrics["Entropy Plain"]/8, abs(metrics["Correlation Plain"])]
enc_vals = [metrics["Entropy Encrypted"]/8, abs(metrics["Correlation Encrypted"])]
dec_vals = [metrics["Entropy Decrypted"]/8, abs(metrics["Correlation Decrypted"])]

x = np.arange(len(labels))

plt.figure(figsize=(8,5))
plt.bar(x-0.2, plain_vals, 0.2, label="Plain")
plt.bar(x, enc_vals, 0.2, label="Encrypted")
plt.bar(x+0.2, dec_vals, 0.2, label="Decrypted")

plt.xticks(x, labels)
plt.ylabel("Normalized Value")
plt.title("Normalized Metric Comparison")
plt.legend()
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/metric_comparison.png")
plt.close()

# =========================
# Figure 5: NPCR/UACI
# =========================
plt.figure(figsize=(6,5))

vals = [metrics["NPCR (%)"], metrics["UACI (%)"]]
labels = ["NPCR", "UACI"]

plt.bar(labels, vals)
plt.ylabel("Percentage")
plt.title("Differential Attack Metrics")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/npcr_uaci.png")
plt.close()

# =========================
# Console Output
# =========================
for k, v in metrics.items():
    print(f"{k}: {v:.4f}")

print("Execution complete. All plots saved.")