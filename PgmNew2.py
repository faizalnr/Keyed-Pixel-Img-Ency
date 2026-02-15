# =====================================================
# code authored by Faizal Nujumudeen
# Presidency University, Bengaluru
# =====================================================

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import math

# =========================
# Configuration
# =========================
IMAGE_PATH = "Giza.jpg"   # change this
OUTPUT_DIR = "output6"
KEY = 123456

os.makedirs(OUTPUT_DIR, exist_ok=True)

# =========================
# Utility Functions
# =========================
def entropy(img):
    hist = np.histogram(img.flatten(), 256, (0,256))[0]
    prob = hist / np.sum(hist)
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
    return np.mean(np.abs(img1.astype(int) - img2.astype(int))) / 255 * 100

# =========================
# Key-dependent orbit generation
# =========================
def generate_global_orbits(h, w, key, orbit_len=64):
    np.random.seed(key)
    coords = [(i, j) for i in range(h) for j in range(w)]
    np.random.shuffle(coords)

    orbits = []
    for i in range(0, len(coords), orbit_len):
        orbit = coords[i:i+orbit_len]
        if len(orbit) > 1:
            np.random.shuffle(orbit)  # random traversal
            orbits.append(orbit)
    return orbits

# =========================
# Encryption
# =========================
def encrypt_strong(img, key):
    h, w = img.shape[:2]
    ch = img.shape[2] if img.ndim == 3 else 1

    orbits = generate_global_orbits(h, w, key)
    permuted = img.copy()

    # Orbit rotation
    for orbit in orbits:
        vals = [img[x,y].copy() for x,y in orbit]
        s = key % len(vals)
        vals = vals[s:] + vals[:s]
        for (x,y), v in zip(orbit, vals):
            permuted[x,y] = v

    # 2D + channel diffusion
    encrypted = permuted.copy()
    for i in range(h):
        for j in range(w):
            for c in range(ch):
                left = encrypted[i, j-1, c] if j > 0 else 0
                up   = encrypted[i-1, j, c] if i > 0 else 0
                encrypted[i,j,c] ^= left ^ up

    # Channel coupling
    if ch == 3:
        encrypted[:,:,0] ^= encrypted[:,:,1]
        encrypted[:,:,1] ^= encrypted[:,:,2]
        encrypted[:,:,2] ^= encrypted[:,:,0]

    return permuted, encrypted, orbits

# =========================
# Decryption
# =========================
def decrypt_strong(enc, orbits, key):
    h, w = enc.shape[:2]
    ch = enc.shape[2] if enc.ndim == 3 else 1
    dec = enc.copy()

    # Reverse channel coupling
    if ch == 3:
        dec[:,:,2] ^= dec[:,:,0]
        dec[:,:,1] ^= dec[:,:,2]
        dec[:,:,0] ^= dec[:,:,1]

    # Reverse diffusion
    for i in reversed(range(h)):
        for j in reversed(range(w)):
            for c in range(ch):
                left = dec[i, j-1, c] if j > 0 else 0
                up   = dec[i-1, j, c] if i > 0 else 0
                dec[i,j,c] ^= left ^ up

    # Reverse orbit rotation
    for orbit in orbits:
        vals = [dec[x,y].copy() for x,y in orbit]
        s = key % len(vals)
        vals = vals[-s:] + vals[:-s]
        for (x,y), v in zip(orbit, vals):
            dec[x,y] = v

    return dec

# =========================
# Main Execution
# =========================
img = cv2.imread(IMAGE_PATH)
if img is None:
    raise ValueError("Image not found")

img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
gray = img[:,:,0]
h,w = img.shape[:2]
permuted, encrypted, orbits = encrypt_strong(img, KEY)
decrypted = decrypt_strong(encrypted, orbits, KEY)

# Save images
cv2.imwrite(f"{OUTPUT_DIR}/plain.png", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
cv2.imwrite(f"{OUTPUT_DIR}/encrypted.png", cv2.cvtColor(encrypted, cv2.COLOR_RGB2BGR))
cv2.imwrite(f"{OUTPUT_DIR}/decrypted.png", cv2.cvtColor(decrypted, cv2.COLOR_RGB2BGR))

# =========================
# Metrics
# =========================
metrics = {
    "Entropy Plain": entropy(gray),
    "Entropy Encrypted": entropy(encrypted[:,:,0]),
    "Entropy Decrypted": entropy(decrypted[:,:,0]),
    "Correlation Plain": correlation(gray),
    "Correlation Encrypted": correlation(encrypted[:,:,0]),
    "Correlation Decrypted": correlation(decrypted[:,:,0]),
    "MSE (Plain-Encrypted)": mse(gray, encrypted[:,:,0]),
    "MSE (Plain-Decrypted)": mse(gray, decrypted[:,:,0]),
    "PSNR (Plain-Encrypted)": psnr(gray, encrypted[:,:,0]),
    "PSNR (Plain-Decrypted)": psnr(gray, decrypted[:,:,0]),
    "NPCR (%)": npcr(gray, encrypted[:,:,0]),
    "UACI (%)": uaci(gray, encrypted[:,:,0])
}

with open(f"{OUTPUT_DIR}/metrics.txt", "w") as f:
    for k,v in metrics.items():
        f.write(f"{k}: {v:.4f}\n")

# =========================
# Histogram Comparison (Normalized)
# =========================
plt.figure()
for data, label in zip([gray, encrypted[:,:,0], decrypted[:,:,0]],
                       ["Plain", "Encrypted", "Decrypted"]):
    hist = np.histogram(data.flatten(), 256, (0,256))[0]
    plt.plot(hist / hist.sum(), label=label)

plt.title("Normalized Histogram Comparison")
plt.legend()
plt.savefig(f"{OUTPUT_DIR}/histogram_comparison.png")
plt.close()

# =========================
# Correlation Comparison
# =========================
plt.figure(figsize=(12,4))
images = [gray, encrypted[:,:,0], decrypted[:,:,0]]
titles = ["Plain", "Encrypted", "Decrypted"]

for i,(img_c,title) in enumerate(zip(images,titles)):
    plt.subplot(1,3,i+1)
    plt.scatter(img_c[:,:-1], img_c[:,1:], s=1)
    plt.title(title)
    plt.axis("off")

plt.suptitle("Horizontal Pixel Correlation Comparison")
plt.savefig(f"{OUTPUT_DIR}/correlation_comparison.png")
plt.close()

# =========================
# Metric Bar Plot (Normalized)
# =========================
labels = ["Entropy", "Correlation"]
plain_vals = [metrics["Entropy Plain"]/8, abs(metrics["Correlation Plain"])]
enc_vals = [metrics["Entropy Encrypted"]/8, abs(metrics["Correlation Encrypted"])]
dec_vals = [metrics["Entropy Decrypted"]/8, abs(metrics["Correlation Decrypted"])]

x = np.arange(len(labels))
plt.figure()
plt.bar(x-0.2, plain_vals, 0.2, label="Plain")
plt.bar(x, enc_vals, 0.2, label="Encrypted")
plt.bar(x+0.2, dec_vals, 0.2, label="Decrypted")
plt.xticks(x, labels)
plt.legend()
plt.title("Normalized Metric Comparison")
plt.savefig(f"{OUTPUT_DIR}/metric_comparison.png")
plt.close()

print("Execution completed. Comparison results saved.")

# =====================================================
# "If you want to shine like a sun, first burn like a sun" - Dr. APJ Abdul Kalam.
# Success is a continuous process
# =====================================================