# =====================================================
# code authored by Faizal Nujumudeen
# Presidency University, Bengaluru
# =====================================================

import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# =========================
# Configuration
# =========================
DATASET_ROOT = "dataset_subset -small"
PLAIN_DIR = os.path.join(DATASET_ROOT, "plain")
ENC_DIR   = os.path.join(DATASET_ROOT, "encrypted")
DEC_DIR   = os.path.join(DATASET_ROOT, "decrypted")

OUTPUT_PLOTS = "dataset_subset -small/dataset_figures"
os.makedirs(OUTPUT_PLOTS, exist_ok=True)

# =========================
# Metric Functions
# =========================
def entropy(img):
    hist,_ = np.histogram(img.flatten(), 256, (0,256))
    p = hist / np.sum(hist)
    p = p[p > 0]
    return -np.sum(p * np.log2(p))

def correlation(img):
    x = img[:, :-1].flatten()
    y = img[:, 1:].flatten()
    return np.corrcoef(x, y)[0,1]

def NPCR(a, b):
    return np.sum(a != b) / a.size * 100

def UACI(a, b):
    return np.mean(np.abs(a.astype(int) - b.astype(int))) / 255 * 100

# =========================
# Dataset Evaluation
# =========================
records = []

files = sorted(os.listdir(ENC_DIR))

for f in files:
    if not f.lower().endswith((".png",".jpg",".jpeg")):
        continue

    plain = cv2.imread(os.path.join(PLAIN_DIR, f), cv2.IMREAD_GRAYSCALE)
    enc   = cv2.imread(os.path.join(ENC_DIR, f),   cv2.IMREAD_GRAYSCALE)
    dec   = cv2.imread(os.path.join(DEC_DIR, f),   cv2.IMREAD_GRAYSCALE)

    if plain is None or enc is None or dec is None:
        continue

    # Differential test
    mod = plain.copy()
    mod[0,0] ^= 1

    enc_mod = enc.copy()
    enc_mod[0,0] ^= 1   # If you have true differential encryption, replace this

    records.append({
        "Entropy_Plain": entropy(plain),
        "Entropy_Enc": entropy(enc),
        "Entropy_Dec": entropy(dec),
        "Corr_Plain": correlation(plain),
        "Corr_Enc": correlation(enc),
        "Corr_Dec": correlation(dec),
        "NPCR": NPCR(enc, enc_mod),
        "UACI": UACI(enc, enc_mod)
    })

df = pd.DataFrame(records)

# =========================
# Figure 8 – Entropy Distribution
# =========================
plt.figure(figsize=(6,4))
plt.hist(df["Entropy_Plain"], bins=30, alpha=0.6, label="Plain", density=True)
plt.hist(df["Entropy_Enc"],   bins=30, alpha=0.6, label="Encrypted", density=True)
plt.hist(df["Entropy_Dec"],   bins=30, alpha=0.6, label="Decrypted", density=True)
plt.xlabel("Entropy")
plt.ylabel("Probability Density")
plt.title("Figure 8. Entropy Distribution Across Dataset")
plt.legend()
plt.grid(alpha=0.3)
plt.savefig(f"{OUTPUT_PLOTS}/figure_8_entropy_distribution.png", dpi=300)
plt.close()

# =========================
# Figure 9 – Correlation Distribution
# =========================
plt.figure(figsize=(6,4))
plt.hist(df["Corr_Plain"], bins=30, alpha=0.6, label="Plain", density=True)
plt.hist(df["Corr_Enc"],   bins=30, alpha=0.6, label="Encrypted", density=True)
plt.hist(df["Corr_Dec"],   bins=30, alpha=0.6, label="Decrypted", density=True)
plt.xlabel("Adjacent Pixel Correlation")
plt.ylabel("Probability Density")
plt.title("Figure 9. Correlation Distribution Across Dataset")
plt.legend()
plt.grid(alpha=0.3)
plt.savefig(f"{OUTPUT_PLOTS}/figure_9_correlation_distribution.png", dpi=300)
plt.close()

# =========================
# Figure 10 – NPCR Distribution
# =========================
plt.figure(figsize=(6,4))
plt.hist(df["NPCR"], bins=30, density=True)
plt.xlabel("NPCR (%)")
plt.ylabel("Probability Density")
plt.title("Figure 10. NPCR Distribution Across Dataset")
plt.grid(alpha=0.3)
plt.savefig(f"{OUTPUT_PLOTS}/figure_10_npcr_distribution.png", dpi=300)
plt.close()

# =========================
# Figure 11 – UACI Distribution
# =========================
plt.figure(figsize=(6,4))
plt.hist(df["UACI"], bins=30, density=True)
plt.xlabel("UACI (%)")
plt.ylabel("Probability Density")
plt.title("Figure 11. UACI Distribution Across Dataset")
plt.grid(alpha=0.3)
plt.savefig(f"{OUTPUT_PLOTS}/figure_11_uaci_distribution.png", dpi=300)
plt.close()

# =========================
# Figure X – Dataset-Level Statistical Behaviour
# =========================
plt.figure(figsize=(7,4))
plt.boxplot(
    [df["Entropy_Enc"], df["Corr_Enc"], df["NPCR"], df["UACI"]],
    labels=["Entropy", "Correlation", "NPCR", "UACI"]
)
plt.ylabel("Metric Value")
plt.title("Figure X. Dataset-Level Statistical Behaviour of Encrypted Images")
plt.grid(alpha=0.3)
plt.savefig(f"{OUTPUT_PLOTS}/figure_X_dataset_statistics.png", dpi=300)
plt.close()

print("All figures generated successfully.")

# =====================================================
# "If you want to shine like a sun, first burn like a sun" - Dr. APJ Abdul Kalam.
# Success is a continuous process
# =====================================================