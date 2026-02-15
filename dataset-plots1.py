# =====================================================
# code authored by Faizal Nujumudeen
# Presidency University, Bengaluru
# =====================================================

import pandas as pd
import matplotlib.pyplot as plt
import os

# =========================
# Configuration
# =========================
CSV_FILE = "dataset_subset -small/dataset_metrics1.csv"     # path to your CSV
OUTPUT_DIR = "dataset_subset -small/csv_plots"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# =========================
# Read CSV
# =========================
df = pd.read_csv(CSV_FILE)

print("Columns found:", df.columns.tolist())

# =========================
# 1. Entropy Distribution
# =========================
plt.figure(figsize=(6,4))
plt.hist(df["Entropy"], bins=30, density=True)
plt.xlabel("Entropy")
plt.ylabel("Probability Density")
plt.title("Entropy Distribution (Encrypted Images)")
plt.grid(alpha=0.3)
plt.savefig(f"{OUTPUT_DIR}/entropy_distribution.png", dpi=300)
plt.close()

# =========================
# 2. Correlation Distribution
# =========================
plt.figure(figsize=(6,4))
plt.hist(df["Correlation"], bins=30, density=True)
plt.xlabel("Adjacent Pixel Correlation")
plt.ylabel("Probability Density")
plt.title("Correlation Distribution (Encrypted Images)")
plt.grid(alpha=0.3)
plt.savefig(f"{OUTPUT_DIR}/correlation_distribution.png", dpi=300)
plt.close()

# =========================
# 3. NPCR Distribution
# =========================
plt.figure(figsize=(6,4))
plt.hist(df["NPCR"], bins=30, density=True)
plt.xlabel("NPCR (%)")
plt.ylabel("Probability Density")
plt.title("NPCR Distribution Across Dataset")
plt.grid(alpha=0.3)
plt.savefig(f"{OUTPUT_DIR}/npcr_distribution.png", dpi=300)
plt.close()

# =========================
# 4. UACI Distribution
# =========================
plt.figure(figsize=(6,4))
plt.hist(df["UACI"], bins=30, density=True)
plt.xlabel("UACI (%)")
plt.ylabel("Probability Density")
plt.title("UACI Distribution Across Dataset")
plt.grid(alpha=0.3)
plt.savefig(f"{OUTPUT_DIR}/uaci_distribution.png", dpi=300)
plt.close()

# =========================
# 5. PSNR Distribution
# =========================
plt.figure(figsize=(6,4))
plt.hist(df["PSNR"].replace(float("inf"), None).dropna(), bins=30)
plt.xlabel("PSNR (dB)")
plt.ylabel("Frequency")
plt.title("PSNR Distribution (Plain vs Decrypted)")
plt.grid(alpha=0.3)
plt.savefig(f"{OUTPUT_DIR}/psnr_distribution.png", dpi=300)
plt.close()

# =========================
# 6. MSE Distribution
# =========================
plt.figure(figsize=(6,4))
plt.hist(df["MSE"], bins=30)
plt.xlabel("MSE")
plt.ylabel("Frequency")
plt.title("MSE Distribution (Plain vs Decrypted)")
plt.grid(alpha=0.3)
plt.savefig(f"{OUTPUT_DIR}/mse_distribution.png", dpi=300)
plt.close()

# =========================
# 7. Metric Stability Plot
# =========================
plt.figure(figsize=(8,4))
plt.plot(df["NPCR"].values, label="NPCR (%)")
plt.plot(df["UACI"].values, label="UACI (%)")
plt.xlabel("Image Index")
plt.ylabel("Metric Value")
plt.title("Differential Metric Stability Across Dataset")
plt.legend()
plt.grid(alpha=0.3)
plt.savefig(f"{OUTPUT_DIR}/npcr_uaci_stability.png", dpi=300)
plt.close()

# =========================
# 8. Dataset-Level Statistical Behaviour (Boxplot)
# =========================
plt.figure(figsize=(7,4))
plt.boxplot(
    [df["Entropy"], df["Correlation"], df["NPCR"], df["UACI"]],
    labels=["Entropy", "Correlation", "NPCR", "UACI"]
)
plt.ylabel("Metric Value")
plt.title("Dataset-Level Statistical Behaviour")
plt.grid(alpha=0.3)
plt.savefig(f"{OUTPUT_DIR}/dataset_statistics_boxplot.png", dpi=300)
plt.close()

print("All plots generated successfully in:", OUTPUT_DIR)

# =====================================================
# "If you want to shine like a sun, first burn like a sun" - Dr. APJ Abdul Kalam.
# Success is a continuous process
# =====================================================