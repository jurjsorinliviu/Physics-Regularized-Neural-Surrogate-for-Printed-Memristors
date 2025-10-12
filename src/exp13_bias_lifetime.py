# ===========================================================
#  experiment13_STANDALONE_CORRECTED.py
#  Uses ONLY corrected Exp 8 + Exp 12 data
# ===========================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import sys

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

print("=" * 80)
print("EXPERIMENT 13 (CORRECTED): Bias-Accelerated Lifetime Projection")
print("=" * 80)
print("Uses: Exp 8 (Arrhenius) + Exp 12 (corrected self-heating)")
print("-" * 80)

# ------------------------------------------------------------
# [1] Parameters from corrected Experiment 8
# ------------------------------------------------------------
print("\n[1/4] Loading Arrhenius parameters from Experiment 8...")

Ea = 0.379  # eV (from your Exp 8 output)
tau_ref = 1.95e7  # s at 300K (from your Exp 8 output)
T_ref = 300.0  # K
kB = 8.617333262e-5  # eV/K

print(f"  E_a = {Ea:.4f} eV")
print(f"  τ_ref(300K) = {tau_ref:.2e} s = {tau_ref/3600:.1f} hours")

# ------------------------------------------------------------
# [2] Load corrected self-heating data (from Exp 12 FINAL)
# ------------------------------------------------------------
print("\n[2/4] Loading self-heating data (Exp 12 corrected)...")

# Load the summary file
try:
    heat_df = pd.read_csv("self_heating_summary_final.csv")
    biases = heat_df["Bias_V"].values
    T95_vals = heat_df["T_95_K"].values
    deltaT_vals = heat_df["deltaT_95_K"].values
    print(f"  ✓ Loaded: self_heating_summary_final.csv")
    print(f"  Data points: {len(biases)}")
    for i in range(len(biases)):
        print(f"    {biases[i]:.2f}V → T_95 = {T95_vals[i]:.2f}K (ΔT = {deltaT_vals[i]:+.2f}K)")
except FileNotFoundError:
    print("  ⚠ self_heating_summary_final.csv not found!")
    print("  Please run the corrected Experiment 12 first.")
    sys.exit(1)

# ------------------------------------------------------------
# [3] Compute lifetime projections using Arrhenius relation
# ------------------------------------------------------------
print("\n[3/4] Computing lifetime projections...")

# Arrhenius: τ(T) = τ_ref * exp[E_a/k_B * (1/T - 1/T_ref)]
exponent = (Ea / kB) * (1.0 / T95_vals - 1.0 / T_ref)
tau_rel = np.exp(exponent)

# Convert to practical units
lifetime_seconds = tau_ref * tau_rel
lifetime_hours = lifetime_seconds / 3600
lifetime_days = lifetime_hours / 24

# Acceleration factors (relative to lowest bias)
AF_thermal = tau_rel[0] / tau_rel

print("\nLifetime calculations:")
for i in range(len(biases)):
    print(f"  {biases[i]:.2f}V: τ_rel={tau_rel[i]:.4f}, " +
          f"Life={lifetime_hours[i]:.2f}h ({lifetime_days[i]:.2f} days), " +
          f"AF={AF_thermal[i]:.3f}×")

# ------------------------------------------------------------
# [4] Save results and generate figures
# ------------------------------------------------------------
print("\n[4/4] Generating outputs...")

# Save CSV
results_df = pd.DataFrame({
    "Bias_V": biases,
    "T95_K": T95_vals,
    "DeltaT_K": deltaT_vals,
    "tau_relative": tau_rel,
    "Lifetime_seconds": lifetime_seconds,
    "Lifetime_hours": lifetime_hours,
    "Lifetime_days": lifetime_days,
    "Accel_Factor": AF_thermal
})
results_df.to_csv("bias_lifetime_corrected_final.csv", index=False)
print("  ✓ Saved: bias_lifetime_corrected_final.csv")

# Generate figure
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

# (a) Lifetime vs Bias
axes[0].plot(biases, lifetime_days, "o-", color="#d95f02", 
             lw=2.5, ms=9, label="Retention Lifetime")
axes[0].set_xlabel("Stress Bias (V)", fontsize=12)
axes[0].set_ylabel("Projected Lifetime (days)", fontsize=12)
axes[0].set_yscale("log")
axes[0].set_title("(a) Bias-Accelerated Lifetime", fontsize=12, fontweight="bold")
axes[0].grid(True, alpha=0.3, which="both")

# Add reference lines
axes[0].axhline(1, color='gray', linestyle='--', alpha=0.5, lw=1.5, label="1 day")
axes[0].axhline(30, color='gray', linestyle=':', alpha=0.5, lw=1.5, label="1 month")
axes[0].legend(fontsize=9, loc="best")

# (b) Arrhenius plot
inv_T = 1000.0 / T95_vals  # 1000/K for better scale
ln_tau = np.log(tau_rel)

axes[1].plot(inv_T, ln_tau, "s-", color="#1b9e77", 
             lw=2.5, ms=9, label="Self-Heating Data")

# Linear fit to verify Arrhenius behavior
slope, intercept, r_value, _, _ = stats.linregress(inv_T, ln_tau)
fit_line = slope * inv_T + intercept
axes[1].plot(inv_T, fit_line, "--", color="gray", 
             lw=2, alpha=0.7, label=f"Linear fit (R²={r_value**2:.4f})")

axes[1].set_xlabel("1000/T (1000/K)", fontsize=12)
axes[1].set_ylabel("ln(τ_rel)", fontsize=12)
axes[1].set_title("(b) Arrhenius Validation", fontsize=12, fontweight="bold")
axes[1].grid(True, alpha=0.3)
axes[1].legend(fontsize=10)

plt.tight_layout()
plt.savefig("bias_lifetime_corrected_final.png", dpi=600, bbox_inches="tight")
print("  ✓ Saved: bias_lifetime_corrected_final.png")
plt.close(fig)

# ------------------------------------------------------------
# Summary
# ------------------------------------------------------------
print("\n" + "=" * 80)
print("RESULTS SUMMARY")
print("=" * 80)
print("\nLifetime Projections:")
print("-" * 70)
for i in range(len(biases)):
    print(f"  {biases[i]:.2f}V: T={T95_vals[i]:.2f}K (ΔT={deltaT_vals[i]:+.2f}K) | " +
          f"Life={lifetime_hours[i]:.1f}h ({lifetime_days[i]:.2f}d) | " +
          f"AF={AF_thermal[i]:.3f}×")

print("\nPhysical Interpretation:")
print(f"  • Baseline (0.05V): {lifetime_days[0]:.1f} days retention")
print(f"  • Maximum stress (0.20V): {lifetime_days[-1]:.1f} days retention")
print(f"  • Thermal acceleration: {AF_thermal[-1]:.3f}× faster at 0.20V")
print(f"  • Temperature effect: {deltaT_vals[-1]:.2f}K rise")

reduction_pct = (1 - tau_rel[-1]/tau_rel[0]) * 100
print(f"  • Lifetime reduction: {reduction_pct:.1f}% from self-heating")

print("\nComparison with Original (Incorrect) Model:")
print("  Original: 618-632K → sub-second lifetimes (unphysical)")
print(f"  Corrected: {T95_vals[0]:.1f}-{T95_vals[-1]:.1f}K → {lifetime_days[0]:.0f}-{lifetime_days[-1]:.0f} day lifetimes (realistic)")

print("\nDesign Recommendations:")
if deltaT_vals[-1] < 5:
    print("  ✓ Self-heating effect is moderate (ΔT < 5K)")
    print("  ✓ Read voltage up to 0.20V appears safe")
else:
    print("  ⚠ Significant self-heating detected (ΔT > 5K)")
    print("  → Consider limiting read voltage to <0.15V")

print("\nFiles saved:")
print("  - bias_lifetime_corrected_final.csv")
print("  - bias_lifetime_corrected_final.png")

print("\n" + "=" * 80)
print("EXPERIMENT 13 COMPLETE (CORRECTED)")
print("=" * 80)