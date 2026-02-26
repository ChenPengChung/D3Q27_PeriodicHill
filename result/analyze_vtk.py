"""Analyze VTK velocity field for GILBM divergence diagnosis."""
import sys
import numpy as np

vtk_file = sys.argv[1] if len(sys.argv) > 1 else "velocity_merged_036001.vtk"

print(f"=== Analyzing {vtk_file} ===\n")

# Read VTK
with open(vtk_file, 'r') as f:
    lines = f.readlines()

# Find VECTORS line
vec_line = None
for idx, line in enumerate(lines):
    if line.startswith("VECTORS"):
        vec_line = idx + 1
        break

NX, NY, NZ = 33, 129, 128  # DIMENSIONS from VTK
total = NX * NY * NZ

# Parse velocity data — skip blank lines
vel = np.zeros((total, 3))
p = 0
line_idx = vec_line
while p < total and line_idx < len(lines):
    parts = lines[line_idx].strip().split()
    line_idx += 1
    if len(parts) < 3:
        continue
    vel[p, 0] = float(parts[0])
    vel[p, 1] = float(parts[1])
    vel[p, 2] = float(parts[2])
    p += 1
print(f"  Parsed {p} velocity vectors")

# Reshape: VTK structured grid order is (k, j, i) fastest-varying = i
# vel[k * NY * NX + j * NX + i]
vel3d_u = vel[:, 0].reshape((NZ, NY, NX))
vel3d_v = vel[:, 1].reshape((NZ, NY, NX))
vel3d_w = vel[:, 2].reshape((NZ, NY, NX))
vmag = np.sqrt(vel3d_u**2 + vel3d_v**2 + vel3d_w**2)

# ======= 1. Global Statistics =======
print("=" * 60)
print("1. GLOBAL VELOCITY STATISTICS")
print("=" * 60)
for name, arr in [("u (x)", vel3d_u), ("v (y, streamwise)", vel3d_v), ("w (z)", vel3d_w), ("|vel|", vmag)]:
    print(f"  {name:20s}: min={arr.min():.6e}  max={arr.max():.6e}  mean={arr.mean():.6e}")
    # Find location of max |value|
    if name != "|vel|":
        amax = np.unravel_index(np.argmax(np.abs(arr)), arr.shape)
        print(f"    max|{name.split()[0]}| at (k={amax[0]}, j={amax[1]}, i={amax[2]}) = {arr[amax]:.6e}")
    else:
        amax = np.unravel_index(np.argmax(arr), arr.shape)
        print(f"    max at (k={amax[0]}, j={amax[1]}, i={amax[2]}) = {arr[amax]:.6e}")

# Check for NaN/Inf
nan_count = np.sum(np.isnan(vel))
inf_count = np.sum(np.isinf(vel))
print(f"\n  NaN count: {nan_count}  Inf count: {inf_count}")

# ======= 2. Bottom Wall Analysis (k=0,1,2) =======
print("\n" + "=" * 60)
print("2. BOTTOM WALL ANALYSIS (k=0,1,2)")
print("=" * 60)

# GPU boundaries in merged j: GPU0=[0..31], GPU1=[32..63], GPU2=[64..95], GPU3=[96..128]
# Local j=33 in merged coords: GPU0→j=30, GPU1→j=62, GPU2→j=94, GPU3→j=126

for k_layer in [0, 1, 2]:
    print(f"\n  --- k={k_layer} ---")
    print(f"  {'j':>5s}  {'GPU':>4s}  {'local_j':>7s}  {'|u|':>12s}  {'|v|':>12s}  {'|w|':>12s}  {'|vel|':>12s}")
    
    key_js = [0, 15, 30, 31, 32, 33, 50, 62, 63, 64, 65, 80, 93, 94, 95, 96, 97, 110, 126, 127, 128]
    for j in key_js:
        gpu = j // 32 if j < 128 else 3
        local_j = j - gpu * 32 + 3  # approximate local j
        u_max = np.max(np.abs(vel3d_u[k_layer, j, :]))
        v_max = np.max(np.abs(vel3d_v[k_layer, j, :]))
        w_max = np.max(np.abs(vel3d_w[k_layer, j, :]))
        vm = np.max(vmag[k_layer, j, :])
        print(f"  {j:5d}  {gpu:4d}  {local_j:7d}  {u_max:12.6e}  {v_max:12.6e}  {w_max:12.6e}  {vm:12.6e}")

# ======= 3. GPU Boundary Cross-sections =======
print("\n" + "=" * 60)
print("3. GPU BOUNDARY CROSS-SECTIONS (k=1)")
print("=" * 60)

k_probe = 1
for gpu_bnd, j_center in [("GPU0/1", 32), ("GPU1/2", 64), ("GPU2/3", 96)]:
    print(f"\n  --- {gpu_bnd} boundary (merged j={j_center}) ---")
    print(f"  {'j':>5s}  {'max|w|':>12s}  {'max|v|':>12s}  {'max|vel|':>12s}  {'mean_v':>12s}")
    for j in range(max(0, j_center - 5), min(NY, j_center + 6)):
        w_max = np.max(np.abs(vel3d_w[k_probe, j, :]))
        v_max = np.max(np.abs(vel3d_v[k_probe, j, :]))
        vm = np.max(vmag[k_probe, j, :])
        v_mean = np.mean(vel3d_v[k_probe, j, :])
        print(f"  {j:5d}  {w_max:12.6e}  {v_max:12.6e}  {vm:12.6e}  {v_mean:12.6e}")

# ======= 4. Vertical Profile at Worst Point =======
print("\n" + "=" * 60)
print("4. VERTICAL PROFILE AT WORST j-COLUMNS")
print("=" * 60)

# Find worst j at k=1
w_abs_k1 = np.max(np.abs(vel3d_w[1, :, :]), axis=1)  # max|w| per j at k=1
worst_js = np.argsort(w_abs_k1)[-5:][::-1]  # top 5

for j_worst in worst_js:
    print(f"\n  j={j_worst} (GPU{j_worst//32}, local_j≈{j_worst%32+3}): max|w| at k=1 = {w_abs_k1[j_worst]:.6e}")
    print(f"  {'k':>5s}  {'max|w|':>12s}  {'max|v|':>12s}  {'w(i=16)':>12s}")
    for k in [0, 1, 2, 3, 5, 10, 20, 40, 64, 100, 127]:
        if k < NZ:
            w_max = np.max(np.abs(vel3d_w[k, j_worst, :]))
            v_max = np.max(np.abs(vel3d_v[k, j_worst, :]))
            w_mid = vel3d_w[k, j_worst, 16]
            print(f"  {k:5d}  {w_max:12.6e}  {v_max:12.6e}  {w_mid:12.6e}")

# ======= 5. Oscillation Check — x direction at worst point =======
print("\n" + "=" * 60)
print("5. X-DIRECTION OSCILLATION CHECK (k=1)")
print("=" * 60)

for j in worst_js[:3]:
    print(f"\n  j={j}, k=1: w-component across all i:")
    w_line = vel3d_w[1, j, :]
    for i in range(NX):
        marker = " ***" if abs(w_line[i]) > 0.01 else ""
        print(f"    i={i:2d}: w={w_line[i]:+.6e}{marker}")

# ======= 6. j-profile of max|w| across entire flow at several k levels =======
print("\n" + "=" * 60)
print("6. j-PROFILE OF max|w| (Flow-wide, at k=0,1,2,5,10,64)")
print("=" * 60)

for k in [0, 1, 2, 5, 10, 64]:
    w_prof = np.max(np.abs(vel3d_w[k, :, :]), axis=1)
    # Find peaks
    top5 = np.argsort(w_prof)[-5:][::-1]
    print(f"  k={k:3d}: top5 j = {list(top5)}, max|w| = {[f'{w_prof[t]:.4e}' for t in top5]}")

# ======= 7. Mid-height flow check (should be healthy) =======
print("\n" + "=" * 60)
print("7. MID-HEIGHT (k=64) FLOW CHECK")
print("=" * 60)
print(f"  v range: [{vel3d_v[64,:,:].min():.6e}, {vel3d_v[64,:,:].max():.6e}]")
print(f"  u range: [{vel3d_u[64,:,:].min():.6e}, {vel3d_u[64,:,:].max():.6e}]")
print(f"  w range: [{vel3d_w[64,:,:].min():.6e}, {vel3d_w[64,:,:].max():.6e}]")
print(f"  mean v = {vel3d_v[64,:,:].mean():.6e} (bulk flow)")

# ======= 8. Local j=33 comparison across GPUs =======
print("\n" + "=" * 60)
print("8. LOCAL j=33 COMPARISON (stencil touches halo)")
print("=" * 60)
print(f"  GPU  merged_j  k=0_max|w|  k=1_max|w|  k=1_max|vel|  region")
for gpu, mj, region in [(0, 30, "hill→flat"), (1, 62, "flat"), (2, 94, "flat"), (3, 126, "flat→hill")]:
    w0 = np.max(np.abs(vel3d_w[0, mj, :]))
    w1 = np.max(np.abs(vel3d_w[1, mj, :]))
    vm1 = np.max(vmag[1, mj, :])
    print(f"  {gpu:3d}  {mj:8d}  {w0:11.4e}  {w1:11.4e}  {vm1:12.4e}  {region}")

print("\n=== Analysis Complete ===")
