"""Quick W-oscillation diagnostic for current VTK files."""
import os, re, glob
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
VTK_PATTERN = os.path.join(SCRIPT_DIR, "velocity_merged_*.vtk")

def read_vtk(filepath):
    with open(filepath, 'r') as f:
        header2 = f.readline(); header2 = f.readline().strip()
        step_m = re.search(r'step=(\d+)', header2)
        force_m = re.search(r'Force=([\d.eE+\-]+)', header2)
        step = int(step_m.group(1)) if step_m else -1
        force = float(force_m.group(1)) if force_m else 0.0
        for line in f:
            if line.startswith("DIMENSIONS"):
                dims = list(map(int, line.split()[1:])); break
        nx, ny, nz = dims; npts = nx*ny*nz
        for line in f:
            if line.startswith("POINTS"): break
        coords = np.empty((npts, 3))
        for i in range(npts): coords[i] = list(map(float, f.readline().split()))
        coords = coords.reshape(nz, ny, nx, 3)
        for line in f:
            if line.startswith("VECTORS"): break
        vel = np.empty((npts, 3))
        for i in range(npts): vel[i] = list(map(float, f.readline().split()))
        vel = vel.reshape(nz, ny, nx, 3)
    return {'step': step, 'force': force, 'dims': (nx,ny,nz),
            'coords': coords, 'u': vel[:,:,:,0], 'v': vel[:,:,:,1], 'w': vel[:,:,:,2]}

vtk_files = sorted(glob.glob(VTK_PATTERN))
print(f"Found {len(vtk_files)} VTK files\n")

data_list = []
for vf in vtk_files:
    print(f"  Reading {os.path.basename(vf)} ...", end='', flush=True)
    d = read_vtk(vf); data_list.append(d)
    print(f"  step={d['step']}  Force={d['force']:.5e}")

data_list.sort(key=lambda d: d['step'])
nx, ny, nz = data_list[0]['dims']
coords = data_list[0]['coords']

print(f"\n{'='*120}")
print(f"{'Step':>7} {'Force':>12} {'W_max':>12} {'W_min':>12} {'W_rms':>12} "
      f"{'V_rms':>12} {'|V|_max':>12} {'W/V%':>8} {'Wmax_loc':>25}")
print(f"{'-'*120}")

for d in data_list:
    u,v,w = d['u'],d['v'],d['w']
    wrms = np.sqrt(np.mean(w**2)); vrms = np.sqrt(np.mean(v**2))
    vmag = np.sqrt(u**2+v**2+w**2).max()
    wabs = np.abs(w); idx = np.argmax(wabs)
    kk,jj,ii = np.unravel_index(idx, w.shape)
    z_,y_ = coords[kk,jj,ii,2], coords[kk,jj,ii,1]
    ratio = wrms/max(vrms,1e-20)*100
    print(f"{d['step']:7d} {d['force']:12.5e} {w.max():+12.5e} {w.min():+12.5e} {wrms:12.5e} "
          f"{vrms:12.5e} {vmag:12.5e} {ratio:7.2f}% k={kk:3d} j={jj:3d} y={y_:.2f} z={z_:.3f}")

# Per-step detailed analysis
for d in data_list:
    w = d['w']; step = d['step']
    print(f"\n{'='*100}")
    print(f"Step {step}  Force={d['force']:.5e}")
    print(f"{'='*100}")
    
    # W_rms per z-level (bottom 10 + top 5)
    print(f"  {'k':>4} {'z':>8} {'W_rms':>12} {'|W|_max':>12} {'W_mean':>12} {'V_rms':>12}")
    for k in list(range(min(12,nz))) + list(range(max(12,nz-5),nz)):
        z_val = coords[k, ny//2, nx//2, 2]
        wk = w[k]; vk = d['v'][k]
        print(f"  {k:4d} {z_val:8.4f} {np.sqrt(np.mean(wk**2)):12.5e} "
              f"{np.abs(wk).max():12.5e} {np.mean(wk):+12.5e} {np.sqrt(np.mean(vk**2)):12.5e}")
    
    # Checkerboard in k-direction for W
    w_alt_b = 0; w_tot_b = 0
    for j in range(ny):
        for i in range(nx):
            for k in range(1, min(10, nz)):
                if abs(w[k,j,i]) > 1e-10 and abs(w[k-1,j,i]) > 1e-10:
                    w_tot_b += 1
                    if w[k,j,i]*w[k-1,j,i] < 0: w_alt_b += 1
    print(f"\n  W checkerboard (bottom 10): {w_alt_b}/{w_tot_b} = {w_alt_b/max(w_tot_b,1)*100:.1f}%")
    
    # Where is |W|_max? Show top 10 locations
    wflat = np.abs(w).flatten()
    top_idx = np.argsort(wflat)[-10:][::-1]
    print(f"\n  Top 10 |W| locations:")
    for rank_i, fi in enumerate(top_idx):
        kk,jj,ii = np.unravel_index(fi, w.shape)
        z_,y_,x_ = coords[kk,jj,ii,2], coords[kk,jj,ii,1], coords[kk,jj,ii,0]
        print(f"    #{rank_i+1}: W={w[kk,jj,ii]:+.5e} at k={kk:3d} j={jj:3d} i={ii:3d} (y={y_:.3f} z={z_:.4f})")

    # Near-wall W profile at worst j location
    worst_j = np.argmax(np.max(np.abs(w[:10,:,:]), axis=(0,2)))
    y_worst = coords[0, worst_j, 0, 1]
    print(f"\n  Near-wall W profile at worst j={worst_j} (y={y_worst:.3f}):")
    for k in range(min(12, nz)):
        z_val = coords[k, worst_j, nx//2, 2]
        w_vals = w[k, worst_j, :]
        print(f"    k={k:3d} z={z_val:.4f}  W_avg={np.mean(w_vals):+.4e}  "
              f"|W|_max={np.abs(w_vals).max():.4e}  W_rms={np.sqrt(np.mean(w_vals**2)):.4e}")

print("\nDone.")
