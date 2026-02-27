"""Analyze VTK flow field for wall oscillation issues."""
import numpy as np

vtk_file = r"c:\Users\88697.CHENPENGCHUNG12\Desktop\GitHub-PeriodicHill\D3Q27_PeriodicHill\result\velocity_merged_001001.vtk"

with open(vtk_file, 'r') as f:
    for line in f:
        if line.startswith("DIMENSIONS"):
            dims = list(map(int, line.split()[1:]))
            break
    nx, ny, nz = dims
    npts = nx * ny * nz
    print(f"Grid: {nx} x {ny} x {nz} = {npts} points")
    
    for line in f:
        if line.startswith("POINTS"):
            break
    coords = []
    for i in range(npts):
        coords.append(list(map(float, f.readline().split())))
    coords = np.array(coords).reshape(nz, ny, nx, 3)
    
    for line in f:
        if line.startswith("VECTORS"):
            break
    vel = []
    for i in range(npts):
        vel.append(list(map(float, f.readline().split())))
    vel = np.array(vel).reshape(nz, ny, nx, 3)

u = vel[:,:,:,0]  # spanwise
v = vel[:,:,:,1]  # streamwise
w = vel[:,:,:,2]  # wall-normal

print(f"Coords: x=[{coords[:,:,:,0].min():.3f},{coords[:,:,:,0].max():.3f}]  y=[{coords[:,:,:,1].min():.3f},{coords[:,:,:,1].max():.3f}]  z=[{coords[:,:,:,2].min():.3f},{coords[:,:,:,2].max():.3f}]")
print(f"Velocity: u=[{u.min():.6e},{u.max():.6e}]  v=[{v.min():.6e},{v.max():.6e}]  w=[{w.min():.6e},{w.max():.6e}]")

print(f"\n{'='*70}")
print("WALL VELOCITY (should be ~0)")
print(f"{'='*70}")
for kw, nm in [(0,"Bottom k=0"), (1,"k=1"), (nz-2,"k=nz-2"), (nz-1,"Top k=nz-1")]:
    print(f"  {nm}: v_rms={np.sqrt(np.mean(v[kw]**2)):.4e}  u_rms={np.sqrt(np.mean(u[kw]**2)):.4e}  w_rms={np.sqrt(np.mean(w[kw]**2)):.4e}  v_max={np.abs(v[kw]).max():.4e}")

print(f"\n{'='*70}")
print("NEAR-WALL PROFILE (x-averaged, i=nx/2 slice)")
print(f"{'='*70}")
for j_idx in [0, ny//4, ny//2, 3*ny//4, ny-1]:
    y_val = coords[0, j_idx, 0, 1]
    print(f"\n  j={j_idx} (y={y_val:.3f}):")
    for k in range(min(10, nz)):
        z_val = coords[k, j_idx, nx//2, 2]
        v_avg = np.mean(v[k, j_idx, :])
        print(f"    k={k:3d}  z={z_val:8.5f}  v_avg={v_avg:+12.6e}  v_max={np.abs(v[k,j_idx,:]).max():12.6e}")

print(f"\n{'='*70}")
print("OSCILLATION: v at bottom 8 levels, several points")
print(f"{'='*70}")
for ii in [0, nx//2]:
    for jj in [0, ny//4, ny//2, 3*ny//4]:
        vals = [v[k, jj, ii] for k in range(8)]
        signs = ''.join(['+' if x>0 else ('-' if x<0 else '0') for x in vals])
        print(f"  i={ii:2d} j={jj:3d}  y={coords[0,jj,ii,1]:.3f}  v[k=0..7]=[{', '.join(f'{x:+.3e}' for x in vals)}]  signs={signs}")

print(f"\n{'='*70}")
print("|V|_max PER Z-LEVEL (first 15 + last 15)")
print(f"{'='*70}")
for k in list(range(min(15,nz))) + list(range(max(15,nz-15), nz)):
    vmag = np.sqrt(u[k]**2 + v[k]**2 + w[k]**2)
    z_val = coords[k, ny//2, nx//2, 2]
    print(f"  k={k:3d}  z={z_val:8.5f}  |V|_max={vmag.max():.6e}  v_rms={np.sqrt(np.mean(v[k]**2)):.6e}")

# Sign alternation count
print(f"\n{'='*70}")
print("SIGN ALTERNATION COUNT (bottom 6 levels)")
print(f"{'='*70}")
osc = 0
total = 0
for j in range(ny):
    for i in range(nx):
        for k in range(1, min(6,nz)):
            total += 1
            if v[k,j,i] * v[k-1,j,i] < 0 and abs(v[k,j,i]) > 1e-10:
                osc += 1
print(f"  Bottom: {osc}/{total} = {osc/total*100:.1f}% sign alternations")

osc_top = 0
total_top = 0
for j in range(ny):
    for i in range(nx):
        for k in range(nz-6, nz-1):
            total_top += 1
            if v[k,j,i] * v[k+1,j,i] < 0 and abs(v[k,j,i]) > 1e-10:
                osc_top += 1
print(f"  Top:    {osc_top}/{total_top} = {osc_top/total_top*100:.1f}% sign alternations")
