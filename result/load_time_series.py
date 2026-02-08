# ============================================================
# ParaView Python Script: Load VTK Time Series Animation
# ============================================================
# Usage:
#   1. Open ParaView
#   2. Tools -> Python Shell
#   3. Click "Run Script" button
#   4. Select this file
#   5. Press Play button to animate
# ============================================================

import os
import re

# --- Path to VTK files ---
vtk_dir = r"C:\Users\88697.CHENPENGCHUNG12\Desktop\GitHub-PeriodicHill\D3Q27_PeriodicHill\result"

# Only load 6-digit format: velocity_merged_XXXXXX.vtk
pattern = re.compile(r'^velocity_merged_\d{6}\.vtk$')
vtk_files = sorted([
    os.path.join(vtk_dir, f) for f in os.listdir(vtk_dir)
    if pattern.match(f)
])

# Report skipped files (wrong format)
all_vtk = [f for f in os.listdir(vtk_dir) if f.startswith("velocity_merged_") and f.endswith(".vtk")]
skipped = [f for f in all_vtk if not pattern.match(f)]
if skipped:
    print("WARNING: Skipped {} non-6-digit files:".format(len(skipped)))
    for f in skipped[:10]:
        print("  SKIP: {}".format(f))

if not vtk_files:
    raise RuntimeError("No valid 6-digit VTK files found in: " + vtk_dir)

print("Found {} valid VTK files".format(len(vtk_files)))
print("  First: {}".format(os.path.basename(vtk_files[0])))
print("  Last:  {}".format(os.path.basename(vtk_files[-1])))

# --- Load as time series using LegacyVTKReader ---
from paraview.simple import *

reader = LegacyVTKReader(FileNames=vtk_files)

# Setup animation timeline
animationScene = GetAnimationScene()
animationScene.UpdateAnimationUsingDataTimeSteps()

# Set playback speed
animationScene.PlayMode = 'Sequence'
animationScene.NumberOfFrames = 100

# Show in render view
renderView = GetActiveViewOrCreate('RenderView')
display = Show(reader, renderView)
display.Representation = 'Surface'

# Color by velocity magnitude
ColorBy(display, ('POINTS', 'velocity', 'Magnitude'))
display.RescaleTransferFunctionToDataRange(True, False)

# Apply rainbow color map
velocityLUT = GetColorTransferFunction('velocity')
velocityLUT.ApplyPreset('jet', True)

renderView.ResetCamera()
Render()

print("\n=== Done! ===")
print("  Timesteps: {}".format(len(vtk_files)))
print("  Mode: Snap To TimeSteps (every frame shown)")
print("  Press PLAY button to start animation")
