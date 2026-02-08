# ============================================================
# ParaView Python Script: Load VTK Time Series Animation
# ============================================================
# Usage:
#   1. Open ParaView
#   2. Tools -> Python Shell
#   3. Click "Run Script" button
#   4. Select this file
#   5. 動畫自動循環播放，或匯出影片
#
# 設定區 (修改這裡控制行為):
#   DURATION_SEC  = 動畫總秒數 (越小越快)
#   LOOP          = True 連續循環 / False 播一次
#   EXPORT_VIDEO  = True 匯出 MP4 / False 只在視窗播放
#   VIDEO_FPS     = 匯出影片的 FPS
# ============================================================

# ==================== 使用者設定 ====================
DURATION_SEC = 3        # 全部 timestep 在幾秒內播完
LOOP         = True     # 是否連續循環播放
EXPORT_VIDEO = True     # True = 匯出影片檔, False = 僅畫面播放
VIDEO_FPS    = 30       # 匯出影片 FPS
MAX_FILES    = 0        # 限制載入檔案數 (0 = 全部載入)
# ====================================================

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

# 限制載入數量 (預覽用)
if MAX_FILES > 0:
    vtk_files = vtk_files[:MAX_FILES]
    print("PREVIEW MODE: only loading first {} files".format(MAX_FILES))

print("Found {} valid VTK files".format(len(vtk_files)))
print("  First: {}".format(os.path.basename(vtk_files[0])))
print("  Last:  {}".format(os.path.basename(vtk_files[-1])))

# --- Load as time series using LegacyVTKReader ---
from paraview.simple import *

reader = LegacyVTKReader(FileNames=vtk_files)

# Setup animation timeline
animationScene = GetAnimationScene()
animationScene.UpdateAnimationUsingDataTimeSteps()

# 播放設定: Sequence 模式
animationScene.PlayMode = 'Sequence'
animationScene.NumberOfFrames = max(10, int(DURATION_SEC * 10))
animationScene.Loop = 1 if LOOP else 0

# Show in render view
renderView = GetActiveViewOrCreate('RenderView')
display = Show(reader, renderView)
display.Representation = 'Surface'

# Color by velocity magnitude
ColorBy(display, ('POINTS', 'velocity', 'Magnitude'))
display.RescaleTransferFunctionToDataRange(True, False)

# Apply Rainbow Desaturated color map
velocityLUT = GetColorTransferFunction('velocity')
velocityLUT.ApplyPreset('Rainbow Desaturated', True)

# --- 顯示 Timestep 標籤 (從檔名數字) ---
# 提取每個檔案的 step 數字
step_numbers = [int(re.search(r'velocity_merged_(\d+)', os.path.basename(f)).group(1))
                for f in vtk_files]
first_step = step_numbers[0]
step_interval = step_numbers[1] - step_numbers[0] if len(step_numbers) > 1 else 1000

# 用 PythonAnnotation 即時顯示當前 step
annotation = PythonAnnotation(Input=reader)
annotation.Expression = "'Step = %d' % ({} + int(t_index) * {})".format(first_step, step_interval)
annotation.ArrayAssociation = 'Point Data'

annDisplay = Show(annotation, renderView)
annDisplay.FontSize = 48
annDisplay.Bold = 1
annDisplay.Color = [0.0, 0.0, 0.0]       # 黑色文字
annDisplay.FontFamily = 'Times'           # Times New Roman
annDisplay.WindowLocation = 'Upper Center' # 主體上方中央

print("  Step range: {} ~ {} (interval={})".format(
    step_numbers[0], step_numbers[-1], step_interval))

# --- 3D 視角設定 ---
renderView.ResetCamera()

# 使用實測的攝影機參數
renderView.CameraPosition = [18.0019, -8.2673, 2.3630]
renderView.CameraFocalPoint = [2.25, 4.5, 1.518]
renderView.CameraViewUp = [-0.03047, 0.02853, 0.99913]

# 先 Reset Camera Closest 再稍微拉遠，主體與邊框留一點間距
renderView.ResetCamera(True)
camera = renderView.GetActiveCamera()
camera.Dolly(1.05)  # >1 拉近，主體更大

# 跳到最後一幀抓色彩範圍 (避免 Step 1 速度太小看不到顏色)
animationScene.GoToLast()
display.RescaleTransferFunctionToDataRange(True, False)

# 回到第一幀
animationScene.GoToFirst()
Render()

# 顯示色條 (Velocity Magnitude)
velocityLUT = GetColorTransferFunction('velocity')
colorBar = GetScalarBar(velocityLUT, renderView)
colorBar.Title = 'Velocity Magnitude'
colorBar.ComponentTitle = ''
colorBar.Visibility = 1

# 設定影片解析度
renderView.ViewSize = [1920, 1080]

# 白色背景 (搭配黑色文字)
renderView.Background = [1.0, 1.0, 1.0]

Render()

# --- 匯出影片 (EXPORT_VIDEO=True 時) ---
if EXPORT_VIDEO:
    video_path = os.path.join(vtk_dir, "animation.avi")
    total_frames = len(vtk_files)
    # 切換為 Snap To TimeSteps，確保每個 timestep 都輸出一幀
    animationScene.PlayMode = 'Snap To TimeSteps'
    print("\nExporting {} frames (1 frame per VTK file)...".format(total_frames))
    SaveAnimation(video_path, renderView,
                  FrameRate=VIDEO_FPS,
                  FrameWindow=[0, total_frames - 1])
    # 驗證
    print("\n=== Video exported! ===")
    print("  File: {}".format(video_path))
    print("  Valid VTK files: {}".format(total_frames))
    if skipped:
        print("  Skipped files:   {} (not included)".format(len(skipped)))
    print("  Exported frames: {} (1:1 match)".format(total_frames))
    print("  FPS: {} -> Duration: {:.1f} sec".format(VIDEO_FPS, total_frames / VIDEO_FPS))
else:
    # 自動開始播放
    animationScene.Play()

print("\n=== Done! ===")
print("  Timesteps: {}".format(len(vtk_files)))
print("  Duration: {} sec, Loop: {}".format(DURATION_SEC, LOOP))
if not EXPORT_VIDEO:
    print("  Animation is playing! Close ParaView to stop.")
