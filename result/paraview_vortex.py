"""
ParaView Python Script: 渦流結構可視化 (Q-criterion isosurface)
================================================================
用法:
  1. 開啟 ParaView
  2. Tools → Python Shell → Run Script → 選此檔案
  或：
  pvpython paraview_vortex.py

輸出:
  - 互動式 3D 渦流結構視圖 (Q-criterion 等值面, w 著色)
  - 自動截圖存為 result/vortex_structure.png

參考: Periodic Hill (LX=4.5, LY=9.0, LZ=3.036, Re=700)
"""

from paraview.simple import *
import os

# ============================================================
# 設定區 (可自行調整)
# ============================================================
VTK_FILE    = "velocity_merged_178001.vtk"   # VTK 檔案名稱
Q_THRESHOLD = 0.001     # 渦度量值等值面閾值 (低 Re/低 Uref 流場需要很低的值)
                         # 太大 → 看不到結構; 太小 → 全場都是
                         # 先看結果再微調
OPACITY     = 0.8        # 等值面透明度 (0~1)
BG_COLOR    = [1, 1, 1]  # 背景色 (白色)
IMG_SIZE    = [1920, 1080]  # 輸出圖片解析度

# 色標範圍 (w 速度, 用於著色)
W_RANGE     = [-0.02, 0.02]  # 配合參考圖的 w 範圍

# ============================================================
# 自動偵測檔案路徑
# ============================================================
script_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in dir() else os.getcwd()

# 嘗試多個可能路徑
possible_paths = [
    os.path.join(script_dir, VTK_FILE),
    os.path.join(script_dir, "..", "result", VTK_FILE),
    os.path.join(script_dir, "result", VTK_FILE),
    VTK_FILE,
]

vtk_path = None
for p in possible_paths:
    if os.path.exists(p):
        vtk_path = p
        break

if vtk_path is None:
    # 如果找不到, 用預設路徑讓使用者知道
    vtk_path = os.path.join(script_dir, VTK_FILE)
    print(f"WARNING: VTK file not found. Expected at: {vtk_path}")
    print(f"         Please modify VTK_FILE variable in script.")

print(f"Loading: {vtk_path}")

# ============================================================
# Step 1: 載入 VTK 檔案
# ============================================================
reader = LegacyVTKReader(FileNames=[vtk_path])
reader.UpdatePipeline()

# 印出可用的陣列
info = reader.GetDataInformation()
pd = info.GetPointDataInformation()
print(f"\n=== VTK Data Info ===")
print(f"Points: {info.GetNumberOfPoints()}")
print(f"Available arrays:")
for i in range(pd.GetNumberOfArrays()):
    arr = pd.GetArrayInformation(i)
    print(f"  - {arr.GetName()} ({arr.GetNumberOfComponents()} components)")

# ============================================================
# Step 2: 計算渦度量值 (vorticity magnitude) 用於等值面
# ============================================================
# VTK 已有 vorticity 向量, 用 Calculator 計算其量值
calc_mag = Calculator(Input=reader)
calc_mag.Function = 'mag(vorticity)'
calc_mag.ResultArrayName = 'VorticityMagnitude'
calc_mag.UpdatePipeline()

# 印出渦度量值範圍, 幫助使用者選擇閾值
vm_info = calc_mag.GetDataInformation().GetPointDataInformation().GetArrayInformation('VorticityMagnitude')
vm_range = vm_info.GetComponentRange(0)
print(f"\n=== Vorticity Magnitude Range ===")
print(f"  Min: {vm_range[0]:.8f}")
print(f"  Max: {vm_range[1]:.8f}")

# 自動設定閾值: 使用 Max 的 10% 作為起始值 (忽略手動 Q_THRESHOLD)
Q_THRESHOLD = vm_range[1] * 0.10
print(f"  Auto threshold (10% of max): {Q_THRESHOLD:.8f}")

# ============================================================
# Step 3: 渦度量值等值面 (Contour)
# ============================================================
contour = Contour(Input=calc_mag)
contour.ContourBy = ['POINTS', 'VorticityMagnitude']
contour.Isosurfaces = [Q_THRESHOLD]
contour.ComputeNormals = 1
contour.ComputeScalars = 1
contour.UpdatePipeline()

n_cells = contour.GetDataInformation().GetNumberOfCells()
print(f"\n=== Contour Info ===")
print(f"  Isosurface cells: {n_cells}")

# 逐步降低閾值直到有足夠的等值面
for fraction in [0.05, 0.02, 0.01, 0.005, 0.002, 0.001]:
    if n_cells >= 5000:
        break
    Q_THRESHOLD = vm_range[1] * fraction
    print(f"  Trying {fraction*100:.1f}% of max = {Q_THRESHOLD:.8f}")
    contour.Isosurfaces = [Q_THRESHOLD]
    contour.UpdatePipeline()
    n_cells = contour.GetDataInformation().GetNumberOfCells()
    print(f"  -> cells: {n_cells}")

print(f"\n  FINAL threshold: {Q_THRESHOLD:.8f}  ({n_cells} cells)")

# ============================================================
# Step 4: 提取 w 分量用於著色
# ============================================================
# 用 Calculator 從 velocity 向量提取 w (第 3 分量)
calc_w = Calculator(Input=contour)
calc_w.Function = 'velocity_Z'       # ParaView 語法: velocity 的 Z 分量
calc_w.ResultArrayName = 'w_velocity'
calc_w.UpdatePipeline()

# ============================================================
# Step 5: 設定渲染視圖
# ============================================================
renderView = GetActiveViewOrCreate('RenderView')
renderView.ViewSize = IMG_SIZE
renderView.Background = BG_COLOR

# 顯示等值面
display = Show(calc_w, renderView)
display.Representation = 'Surface'
display.Opacity = OPACITY

# 以 w 速度著色
ColorBy(display, ('POINTS', 'w_velocity'))
w_lut = GetColorTransferFunction('w_velocity')
w_lut.RescaleTransferFunction(W_RANGE[0], W_RANGE[1])

# 使用 Cool to Warm 色標 (藍-白-紅, 與參考圖一致)
w_lut.ApplyPreset('Cool to Warm', True)

# 顯示色標列
w_bar = GetScalarBar(w_lut, renderView)
w_bar.Title = 'w'
w_bar.ComponentTitle = ''
w_bar.Visibility = 1
w_bar.TitleFontSize = 16
w_bar.LabelFontSize = 14

# ============================================================
# Step 6: 設定相機角度 (斜俯視, 類似參考圖)
# ============================================================
renderView.ResetCamera()
camera = renderView.GetActiveCamera()

# 斜俯視角度 (從右上方看向左下)
# 物理域: x=[0,4.5], y=[0,9], z=[0,3.036]
domain_center = [2.25, 4.5, 1.5]  # 域中心

camera.SetFocalPoint(*domain_center)
camera.SetPosition(12, -8, 12)     # 相機位置 (右前上方)
camera.SetViewUp(0, 0, 1)          # z 軸朝上

renderView.ResetCamera()
# 稍微拉遠
camera.Dolly(0.85)

# ============================================================
# Step 7: 加入座標軸標註
# ============================================================
renderView.AxesGrid.Visibility = 1
renderView.AxesGrid.XTitle = 'X Axis'
renderView.AxesGrid.YTitle = 'Y Axis'
renderView.AxesGrid.ZTitle = 'Z Axis'
renderView.AxesGrid.XTitleFontSize = 14
renderView.AxesGrid.YTitleFontSize = 14
renderView.AxesGrid.ZTitleFontSize = 14

# ============================================================
# Step 8: 加入半透明底部山丘輪廓 (用壁面 slice)
# ============================================================
# 底壁 = 計算域最底面, 用 Clip 取出 z < hill_height 的區域
# 或直接顯示原始網格的外表面作為參考
try:
    outline = Show(reader, renderView)
    outline.Representation = 'Outline'
    outline.AmbientColor = [0.3, 0.3, 0.3]
    outline.DiffuseColor = [0.3, 0.3, 0.3]
    outline.LineWidth = 1.5
    outline.Opacity = 0.3
except:
    pass

# ============================================================
# Step 9: 渲染 + 截圖
# ============================================================
Render()

# 儲存截圖
output_dir = script_dir
output_png = os.path.join(output_dir, "vortex_structure.png")
SaveScreenshot(output_png, renderView,
               ImageResolution=IMG_SIZE,
               TransparentBackground=0)
print(f"\n=== Screenshot saved: {output_png} ===")

# ============================================================
# Step 10: 額外視角 — 俯視圖 (top view)
# ============================================================
camera.SetPosition(2.25, 4.5, 20)   # 從正上方看
camera.SetFocalPoint(*domain_center)
camera.SetViewUp(0, 1, 0)           # y 軸朝上
renderView.ResetCamera()

output_top = os.path.join(output_dir, "vortex_structure_top.png")
SaveScreenshot(output_top, renderView,
               ImageResolution=IMG_SIZE,
               TransparentBackground=0)
print(f"=== Top view saved: {output_top} ===")

# 恢復斜俯視
camera.SetPosition(12, -8, 12)
camera.SetFocalPoint(*domain_center)
camera.SetViewUp(0, 0, 1)
renderView.ResetCamera()
camera.Dolly(0.85)

print(f"""
======================================================
  渦流結構可視化完成！
  
  Vorticity Magnitude 範圍: [{vm_range[0]:.8f}, {vm_range[1]:.8f}]
  使用閾值: {Q_THRESHOLD:.8f}
  等值面 cells: {n_cells}
  色標: w velocity [{W_RANGE[0]}, {W_RANGE[1]}]
  
  調整建議:
  - 結構太少/看不到 → 降低 Q_THRESHOLD
  - 結構太多/糊成一團 → 提高 Q_THRESHOLD
  - 色標範圍 → 修改 W_RANGE
  
  輸出檔案:
  - {output_png}
  - {output_top}
======================================================
""")

# 完成 — 在 ParaView GUI 中可直接互動操作視圖
