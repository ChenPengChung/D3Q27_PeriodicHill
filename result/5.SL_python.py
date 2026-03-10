# -*- coding: utf-8 -*-
"""
pvpython 離線渲染：YZ 中間剖面 Contour + 流線，直接輸出 PNG。
用法：
  pvpython render_yz_contour.py
"""
import os, sys, glob, math, time

# 指定檔名或 None：None 時自動偵測資料夾內「最新」的 .vtk（依修改時間）
VTK_FILE = None
FOLDER = os.path.dirname(os.path.abspath(__file__))
if len(sys.argv) > 1:
    FOLDER = os.path.abspath(sys.argv[1])
OUTPUT_PNG = os.path.join(FOLDER, "yz_contour_streamlines.png")
IMAGE_W, IMAGE_H = 2800, 800

NUM_UNIFORM_SEEDS = 30
# 渦流/回流區偵測：速度大小低於此百分位的區域視為渦流回流區
VORTEX_LOW_PERCENTILE = 10
MAX_VORTEX_SEEDS = 300
MAX_STREAMLINE_LEN = 20.0
SEED_STEP = 0.02
MAX_STEPS = 4000

def log(msg):
    print(msg, flush=True)

from paraview.simple import *

# ============================================================
# 1) 讀取（自動偵測最新 .vtk）
# ============================================================
candidates = sorted(glob.glob(os.path.join(FOLDER, "*.vtk")), key=os.path.getmtime)
if not candidates:
    log("ERROR: no .vtk in " + FOLDER); sys.exit(1)
if VTK_FILE:
    specified = os.path.join(FOLDER, VTK_FILE)
    vtk_path = specified if os.path.isfile(specified) else candidates[-1]
else:
    vtk_path = candidates[-1]
log("Loading (latest): " + vtk_path)

reader = LegacyVTKReader(FileNames=[vtk_path])
reader.UpdatePipeline()

bounds = reader.GetDataInformation().GetBounds()
xmin, xmax = bounds[0], bounds[1]
ymin, ymax = bounds[2], bounds[3]
zmin, zmax = bounds[4], bounds[5]
xmid = (xmin + xmax) * 0.5
log("Bounds  X:[%.3f,%.3f]  Y:[%.3f,%.3f]  Z:[%.3f,%.3f]" % (xmin,xmax,ymin,ymax,zmin,zmax))
log("Slice at X = %.3f" % xmid)

# ============================================================
# 2) 中間 YZ 剖面
# ============================================================
sliceF = Slice(Input=reader)
sliceF.SliceType.Normal = [1, 0, 0]
sliceF.SliceType.Origin = [xmid, (ymin+ymax)/2, (zmin+zmax)/2]
sliceF.UpdatePipeline()

# Y 方向速度分量 = 流向速度 u（對應參考圖的 u-bar）
calcU = Calculator(Input=sliceF)
calcU.ResultArrayName = "u_streamwise"
calcU.Function = "velocity_Y"
calcU.UpdatePipeline()

# 速度大小（供渦流偵測用）
calcMag = Calculator(Input=calcU)
calcMag.ResultArrayName = "VelMag"
calcMag.Function = "mag(velocity)"
calcMag.UpdatePipeline()

# ============================================================
# 3) 流線（入口處的垂直線種子，從左邊射入）
# ============================================================
lineSeed = Line()
lineSeed.Point1 = [xmid, ymin + 0.01, zmin]
lineSeed.Point2 = [xmid, ymin + 0.01, zmax]
lineSeed.Resolution = NUM_UNIFORM_SEEDS
log("Seed line: %d points at Y=%.3f" % (NUM_UNIFORM_SEEDS, ymin+0.01))

st1 = StreamTracerWithCustomSource(Input=reader, SeedSource=lineSeed)
st1.Vectors = ["POINTS", "velocity"]
st1.MaximumStreamlineLength = MAX_STREAMLINE_LEN
st1.InitialStepLength = SEED_STEP
st1.MaximumSteps = MAX_STEPS
st1.IntegrationDirection = "FORWARD"
st1.UpdatePipeline()
log("Uniform streamlines done")

# ============================================================
# 4) 渦流區加密流線
# ============================================================
st2 = None
try:
    from paraview.servermanager import Fetch
    data = Fetch(calcMag)
    arr = data.GetPointData().GetArray("VelMag") if data else None
    if arr:
        npts = arr.GetNumberOfTuples()
        vals = sorted([arr.GetValue(i) for i in range(npts)])
        # 低速區 = 回流/渦流區（速度大小在最低 15%）
        k = int((npts - 1) * VORTEX_LOW_PERCENTILE / 100.0)
        thresh_lo = vals[k]
        vmin = vals[0]
        if thresh_lo > vmin:
            threshF = Threshold(Input=calcMag)
            threshF.Scalars = ["POINTS", "VelMag"]
            threshF.LowerThreshold = vmin
            threshF.UpperThreshold = thresh_lo
            threshF.ThresholdMethod = "Between"
            threshF.UpdatePipeline()
            nv = threshF.GetDataInformation().GetNumberOfPoints()
            if nv > 0:
                seedSource = threshF
                if nv > MAX_VORTEX_SEEDS:
                    mask = MaskPoints(Input=threshF)
                    mask.OnRatio = max(2, nv // MAX_VORTEX_SEEDS)
                    mask.RandomSampling = 1
                    mask.UpdatePipeline()
                    seedSource = mask
                    nv = seedSource.GetDataInformation().GetNumberOfPoints()
                st2 = StreamTracerWithCustomSource(Input=reader, SeedSource=seedSource)
                st2.Vectors = ["POINTS", "velocity"]
                st2.MaximumStreamlineLength = MAX_STREAMLINE_LEN * 0.4
                st2.InitialStepLength = SEED_STEP * 0.5
                st2.MaximumSteps = MAX_STEPS
                st2.IntegrationDirection = "BOTH"
                st2.UpdatePipeline()
                log("Vortex (low-speed) streamlines: %d seeds" % nv)
            else:
                log("Low-speed points=%d, skip vortex seeds" % nv)
        else:
            log("No clear recirculation region detected")
except Exception as e:
    log("Vortex seed fallback: " + str(e))

# ============================================================
# 5) 離線渲染
# ============================================================
ren = CreateView("RenderView")
ren.ViewSize = [IMAGE_W, IMAGE_H]
ren.Background = [1, 1, 1]

# Contour 填色（u_streamwise，類似參考圖的 u-bar 藍→紅）
disp = Show(calcMag, ren)
disp.Representation = "Surface"
disp.ColorArrayName = ["POINTS", "u_streamwise"]

lut = GetColorTransferFunction("u_streamwise")
info = calcMag.GetDataInformation().GetPointDataInformation().GetArrayInformation("u_streamwise")
if info:
    lo = info.GetComponentRange(0)[0]
    hi = info.GetComponentRange(0)[1]
else:
    lo, hi = 0.0, 1.0
log("u_streamwise range: [%.4f, %.4f]" % (lo, hi))

lut.ColorSpace = "RGB"
lut.RGBPoints = [
    lo,                       0.0, 0.0, 0.5,
    lo + (hi-lo)*0.125,       0.0, 0.0, 1.0,
    lo + (hi-lo)*0.25,        0.0, 0.5, 1.0,
    lo + (hi-lo)*0.375,       0.0, 1.0, 1.0,
    lo + (hi-lo)*0.5,         0.5, 1.0, 0.5,
    lo + (hi-lo)*0.625,       1.0, 1.0, 0.0,
    lo + (hi-lo)*0.75,        1.0, 0.5, 0.0,
    lo + (hi-lo)*0.875,       1.0, 0.0, 0.0,
    hi,                       0.5, 0.0, 0.0,
]
disp.LookupTable = lut
disp.SetScalarBarVisibility(ren, True)

bar = GetScalarBar(lut, ren)
bar.Title = "u_streamwise"
bar.ComponentTitle = ""
bar.TitleFontSize = 18
bar.LabelFontSize = 14
bar.Orientation = "Vertical"
# 將 colorbar 移到 contour 外側（右邊緣）
bar.WindowLocation = "Any Location"
bar.Position = [0.90, 0.05]
bar.ScalarBarLength = 0.85

# 流線：黑色線條（在 jet 背景上清楚可見）
sd1 = Show(st1, ren)
sd1.Representation = "Wireframe"
sd1.LineWidth = 1.2
sd1.AmbientColor = [0.1, 0.1, 0.1]
sd1.DiffuseColor = [0.1, 0.1, 0.1]
ColorBy(sd1, None)
sd1.SetScalarBarVisibility(ren, False)

if st2:
    sd2 = Show(st2, ren)
    sd2.Representation = "Wireframe"
    sd2.LineWidth = 0.8
    sd2.AmbientColor = [0.15, 0.15, 0.15]
    sd2.DiffuseColor = [0.15, 0.15, 0.15]
    ColorBy(sd2, None)
    sd2.SetScalarBarVisibility(ren, False)

# 相機：正對 YZ 平面（Y 水平、Z 垂直）
cam = ren.GetActiveCamera()
cam.SetPosition(xmid + 20, (ymin+ymax)/2, (zmin+zmax)/2)
cam.SetFocalPoint(xmid,     (ymin+ymax)/2, (zmin+zmax)/2)
cam.SetViewUp(0, 0, 1)
cam.SetParallelProjection(True)
ResetCamera()
cam.SetParallelScale((zmax - zmin) * 0.55)

Render(ren)

# ============================================================
# 6) 儲存
# ============================================================
SaveScreenshot(OUTPUT_PNG, ren, ImageResolution=[IMAGE_W, IMAGE_H])
log("Image saved: " + OUTPUT_PNG)
log("Done.")
