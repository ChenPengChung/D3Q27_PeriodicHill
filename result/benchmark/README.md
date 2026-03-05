# Benchmark Data: Periodic Hill Re=700

## Source
Breuer, M., Peller, N., Rapp, Ch., Manhart, M. (2009).
"Flow over periodic hills – Numerical and experimental study in a wide range of Reynolds numbers."
*Computers & Fluids*, 38, pp. 433–457.

## Download
1. ERCOFTAC Knowledge Base Wiki (UFR 3-30):
   https://kbwiki.ercoftac.org/w/index.php?title=Abstr:2D_Periodic_Hill_Flow
   → 選擇 Re=700 → 下載 DNS mean velocity profiles

2. Breuer 研究室 (Universität der Bundeswehr München):
   https://www.unibw.de/lrt1 → Research → Data → Periodic Hill Flow

## File Format
每個檔案包含一個 ｄx/h 站位的垂直剖面：

```
# breuer_re700_xh05.dat  (x/h = 0.5)
# Column 1: (z - z_wall) / h   (wall-normal, normalized by hill height)
# Column 2: U / Ub             (streamwise velocity / bulk velocity)
0.0000  0.0000
0.0500  0.1234
...
```

## Naming Convention
| x/h station | Filename               |
|-------------|------------------------|
| 0.5         | breuer_re700_xh05.dat  |
| 2.0         | breuer_re700_xh20.dat  |
| 4.0         | breuer_re700_xh40.dat  |
| 6.0         | breuer_re700_xh60.dat  |
| 8.0         | breuer_re700_xh80.dat  |

## Alternative naming (also supported)
- `Re700_x0.5.dat`, `Re700_x2.0.dat`, ...
- `xh05.dat`, `xh20.dat`, ...
