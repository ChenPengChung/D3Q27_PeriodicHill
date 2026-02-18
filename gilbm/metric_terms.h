#ifndef METRIC_TERMS_FILE
#define METRIC_TERMS_FILE

#include <fstream>
#include <iomanip>
#include <cmath>
#include <cstdlib>

using namespace std ;

// Phase 0: 座標轉換度量項計算（Imamura 2005 左側元素）
// 計算座標 = 網格索引 (i, j, k)，均勻間距 Δ=1
//
// 度量項（物理→計算空間映射，用於逆變速度公式）：
//   ∂ζ/∂z = dk_dz = 1 / (∂z/∂k)
//   ∂ζ/∂y = dk_dy = -(∂z/∂j) / (dy · ∂z/∂k)
//
// 計算方法：先用中心差分求 ∂z/∂k, ∂z/∂j（正 Jacobian），再求逆得到度量項
// 最終存儲的是文獻所需的左側元素 ∂ζ/∂z, ∂ζ/∂y

void ComputeMetricTerms(
    double *dk_dz_h,    // output [NYD6*NZ6]
    double *dk_dy_h,    // output [NYD6*NZ6]
    const double *z_h,  // input  [NYD6*NZ6]
    const double *y_h,  // input  [NYD6]
    int NYD6_local,
    int NZ6_local
) {
    //公視告用Jacobian轉換
    double dy = y_h[4] - y_h[3];  // 均勻 Y 格距

    for (int j = 3; j < NYD6_local - 3; j++) {
        for (int k = 3; k < NZ6_local - 3; k++) {
            int idx = j * NZ6_local + k;

            // ∂z/∂k：Z 方向中心差分（固定 j）
            double dz_dk = (z_h[j * NZ6_local + (k + 1)] -
                            z_h[j * NZ6_local + (k - 1)]) / 2.0;

            // ∂z/∂j：Y 方向中心差分（固定 k）
            double dz_dj = (z_h[(j + 1) * NZ6_local + k] -
                            z_h[(j - 1) * NZ6_local + k]) / 2.0;

            // 度量項（左側元素）
            //J = (dz_dk * dy_dj) - (dz_dj*dy_dk) = dz_dk*dy - dz_dj*0 = dz_dk*dy 
            dk_dz_h[idx] = dy / (dy * dz_dk);
            dk_dy_h[idx] = -dz_dj / (dy * dz_dk);
        }
    }
}


// ======== Phase 0 診斷輸出 ========
// 在 rank 0 內部重建全域座標，計算全域度量項並輸出診斷文件
// 在 GenerateMesh_Z() 之後調用
void DiagnoseMetricTerms(int myid) {
    // 只在 rank 0 輸出
    if (myid != 0) return;

    int bfr = 3;
    double dy = LY / (double)(NY6 - 2*bfr - 1);
    double dx = LX / (double)(NX6 - 2*bfr - 1);

    // ====== 重建全域座標（與 GenerateMesh_Y/Z 相同公式）======
    double *y_g  = (double *)malloc(NY6 * sizeof(double));
    double *z_g  = (double *)malloc(NY6 * NZ6 * sizeof(double));
    double *dk_dz_g = (double *)malloc(NY6 * NZ6 * sizeof(double));
    double *dk_dy_g = (double *)malloc(NY6 * NZ6 * sizeof(double));

    // Y 座標
    for (int j = 0; j < NY6; j++) {
        y_g[j] = dy * ((double)(j - bfr));
    }

    // Z 座標（非均勻 tanh 拉伸 + 山丘地形）
    double a = GetNonuniParameter();
    for (int j = 0; j < NY6; j++) {
        double total = LZ - HillFunction(y_g[j]) - minSize;
        for (int k = bfr; k < NZ6 - bfr; k++) {
            z_g[j*NZ6+k] = tanhFunction(total, minSize, a, (k-3), (NZ6-7))
                         + HillFunction(y_g[j]);
        }
        z_g[j*NZ6+2] = HillFunction(y_g[j]);
        z_g[j*NZ6+(NZ6-3)] = (double)LZ;
    }

    // ====== 計算全域度量項 ======
    ComputeMetricTerms(dk_dz_g, dk_dy_g, z_g, y_g, NY6, NZ6);

    // ====== 輸出 1: 全場Jacibian轉換係數 ======
    ofstream fout("gilbm_metrics.dat");
    fout << "# j  k  y  z  H(y)  dz_dk  dk_dz  dz_dj  dk_dy  J\n";
    for (int j = bfr; j < NY6 - bfr; j++) {
        double Hy = HillFunction(y_g[j]);
        for (int k = bfr; k < NZ6 - bfr; k++) {
            int idx = j * NZ6 + k;
            double dz_dk = 1.0 / dk_dz_g[idx];
            double dz_dj = -dk_dy_g[idx] * dy * dz_dk;
            double J = dx * dy * dz_dk;  // Jacobian 行列式

            fout << setw(4) << j << " "
                 << setw(4) << k << " "
                 << setw(12) << fixed << setprecision(6) << y_g[j] << " "
                 << setw(12) << z_g[idx] << " "
                 << setw(12) << Hy << " "
                 << setw(12) << scientific << setprecision(6) << dz_dk << " "
                 << setw(12) << dk_dz_g[idx] << " "
                 << setw(12) << dz_dj << " "
                 << setw(12) << dk_dy_g[idx] << " "
                 << setw(12) << J << "\n";
        }
    }
    fout.close();

    // ====== 輸出 2: 選定位置的剖面 ======
    // 找出三個特徵 j 值（在全域範圍搜索）
    int j_flat = -1, j_peak = -1, j_slope = -1;
    double H_max = 0.0, dH_max = 0.0;

    for (int j = bfr ; j < NY6 - bfr - 1; j++) {
        double Hy = HillFunction(y_g[j]);
        //discrete dirivative of H(y) for slope detection
        double dHdy = (HillFunction(y_g[j + 1]) - HillFunction(y_g[j - 1])) / (2.0 * dy);
        
        if (Hy < 0.01 && j_flat < 0) j_flat = j; //尋找第一個平坦點 (H≈0)
        /*山丘高度為零，底壁是平的
        預期：∂z/∂j ≈ 0 → dk_dy ≈ 0，座標系退化為正交（無扭曲）
        驗證用途：判據 3 檢查 dk_dy ≈ 0；判據 5 檢查壁面 BC 方向恰好是標準的 5 個*/
        if (Hy > H_max) { H_max = Hy; j_peak = j; } //尋找山丘最高點 (argmax H)
        /*物理空間被壓縮最嚴重（天花板到山丘頂的間距最小）
        預期：dz_dk 最小（格點擠在一起）、dk_dz 最大
        驗證用途：輸出 2 的剖面圖，確認度量項在極端壓縮處的數值是否合理*/
        if (fabs(dHdy) > dH_max && Hy > 0.1) { dH_max = fabs(dHdy); j_slope = j; } //尋找最陡斜面 (argmax |H'|, 排除平坦段)的j值（正規化）
        /*座標系扭曲最嚴重的位置（z 網格線不再垂直，而是傾斜）
        預期 ：|dk_dy| 最大（座標耦合最強）；壁面 BC 需要額外方向（判據 6）
        驗證用途：如果這個最極端的位置度量項都正確，其他位置更不會有問題*/
    }

    cout << "\n===== Phase 0: Metric Terms Diagnostics (Global) =====\n";
    cout << "j_flat  = " << setw(4) << j_flat << "  (y=" << fixed << setprecision(4) << setw(8) << y_g[j_flat] << ", H=" << setw(8) << HillFunction(y_g[j_flat]) << ")\n";
    cout << "j_peak  = " << setw(4) << j_peak << "  (y=" << fixed << setprecision(4) << setw(8) << y_g[j_peak] << ", H=" << setw(8) << HillFunction(y_g[j_peak]) << ")\n";
    cout << "j_slope = " << setw(4) << j_slope << "  (y=" << fixed << setprecision(4) << setw(8) << y_g[j_slope] << ", H=" << setw(8) << HillFunction(y_g[j_slope]) << ", |H'|=" << setw(8) << dH_max << ")\n";


    // ====== 輸出 3: 壁面方向判別 ======
    // D3Q19 速度集
    double e[19][3] = {
        {0,0,0},
        {1,0,0},{-1,0,0},{0,1,0},{0,-1,0},{0,0,1},{0,0,-1},
        {1,1,0},{-1,1,0},{1,-1,0},{-1,-1,0},
        {1,0,1},{-1,0,1},{1,0,-1},{-1,0,-1},
        {0,1,1},{0,-1,1},{0,1,-1},{0,-1,-1}
    };

    ofstream fwall;
    fwall.open("gilbm_contravariant_wall.dat");

    fwall << "# j  y  H(y)  dH/dy  num_BC_dirs  BC_directions\n";

    int pass_flat_5dirs = 1;  // 用於判據 5
    int pass_slope_extra = 0; // 用於判據 6

    for (int j = bfr + 1; j < NY6 - bfr - 1; j++) {
        int k_wall = bfr;  // 底壁
        int idx = j * NZ6 + k_wall;
        double Hy = HillFunction(y_g[j]);
        double dHdy = (HillFunction(y_g[j + 1]) - HillFunction(y_g[j - 1])) / (2.0 * dy);

        int num_bc = 0;
        fwall << setw(4) << j << " "
              << setw(10) << fixed << setprecision(5) << y_g[j] << " "
              << setw(10) << Hy << " "
              << setw(10) << dHdy << "  ";

        // 暫存需要 BC 的方向
        int bc_dirs[19];
        for (int alpha = 0; alpha < 19; alpha++) {
            double e_tilde_zeta = e[alpha][1] * dk_dy_g[idx]
                                + e[alpha][2] * dk_dz_g[idx];
            if (e_tilde_zeta > 0.0) {
                bc_dirs[num_bc++] = alpha;
            }
        }

        fwall << setw(2) << num_bc << "  ";
        for (int n = 0; n < num_bc; n++) {
            fwall << bc_dirs[n] << " ";
        }
        fwall << "\n";

        // 判據 5：平坦段應恰好 5 方向 {5, 11, 12, 15, 16}
        if (fabs(Hy) < 0.01 && fabs(dHdy) < 0.01) {
            if (num_bc != 5) {
                pass_flat_5dirs = 0;
                printf("  FAIL criteria 5: j=%d (flat, H=%.4f), num_BC=%d (expected 5)\n",
                       j, Hy, num_bc);
            }
        }

        // 判據 6：斜面應有額外方向
        if (fabs(dHdy) > 0.1 && num_bc > 5) {
            pass_slope_extra = 1;
        }
    }
    fwall.close();

    // ====== Pass/Fail 判據匯總 ======
    printf("\n----- Pass/Fail Criteria -----\n");

    // 判據 1: dk_dz > 0 全場
    int pass1 = 1;
    for (int j = bfr; j < NY6 - bfr; j++) {
        for (int k = bfr; k < NZ6 - bfr; k++) {
            if (dk_dz_g[j * NZ6 + k] <= 0.0) {
                pass1 = 0;
                printf("  FAIL: dk_dz <= 0 at j=%d, k=%d\n", j, k);
            }
        }
    }
    printf("[%s] Criteria 1: dk_dz > 0 everywhere\n", pass1 ? "PASS" : "FAIL");

    // 判據 2: 壁面附近 dz_dk ≈ minSize
    int pass2 = 1;
    for (int j = bfr; j < NY6 - bfr; j++) {
        double dz_dk_wall = 1.0 / dk_dz_g[j * NZ6 + bfr];
        double rel_err = fabs(dz_dk_wall - minSize) / minSize;
        if (rel_err > 0.1) {
            pass2 = 0;
            printf("  FAIL: j=%d, dz_dk[wall]=%.6e, minSize=%.6e, rel_err=%.2f%%\n",
                   j, dz_dk_wall, minSize, rel_err * 100);
        }
    }
    printf("[%s] Criteria 2: dz_dk(wall) ≈ minSize (within 10%%)\n", pass2 ? "PASS" : "FAIL");

    // 判據 3: 平坦段 dk_dy ≈ 0
    int pass3 = 1;
    if (j_flat >= 0) {
        for (int k = bfr; k < NZ6 - bfr; k++) {
            if (fabs(dk_dy_g[j_flat * NZ6 + k]) > 0.1) {
                pass3 = 0;
                printf("  FAIL: flat region j=%d k=%d, dk_dy=%.6e (expected ~0)\n",
                       j_flat, k, dk_dy_g[j_flat * NZ6 + k]);
            }
        }
    }
    printf("[%s] Criteria 3: dk_dy ≈ 0 at flat region\n", pass3 ? "PASS" : "FAIL");

    // 判據 4: 斜面 dk_dy 符號正確
    int pass4 = 1;
    if (j_slope >= 0) {
        double dHdy_slope = (HillFunction(y_g[j_slope + 1]) -
                             HillFunction(y_g[j_slope - 1])) / (2.0 * dy);
        int k_mid = NZ6 / 2;
        double dk_dy_val = dk_dy_g[j_slope * NZ6 + k_mid];
        // 當 H'(y)>0（山丘上升段），dz_dj>0 → dk_dy<0
        if (dHdy_slope > 0 && dk_dy_val > 0) {
            pass4 = 0;
            printf("  FAIL: slope j=%d, H'>0 but dk_dy>0 (sign wrong)\n", j_slope);
        }
        if (dHdy_slope < 0 && dk_dy_val < 0) {
            pass4 = 0;
            printf("  FAIL: slope j=%d, H'<0 but dk_dy<0 (sign wrong)\n", j_slope);
        }
    }
    printf("[%s] Criteria 4: dk_dy sign consistent with -H'(y)\n", pass4 ? "PASS" : "FAIL");

    // 判據 5：平坦段壁面恰好 5 個方向需要 BC
    printf("[%s] Criteria 5: flat wall has exactly 5 BC directions\n",
           pass_flat_5dirs ? "PASS" : "FAIL");

    // 判據 6：斜面有額外方向
    printf("[%s] Criteria 6: slope wall has >5 BC directions\n",
           pass_slope_extra ? "PASS" : "WARN(may be ok if slope is gentle)");

    printf("\nDiagnostic files written:\n");
    printf("  gilbm_metrics.dat           — full field metric terms\n");
    printf("  gilbm_metrics_selected.dat  — profiles at 3 characteristic j\n");
    printf("  gilbm_contravariant_wall.dat — wall direction classification\n");
    printf("===== End Phase 0 Diagnostics =====\n\n");

    free(y_g);
    free(z_g);
    free(dk_dz_g);
    free(dk_dy_g);
}

#endif
