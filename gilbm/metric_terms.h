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
    int pass_slope_extra = 1; // 用於判據 6（預設 PASS，任何斜面點 num_bc<=5 則 FAIL）
    int found_any_slope = 0;  // 是否找到任何斜面點 (|dHdy| > 0.1)

    for (int j = bfr ; j < NY6 - bfr - 1; j++) { //由左到右掃描
        int k_wall = bfr;  //壁面位置（k=3，對應 z=H(y)+0.5minSize）
        int idx = j * NZ6 + k_wall;//計算空間最下面一排的計算點
        double Hy = HillFunction(y_g[j]);
        double dHdy = (HillFunction(y_g[j + 1]) - HillFunction(y_g[j - 1])) / (2.0 * dy);

        //分別印出 編號：全域y座標：H(y)：dH/dy
        fwall << setw(4) << j << " "
              << setw(10) << fixed << setprecision(5) << y_g[j] << " "
              << setw(10) << Hy << " "
              << setw(10) << dHdy << "  ";
        //====外迴圈：掃描各個底層計算點，內迴圈：掃描不同離散速度的方向
        // 暫存需要 BC 的方向
        int num_bc = 0;
        int bc_dirs[19];
        //不同離散速度分開計算
        for (int alpha = 0; alpha < 19; alpha++) {
            //計算座標變換下的離散速度集k分量(k = zeta分量)
            double e_tilde_k = e[alpha][1] * dk_dy_g[idx] + e[alpha][2] * dk_dz_g[idx];//k = zeta 為方向座標變換
            if (e_tilde_k > 0.0) {//在哪一個y列的下面一點上，離散速度的zeta分量不為零，則此點為該編號alpha的邊界計算點
                // 將當前方向索引 alpha 記錄到邊界條件方向陣列 bc_dirs 中，
                // 同時將邊界條件計數器 num_bc 遞增 1。
                bc_dirs[num_bc] = alpha;
                num_bc = num_bc + 1;
            }
        }
        fwall << setw(2) << num_bc << "  "; //一共有幾個需要邊界處理 
        for (int n = 0; n < num_bc; n++) {
            fwall << bc_dirs[n] << " "; //引出哪一些編號需要邊界處理
        }
        fwall << "\n";
        // ============================================================================
        // 判據 5 (Criteria 5): 平坦區域方向數驗證
        // ----------------------------------------------------------------------------
        // 目的：驗證在幾何形狀的平坦段（水平區域），邊界條件應恰好使用 5 個方向
        // 
        // 判斷條件：
        //   - |Hy| < 0.01     : 高度值接近零，表示位於平坦區域
        //   - |dHdy| < 0.01   : 高度梯度接近零，表示無斜率變化
        // 
        // 預期結果：
        //   - 平坦段的邊界條件方向數應為 5，對應 D3Q27 中的方向 {5, 11, 12, 15, 16}
        //   - 這些方向代表與平坦底面相交的離散速度方向
        // 
        // 物理意義：
        //   在 LBM 的曲線邊界處理中，平坦表面只需處理垂直於表面的速度分量，
        //   因此邊界條件方向數是固定且已知的
        // ============================================================================
        // 判據 5：平坦段應恰好 5 方向 {5, 11, 12, 15, 16}
        if (fabs(Hy) < 0.01 && fabs(dHdy) < 0.01) { //平坦區段判斷條件
            if (num_bc != 5) {
                pass_flat_5dirs = 0;
                cout << " FAIL criteria 5: j=" << j << " (flat, H=" << fixed << setprecision(4) << Hy << "), num_BC=" << num_bc << " (expected 5)\n";
            }
        }//需要做邊界處理的編號為反向牆面編號
        // ============================================================================
        // 判據 6 (Criteria 6): 斜面額外方向驗證
        // ----------------------------------------------------------------------------
        // 目的：驗證在幾何形狀的斜面段，邊界條件應包含額外的方向
        // 
        // 判斷條件：
        //   - |dHdy| > 0.1    : 高度梯度顯著，表示存在斜率
        //   - num_bc > 5      : 邊界條件方向數超過平坦段的 5 個
        // 
        // 預期結果：
        //   - 斜面區域需要處理更多的離散速度方向
        //   - 因為傾斜表面會與更多的速度向量相交
        // 
        // 物理意義：
        //   在 GILBM（Generalized Interpolation-based LBM）中，曲線/斜面邊界
        //   需要額外的插值方向來準確表示流體與傾斜壁面的相互作用
        // ============================================================================
        // 判據 6：斜面應有額外方向 (num_bc > 5)
        if (fabs(dHdy) > 0.1) {  // 這是一個斜面點（山坡梯度顯著）
            found_any_slope = 1;
            if (num_bc <= 5) {  // 斜面點卻只有 ≤5 個方向 → 該下邊界計算點的某一個度量項可能有誤
                pass_slope_extra = 0;
                cout << "  FAIL criteria 6: j=" << j
                     << " (slope, |dH/dy|=" << fixed << setprecision(4) << fabs(dHdy)
                     << "), num_BC=" << num_bc << " (expected >5)\n"; 
            }
        }
    }
    fwall.close();







    // ====== Pass/Fail 判據匯總 ======
    cout << "\n----- Pass/Fail Criteria -----\n";

    // 判據 1: dk_dz > 0 全場 //因為每一個點都應該存在hyperbolic tangent 伸縮程度 
    int pass1 = 1;
    for (int j = bfr; j < NY6 - bfr; j++) {
        for (int k = bfr; k < NZ6 - bfr; k++) {
            if (dk_dz_g[j * NZ6 + k] <= 0.0) {
                pass1 = 0;
                cout << "  FAIL: dk_dz <= 0 at j=" << j << ", k=" << k << "\n";
            }
        }
    }
    cout << "[" << (pass1 ? "PASS" : "FAIL") << "] Criteria 1: dk_dz > 0 everywhere\n";

    // 判據 2:  下壁面附近 dz_dk ≈ minSize  = discrete jacobian \frac{z_h[idx_xi+1] - z_h[idx_xi應義-1]}{2}
    int pass2 = 1;
    for (int j = bfr; j < NY6 - bfr; j++) {
        double dz_dk_wall = 1.0 / dk_dz_g[j * NZ6 + bfr];//第四個度量係數在下邊界的判斷
        double rel_err = fabs(dz_dk_wall - minSize) / minSize;
        if (rel_err > 0.1) {
            pass2 = 0;
            cout << "  FAIL: j=" << j
                 << ", dz_dk[wall]=" << scientific << setprecision(6) << dz_dk_wall
                 << ", minSize=" << minSize
                 << ", rel_err=" << fixed << setprecision(2) << rel_err * 100 << "%\n";
        }
    }
    cout << "[" << (pass2 ? "PASS" : "FAIL") << "] Criteria 2: dz_dk(wall) ≈ minSize (within 10%)\n";

    // 判據 3: 平坦段 dk_dy ≈ 0（掃描所有平坦 j，與判據 5 同條件）
    //  由於山坡曲率存在所引起的度量係數，在平坦區域應趨近於零
    int pass3 = 1;
    for (int j = bfr; j < NY6 - bfr - 1; j++) {
        double Hy_c3 = HillFunction(y_g[j]);
        double dHdy_c3 = (HillFunction(y_g[j + 1]) - HillFunction(y_g[j - 1])) / (2.0 * dy);
        if (fabs(Hy_c3) < 0.01 && fabs(dHdy_c3) < 0.01) {//如果在平探區域內部// 同判據 5 的平坦條件
            for (int k = bfr; k < NZ6 - bfr; k++) {
                if (fabs(dk_dy_g[j * NZ6 + k]) > 0.1) {
                    pass3 = 0;
                    cout << "  FAIL: flat region j=" << j << " k=" << k
                         << ", dk_dy=" << scientific << setprecision(6) << dk_dy_g[j * NZ6 + k]
                         << " (expected ~0)\n";
                }
            }
        }
    }
    cout << "[" << (pass3 ? "PASS" : "FAIL") << "] Criteria 3: dk_dy ≈ 0 at flat region\n";

    // 判據 4: 斜面 dk_dy 符號正確//dk_dy為因為山坡曲率而存在的度量係數
    int pass4 = 1;
    if (j_slope >= 0) { // j_slope 為最陡峭區域的 j 值
        double dHdy_slope = (HillFunction(y_g[j_slope + 1]) -
                             HillFunction(y_g[j_slope - 1])) / (2.0 * dy); //該j值點的山坡導數
        int k_mid = NZ6 / 2;//選最陡峭ｊ點的垂直中點
        double dk_dy_val = dk_dy_g[j_slope * NZ6 + k_mid];
        // 當 H'(y)>0（山丘上升段），dz_dj>0 → dk_dy<0
        if (dHdy_slope > 0 && dk_dy_val > 0) { //如果最陡峭j值為上升階段，則度量係數應該要<0（因為Jacobian Determination <0)
            pass4 = 0;
            cout << "  FAIL: slope j=" << j_slope << ", H'>0 but dk_dy>0 (sign wrong)\n";
        }
        if (dHdy_slope < 0 && dk_dy_val < 0) { //如果最陡峭區域j值處在下降階段 , 則度量係數>0 (因為因為Jacobian Determination <0)
            pass4 = 0;
            cout << "  FAIL: slope j=" << j_slope << ", H'<0 but dk_dy<0 (sign wrong)\n";
        }
    }
    //判斷4的意義：判斷因為山坡曲率而存在的度量係數是否計算錯誤 。
    cout << "[" << (pass4 ? "PASS" : "FAIL") << "] Criteria 4: dk_dy sign consistent with -H'(y)\n";

    // 判據 5：平坦段壁面恰好 5 個方向需要 BC
    cout << "[" << (pass_flat_5dirs ? "PASS" : "FAIL") << "] Criteria 5: flat wall has exactly 5 BC directions\n";

    // 判據 6：斜面有額外方向（三態：PASS / FAIL / SKIP）
    if (found_any_slope) {
        cout << "[" << (pass_slope_extra ? "PASS" : "FAIL") << "] Criteria 6: slope wall has >5 BC directions\n";
    } else {
        cout << "[SKIP] Criteria 6: no significant slope found (|dH/dy| > 0.1)\n";
    }

    cout << "\nDiagnostic files written:\n";
    cout << "  gilbm_metrics.dat           — full field metric terms\n";
    cout << "  gilbm_metrics_selected.dat  — profiles at 3 characteristic j\n";
    cout << "  gilbm_contravariant_wall.dat — wall direction classification\n";
    cout << "===== End Phase 0 Diagnostics =====\n\n";

    free(y_g);
    free(z_g);
    free(dk_dz_g);
    free(dk_dy_g);
}

#endif
