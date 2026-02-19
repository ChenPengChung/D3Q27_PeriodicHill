#ifndef GILBM_PRECOMPUTE_H
#define GILBM_PRECOMPUTE_H

// Phase 1.5: GILBM RK2 upwind displacement precomputation (Imamura 2005 Eq. 19-20)
// Split into ξ-direction (constant for uniform y) and ζ-direction (space-varying).
//
// δξ[α] = dt · ẽ^ξ_α = dt · e_y[α] / dy   → constant per alpha (19 values)
// δζ[α,j,k] = dt · ẽ^ζ_α(k_half)           → RK2 midpoint evaluation [19*NYD6*NZ6]
//
// Called once during initialization; results copied to GPU.

// ============================================================================
// PrecomputeGILBM_DeltaXi: ξ-direction displacement (constant for uniform y)
// ============================================================================
// For uniform y-grid: dξ/dy = 1/dy (constant), dξ/dz = 0
// → ẽ^ξ_α = e_y[α] · (1/dy) = e_y[α] / dy
// → δξ[α] = dt · e_y[α] / dy
// No RK2 correction needed (metric is constant → midpoint = endpoint).
//
// When y becomes non-uniform, promote delta_xi from [19] to [19*NYD6*NZ6]
// and add RK2 midpoint interpolation in j-direction.
void PrecomputeGILBM_DeltaXi(
    double *delta_xi_h,    // 輸出: [19]，ξ 方向位移量（常數）
    double dy_val          // 輸入: uniform grid spacing dy = LY/(NY6-7)
) {
    // D3Q19 離散速度集（與 initialization.h 中一致）
    double e_y[19] = {
        0,
        0, 0, 1, -1, 0, 0,
        1, 1, -1, -1,
        0, 0, 0, 0,
        1, -1, 1, -1
    };

    for (int alpha = 0; alpha < 19; alpha++) {
        delta_xi_h[alpha] = dt * e_y[alpha] / dy_val;
    }
}

// ============================================================================
// PrecomputeGILBM_DeltaZeta: ζ-direction RK2 displacement (space-varying)
// ============================================================================
// Imamura 2005 Eq. 19-20:
//   Step 1: ẽ^ζ_α(k) = e_y·(dk/dy) + e_z·(dk/dz)  at current point
//   Step 2: k_half = k - 0.5·dt·ẽ^ζ_α(k)            RK2 midpoint
//   Step 3: Interpolate dk/dy, dk/dz at k_half
//   Step 4: δζ[α] = dt · ẽ^ζ_α(k_half)              full RK2 displacement
void PrecomputeGILBM_DeltaZeta(
    double *delta_zeta_h,    // 輸出: [19 * NYD6 * NZ6]，預計算的位移量
    const double *dk_dz_h,   // 輸入: 度量項 dk/dz [NYD6*NZ6]
    const double *dk_dy_h,   // 輸入: 度量項 dk/dy [NYD6*NZ6]
    int NYD6_local,
    int NZ6_local
) {
    // D3Q19 離散速度集
    double e[19][3] = {
        {0,0,0},
        {1,0,0},{-1,0,0},{0,1,0},{0,-1,0},{0,0,1},{0,0,-1},
        {1,1,0},{-1,1,0},{1,-1,0},{-1,-1,0},
        {1,0,1},{-1,0,1},{1,0,-1},{-1,0,-1},
        {0,1,1},{0,-1,1},{0,1,-1},{0,-1,-1}
    };

    int sz = NYD6_local * NZ6_local;

    // 將整個輸出陣列初始化為零
    for (int n = 0; n < 19 * sz; n++) {
        delta_zeta_h[n] = 0.0;
    }

    // 遍歷所有離散速度方向（alpha=0 為靜止方向，跳過）
    for (int alpha = 1; alpha < 19; alpha++) {
        // 若該方向在 y 和 z 分量皆為零，則逆變速度 e_tilde_zeta 恆為零
        if (e[alpha][1] == 0.0 && e[alpha][2] == 0.0) continue;

        for (int j = 3; j < NYD6_local - 4; j++) { // 跳過 y 方向 buffer layer
            for (int k = 2; k < NZ6_local - 2; k++) {
                int idx_jk = j * NZ6_local + k;

                // 步驟一：計算當前格點 (j, k) 處的 ζ 方向逆變速度
                double e_tilde_zeta0 = e[alpha][1] * dk_dy_h[idx_jk]
                                     + e[alpha][2] * dk_dz_h[idx_jk];

                // 步驟二：RK2 半步位移與中間點位置
                double dk_half = 0.5 * dt * e_tilde_zeta0;
                double k_half = (double)k - dk_half;

                // 在中間點位置進行度量項的線性插值
                int k_lo = (int)floor(k_half);
                if (k_lo < 2) k_lo = 2;
                if (k_lo > NZ6_local - 4) k_lo = NZ6_local - 4;
                double frac = k_half - (double)k_lo;
                if (frac < 0.0) frac = 0.0;
                if (frac > 1.0) frac = 1.0;

                int idx_lo = j * NZ6_local + k_lo;
                int idx_hi = j * NZ6_local + k_lo + 1;

                // 插值得到中間點處的度量項
                double dk_dy_half = (1.0 - frac) * dk_dy_h[idx_lo]
                                  + frac * dk_dy_h[idx_hi];
                double dk_dz_half = (1.0 - frac) * dk_dz_h[idx_lo]
                                  + frac * dk_dz_h[idx_hi];

                // 步驟三：計算中間點處的逆變速度
                double e_tilde_zeta_half = e[alpha][1] * dk_dy_half
                                         + e[alpha][2] * dk_dz_half;

                // 步驟四：完整 RK2 位移量（Imamura 2005 Eq.20）
                delta_zeta_h[alpha * sz + idx_jk] = dt * e_tilde_zeta_half;
            }
        }
    }
}

// ============================================================================
// Wrapper: precompute both ξ and ζ displacements
// ============================================================================
void PrecomputeGILBM_DeltaXiZeta(
    double *delta_xi_h,      // 輸出: [19]，ξ 方向位移量
    double *delta_zeta_h,    // 輸出: [19 * NYD6 * NZ6]，ζ 方向位移量
    const double *dk_dz_h,   // 輸入: 度量項 dk/dz [NYD6*NZ6]
    const double *dk_dy_h,   // 輸入: 度量項 dk/dy [NYD6*NZ6]
    int NYD6_local,
    int NZ6_local
) {
    double dy_val = LY / (double)(NY6 - 7);

    PrecomputeGILBM_DeltaXi(delta_xi_h, dy_val);
    PrecomputeGILBM_DeltaZeta(delta_zeta_h, dk_dz_h, dk_dy_h, NYD6_local, NZ6_local);
}

#endif
/*
在曲線座標下的遷移距離計算應該要分開編號，分裏量計算，或者創建二維度陣咧儲存
Delta[alpha][0=\eta] = dt*e_x[alpha] / dx
Delta[alpha][1=\xi]  = dt*e_y[alpha] / dy
Delta[alpha][2=\zeta][idx_xi] = 利用二階RK方法計算每個點不同，或許應該擴展陣陣列為每個空間計算點都存放一值
\documentclass[12pt,a4paper]{article}




%--- 套件 ---

\usepackage{fontenc}

\usepackage[utf8]{inputenc}

\usepackage{amsmath, amssymb, amsthm}

\usepackage{geometry}

\usepackage{booktabs}

\usepackage{listings}

\usepackage{xcolor}

\usepackage{hyperref}

\usepackage{ctex} % 若不需中文可移除




\geometry{margin=2.5cm}




%--- 程式碼樣式 ---

\lstset{

language=C++,

basicstyle=\ttfamily\small,

keywordstyle=\color{blue}\bfseries,

commentstyle=\color{gray},

stringstyle=\color{orange},

numbers=left,

numberstyle=\tiny\color{gray},

breaklines=true,

frame=single,

tabsize=4

}




\title{\textbf{GILBM Departure Point CFL 驗證與計算點佈局調整}\\

\large Prompt for AI-Assisted CFD Code Development}

\author{}

\date{}




\begin{document}

\maketitle




%=============================================================

\section*{背景與問題描述}

%=============================================================




在 GILBM（General Interpolation Lattice Boltzmann Method）曲線座標系下進行遷移步驟時，遷移前空間點（departure point）的位置需要透過逆變速度回推，但可能出現以下兩個耦合問題：




\subsection*{問題一：Departure Point 落入固體區域（山坡內部）}




當計算點 $k=3$（第一層內點，$k=2$ 為邊界計算點）向後追蹤 departure point 時，若逆變速度

$\tilde{e}_{\zeta}^{(\alpha)}$ 在 $\zeta$ 方向分量夠大，departure point 可能穿越邊界進入固體區域，這在物理上是錯誤的。




\subsection*{問題二：網格解析度不足以支撐 CFL 條件}




在 $\zeta$ 方向使用 Hyperbolic Tangent 拉伸後，靠近壁面的最小網格間距

\[

\Delta\zeta_{\min} = \zeta[k=4] - \zeta[k=3]

\]

若不滿足

\begin{equation}

\Delta\zeta_{\min} > \max_{\alpha,\,j}\!\left(\bigl|\tilde{e}_{\zeta}^{(\alpha)}\bigr|\cdot\Delta t\right),

\label{eq:cfl_cond}

\end{equation}

則 departure point 必然越過相鄰格點甚至穿入固體，插值失效。




%=============================================================

\section*{目標}

%=============================================================




設計一套\textbf{計算點佈局調整方案}，滿足以下條件：

\begin{enumerate}

\item 維持 $k=2$ 為邊界計算點（Boundary Node），$k=3$ 為第一層內點（Interior Node），對其做一般 Lagrange 插值。

\item 透過調整 Hyperbolic Tangent 拉伸參數 $a$，確保對所有 $j$（$\xi$ 列）、所有離散速度方向 $\alpha$，$\zeta$ 方向最小網格間距滿足 CFL 約束式~\eqref{eq:cfl_cond}。

\item 提供自動化驗證機制，在初始化階段即可偵測違規情形。

\end{enumerate}




%=============================================================

\section*{A\quad 判斷條件（數學表述）}

%=============================================================




對每一個 $j$，定義 $\zeta$ 方向的局部 CFL 數為：

\begin{equation}

\mathrm{CFL}_{\zeta}(j)

= \frac{\displaystyle\max_{\alpha}\bigl|\tilde{e}_{\zeta}^{(\alpha)}(j,\,k_{\min})\bigr|\cdot\Delta t}

{\Delta\zeta_{\min}(j)},

\end{equation}

其中

\begin{align}

\tilde{e}_{\zeta}^{(\alpha)}

&= e_y\,\frac{\partial\zeta}{\partial y}

+ e_z\,\frac{\partial\zeta}{\partial z}, \\[4pt]

\Delta\zeta_{\min}(j)

&= \zeta\!\left[j\cdot N_{Z6}+4\right]

\zeta\!\left[j\cdot N_{Z6}+3\right].

\end{align}




安全條件為

\begin{equation}

\boxed{\mathrm{CFL}_{\zeta}(j) < 1 \quad \forall\; j}

\qquad(\text{建議保留 20\% 安全餘裕：}\mathrm{CFL}_{\zeta} < 0.8).

\end{equation}




%=============================================================

\section*{B\quad 驗證函數（C++ 實作）}

%=============================================================




\begin{lstlisting}

bool ValidateDeparturePoints(

const double zeta_h, // 曲線座標 zeta [NYD6NZ6]

const double dzeta_dy_h, // 度量項 [NYD6NZ6]

const double *dzeta_dz_h,

int NYD6, int NZ6,

double dt,

double &CFL_max_out,

int &j_violate, int &k_violate, int &alpha_violate

) {

// D3Q19 離散速度

double e[19][3] = {

{0,0,0},

{1,0,0},{-1,0,0},{0,1,0},{0,-1,0},{0,0,1},{0,0,-1},

{1,1,0},{-1,1,0},{1,-1,0},{-1,-1,0},

{1,0,1},{-1,0,1},{1,0,-1},{-1,0,-1},

{0,1,1},{0,-1,1},{0,1,-1},{0,-1,-1}

};




double CFL_max = 0.0;

j_violate = -1; k_violate = -1; alpha_violate = -1;

bool valid = true;




for (int j = 3; j < NYD6 - 4; j++) {

// 最小網格間距：k=3 到 k=4

int idx3 = j * NZ6 + 3;

int idx4 = j * NZ6 + 4;

double dZeta_min = zeta_h[idx4] - zeta_h[idx3];




for (int alpha = 1; alpha < 19; alpha++) {

if (e[alpha][1] == 0.0 && e[alpha][2] == 0.0) continue;




// 在 k=3 處計算逆變速度

double e_tilde_zeta = e[alpha][1] * dzeta_dy_h[idx3]

+ e[alpha][2] * dzeta_dz_h[idx3];




double displacement = fabs(e_tilde_zeta) * dt;

double CFL_local = displacement / dZeta_min;




if (CFL_local > CFL_max) {

CFL_max = CFL_local;

j_violate = j;

k_violate = 3;

alpha_violate = alpha;

}




if (CFL_local >= 1.0) {

valid = false;

printf("[VIOLATION] j=%d, alpha=%d: CFL=%.4f, "

"|e_tilde_zeta|*dt=%.6f, dZeta_min=%.6f\n",

j, alpha, CFL_local, displacement, dZeta_min);

}

}

}




CFL_max_out = CFL_max;

printf("[CFL CHECK] Max CFL_zeta = %.4f at j=%d, k=%d, alpha=%d\n",

CFL_max, j_violate, k_violate, alpha_violate);

return valid;

}

\end{lstlisting}




%=============================================================

\section*{C\quad Hyperbolic Tangent 參數調整指引}

%=============================================================




$\zeta$ 方向的拉伸函數通常為：

\begin{equation}

\zeta(k) = L_z\!\left[1 - \frac{\tanh\!\left(a\cdot(1-k/N)\right)}{\tanh(a)}\right].

\end{equation}




靠近壁面（$k\approx 2$）的最小網格間距近似為：

\begin{equation}

\Delta\zeta_{\min} \approx \frac{L_z\,a}{\tanh(a)}\cdot\frac{1}{N}.

\end{equation}




要滿足 CFL 條件，需要：

\begin{equation}

a \;\leq\; \frac{\tanh(a)\cdot N\cdot\Delta\zeta_{\mathrm{safe}}}{L_z}.

\end{equation}




\textbf{實務調整流程：}

\begin{enumerate}

\item 從較小的 $a$ 值出發（拉伸程度較弱）。

\item 逐步增大 $a$（拉伸程度增強，壁面附近網格加密）。

\item 每次重新執行 \texttt{ValidateDeparturePoints}。

\item 直到 $\mathrm{CFL}_{\max} < 0.8$（保留 20\% 安全餘裕）為止。

\end{enumerate}




%=============================================================

\section*{D\quad 偵錯輸出建議}

%=============================================================




在初始化函數末尾加入：




\begin{lstlisting}

double CFL_max;

int jv, kv, av;

bool ok = ValidateDeparturePoints(

zeta_h, dzeta_dy_h, dzeta_dz_h,

NYD6, NZ6, dt, CFL_max, jv, kv, av);




if (!ok) {

fprintf(stderr,

"[ERROR] CFL violation detected! "

"Increase HypTan parameter 'a' or reduce dt.\n"

"Current max CFL = %.4f\n", CFL_max);

exit(EXIT_FAILURE);

}

\end{lstlisting}




%=============================================================

\section*{預期結果}

%=============================================================




\begin{center}

\begin{tabular}{lc}

\toprule

\textbf{驗證項目} & \textbf{通過條件} \\

\midrule

所有 $j$ 列 $\mathrm{CFL}_\zeta$ & $< 1.0$（建議 $< 0.8$） \\

Departure point $\zeta$ 座標 & $> \zeta[k=2]$（不穿入固體） \\

最小網格間距 & $> \max|\tilde{e}_\zeta|\cdot\Delta t$ \\

\bottomrule

\end{tabular}

\end{center}




\end{document}
*/
