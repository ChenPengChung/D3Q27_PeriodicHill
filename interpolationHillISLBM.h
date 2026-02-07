#ifndef INTERPOLATIONHILLISLBM_FILE
#define INTERPOLATIONHILLISLBM_FILE

#define Intrpl7(f1, a1, f2, a2, f3, a3, f4, a4, f5, a5, f6, a6, f7, a7)     \
(       \
    f1*a1 + f2*a2 + f3*a3 + f4*a4 + f5*a5 + f6*a6 + f7*a7   \
)

#define F0_Intrpl7(f, i, j, k)      \
    idx = j*nface + k*nline + i;    \
    F0_in = ( f[idx] );

#define F1_Intrpl7(f, i, j, k, i_c, j_c, k_c, idx_x, idx_y, idx_xi, x_0, x_1, x_2, x_3, x_4, x_5, x_6)    \
    idx = j*nface + k*nline + i_c;  \
    F1_in = Intrpl7( f[idx], x_0[idx_x], f[idx+1], x_1[idx_x], f[idx+2], x_2[idx_x], f[idx+3], x_3[idx_x], f[idx+4], x_4[idx_x], f[idx+5], x_5[idx_x], f[idx+6], x_6[idx_x] );

#define F2_Intrpl7(f, i, j, k, i_c, j_c, k_c, idx_x, idx_y, idx_xi, x_0, x_1, x_2, x_3, x_4, x_5, x_6)    \
    idx = j*nface + k*nline + i_c;  \
    F2_in = Intrpl7( f[idx], x_0[idx_x], f[idx+1], x_1[idx_x], f[idx+2], x_2[idx_x], f[idx+3], x_3[idx_x], f[idx+4], x_4[idx_x], f[idx+5], x_5[idx_x], f[idx+6], x_6[idx_x] );

#define F3_Intrpl7(f, i, j, k, i_c, j_c, k_c, idx_x, idx_y, idx_xi, y_0, y_1, y_2, y_3, y_4, y_5, y_6, xi_0, xi_1, xi_2, xi_3, xi_4, xi_5, xi_6)    \
    idx = j_c*nface + k_c*nline + i;    \
    F3_in = Intrpl7(    \
        Intrpl7( f[idx],         xi_0[idx_xi], f[idx+nline],         xi_1[idx_xi], f[idx+2*nline],         xi_2[idx_xi], f[idx+3*nline],         xi_3[idx_xi], f[idx+4*nline],         xi_4[idx_xi], f[idx+5*nline],         xi_5[idx_xi], f[idx+6*nline],         xi_6[idx_xi] ), y_0[idx_y],   \
        Intrpl7( f[idx+nface],   xi_0[idx_xi], f[idx+nline+nface],   xi_1[idx_xi], f[idx+2*nline+nface],   xi_2[idx_xi], f[idx+3*nline+nface],   xi_3[idx_xi], f[idx+4*nline+nface],   xi_4[idx_xi], f[idx+5*nline+nface],   xi_5[idx_xi], f[idx+6*nline+nface],   xi_6[idx_xi] ), y_1[idx_y],   \
        Intrpl7( f[idx+2*nface], xi_0[idx_xi], f[idx+nline+2*nface], xi_1[idx_xi], f[idx+2*nline+2*nface], xi_2[idx_xi], f[idx+3*nline+2*nface], xi_3[idx_xi], f[idx+4*nline+2*nface], xi_4[idx_xi], f[idx+5*nline+2*nface], xi_5[idx_xi], f[idx+6*nline+2*nface], xi_6[idx_xi] ), y_2[idx_y],   \
        Intrpl7( f[idx+3*nface], xi_0[idx_xi], f[idx+nline+3*nface], xi_1[idx_xi], f[idx+2*nline+3*nface], xi_2[idx_xi], f[idx+3*nline+3*nface], xi_3[idx_xi], f[idx+4*nline+3*nface], xi_4[idx_xi], f[idx+5*nline+3*nface], xi_5[idx_xi], f[idx+6*nline+3*nface], xi_6[idx_xi] ), y_3[idx_y],   \
        Intrpl7( f[idx+4*nface], xi_0[idx_xi], f[idx+nline+4*nface], xi_1[idx_xi], f[idx+2*nline+4*nface], xi_2[idx_xi], f[idx+3*nline+4*nface], xi_3[idx_xi], f[idx+4*nline+4*nface], xi_4[idx_xi], f[idx+5*nline+4*nface], xi_5[idx_xi], f[idx+6*nline+4*nface], xi_6[idx_xi] ), y_4[idx_y],   \
        Intrpl7( f[idx+5*nface], xi_0[idx_xi], f[idx+nline+5*nface], xi_1[idx_xi], f[idx+2*nline+5*nface], xi_2[idx_xi], f[idx+3*nline+5*nface], xi_3[idx_xi], f[idx+4*nline+5*nface], xi_4[idx_xi], f[idx+5*nline+5*nface], xi_5[idx_xi], f[idx+6*nline+5*nface], xi_6[idx_xi] ), y_5[idx_y],   \
        Intrpl7( f[idx+6*nface], xi_0[idx_xi], f[idx+nline+6*nface], xi_1[idx_xi], f[idx+2*nline+6*nface], xi_2[idx_xi], f[idx+3*nline+6*nface], xi_3[idx_xi], f[idx+4*nline+6*nface], xi_4[idx_xi], f[idx+5*nline+6*nface], xi_5[idx_xi], f[idx+6*nline+6*nface], xi_6[idx_xi] ), y_6[idx_y]    \
    );

#define F4_Intrpl7(f, i, j, k, i_c, j_c, k_c, idx_x, idx_y, idx_xi, y_0, y_1, y_2, y_3, y_4, y_5, y_6, xi_0, xi_1, xi_2, xi_3, xi_4, xi_5, xi_6)    \
    idx = j_c*nface + k_c*nline + i;    \
    F4_in = Intrpl7(    \
        Intrpl7( f[idx],         xi_0[idx_xi], f[idx+nline],         xi_1[idx_xi], f[idx+2*nline],         xi_2[idx_xi], f[idx+3*nline],         xi_3[idx_xi], f[idx+4*nline],         xi_4[idx_xi], f[idx+5*nline],         xi_5[idx_xi], f[idx+6*nline],         xi_6[idx_xi] ), y_0[idx_y],   \
        Intrpl7( f[idx+nface],   xi_0[idx_xi], f[idx+nline+nface],   xi_1[idx_xi], f[idx+2*nline+nface],   xi_2[idx_xi], f[idx+3*nline+nface],   xi_3[idx_xi], f[idx+4*nline+nface],   xi_4[idx_xi], f[idx+5*nline+nface],   xi_5[idx_xi], f[idx+6*nline+nface],   xi_6[idx_xi] ), y_1[idx_y],   \
        Intrpl7( f[idx+2*nface], xi_0[idx_xi], f[idx+nline+2*nface], xi_1[idx_xi], f[idx+2*nline+2*nface], xi_2[idx_xi], f[idx+3*nline+2*nface], xi_3[idx_xi], f[idx+4*nline+2*nface], xi_4[idx_xi], f[idx+5*nline+2*nface], xi_5[idx_xi], f[idx+6*nline+2*nface], xi_6[idx_xi] ), y_2[idx_y],   \
        Intrpl7( f[idx+3*nface], xi_0[idx_xi], f[idx+nline+3*nface], xi_1[idx_xi], f[idx+2*nline+3*nface], xi_2[idx_xi], f[idx+3*nline+3*nface], xi_3[idx_xi], f[idx+4*nline+3*nface], xi_4[idx_xi], f[idx+5*nline+3*nface], xi_5[idx_xi], f[idx+6*nline+3*nface], xi_6[idx_xi] ), y_3[idx_y],   \
        Intrpl7( f[idx+4*nface], xi_0[idx_xi], f[idx+nline+4*nface], xi_1[idx_xi], f[idx+2*nline+4*nface], xi_2[idx_xi], f[idx+3*nline+4*nface], xi_3[idx_xi], f[idx+4*nline+4*nface], xi_4[idx_xi], f[idx+5*nline+4*nface], xi_5[idx_xi], f[idx+6*nline+4*nface], xi_6[idx_xi] ), y_4[idx_y],   \
        Intrpl7( f[idx+5*nface], xi_0[idx_xi], f[idx+nline+5*nface], xi_1[idx_xi], f[idx+2*nline+5*nface], xi_2[idx_xi], f[idx+3*nline+5*nface], xi_3[idx_xi], f[idx+4*nline+5*nface], xi_4[idx_xi], f[idx+5*nline+5*nface], xi_5[idx_xi], f[idx+6*nline+5*nface], xi_6[idx_xi] ), y_5[idx_y],   \
        Intrpl7( f[idx+6*nface], xi_0[idx_xi], f[idx+nline+6*nface], xi_1[idx_xi], f[idx+2*nline+6*nface], xi_2[idx_xi], f[idx+3*nline+6*nface], xi_3[idx_xi], f[idx+4*nline+6*nface], xi_4[idx_xi], f[idx+5*nline+6*nface], xi_5[idx_xi], f[idx+6*nline+6*nface], xi_6[idx_xi] ), y_6[idx_y]    \
    );

#define F5_Intrpl7(f, i, j, k, i_c, j_c, k_c, idx_x, idx_y, idx_xi, xi_0, xi_1, xi_2, xi_3, xi_4, xi_5, xi_6)    \
    idx = j*nface + k_c*nline + i;    \
    F5_in = Intrpl7( f[idx], xi_0[idx_xi], f[idx+nline], xi_1[idx_xi], f[idx+2*nline], xi_2[idx_xi], f[idx+3*nline], xi_3[idx_xi], f[idx+4*nline], xi_4[idx_xi], f[idx+5*nline], xi_5[idx_xi], f[idx+6*nline], xi_6[idx_xi] );

#define F6_Intrpl7(f, i, j, k, i_c, j_c, k_c, idx_x, idx_y, idx_xi, xi_0, xi_1, xi_2, xi_3, xi_4, xi_5, xi_6)    \
    idx = j*nface + k_c*nline + i;    \
    F6_in = Intrpl7( f[idx], xi_0[idx_xi], f[idx+nline], xi_1[idx_xi], f[idx+2*nline], xi_2[idx_xi], f[idx+3*nline], xi_3[idx_xi], f[idx+4*nline], xi_4[idx_xi], f[idx+5*nline], xi_5[idx_xi], f[idx+6*nline], xi_6[idx_xi] );

#define X_Y_XI_Intrpl7(f, F_in, i, j, k, i_c, j_c, k_c, idx_x, idx_y, idx_xi, x_0, x_1, x_2, x_3, x_4, x_5, x_6, y_0, y_1, y_2, y_3, y_4, y_5, y_6, xi_0, xi_1, xi_2, xi_3, xi_4, xi_5, xi_6)     \
    idx = j_c*nface + k_c*nline + i_c;      \
    F_in = Intrpl7(         \
        Intrpl7(            \
            Intrpl7( f[idx],                 x_0[idx_x], f[idx+1],                 x_1[idx_x], f[idx+2],                 x_2[idx_x], f[idx+3],                 x_3[idx_x], f[idx+4],                 x_4[idx_x], f[idx+5],                 x_5[idx_x], f[idx+6],                 x_6[idx_x] ), y_0[idx_y],   \
            Intrpl7( f[idx+nface],           x_0[idx_x], f[idx+1+nface],           x_1[idx_x], f[idx+2+nface],           x_2[idx_x], f[idx+3+nface],           x_3[idx_x], f[idx+4+nface],           x_4[idx_x], f[idx+5+nface],           x_5[idx_x], f[idx+6+nface],           x_6[idx_x] ), y_1[idx_y],   \
            Intrpl7( f[idx+2*nface],         x_0[idx_x], f[idx+1+2*nface],         x_1[idx_x], f[idx+2+2*nface],         x_2[idx_x], f[idx+3+2*nface],         x_3[idx_x], f[idx+4+2*nface],         x_4[idx_x], f[idx+5+2*nface],         x_5[idx_x], f[idx+6+2*nface],         x_6[idx_x] ), y_2[idx_y],   \
            Intrpl7( f[idx+3*nface],         x_0[idx_x], f[idx+1+3*nface],         x_1[idx_x], f[idx+2+3*nface],         x_2[idx_x], f[idx+3+3*nface],         x_3[idx_x], f[idx+4+3*nface],         x_4[idx_x], f[idx+5+3*nface],         x_5[idx_x], f[idx+6+3*nface],         x_6[idx_x] ), y_3[idx_y],   \
            Intrpl7( f[idx+4*nface],         x_0[idx_x], f[idx+1+4*nface],         x_1[idx_x], f[idx+2+4*nface],         x_2[idx_x], f[idx+3+4*nface],         x_3[idx_x], f[idx+4+4*nface],         x_4[idx_x], f[idx+5+4*nface],         x_5[idx_x], f[idx+6+4*nface],         x_6[idx_x] ), y_4[idx_y],   \
            Intrpl7( f[idx+5*nface],         x_0[idx_x], f[idx+1+5*nface],         x_1[idx_x], f[idx+2+5*nface],         x_2[idx_x], f[idx+3+5*nface],         x_3[idx_x], f[idx+4+5*nface],         x_4[idx_x], f[idx+5+5*nface],         x_5[idx_x], f[idx+6+5*nface],         x_6[idx_x] ), y_5[idx_y],   \
            Intrpl7( f[idx+6*nface],         x_0[idx_x], f[idx+1+6*nface],         x_1[idx_x], f[idx+2+6*nface],         x_2[idx_x], f[idx+3+6*nface],         x_3[idx_x], f[idx+4+6*nface],         x_4[idx_x], f[idx+5+6*nface],         x_5[idx_x], f[idx+6+6*nface],         x_6[idx_x] ), y_6[idx_y]    \
        ), xi_0[idx_xi],    \
        Intrpl7(            \
            Intrpl7( f[idx+nline],           x_0[idx_x], f[idx+1+nline],           x_1[idx_x], f[idx+2+nline],           x_2[idx_x], f[idx+3+nline],           x_3[idx_x], f[idx+4+nline],           x_4[idx_x], f[idx+5+nline],           x_5[idx_x], f[idx+6+nline],           x_6[idx_x] ), y_0[idx_y],  \
            Intrpl7( f[idx+nline+nface],     x_0[idx_x], f[idx+1+nline+nface],     x_1[idx_x], f[idx+2+nline+nface],     x_2[idx_x], f[idx+3+nline+nface],     x_3[idx_x], f[idx+4+nline+nface],     x_4[idx_x], f[idx+5+nline+nface],     x_5[idx_x], f[idx+6+nline+nface],     x_6[idx_x] ), y_1[idx_y],  \
            Intrpl7( f[idx+nline+2*nface],   x_0[idx_x], f[idx+1+nline+2*nface],   x_1[idx_x], f[idx+2+nline+2*nface],   x_2[idx_x], f[idx+3+nline+2*nface],   x_3[idx_x], f[idx+4+nline+2*nface],   x_4[idx_x], f[idx+5+nline+2*nface],   x_5[idx_x], f[idx+6+nline+2*nface],   x_6[idx_x] ), y_2[idx_y],  \
            Intrpl7( f[idx+nline+3*nface],   x_0[idx_x], f[idx+1+nline+3*nface],   x_1[idx_x], f[idx+2+nline+3*nface],   x_2[idx_x], f[idx+3+nline+3*nface],   x_3[idx_x], f[idx+4+nline+3*nface],   x_4[idx_x], f[idx+5+nline+3*nface],   x_5[idx_x], f[idx+6+nline+3*nface],   x_6[idx_x] ), y_3[idx_y],  \
            Intrpl7( f[idx+nline+4*nface],   x_0[idx_x], f[idx+1+nline+4*nface],   x_1[idx_x], f[idx+2+nline+4*nface],   x_2[idx_x], f[idx+3+nline+4*nface],   x_3[idx_x], f[idx+4+nline+4*nface],   x_4[idx_x], f[idx+5+nline+4*nface],   x_5[idx_x], f[idx+6+nline+4*nface],   x_6[idx_x] ), y_4[idx_y],  \
            Intrpl7( f[idx+nline+5*nface],   x_0[idx_x], f[idx+1+nline+5*nface],   x_1[idx_x], f[idx+2+nline+5*nface],   x_2[idx_x], f[idx+3+nline+5*nface],   x_3[idx_x], f[idx+4+nline+5*nface],   x_4[idx_x], f[idx+5+nline+5*nface],   x_5[idx_x], f[idx+6+nline+5*nface],   x_6[idx_x] ), y_5[idx_y],  \
            Intrpl7( f[idx+nline+6*nface],   x_0[idx_x], f[idx+1+nline+6*nface],   x_1[idx_x], f[idx+2+nline+6*nface],   x_2[idx_x], f[idx+3+nline+6*nface],   x_3[idx_x], f[idx+4+nline+6*nface],   x_4[idx_x], f[idx+5+nline+6*nface],   x_5[idx_x], f[idx+6+nline+6*nface],   x_6[idx_x] ), y_6[idx_y]   \
        ), xi_1[idx_xi],    \
        Intrpl7(            \
            Intrpl7( f[idx+2*nline],         x_0[idx_x], f[idx+1+2*nline],         x_1[idx_x], f[idx+2+2*nline],         x_2[idx_x], f[idx+3+2*nline],         x_3[idx_x], f[idx+4+2*nline],         x_4[idx_x], f[idx+5+2*nline],         x_5[idx_x], f[idx+6+2*nline],         x_6[idx_x] ), y_0[idx_y],  \
            Intrpl7( f[idx+2*nline+nface],   x_0[idx_x], f[idx+1+2*nline+nface],   x_1[idx_x], f[idx+2+2*nline+nface],   x_2[idx_x], f[idx+3+2*nline+nface],   x_3[idx_x], f[idx+4+2*nline+nface],   x_4[idx_x], f[idx+5+2*nline+nface],   x_5[idx_x], f[idx+6+2*nline+nface],   x_6[idx_x] ), y_1[idx_y],  \
            Intrpl7( f[idx+2*nline+2*nface], x_0[idx_x], f[idx+1+2*nline+2*nface], x_1[idx_x], f[idx+2+2*nline+2*nface], x_2[idx_x], f[idx+3+2*nline+2*nface], x_3[idx_x], f[idx+4+2*nline+2*nface], x_4[idx_x], f[idx+5+2*nline+2*nface], x_5[idx_x], f[idx+6+2*nline+2*nface], x_6[idx_x] ), y_2[idx_y],  \
            Intrpl7( f[idx+2*nline+3*nface], x_0[idx_x], f[idx+1+2*nline+3*nface], x_1[idx_x], f[idx+2+2*nline+3*nface], x_2[idx_x], f[idx+3+2*nline+3*nface], x_3[idx_x], f[idx+4+2*nline+3*nface], x_4[idx_x], f[idx+5+2*nline+3*nface], x_5[idx_x], f[idx+6+2*nline+3*nface], x_6[idx_x] ), y_3[idx_y],  \
            Intrpl7( f[idx+2*nline+4*nface], x_0[idx_x], f[idx+1+2*nline+4*nface], x_1[idx_x], f[idx+2+2*nline+4*nface], x_2[idx_x], f[idx+3+2*nline+4*nface], x_3[idx_x], f[idx+4+2*nline+4*nface], x_4[idx_x], f[idx+5+2*nline+4*nface], x_5[idx_x], f[idx+6+2*nline+4*nface], x_6[idx_x] ), y_4[idx_y],  \
            Intrpl7( f[idx+2*nline+5*nface], x_0[idx_x], f[idx+1+2*nline+5*nface], x_1[idx_x], f[idx+2+2*nline+5*nface], x_2[idx_x], f[idx+3+2*nline+5*nface], x_3[idx_x], f[idx+4+2*nline+5*nface], x_4[idx_x], f[idx+5+2*nline+5*nface], x_5[idx_x], f[idx+6+2*nline+5*nface], x_6[idx_x] ), y_5[idx_y],  \
            Intrpl7( f[idx+2*nline+6*nface], x_0[idx_x], f[idx+1+2*nline+6*nface], x_1[idx_x], f[idx+2+2*nline+6*nface], x_2[idx_x], f[idx+3+2*nline+6*nface], x_3[idx_x], f[idx+4+2*nline+6*nface], x_4[idx_x], f[idx+5+2*nline+6*nface], x_5[idx_x], f[idx+6+2*nline+6*nface], x_6[idx_x] ), y_6[idx_y]   \
        ), xi_2[idx_xi],    \
        Intrpl7(            \
            Intrpl7( f[idx+3*nline],         x_0[idx_x], f[idx+1+3*nline],         x_1[idx_x], f[idx+2+3*nline],         x_2[idx_x], f[idx+3+3*nline],         x_3[idx_x], f[idx+4+3*nline],         x_4[idx_x], f[idx+5+3*nline],         x_5[idx_x], f[idx+6+3*nline],         x_6[idx_x] ), y_0[idx_y],  \
            Intrpl7( f[idx+3*nline+nface],   x_0[idx_x], f[idx+1+3*nline+nface],   x_1[idx_x], f[idx+2+3*nline+nface],   x_2[idx_x], f[idx+3+3*nline+nface],   x_3[idx_x], f[idx+4+3*nline+nface],   x_4[idx_x], f[idx+5+3*nline+nface],   x_5[idx_x], f[idx+6+3*nline+nface],   x_6[idx_x] ), y_1[idx_y],  \
            Intrpl7( f[idx+3*nline+2*nface], x_0[idx_x], f[idx+1+3*nline+2*nface], x_1[idx_x], f[idx+2+3*nline+2*nface], x_2[idx_x], f[idx+3+3*nline+2*nface], x_3[idx_x], f[idx+4+3*nline+2*nface], x_4[idx_x], f[idx+5+3*nline+2*nface], x_5[idx_x], f[idx+6+3*nline+2*nface], x_6[idx_x] ), y_2[idx_y],  \
            Intrpl7( f[idx+3*nline+3*nface], x_0[idx_x], f[idx+1+3*nline+3*nface], x_1[idx_x], f[idx+2+3*nline+3*nface], x_2[idx_x], f[idx+3+3*nline+3*nface], x_3[idx_x], f[idx+4+3*nline+3*nface], x_4[idx_x], f[idx+5+3*nline+3*nface], x_5[idx_x], f[idx+6+3*nline+3*nface], x_6[idx_x] ), y_3[idx_y],  \
            Intrpl7( f[idx+3*nline+4*nface], x_0[idx_x], f[idx+1+3*nline+4*nface], x_1[idx_x], f[idx+2+3*nline+4*nface], x_2[idx_x], f[idx+3+3*nline+4*nface], x_3[idx_x], f[idx+4+3*nline+4*nface], x_4[idx_x], f[idx+5+3*nline+4*nface], x_5[idx_x], f[idx+6+3*nline+4*nface], x_6[idx_x] ), y_4[idx_y],  \
            Intrpl7( f[idx+3*nline+5*nface], x_0[idx_x], f[idx+1+3*nline+5*nface], x_1[idx_x], f[idx+2+3*nline+5*nface], x_2[idx_x], f[idx+3+3*nline+5*nface], x_3[idx_x], f[idx+4+3*nline+5*nface], x_4[idx_x], f[idx+5+3*nline+5*nface], x_5[idx_x], f[idx+6+3*nline+5*nface], x_6[idx_x] ), y_5[idx_y],  \
            Intrpl7( f[idx+3*nline+6*nface], x_0[idx_x], f[idx+1+3*nline+6*nface], x_1[idx_x], f[idx+2+3*nline+6*nface], x_2[idx_x], f[idx+3+3*nline+6*nface], x_3[idx_x], f[idx+4+3*nline+6*nface], x_4[idx_x], f[idx+5+3*nline+6*nface], x_5[idx_x], f[idx+6+3*nline+6*nface], x_6[idx_x] ), y_6[idx_y]   \
        ), xi_3[idx_xi],    \
        Intrpl7(            \
            Intrpl7( f[idx+4*nline],         x_0[idx_x], f[idx+1+4*nline],         x_1[idx_x], f[idx+2+4*nline],         x_2[idx_x], f[idx+3+4*nline],         x_3[idx_x], f[idx+4+4*nline],         x_4[idx_x], f[idx+5+4*nline],         x_5[idx_x], f[idx+6+4*nline],         x_6[idx_x] ), y_0[idx_y],  \
            Intrpl7( f[idx+4*nline+nface],   x_0[idx_x], f[idx+1+4*nline+nface],   x_1[idx_x], f[idx+2+4*nline+nface],   x_2[idx_x], f[idx+3+4*nline+nface],   x_3[idx_x], f[idx+4+4*nline+nface],   x_4[idx_x], f[idx+5+4*nline+nface],   x_5[idx_x], f[idx+6+4*nline+nface],   x_6[idx_x] ), y_1[idx_y],  \
            Intrpl7( f[idx+4*nline+2*nface], x_0[idx_x], f[idx+1+4*nline+2*nface], x_1[idx_x], f[idx+2+4*nline+2*nface], x_2[idx_x], f[idx+3+4*nline+2*nface], x_3[idx_x], f[idx+4+4*nline+2*nface], x_4[idx_x], f[idx+5+4*nline+2*nface], x_5[idx_x], f[idx+6+4*nline+2*nface], x_6[idx_x] ), y_2[idx_y],  \
            Intrpl7( f[idx+4*nline+3*nface], x_0[idx_x], f[idx+1+4*nline+3*nface], x_1[idx_x], f[idx+2+4*nline+3*nface], x_2[idx_x], f[idx+3+4*nline+3*nface], x_3[idx_x], f[idx+4+4*nline+3*nface], x_4[idx_x], f[idx+5+4*nline+3*nface], x_5[idx_x], f[idx+6+4*nline+3*nface], x_6[idx_x] ), y_3[idx_y],  \
            Intrpl7( f[idx+4*nline+4*nface], x_0[idx_x], f[idx+1+4*nline+4*nface], x_1[idx_x], f[idx+2+4*nline+4*nface], x_2[idx_x], f[idx+3+4*nline+4*nface], x_3[idx_x], f[idx+4+4*nline+4*nface], x_4[idx_x], f[idx+5+4*nline+4*nface], x_5[idx_x], f[idx+6+4*nline+4*nface], x_6[idx_x] ), y_4[idx_y],  \
            Intrpl7( f[idx+4*nline+5*nface], x_0[idx_x], f[idx+1+4*nline+5*nface], x_1[idx_x], f[idx+2+4*nline+5*nface], x_2[idx_x], f[idx+3+4*nline+5*nface], x_3[idx_x], f[idx+4+4*nline+5*nface], x_4[idx_x], f[idx+5+4*nline+5*nface], x_5[idx_x], f[idx+6+4*nline+5*nface], x_6[idx_x] ), y_5[idx_y],  \
            Intrpl7( f[idx+4*nline+6*nface], x_0[idx_x], f[idx+1+4*nline+6*nface], x_1[idx_x], f[idx+2+4*nline+6*nface], x_2[idx_x], f[idx+3+4*nline+6*nface], x_3[idx_x], f[idx+4+4*nline+6*nface], x_4[idx_x], f[idx+5+4*nline+6*nface], x_5[idx_x], f[idx+6+4*nline+6*nface], x_6[idx_x] ), y_6[idx_y]   \
        ), xi_4[idx_xi],    \
        Intrpl7(            \
            Intrpl7( f[idx+5*nline],         x_0[idx_x], f[idx+1+5*nline],         x_1[idx_x], f[idx+2+5*nline],         x_2[idx_x], f[idx+3+5*nline],         x_3[idx_x], f[idx+4+5*nline],         x_4[idx_x], f[idx+5+5*nline],         x_5[idx_x], f[idx+6+5*nline],         x_6[idx_x] ), y_0[idx_y],    \
            Intrpl7( f[idx+5*nline+nface],   x_0[idx_x], f[idx+1+5*nline+nface],   x_1[idx_x], f[idx+2+5*nline+nface],   x_2[idx_x], f[idx+3+5*nline+nface],   x_3[idx_x], f[idx+4+5*nline+nface],   x_4[idx_x], f[idx+5+5*nline+nface],   x_5[idx_x], f[idx+6+5*nline+nface],   x_6[idx_x] ), y_1[idx_y],    \
            Intrpl7( f[idx+5*nline+2*nface], x_0[idx_x], f[idx+1+5*nline+2*nface], x_1[idx_x], f[idx+2+5*nline+2*nface], x_2[idx_x], f[idx+3+5*nline+2*nface], x_3[idx_x], f[idx+4+5*nline+2*nface], x_4[idx_x], f[idx+5+5*nline+2*nface], x_5[idx_x], f[idx+6+5*nline+2*nface], x_6[idx_x] ), y_2[idx_y],    \
            Intrpl7( f[idx+5*nline+3*nface], x_0[idx_x], f[idx+1+5*nline+3*nface], x_1[idx_x], f[idx+2+5*nline+3*nface], x_2[idx_x], f[idx+3+5*nline+3*nface], x_3[idx_x], f[idx+4+5*nline+3*nface], x_4[idx_x], f[idx+5+5*nline+3*nface], x_5[idx_x], f[idx+6+5*nline+3*nface], x_6[idx_x] ), y_3[idx_y],    \
            Intrpl7( f[idx+5*nline+4*nface], x_0[idx_x], f[idx+1+5*nline+4*nface], x_1[idx_x], f[idx+2+5*nline+4*nface], x_2[idx_x], f[idx+3+5*nline+4*nface], x_3[idx_x], f[idx+4+5*nline+4*nface], x_4[idx_x], f[idx+5+5*nline+4*nface], x_5[idx_x], f[idx+6+5*nline+4*nface], x_6[idx_x] ), y_4[idx_y],    \
            Intrpl7( f[idx+5*nline+5*nface], x_0[idx_x], f[idx+1+5*nline+5*nface], x_1[idx_x], f[idx+2+5*nline+5*nface], x_2[idx_x], f[idx+3+5*nline+5*nface], x_3[idx_x], f[idx+4+5*nline+5*nface], x_4[idx_x], f[idx+5+5*nline+5*nface], x_5[idx_x], f[idx+6+5*nline+5*nface], x_6[idx_x] ), y_5[idx_y],    \
            Intrpl7( f[idx+5*nline+6*nface], x_0[idx_x], f[idx+1+5*nline+6*nface], x_1[idx_x], f[idx+2+5*nline+6*nface], x_2[idx_x], f[idx+3+5*nline+6*nface], x_3[idx_x], f[idx+4+5*nline+6*nface], x_4[idx_x], f[idx+5+5*nline+6*nface], x_5[idx_x], f[idx+6+5*nline+6*nface], x_6[idx_x] ), y_6[idx_y]     \
        ), xi_5[idx_xi],    \
        Intrpl7(            \
            Intrpl7( f[idx+6*nline],         x_0[idx_x], f[idx+1+6*nline],         x_1[idx_x], f[idx+2+6*nline],         x_2[idx_x], f[idx+3+6*nline],         x_3[idx_x], f[idx+4+6*nline],         x_4[idx_x], f[idx+5+6*nline],         x_5[idx_x], f[idx+6+6*nline],         x_6[idx_x] ), y_0[idx_y],  \
            Intrpl7( f[idx+6*nline+nface],   x_0[idx_x], f[idx+1+6*nline+nface],   x_1[idx_x], f[idx+2+6*nline+nface],   x_2[idx_x], f[idx+3+6*nline+nface],   x_3[idx_x], f[idx+4+6*nline+nface],   x_4[idx_x], f[idx+5+6*nline+nface],   x_5[idx_x], f[idx+6+6*nline+nface],   x_6[idx_x] ), y_1[idx_y],  \
            Intrpl7( f[idx+6*nline+2*nface], x_0[idx_x], f[idx+1+6*nline+2*nface], x_1[idx_x], f[idx+2+6*nline+2*nface], x_2[idx_x], f[idx+3+6*nline+2*nface], x_3[idx_x], f[idx+4+6*nline+2*nface], x_4[idx_x], f[idx+5+6*nline+2*nface], x_5[idx_x], f[idx+6+6*nline+2*nface], x_6[idx_x] ), y_2[idx_y],  \
            Intrpl7( f[idx+6*nline+3*nface], x_0[idx_x], f[idx+1+6*nline+3*nface], x_1[idx_x], f[idx+2+6*nline+3*nface], x_2[idx_x], f[idx+3+6*nline+3*nface], x_3[idx_x], f[idx+4+6*nline+3*nface], x_4[idx_x], f[idx+5+6*nline+3*nface], x_5[idx_x], f[idx+6+6*nline+3*nface], x_6[idx_x] ), y_3[idx_y],  \
            Intrpl7( f[idx+6*nline+4*nface], x_0[idx_x], f[idx+1+6*nline+4*nface], x_1[idx_x], f[idx+2+6*nline+4*nface], x_2[idx_x], f[idx+3+6*nline+4*nface], x_3[idx_x], f[idx+4+6*nline+4*nface], x_4[idx_x], f[idx+5+6*nline+4*nface], x_5[idx_x], f[idx+6+6*nline+4*nface], x_6[idx_x] ), y_4[idx_y],  \
            Intrpl7( f[idx+6*nline+5*nface], x_0[idx_x], f[idx+1+6*nline+5*nface], x_1[idx_x], f[idx+2+6*nline+5*nface], x_2[idx_x], f[idx+3+6*nline+5*nface], x_3[idx_x], f[idx+4+6*nline+5*nface], x_4[idx_x], f[idx+5+6*nline+5*nface], x_5[idx_x], f[idx+6+6*nline+5*nface], x_6[idx_x] ), y_5[idx_y],  \
            Intrpl7( f[idx+6*nline+6*nface], x_0[idx_x], f[idx+1+6*nline+6*nface], x_1[idx_x], f[idx+2+6*nline+6*nface], x_2[idx_x], f[idx+3+6*nline+6*nface], x_3[idx_x], f[idx+4+6*nline+6*nface], x_4[idx_x], f[idx+5+6*nline+6*nface], x_5[idx_x], f[idx+6+6*nline+6*nface], x_6[idx_x] ), y_6[idx_y]   \
        ), xi_6[idx_xi]     \
    );

#define X_XI_Intrpl7(f, F_in, i, j, k, i_c, j_c, k_c, idx_x, idx_y, idx_xi, x_0, x_1, x_2, x_3, x_4, x_5, x_6, xi_0, xi_1, xi_2, xi_3, xi_4, xi_5, xi_6)  \
    idx = j*nface + k_c*nline + i_c;  \
    F_in = Intrpl7( \
        Intrpl7( f[idx],         x_0[idx_x], f[idx+1],         x_1[idx_x], f[idx+2],         x_2[idx_x], f[idx+3],         x_3[idx_x], f[idx+4],         x_4[idx_x], f[idx+5],         x_5[idx_x], f[idx+6],         x_6[idx_x] ), xi_0[idx_xi],    \
        Intrpl7( f[idx+nline],   x_0[idx_x], f[idx+1+nline],   x_1[idx_x], f[idx+2+nline],   x_2[idx_x], f[idx+3+nline],   x_3[idx_x], f[idx+4+nline],   x_4[idx_x], f[idx+5+nline],   x_5[idx_x], f[idx+6+nline],   x_6[idx_x] ), xi_1[idx_xi],    \
        Intrpl7( f[idx+2*nline], x_0[idx_x], f[idx+1+2*nline], x_1[idx_x], f[idx+2+2*nline], x_2[idx_x], f[idx+3+2*nline], x_3[idx_x], f[idx+4+2*nline], x_4[idx_x], f[idx+5+2*nline], x_5[idx_x], f[idx+6+2*nline], x_6[idx_x] ), xi_2[idx_xi],    \
        Intrpl7( f[idx+3*nline], x_0[idx_x], f[idx+1+3*nline], x_1[idx_x], f[idx+2+3*nline], x_2[idx_x], f[idx+3+3*nline], x_3[idx_x], f[idx+4+3*nline], x_4[idx_x], f[idx+5+3*nline], x_5[idx_x], f[idx+6+3*nline], x_6[idx_x] ), xi_3[idx_xi],    \
        Intrpl7( f[idx+4*nline], x_0[idx_x], f[idx+1+4*nline], x_1[idx_x], f[idx+2+4*nline], x_2[idx_x], f[idx+3+4*nline], x_3[idx_x], f[idx+4+4*nline], x_4[idx_x], f[idx+5+4*nline], x_5[idx_x], f[idx+6+4*nline], x_6[idx_x] ), xi_4[idx_xi],    \
        Intrpl7( f[idx+5*nline], x_0[idx_x], f[idx+1+5*nline], x_1[idx_x], f[idx+2+5*nline], x_2[idx_x], f[idx+3+5*nline], x_3[idx_x], f[idx+4+5*nline], x_4[idx_x], f[idx+5+5*nline], x_5[idx_x], f[idx+6+5*nline], x_6[idx_x] ), xi_5[idx_xi],    \
        Intrpl7( f[idx+6*nline], x_0[idx_x], f[idx+1+6*nline], x_1[idx_x], f[idx+2+6*nline], x_2[idx_x], f[idx+3+6*nline], x_3[idx_x], f[idx+4+6*nline], x_4[idx_x], f[idx+5+6*nline], x_5[idx_x], f[idx+6+6*nline], x_6[idx_x] ), xi_6[idx_xi]     \
    );

#define Y_XI_Intrpl7(f, F_in, i, j, k, i_c, j_c, k_c, idx_x, idx_y, idx_xi, y_0, y_1, y_2, y_3, y_4, y_5, y_6, xi_0, xi_1, xi_2, xi_3, xi_4, xi_5, xi_6)    \
    idx = j_c*nface + k_c*nline + i;  \
    F_in = Intrpl7(     \
        Intrpl7( f[idx],         y_0[idx_y], f[idx+nface],         y_1[idx_y], f[idx+2*nface],         y_2[idx_y], f[idx+3*nface],         y_3[idx_y], f[idx+4*nface],         y_4[idx_y], f[idx+5*nface],         y_5[idx_y], f[idx+6*nface],         y_6[idx_y] ), xi_0[idx_xi],   \
        Intrpl7( f[idx+nline],   y_0[idx_y], f[idx+nface+nline],   y_1[idx_y], f[idx+2*nface+nline],   y_2[idx_y], f[idx+3*nface+nline],   y_3[idx_y], f[idx+4*nface+nline],   y_4[idx_y], f[idx+5*nface+nline],   y_5[idx_y], f[idx+6*nface+nline],   y_6[idx_y] ), xi_1[idx_xi],   \
        Intrpl7( f[idx+2*nline], y_0[idx_y], f[idx+nface+2*nline], y_1[idx_y], f[idx+2*nface+2*nline], y_2[idx_y], f[idx+3*nface+2*nline], y_3[idx_y], f[idx+4*nface+2*nline], y_4[idx_y], f[idx+5*nface+2*nline], y_5[idx_y], f[idx+6*nface+2*nline], y_6[idx_y] ), xi_2[idx_xi],   \
        Intrpl7( f[idx+3*nline], y_0[idx_y], f[idx+nface+3*nline], y_1[idx_y], f[idx+2*nface+3*nline], y_2[idx_y], f[idx+3*nface+3*nline], y_3[idx_y], f[idx+4*nface+3*nline], y_4[idx_y], f[idx+5*nface+3*nline], y_5[idx_y], f[idx+6*nface+3*nline], y_6[idx_y] ), xi_3[idx_xi],   \
        Intrpl7( f[idx+4*nline], y_0[idx_y], f[idx+nface+4*nline], y_1[idx_y], f[idx+2*nface+4*nline], y_2[idx_y], f[idx+3*nface+4*nline], y_3[idx_y], f[idx+4*nface+4*nline], y_4[idx_y], f[idx+5*nface+4*nline], y_5[idx_y], f[idx+6*nface+4*nline], y_6[idx_y] ), xi_4[idx_xi],   \
        Intrpl7( f[idx+5*nline], y_0[idx_y], f[idx+nface+5*nline], y_1[idx_y], f[idx+2*nface+5*nline], y_2[idx_y], f[idx+3*nface+5*nline], y_3[idx_y], f[idx+4*nface+5*nline], y_4[idx_y], f[idx+5*nface+5*nline], y_5[idx_y], f[idx+6*nface+5*nline], y_6[idx_y] ), xi_5[idx_xi],   \
        Intrpl7( f[idx+6*nline], y_0[idx_y], f[idx+nface+6*nline], y_1[idx_y], f[idx+2*nface+6*nline], y_2[idx_y], f[idx+3*nface+6*nline], y_3[idx_y], f[idx+4*nface+6*nline], y_4[idx_y], f[idx+5*nface+6*nline], y_5[idx_y], f[idx+6*nface+6*nline], y_6[idx_y] ), xi_6[idx_xi]    \
    );

#define Intrpl3(f1, a1, f2, a2, f3, a3)     \
(   \
    f1*a1 + f2*a2 + f3*a3   \
)

#define Y_XI_Intrpl3(f, F_in, i, j, k, i_c, j_c, k_c, idx_x, idx_y, idx_xi, y_0, y_1, y_2, xi_0, xi_1, xi_2)    \
    idx = j_c*nface + k_c*nline + i;  \
    F_in = Intrpl3(     \
        Intrpl3( f[idx],         y_0[idx_y], f[idx+nface],         y_1[idx_y], f[idx+2*nface],         y_2[idx_y]), xi_0[idx_xi],   \
        Intrpl3( f[idx+nline],   y_0[idx_y], f[idx+nface+nline],   y_1[idx_y], f[idx+2*nface+nline],   y_2[idx_y]), xi_1[idx_xi],   \
        Intrpl3( f[idx+2*nline], y_0[idx_y], f[idx+nface+2*nline], y_1[idx_y], f[idx+2*nface+2*nline], y_2[idx_y]), xi_2[idx_xi]    \
    );

#define X_Y_XI_Intrpl3(f, F_in, i, j, k, i_c, j_c, k_c, idx_x, idx_y, idx_xi, x_0, x_1, x_2, y_0, y_1, y_2, xi_0, xi_1, xi_2)   \
    idx = j_c*nface + k_c*nline + i_c;    \
    F_in = Intrpl3(         \
        Intrpl3(            \
            Intrpl3( f[idx],                 x_0[idx_x], f[idx+1],                 x_1[idx_x], f[idx+2],                 x_2[idx_x] ), y_0[idx_y],      \
            Intrpl3( f[idx+nface],           x_0[idx_x], f[idx+1+nface],           x_1[idx_x], f[idx+2+nface],           x_2[idx_x] ), y_1[idx_y],      \
            Intrpl3( f[idx+2*nface],         x_0[idx_x], f[idx+1+2*nface],         x_1[idx_x], f[idx+2+2*nface],         x_2[idx_x] ), y_2[idx_y]       \
        ), xi_0[idx_xi],    \
        Intrpl3(            \
            Intrpl3( f[idx+nline],           x_0[idx_x], f[idx+1+nline],           x_1[idx_x], f[idx+2+nline],           x_2[idx_x] ), y_0[idx_y],      \
            Intrpl3( f[idx+nline+nface],     x_0[idx_x], f[idx+1+nline+nface],     x_1[idx_x], f[idx+2+nline+nface],     x_2[idx_x] ), y_1[idx_y],      \
            Intrpl3( f[idx+nline+2*nface],   x_0[idx_x], f[idx+1+nline+2*nface],   x_1[idx_x], f[idx+2+nline+2*nface],   x_2[idx_x] ), y_2[idx_y]       \
        ), xi_1[idx_xi],    \
        Intrpl3(            \
            Intrpl3( f[idx+2*nline],         x_0[idx_x], f[idx+1+2*nline],         x_1[idx_x], f[idx+2+2*nline],         x_2[idx_x] ), y_0[idx_y],      \
            Intrpl3( f[idx+2*nline+nface],   x_0[idx_x], f[idx+1+2*nline+nface],   x_1[idx_x], f[idx+2+2*nline+nface],   x_2[idx_x] ), y_1[idx_y],      \
            Intrpl3( f[idx+2*nline+2*nface], x_0[idx_x], f[idx+1+2*nline+2*nface], x_1[idx_x], f[idx+2+2*nline+2*nface], x_2[idx_x] ), y_2[idx_y]       \
        ), xi_2[idx_xi]     \
    )

#endif