#ifndef INTERPOLATIONSLLBM_FILE
#define INTERPOLATIONSLLBM_FILE

#define Intrpl7(f1, a1, f2, a2, f3, a3, f4, a4, f5, a5, f6, a6, f7, a7)     \
(       \
    f1*a1 + f2*a2 + f3*a3 + f4*a4 + f5*a5 + f6*a6 + f7*a7   \
)

#define F0_Intrpl7(f, i, j, k)      \
    idx = j*NX6*NZ6 + k*NX6 + i;    \
    F0_in = ( f[idx] );

#define F1_Intrpl7(f, i, j, k, i_c, j_c, k_c, XPara_0, XPara_1, XPara_2, XPara_3, XPara_4, XPara_5, XPara_6)    \
    idx = j*NX6*NZ6 + k*NX6 + i_c;  \
    F1_in = Intrpl7( f[idx], XPara_0[i], f[idx+1], XPara_1[i], f[idx+2], XPara_2[i], f[idx+3], XPara_3[i], f[idx+4], XPara_4[i], f[idx+5], XPara_5[i], f[idx+6], XPara_6[i] );

#define F2_Intrpl7(f, i, j, k, i_c, j_c, k_c, XPara_0, XPara_1, XPara_2, XPara_3, XPara_4, XPara_5, XPara_6)    \
    idx = j*NX6*NZ6 + k*NX6 + i_c;  \
    F2_in = Intrpl7( f[idx], XPara_0[i], f[idx+1], XPara_1[i], f[idx+2], XPara_2[i], f[idx+3], XPara_3[i], f[idx+4], XPara_4[i], f[idx+5], XPara_5[i], f[idx+6], XPara_6[i] );

#define F3_Intrpl7(f, i, j, k, i_c, j_c, k_c, YPara_0, YPara_1, YPara_2, YPara_3, YPara_4, YPara_5, YPara_6)    \
    idx = j_c*NX6*NZ6 + k*NX6 + i;  \
    F3_in = Intrpl7( f[idx], YPara_0[j], f[idx+NX6*NZ6], YPara_1[j], f[idx+2*NX6*NZ6], YPara_2[j], f[idx+3*NX6*NZ6], YPara_3[j], f[idx+4*NX6*NZ6], YPara_4[j], f[idx+5*NX6*NZ6], YPara_5[j], f[idx+6*NX6*NZ6], YPara_6[j] );

#define F4_Intrpl7(f, i, j, k, i_c, j_c, k_c, YPara_0, YPara_1, YPara_2, YPara_3, YPara_4, YPara_5, YPara_6)    \
    idx = j_c*NX6*NZ6 + k*NX6 + i;  \
    F4_in = Intrpl7( f[idx], YPara_0[j], f[idx+NX6*NZ6], YPara_1[j], f[idx+2*NX6*NZ6], YPara_2[j], f[idx+3*NX6*NZ6], YPara_3[j], f[idx+4*NX6*NZ6], YPara_4[j], f[idx+5*NX6*NZ6], YPara_5[j], f[idx+6*NX6*NZ6], YPara_6[j] );

#define F5_Intrpl7(f, i, j, k, i_c, j_c, k_c, ZPara_0, ZPara_1, ZPara_2, ZPara_3, ZPara_4, ZPara_5, ZPara_6)    \
    idx = j*NX6*NZ6 + k_c*NX6 + i;  \
    F5_in = Intrpl7( f[idx], ZPara_0[k], f[idx+NX6], ZPara_1[k], f[idx+2*NX6], ZPara_2[k], f[idx+3*NX6], ZPara_3[k], f[idx+4*NX6], ZPara_4[k], f[idx+5*NX6], ZPara_5[k], f[idx+6*NX6], ZPara_6[k] );

#define F6_Intrpl7(f, i, j, k, i_c, j_c, k_c, ZPara_0, ZPara_1, ZPara_2, ZPara_3, ZPara_4, ZPara_5, ZPara_6)    \
    idx = j*NX6*NZ6 + k_c*NX6 + i;  \
    F6_in = Intrpl7( f[idx], ZPara_0[k], f[idx+NX6], ZPara_1[k], f[idx+2*NX6], ZPara_2[k], f[idx+3*NX6], ZPara_3[k], f[idx+4*NX6], ZPara_4[k], f[idx+5*NX6], ZPara_5[k], f[idx+6*NX6], ZPara_6[k] );

#define F7_Intrpl7(f, i, j, k, i_c, j_c, k_c, XPara_0, XPara_1, XPara_2, XPara_3, XPara_4, XPara_5, XPara_6, YPara_0, YPara_1, YPara_2, YPara_3, YPara_4, YPara_5, YPara_6)    \
    idx = j_c*NX6*NZ6 + k*NX6 + i_c;    \
    F7_in = Intrpl7(     \
        Intrpl7( f[idx],   YPara_0[j], f[idx+NX6*NZ6],   YPara_1[j], f[idx+2*NX6*NZ6],   YPara_2[j], f[idx+3*NX6*NZ6],   YPara_3[j], f[idx+4*NX6*NZ6],   YPara_4[j], f[idx+5*NX6*NZ6],   YPara_5[j], f[idx+6*NX6*NZ6],   YPara_6[j] ), XPara_0[i],       \
        Intrpl7( f[idx+1], YPara_0[j], f[idx+NX6*NZ6+1], YPara_1[j], f[idx+2*NX6*NZ6+1], YPara_2[j], f[idx+3*NX6*NZ6+1], YPara_3[j], f[idx+4*NX6*NZ6+1], YPara_4[j], f[idx+5*NX6*NZ6+1], YPara_5[j], f[idx+6*NX6*NZ6+1], YPara_6[j] ), XPara_1[i],       \
        Intrpl7( f[idx+2], YPara_0[j], f[idx+NX6*NZ6+2], YPara_1[j], f[idx+2*NX6*NZ6+2], YPara_2[j], f[idx+3*NX6*NZ6+2], YPara_3[j], f[idx+4*NX6*NZ6+2], YPara_4[j], f[idx+5*NX6*NZ6+2], YPara_5[j], f[idx+6*NX6*NZ6+2], YPara_6[j] ), XPara_2[i],       \
        Intrpl7( f[idx+3], YPara_0[j], f[idx+NX6*NZ6+3], YPara_1[j], f[idx+2*NX6*NZ6+3], YPara_2[j], f[idx+3*NX6*NZ6+3], YPara_3[j], f[idx+4*NX6*NZ6+3], YPara_4[j], f[idx+5*NX6*NZ6+3], YPara_5[j], f[idx+6*NX6*NZ6+3], YPara_6[j] ), XPara_3[i],       \
        Intrpl7( f[idx+4], YPara_0[j], f[idx+NX6*NZ6+4], YPara_1[j], f[idx+2*NX6*NZ6+4], YPara_2[j], f[idx+3*NX6*NZ6+4], YPara_3[j], f[idx+4*NX6*NZ6+4], YPara_4[j], f[idx+5*NX6*NZ6+4], YPara_5[j], f[idx+6*NX6*NZ6+4], YPara_6[j] ), XPara_4[i],       \
        Intrpl7( f[idx+5], YPara_0[j], f[idx+NX6*NZ6+5], YPara_1[j], f[idx+2*NX6*NZ6+5], YPara_2[j], f[idx+3*NX6*NZ6+5], YPara_3[j], f[idx+4*NX6*NZ6+5], YPara_4[j], f[idx+5*NX6*NZ6+5], YPara_5[j], f[idx+6*NX6*NZ6+5], YPara_6[j] ), XPara_5[i],       \
        Intrpl7( f[idx+6], YPara_0[j], f[idx+NX6*NZ6+6], YPara_1[j], f[idx+2*NX6*NZ6+6], YPara_2[j], f[idx+3*NX6*NZ6+6], YPara_3[j], f[idx+4*NX6*NZ6+6], YPara_4[j], f[idx+5*NX6*NZ6+6], YPara_5[j], f[idx+6*NX6*NZ6+6], YPara_6[j] ), XPara_6[i]        \
    );

#define F8_Intrpl7(f, i, j, k, i_c, j_c, k_c, XPara_0, XPara_1, XPara_2, XPara_3, XPara_4, XPara_5, XPara_6, YPara_0, YPara_1, YPara_2, YPara_3, YPara_4, YPara_5, YPara_6)    \
    idx = j_c*NX6*NZ6 + k*NX6 + i_c;    \
    F8_in = Intrpl7(     \
        Intrpl7( f[idx],   YPara_0[j], f[idx+NX6*NZ6],   YPara_1[j], f[idx+2*NX6*NZ6],   YPara_2[j], f[idx+3*NX6*NZ6],   YPara_3[j], f[idx+4*NX6*NZ6],   YPara_4[j], f[idx+5*NX6*NZ6],   YPara_5[j], f[idx+6*NX6*NZ6],   YPara_6[j] ), XPara_0[i],       \
        Intrpl7( f[idx+1], YPara_0[j], f[idx+NX6*NZ6+1], YPara_1[j], f[idx+2*NX6*NZ6+1], YPara_2[j], f[idx+3*NX6*NZ6+1], YPara_3[j], f[idx+4*NX6*NZ6+1], YPara_4[j], f[idx+5*NX6*NZ6+1], YPara_5[j], f[idx+6*NX6*NZ6+1], YPara_6[j] ), XPara_1[i],       \
        Intrpl7( f[idx+2], YPara_0[j], f[idx+NX6*NZ6+2], YPara_1[j], f[idx+2*NX6*NZ6+2], YPara_2[j], f[idx+3*NX6*NZ6+2], YPara_3[j], f[idx+4*NX6*NZ6+2], YPara_4[j], f[idx+5*NX6*NZ6+2], YPara_5[j], f[idx+6*NX6*NZ6+2], YPara_6[j] ), XPara_2[i],       \
        Intrpl7( f[idx+3], YPara_0[j], f[idx+NX6*NZ6+3], YPara_1[j], f[idx+2*NX6*NZ6+3], YPara_2[j], f[idx+3*NX6*NZ6+3], YPara_3[j], f[idx+4*NX6*NZ6+3], YPara_4[j], f[idx+5*NX6*NZ6+3], YPara_5[j], f[idx+6*NX6*NZ6+3], YPara_6[j] ), XPara_3[i],       \
        Intrpl7( f[idx+4], YPara_0[j], f[idx+NX6*NZ6+4], YPara_1[j], f[idx+2*NX6*NZ6+4], YPara_2[j], f[idx+3*NX6*NZ6+4], YPara_3[j], f[idx+4*NX6*NZ6+4], YPara_4[j], f[idx+5*NX6*NZ6+4], YPara_5[j], f[idx+6*NX6*NZ6+4], YPara_6[j] ), XPara_4[i],       \
        Intrpl7( f[idx+5], YPara_0[j], f[idx+NX6*NZ6+5], YPara_1[j], f[idx+2*NX6*NZ6+5], YPara_2[j], f[idx+3*NX6*NZ6+5], YPara_3[j], f[idx+4*NX6*NZ6+5], YPara_4[j], f[idx+5*NX6*NZ6+5], YPara_5[j], f[idx+6*NX6*NZ6+5], YPara_6[j] ), XPara_5[i],       \
        Intrpl7( f[idx+6], YPara_0[j], f[idx+NX6*NZ6+6], YPara_1[j], f[idx+2*NX6*NZ6+6], YPara_2[j], f[idx+3*NX6*NZ6+6], YPara_3[j], f[idx+4*NX6*NZ6+6], YPara_4[j], f[idx+5*NX6*NZ6+6], YPara_5[j], f[idx+6*NX6*NZ6+6], YPara_6[j] ), XPara_6[i]        \
    );

#define F9_Intrpl7(f, i, j, k, i_c, j_c, k_c, XPara_0, XPara_1, XPara_2, XPara_3, XPara_4, XPara_5, XPara_6, YPara_0, YPara_1, YPara_2, YPara_3, YPara_4, YPara_5, YPara_6)    \
    idx = j_c*NX6*NZ6 + k*NX6 + i_c;    \
    F9_in = Intrpl7(     \
        Intrpl7( f[idx],   YPara_0[j], f[idx+NX6*NZ6],   YPara_1[j], f[idx+2*NX6*NZ6],   YPara_2[j], f[idx+3*NX6*NZ6],   YPara_3[j], f[idx+4*NX6*NZ6],   YPara_4[j], f[idx+5*NX6*NZ6],   YPara_5[j], f[idx+6*NX6*NZ6],   YPara_6[j] ), XPara_0[i],       \
        Intrpl7( f[idx+1], YPara_0[j], f[idx+NX6*NZ6+1], YPara_1[j], f[idx+2*NX6*NZ6+1], YPara_2[j], f[idx+3*NX6*NZ6+1], YPara_3[j], f[idx+4*NX6*NZ6+1], YPara_4[j], f[idx+5*NX6*NZ6+1], YPara_5[j], f[idx+6*NX6*NZ6+1], YPara_6[j] ), XPara_1[i],       \
        Intrpl7( f[idx+2], YPara_0[j], f[idx+NX6*NZ6+2], YPara_1[j], f[idx+2*NX6*NZ6+2], YPara_2[j], f[idx+3*NX6*NZ6+2], YPara_3[j], f[idx+4*NX6*NZ6+2], YPara_4[j], f[idx+5*NX6*NZ6+2], YPara_5[j], f[idx+6*NX6*NZ6+2], YPara_6[j] ), XPara_2[i],       \
        Intrpl7( f[idx+3], YPara_0[j], f[idx+NX6*NZ6+3], YPara_1[j], f[idx+2*NX6*NZ6+3], YPara_2[j], f[idx+3*NX6*NZ6+3], YPara_3[j], f[idx+4*NX6*NZ6+3], YPara_4[j], f[idx+5*NX6*NZ6+3], YPara_5[j], f[idx+6*NX6*NZ6+3], YPara_6[j] ), XPara_3[i],       \
        Intrpl7( f[idx+4], YPara_0[j], f[idx+NX6*NZ6+4], YPara_1[j], f[idx+2*NX6*NZ6+4], YPara_2[j], f[idx+3*NX6*NZ6+4], YPara_3[j], f[idx+4*NX6*NZ6+4], YPara_4[j], f[idx+5*NX6*NZ6+4], YPara_5[j], f[idx+6*NX6*NZ6+4], YPara_6[j] ), XPara_4[i],       \
        Intrpl7( f[idx+5], YPara_0[j], f[idx+NX6*NZ6+5], YPara_1[j], f[idx+2*NX6*NZ6+5], YPara_2[j], f[idx+3*NX6*NZ6+5], YPara_3[j], f[idx+4*NX6*NZ6+5], YPara_4[j], f[idx+5*NX6*NZ6+5], YPara_5[j], f[idx+6*NX6*NZ6+5], YPara_6[j] ), XPara_5[i],       \
        Intrpl7( f[idx+6], YPara_0[j], f[idx+NX6*NZ6+6], YPara_1[j], f[idx+2*NX6*NZ6+6], YPara_2[j], f[idx+3*NX6*NZ6+6], YPara_3[j], f[idx+4*NX6*NZ6+6], YPara_4[j], f[idx+5*NX6*NZ6+6], YPara_5[j], f[idx+6*NX6*NZ6+6], YPara_6[j] ), XPara_6[i]        \
    );

#define F10_Intrpl7(f, i, j, k, i_c, j_c, k_c, XPara_0, XPara_1, XPara_2, XPara_3, XPara_4, XPara_5, XPara_6, YPara_0, YPara_1, YPara_2, YPara_3, YPara_4, YPara_5, YPara_6)   \
    idx = j_c*NX6*NZ6 + k*NX6 + i_c;    \
    F10_in = Intrpl7(     \
        Intrpl7( f[idx],   YPara_0[j], f[idx+NX6*NZ6],   YPara_1[j], f[idx+2*NX6*NZ6],   YPara_2[j], f[idx+3*NX6*NZ6],   YPara_3[j], f[idx+4*NX6*NZ6],   YPara_4[j], f[idx+5*NX6*NZ6],   YPara_5[j], f[idx+6*NX6*NZ6],   YPara_6[j] ), XPara_0[i],       \
        Intrpl7( f[idx+1], YPara_0[j], f[idx+NX6*NZ6+1], YPara_1[j], f[idx+2*NX6*NZ6+1], YPara_2[j], f[idx+3*NX6*NZ6+1], YPara_3[j], f[idx+4*NX6*NZ6+1], YPara_4[j], f[idx+5*NX6*NZ6+1], YPara_5[j], f[idx+6*NX6*NZ6+1], YPara_6[j] ), XPara_1[i],       \
        Intrpl7( f[idx+2], YPara_0[j], f[idx+NX6*NZ6+2], YPara_1[j], f[idx+2*NX6*NZ6+2], YPara_2[j], f[idx+3*NX6*NZ6+2], YPara_3[j], f[idx+4*NX6*NZ6+2], YPara_4[j], f[idx+5*NX6*NZ6+2], YPara_5[j], f[idx+6*NX6*NZ6+2], YPara_6[j] ), XPara_2[i],       \
        Intrpl7( f[idx+3], YPara_0[j], f[idx+NX6*NZ6+3], YPara_1[j], f[idx+2*NX6*NZ6+3], YPara_2[j], f[idx+3*NX6*NZ6+3], YPara_3[j], f[idx+4*NX6*NZ6+3], YPara_4[j], f[idx+5*NX6*NZ6+3], YPara_5[j], f[idx+6*NX6*NZ6+3], YPara_6[j] ), XPara_3[i],       \
        Intrpl7( f[idx+4], YPara_0[j], f[idx+NX6*NZ6+4], YPara_1[j], f[idx+2*NX6*NZ6+4], YPara_2[j], f[idx+3*NX6*NZ6+4], YPara_3[j], f[idx+4*NX6*NZ6+4], YPara_4[j], f[idx+5*NX6*NZ6+4], YPara_5[j], f[idx+6*NX6*NZ6+4], YPara_6[j] ), XPara_4[i],       \
        Intrpl7( f[idx+5], YPara_0[j], f[idx+NX6*NZ6+5], YPara_1[j], f[idx+2*NX6*NZ6+5], YPara_2[j], f[idx+3*NX6*NZ6+5], YPara_3[j], f[idx+4*NX6*NZ6+5], YPara_4[j], f[idx+5*NX6*NZ6+5], YPara_5[j], f[idx+6*NX6*NZ6+5], YPara_6[j] ), XPara_5[i],       \
        Intrpl7( f[idx+6], YPara_0[j], f[idx+NX6*NZ6+6], YPara_1[j], f[idx+2*NX6*NZ6+6], YPara_2[j], f[idx+3*NX6*NZ6+6], YPara_3[j], f[idx+4*NX6*NZ6+6], YPara_4[j], f[idx+5*NX6*NZ6+6], YPara_5[j], f[idx+6*NX6*NZ6+6], YPara_6[j] ), XPara_6[i]        \
    );

#define F11_Intrpl7(f, i, j, k, i_c, j_c, k_c, XPara_0, XPara_1, XPara_2, XPara_3, XPara_4, XPara_5, XPara_6, ZPara_0, ZPara_1, ZPara_2, ZPara_3, ZPara_4, ZPara_5, ZPara_6)   \
    idx = j*NX6*NZ6 + k_c*NX6 + i_c;    \
    F11_in = Intrpl7(    \
        Intrpl7( f[idx],       XPara_0[i], f[idx+1],       XPara_1[i], f[idx+2],       XPara_2[i], f[idx+3],       XPara_3[i], f[idx+4],       XPara_4[i], f[idx+5],       XPara_5[i], f[idx+6],       XPara_6[i] ), ZPara_0[k],     \
        Intrpl7( f[idx+NX6],   XPara_0[i], f[idx+1+NX6],   XPara_1[i], f[idx+2+NX6],   XPara_2[i], f[idx+3+NX6],   XPara_3[i], f[idx+4+NX6],   XPara_4[i], f[idx+5+NX6],   XPara_5[i], f[idx+6+NX6],   XPara_6[i] ), ZPara_1[k],     \
        Intrpl7( f[idx+2*NX6], XPara_0[i], f[idx+1+2*NX6], XPara_1[i], f[idx+2+2*NX6], XPara_2[i], f[idx+3+2*NX6], XPara_3[i], f[idx+4+2*NX6], XPara_4[i], f[idx+5+2*NX6], XPara_5[i], f[idx+6+2*NX6], XPara_6[i] ), ZPara_2[k],     \
        Intrpl7( f[idx+3*NX6], XPara_0[i], f[idx+1+3*NX6], XPara_1[i], f[idx+2+3*NX6], XPara_2[i], f[idx+3+3*NX6], XPara_3[i], f[idx+4+3*NX6], XPara_4[i], f[idx+5+3*NX6], XPara_5[i], f[idx+6+3*NX6], XPara_6[i] ), ZPara_3[k],     \
        Intrpl7( f[idx+4*NX6], XPara_0[i], f[idx+1+4*NX6], XPara_1[i], f[idx+2+4*NX6], XPara_2[i], f[idx+3+4*NX6], XPara_3[i], f[idx+4+4*NX6], XPara_4[i], f[idx+5+4*NX6], XPara_5[i], f[idx+6+4*NX6], XPara_6[i] ), ZPara_4[k],     \
        Intrpl7( f[idx+5*NX6], XPara_0[i], f[idx+1+5*NX6], XPara_1[i], f[idx+2+5*NX6], XPara_2[i], f[idx+3+5*NX6], XPara_3[i], f[idx+4+5*NX6], XPara_4[i], f[idx+5+5*NX6], XPara_5[i], f[idx+6+5*NX6], XPara_6[i] ), ZPara_5[k],     \
        Intrpl7( f[idx+6*NX6], XPara_0[i], f[idx+1+6*NX6], XPara_1[i], f[idx+2+6*NX6], XPara_2[i], f[idx+3+6*NX6], XPara_3[i], f[idx+4+6*NX6], XPara_4[i], f[idx+5+6*NX6], XPara_5[i], f[idx+6+6*NX6], XPara_6[i] ), ZPara_6[k]      \
    );

#define F12_Intrpl7(f, i, j, k, i_c, j_c, k_c, XPara_0, XPara_1, XPara_2, XPara_3, XPara_4, XPara_5, XPara_6, ZPara_0, ZPara_1, ZPara_2, ZPara_3, ZPara_4, ZPara_5, ZPara_6)   \
    idx = j*NX6*NZ6 + k_c*NX6 + i_c;    \
    F12_in = Intrpl7(    \
        Intrpl7( f[idx],       XPara_0[i], f[idx+1],       XPara_1[i], f[idx+2],       XPara_2[i], f[idx+3],       XPara_3[i], f[idx+4],       XPara_4[i], f[idx+5],       XPara_5[i], f[idx+6],       XPara_6[i] ), ZPara_0[k],     \
        Intrpl7( f[idx+NX6],   XPara_0[i], f[idx+1+NX6],   XPara_1[i], f[idx+2+NX6],   XPara_2[i], f[idx+3+NX6],   XPara_3[i], f[idx+4+NX6],   XPara_4[i], f[idx+5+NX6],   XPara_5[i], f[idx+6+NX6],   XPara_6[i] ), ZPara_1[k],     \
        Intrpl7( f[idx+2*NX6], XPara_0[i], f[idx+1+2*NX6], XPara_1[i], f[idx+2+2*NX6], XPara_2[i], f[idx+3+2*NX6], XPara_3[i], f[idx+4+2*NX6], XPara_4[i], f[idx+5+2*NX6], XPara_5[i], f[idx+6+2*NX6], XPara_6[i] ), ZPara_2[k],     \
        Intrpl7( f[idx+3*NX6], XPara_0[i], f[idx+1+3*NX6], XPara_1[i], f[idx+2+3*NX6], XPara_2[i], f[idx+3+3*NX6], XPara_3[i], f[idx+4+3*NX6], XPara_4[i], f[idx+5+3*NX6], XPara_5[i], f[idx+6+3*NX6], XPara_6[i] ), ZPara_3[k],     \
        Intrpl7( f[idx+4*NX6], XPara_0[i], f[idx+1+4*NX6], XPara_1[i], f[idx+2+4*NX6], XPara_2[i], f[idx+3+4*NX6], XPara_3[i], f[idx+4+4*NX6], XPara_4[i], f[idx+5+4*NX6], XPara_5[i], f[idx+6+4*NX6], XPara_6[i] ), ZPara_4[k],     \
        Intrpl7( f[idx+5*NX6], XPara_0[i], f[idx+1+5*NX6], XPara_1[i], f[idx+2+5*NX6], XPara_2[i], f[idx+3+5*NX6], XPara_3[i], f[idx+4+5*NX6], XPara_4[i], f[idx+5+5*NX6], XPara_5[i], f[idx+6+5*NX6], XPara_6[i] ), ZPara_5[k],     \
        Intrpl7( f[idx+6*NX6], XPara_0[i], f[idx+1+6*NX6], XPara_1[i], f[idx+2+6*NX6], XPara_2[i], f[idx+3+6*NX6], XPara_3[i], f[idx+4+6*NX6], XPara_4[i], f[idx+5+6*NX6], XPara_5[i], f[idx+6+6*NX6], XPara_6[i] ), ZPara_6[k]      \
    );

#define F13_Intrpl7(f, i, j, k, i_c, j_c, k_c, XPara_0, XPara_1, XPara_2, XPara_3, XPara_4, XPara_5, XPara_6, ZPara_0, ZPara_1, ZPara_2, ZPara_3, ZPara_4, ZPara_5, ZPara_6)   \
    idx = j*NX6*NZ6 + k_c*NX6 + i_c;    \
    F13_in = Intrpl7(    \
        Intrpl7( f[idx],       XPara_0[i], f[idx+1],       XPara_1[i], f[idx+2],       XPara_2[i], f[idx+3],       XPara_3[i], f[idx+4],       XPara_4[i], f[idx+5],       XPara_5[i], f[idx+6],       XPara_6[i] ), ZPara_0[k],     \
        Intrpl7( f[idx+NX6],   XPara_0[i], f[idx+1+NX6],   XPara_1[i], f[idx+2+NX6],   XPara_2[i], f[idx+3+NX6],   XPara_3[i], f[idx+4+NX6],   XPara_4[i], f[idx+5+NX6],   XPara_5[i], f[idx+6+NX6],   XPara_6[i] ), ZPara_1[k],     \
        Intrpl7( f[idx+2*NX6], XPara_0[i], f[idx+1+2*NX6], XPara_1[i], f[idx+2+2*NX6], XPara_2[i], f[idx+3+2*NX6], XPara_3[i], f[idx+4+2*NX6], XPara_4[i], f[idx+5+2*NX6], XPara_5[i], f[idx+6+2*NX6], XPara_6[i] ), ZPara_2[k],     \
        Intrpl7( f[idx+3*NX6], XPara_0[i], f[idx+1+3*NX6], XPara_1[i], f[idx+2+3*NX6], XPara_2[i], f[idx+3+3*NX6], XPara_3[i], f[idx+4+3*NX6], XPara_4[i], f[idx+5+3*NX6], XPara_5[i], f[idx+6+3*NX6], XPara_6[i] ), ZPara_3[k],     \
        Intrpl7( f[idx+4*NX6], XPara_0[i], f[idx+1+4*NX6], XPara_1[i], f[idx+2+4*NX6], XPara_2[i], f[idx+3+4*NX6], XPara_3[i], f[idx+4+4*NX6], XPara_4[i], f[idx+5+4*NX6], XPara_5[i], f[idx+6+4*NX6], XPara_6[i] ), ZPara_4[k],     \
        Intrpl7( f[idx+5*NX6], XPara_0[i], f[idx+1+5*NX6], XPara_1[i], f[idx+2+5*NX6], XPara_2[i], f[idx+3+5*NX6], XPara_3[i], f[idx+4+5*NX6], XPara_4[i], f[idx+5+5*NX6], XPara_5[i], f[idx+6+5*NX6], XPara_6[i] ), ZPara_5[k],     \
        Intrpl7( f[idx+6*NX6], XPara_0[i], f[idx+1+6*NX6], XPara_1[i], f[idx+2+6*NX6], XPara_2[i], f[idx+3+6*NX6], XPara_3[i], f[idx+4+6*NX6], XPara_4[i], f[idx+5+6*NX6], XPara_5[i], f[idx+6+6*NX6], XPara_6[i] ), ZPara_6[k]      \
    );

#define F14_Intrpl7(f, i, j, k, i_c, j_c, k_c, XPara_0, XPara_1, XPara_2, XPara_3, XPara_4, XPara_5, XPara_6, ZPara_0, ZPara_1, ZPara_2, ZPara_3, ZPara_4, ZPara_5, ZPara_6)   \
    idx = j*NX6*NZ6 + k_c*NX6 + i_c;    \
    F14_in = Intrpl7(    \
        Intrpl7( f[idx],       XPara_0[i], f[idx+1],       XPara_1[i], f[idx+2],       XPara_2[i], f[idx+3],       XPara_3[i], f[idx+4],       XPara_4[i], f[idx+5],       XPara_5[i], f[idx+6],       XPara_6[i] ), ZPara_0[k],     \
        Intrpl7( f[idx+NX6],   XPara_0[i], f[idx+1+NX6],   XPara_1[i], f[idx+2+NX6],   XPara_2[i], f[idx+3+NX6],   XPara_3[i], f[idx+4+NX6],   XPara_4[i], f[idx+5+NX6],   XPara_5[i], f[idx+6+NX6],   XPara_6[i] ), ZPara_1[k],     \
        Intrpl7( f[idx+2*NX6], XPara_0[i], f[idx+1+2*NX6], XPara_1[i], f[idx+2+2*NX6], XPara_2[i], f[idx+3+2*NX6], XPara_3[i], f[idx+4+2*NX6], XPara_4[i], f[idx+5+2*NX6], XPara_5[i], f[idx+6+2*NX6], XPara_6[i] ), ZPara_2[k],     \
        Intrpl7( f[idx+3*NX6], XPara_0[i], f[idx+1+3*NX6], XPara_1[i], f[idx+2+3*NX6], XPara_2[i], f[idx+3+3*NX6], XPara_3[i], f[idx+4+3*NX6], XPara_4[i], f[idx+5+3*NX6], XPara_5[i], f[idx+6+3*NX6], XPara_6[i] ), ZPara_3[k],     \
        Intrpl7( f[idx+4*NX6], XPara_0[i], f[idx+1+4*NX6], XPara_1[i], f[idx+2+4*NX6], XPara_2[i], f[idx+3+4*NX6], XPara_3[i], f[idx+4+4*NX6], XPara_4[i], f[idx+5+4*NX6], XPara_5[i], f[idx+6+4*NX6], XPara_6[i] ), ZPara_4[k],     \
        Intrpl7( f[idx+5*NX6], XPara_0[i], f[idx+1+5*NX6], XPara_1[i], f[idx+2+5*NX6], XPara_2[i], f[idx+3+5*NX6], XPara_3[i], f[idx+4+5*NX6], XPara_4[i], f[idx+5+5*NX6], XPara_5[i], f[idx+6+5*NX6], XPara_6[i] ), ZPara_5[k],     \
        Intrpl7( f[idx+6*NX6], XPara_0[i], f[idx+1+6*NX6], XPara_1[i], f[idx+2+6*NX6], XPara_2[i], f[idx+3+6*NX6], XPara_3[i], f[idx+4+6*NX6], XPara_4[i], f[idx+5+6*NX6], XPara_5[i], f[idx+6+6*NX6], XPara_6[i] ), ZPara_6[k]      \
    );

#define F15_Intrpl7(f, i, j, k, i_c, j_c, k_c, YPara_0, YPara_1, YPara_2, YPara_3, YPara_4, YPara_5, YPara_6, ZPara_0, ZPara_1, ZPara_2, ZPara_3, ZPara_4, ZPara_5, ZPara_6)   \
    idx = j_c*NX6*NZ6 + k_c*NX6 + i;    \
    F15_in = Intrpl7(    \
        Intrpl7( f[idx],       YPara_0[j], f[idx+NX6*NZ6],       YPara_1[j], f[idx+2*NX6*NZ6],       YPara_2[j], f[idx+3*NX6*NZ6],       YPara_3[j], f[idx+4*NX6*NZ6],       YPara_4[j], f[idx+5*NX6*NZ6],       YPara_5[j], f[idx+6*NX6*NZ6],       YPara_6[j] ), ZPara_0[k],       \
        Intrpl7( f[idx+NX6],   YPara_0[j], f[idx+NX6*NZ6+NX6],   YPara_1[j], f[idx+2*NX6*NZ6+NX6],   YPara_2[j], f[idx+3*NX6*NZ6+NX6],   YPara_3[j], f[idx+4*NX6*NZ6+NX6],   YPara_4[j], f[idx+5*NX6*NZ6+NX6],   YPara_5[j], f[idx+6*NX6*NZ6+NX6],   YPara_6[j] ), ZPara_1[k],       \
        Intrpl7( f[idx+2*NX6], YPara_0[j], f[idx+NX6*NZ6+2*NX6], YPara_1[j], f[idx+2*NX6*NZ6+2*NX6], YPara_2[j], f[idx+3*NX6*NZ6+2*NX6], YPara_3[j], f[idx+4*NX6*NZ6+2*NX6], YPara_4[j], f[idx+5*NX6*NZ6+2*NX6], YPara_5[j], f[idx+6*NX6*NZ6+2*NX6], YPara_6[j] ), ZPara_2[k],       \
        Intrpl7( f[idx+3*NX6], YPara_0[j], f[idx+NX6*NZ6+3*NX6], YPara_1[j], f[idx+2*NX6*NZ6+3*NX6], YPara_2[j], f[idx+3*NX6*NZ6+3*NX6], YPara_3[j], f[idx+4*NX6*NZ6+3*NX6], YPara_4[j], f[idx+5*NX6*NZ6+3*NX6], YPara_5[j], f[idx+6*NX6*NZ6+3*NX6], YPara_6[j] ), ZPara_3[k],       \
        Intrpl7( f[idx+4*NX6], YPara_0[j], f[idx+NX6*NZ6+4*NX6], YPara_1[j], f[idx+2*NX6*NZ6+4*NX6], YPara_2[j], f[idx+3*NX6*NZ6+4*NX6], YPara_3[j], f[idx+4*NX6*NZ6+4*NX6], YPara_4[j], f[idx+5*NX6*NZ6+4*NX6], YPara_5[j], f[idx+6*NX6*NZ6+4*NX6], YPara_6[j] ), ZPara_4[k],       \
        Intrpl7( f[idx+5*NX6], YPara_0[j], f[idx+NX6*NZ6+5*NX6], YPara_1[j], f[idx+2*NX6*NZ6+5*NX6], YPara_2[j], f[idx+3*NX6*NZ6+5*NX6], YPara_3[j], f[idx+4*NX6*NZ6+5*NX6], YPara_4[j], f[idx+5*NX6*NZ6+5*NX6], YPara_5[j], f[idx+6*NX6*NZ6+5*NX6], YPara_6[j] ), ZPara_5[k],       \
        Intrpl7( f[idx+6*NX6], YPara_0[j], f[idx+NX6*NZ6+6*NX6], YPara_1[j], f[idx+2*NX6*NZ6+6*NX6], YPara_2[j], f[idx+3*NX6*NZ6+6*NX6], YPara_3[j], f[idx+4*NX6*NZ6+6*NX6], YPara_4[j], f[idx+5*NX6*NZ6+6*NX6], YPara_5[j], f[idx+6*NX6*NZ6+6*NX6], YPara_6[j] ), ZPara_6[k]        \
    );

#define F16_Intrpl7(f, i, j, k, i_c, j_c, k_c, YPara_0, YPara_1, YPara_2, YPara_3, YPara_4, YPara_5, YPara_6, ZPara_0, ZPara_1, ZPara_2, ZPara_3, ZPara_4, ZPara_5, ZPara_6)   \
    idx = j_c*NX6*NZ6 + k_c*NX6 + i;    \
    F16_in = Intrpl7(    \
        Intrpl7( f[idx],       YPara_0[j], f[idx+NX6*NZ6],       YPara_1[j], f[idx+2*NX6*NZ6],       YPara_2[j], f[idx+3*NX6*NZ6],       YPara_3[j], f[idx+4*NX6*NZ6],       YPara_4[j], f[idx+5*NX6*NZ6],       YPara_5[j], f[idx+6*NX6*NZ6],       YPara_6[j] ), ZPara_0[k],       \
        Intrpl7( f[idx+NX6],   YPara_0[j], f[idx+NX6*NZ6+NX6],   YPara_1[j], f[idx+2*NX6*NZ6+NX6],   YPara_2[j], f[idx+3*NX6*NZ6+NX6],   YPara_3[j], f[idx+4*NX6*NZ6+NX6],   YPara_4[j], f[idx+5*NX6*NZ6+NX6],   YPara_5[j], f[idx+6*NX6*NZ6+NX6],   YPara_6[j] ), ZPara_1[k],       \
        Intrpl7( f[idx+2*NX6], YPara_0[j], f[idx+NX6*NZ6+2*NX6], YPara_1[j], f[idx+2*NX6*NZ6+2*NX6], YPara_2[j], f[idx+3*NX6*NZ6+2*NX6], YPara_3[j], f[idx+4*NX6*NZ6+2*NX6], YPara_4[j], f[idx+5*NX6*NZ6+2*NX6], YPara_5[j], f[idx+6*NX6*NZ6+2*NX6], YPara_6[j] ), ZPara_2[k],       \
        Intrpl7( f[idx+3*NX6], YPara_0[j], f[idx+NX6*NZ6+3*NX6], YPara_1[j], f[idx+2*NX6*NZ6+3*NX6], YPara_2[j], f[idx+3*NX6*NZ6+3*NX6], YPara_3[j], f[idx+4*NX6*NZ6+3*NX6], YPara_4[j], f[idx+5*NX6*NZ6+3*NX6], YPara_5[j], f[idx+6*NX6*NZ6+3*NX6], YPara_6[j] ), ZPara_3[k],       \
        Intrpl7( f[idx+4*NX6], YPara_0[j], f[idx+NX6*NZ6+4*NX6], YPara_1[j], f[idx+2*NX6*NZ6+4*NX6], YPara_2[j], f[idx+3*NX6*NZ6+4*NX6], YPara_3[j], f[idx+4*NX6*NZ6+4*NX6], YPara_4[j], f[idx+5*NX6*NZ6+4*NX6], YPara_5[j], f[idx+6*NX6*NZ6+4*NX6], YPara_6[j] ), ZPara_4[k],       \
        Intrpl7( f[idx+5*NX6], YPara_0[j], f[idx+NX6*NZ6+5*NX6], YPara_1[j], f[idx+2*NX6*NZ6+5*NX6], YPara_2[j], f[idx+3*NX6*NZ6+5*NX6], YPara_3[j], f[idx+4*NX6*NZ6+5*NX6], YPara_4[j], f[idx+5*NX6*NZ6+5*NX6], YPara_5[j], f[idx+6*NX6*NZ6+5*NX6], YPara_6[j] ), ZPara_5[k],       \
        Intrpl7( f[idx+6*NX6], YPara_0[j], f[idx+NX6*NZ6+6*NX6], YPara_1[j], f[idx+2*NX6*NZ6+6*NX6], YPara_2[j], f[idx+3*NX6*NZ6+6*NX6], YPara_3[j], f[idx+4*NX6*NZ6+6*NX6], YPara_4[j], f[idx+5*NX6*NZ6+6*NX6], YPara_5[j], f[idx+6*NX6*NZ6+6*NX6], YPara_6[j] ), ZPara_6[k]        \
    );

#define F17_Intrpl7(f, i, j, k, i_c, j_c, k_c, YPara_0, YPara_1, YPara_2, YPara_3, YPara_4, YPara_5, YPara_6, ZPara_0, ZPara_1, ZPara_2, ZPara_3, ZPara_4, ZPara_5, ZPara_6)   \
    idx = j_c*NX6*NZ6 + k_c*NX6 + i;    \
    F17_in = Intrpl7(    \
        Intrpl7( f[idx],       YPara_0[j], f[idx+NX6*NZ6],       YPara_1[j], f[idx+2*NX6*NZ6],       YPara_2[j], f[idx+3*NX6*NZ6],       YPara_3[j], f[idx+4*NX6*NZ6],       YPara_4[j], f[idx+5*NX6*NZ6],       YPara_5[j], f[idx+6*NX6*NZ6],       YPara_6[j] ), ZPara_0[k],       \
        Intrpl7( f[idx+NX6],   YPara_0[j], f[idx+NX6*NZ6+NX6],   YPara_1[j], f[idx+2*NX6*NZ6+NX6],   YPara_2[j], f[idx+3*NX6*NZ6+NX6],   YPara_3[j], f[idx+4*NX6*NZ6+NX6],   YPara_4[j], f[idx+5*NX6*NZ6+NX6],   YPara_5[j], f[idx+6*NX6*NZ6+NX6],   YPara_6[j] ), ZPara_1[k],       \
        Intrpl7( f[idx+2*NX6], YPara_0[j], f[idx+NX6*NZ6+2*NX6], YPara_1[j], f[idx+2*NX6*NZ6+2*NX6], YPara_2[j], f[idx+3*NX6*NZ6+2*NX6], YPara_3[j], f[idx+4*NX6*NZ6+2*NX6], YPara_4[j], f[idx+5*NX6*NZ6+2*NX6], YPara_5[j], f[idx+6*NX6*NZ6+2*NX6], YPara_6[j] ), ZPara_2[k],       \
        Intrpl7( f[idx+3*NX6], YPara_0[j], f[idx+NX6*NZ6+3*NX6], YPara_1[j], f[idx+2*NX6*NZ6+3*NX6], YPara_2[j], f[idx+3*NX6*NZ6+3*NX6], YPara_3[j], f[idx+4*NX6*NZ6+3*NX6], YPara_4[j], f[idx+5*NX6*NZ6+3*NX6], YPara_5[j], f[idx+6*NX6*NZ6+3*NX6], YPara_6[j] ), ZPara_3[k],       \
        Intrpl7( f[idx+4*NX6], YPara_0[j], f[idx+NX6*NZ6+4*NX6], YPara_1[j], f[idx+2*NX6*NZ6+4*NX6], YPara_2[j], f[idx+3*NX6*NZ6+4*NX6], YPara_3[j], f[idx+4*NX6*NZ6+4*NX6], YPara_4[j], f[idx+5*NX6*NZ6+4*NX6], YPara_5[j], f[idx+6*NX6*NZ6+4*NX6], YPara_6[j] ), ZPara_4[k],       \
        Intrpl7( f[idx+5*NX6], YPara_0[j], f[idx+NX6*NZ6+5*NX6], YPara_1[j], f[idx+2*NX6*NZ6+5*NX6], YPara_2[j], f[idx+3*NX6*NZ6+5*NX6], YPara_3[j], f[idx+4*NX6*NZ6+5*NX6], YPara_4[j], f[idx+5*NX6*NZ6+5*NX6], YPara_5[j], f[idx+6*NX6*NZ6+5*NX6], YPara_6[j] ), ZPara_5[k],       \
        Intrpl7( f[idx+6*NX6], YPara_0[j], f[idx+NX6*NZ6+6*NX6], YPara_1[j], f[idx+2*NX6*NZ6+6*NX6], YPara_2[j], f[idx+3*NX6*NZ6+6*NX6], YPara_3[j], f[idx+4*NX6*NZ6+6*NX6], YPara_4[j], f[idx+5*NX6*NZ6+6*NX6], YPara_5[j], f[idx+6*NX6*NZ6+6*NX6], YPara_6[j] ), ZPara_6[k]        \
    );

#define F18_Intrpl7(f, i, j, k, i_c, j_c, k_c, YPara_0, YPara_1, YPara_2, YPara_3, YPara_4, YPara_5, YPara_6, ZPara_0, ZPara_1, ZPara_2, ZPara_3, ZPara_4, ZPara_5, ZPara_6)   \
    idx = j_c*NX6*NZ6 + k_c*NX6 + i;    \
    F18_in = Intrpl7(    \
        Intrpl7( f[idx],       YPara_0[j], f[idx+NX6*NZ6],       YPara_1[j], f[idx+2*NX6*NZ6],       YPara_2[j], f[idx+3*NX6*NZ6],       YPara_3[j], f[idx+4*NX6*NZ6],       YPara_4[j], f[idx+5*NX6*NZ6],       YPara_5[j], f[idx+6*NX6*NZ6],       YPara_6[j] ), ZPara_0[k],       \
        Intrpl7( f[idx+NX6],   YPara_0[j], f[idx+NX6*NZ6+NX6],   YPara_1[j], f[idx+2*NX6*NZ6+NX6],   YPara_2[j], f[idx+3*NX6*NZ6+NX6],   YPara_3[j], f[idx+4*NX6*NZ6+NX6],   YPara_4[j], f[idx+5*NX6*NZ6+NX6],   YPara_5[j], f[idx+6*NX6*NZ6+NX6],   YPara_6[j] ), ZPara_1[k],       \
        Intrpl7( f[idx+2*NX6], YPara_0[j], f[idx+NX6*NZ6+2*NX6], YPara_1[j], f[idx+2*NX6*NZ6+2*NX6], YPara_2[j], f[idx+3*NX6*NZ6+2*NX6], YPara_3[j], f[idx+4*NX6*NZ6+2*NX6], YPara_4[j], f[idx+5*NX6*NZ6+2*NX6], YPara_5[j], f[idx+6*NX6*NZ6+2*NX6], YPara_6[j] ), ZPara_2[k],       \
        Intrpl7( f[idx+3*NX6], YPara_0[j], f[idx+NX6*NZ6+3*NX6], YPara_1[j], f[idx+2*NX6*NZ6+3*NX6], YPara_2[j], f[idx+3*NX6*NZ6+3*NX6], YPara_3[j], f[idx+4*NX6*NZ6+3*NX6], YPara_4[j], f[idx+5*NX6*NZ6+3*NX6], YPara_5[j], f[idx+6*NX6*NZ6+3*NX6], YPara_6[j] ), ZPara_3[k],       \
        Intrpl7( f[idx+4*NX6], YPara_0[j], f[idx+NX6*NZ6+4*NX6], YPara_1[j], f[idx+2*NX6*NZ6+4*NX6], YPara_2[j], f[idx+3*NX6*NZ6+4*NX6], YPara_3[j], f[idx+4*NX6*NZ6+4*NX6], YPara_4[j], f[idx+5*NX6*NZ6+4*NX6], YPara_5[j], f[idx+6*NX6*NZ6+4*NX6], YPara_6[j] ), ZPara_4[k],       \
        Intrpl7( f[idx+5*NX6], YPara_0[j], f[idx+NX6*NZ6+5*NX6], YPara_1[j], f[idx+2*NX6*NZ6+5*NX6], YPara_2[j], f[idx+3*NX6*NZ6+5*NX6], YPara_3[j], f[idx+4*NX6*NZ6+5*NX6], YPara_4[j], f[idx+5*NX6*NZ6+5*NX6], YPara_5[j], f[idx+6*NX6*NZ6+5*NX6], YPara_6[j] ), ZPara_5[k],       \
        Intrpl7( f[idx+6*NX6], YPara_0[j], f[idx+NX6*NZ6+6*NX6], YPara_1[j], f[idx+2*NX6*NZ6+6*NX6], YPara_2[j], f[idx+3*NX6*NZ6+6*NX6], YPara_3[j], f[idx+4*NX6*NZ6+6*NX6], YPara_4[j], f[idx+5*NX6*NZ6+6*NX6], YPara_5[j], f[idx+6*NX6*NZ6+6*NX6], YPara_6[j] ), ZPara_6[k]        \
    );

#endif