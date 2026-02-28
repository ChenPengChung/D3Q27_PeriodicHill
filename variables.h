#ifndef VARIABLES_FILE
#define VARIABLES_FILE

//stream wise : LX, for lid-driven cavity flow
#define     pi     3.14159265358979323846264338327950
#define     LX     (4.5)
#define     LY     (9.0)
#define     LZ     (3.036)
#define     H_HILL (1.0)        // hill height (Re_h reference length)

//global grid numbers of each direction
#define     NX      32
#define     NY      128
#define     NZ      128

#define     jp      4

#define     NX6    (NX+7)
#define     NYD6   (NY/jp+7)
#define     NY6    (NY+7)
#define     NZ6    (NZ+6)

//coefficient for non-uniform grid
#define     CFL                 0.5
#define     minSize             ((LZ-1.0)/(NZ6-6)*CFL)
//1 : Yes,  0 : No
#define     Uniform_In_Xdir     1
#define     Uniform_In_Ydir     1
#define     Uniform_In_Zdir     0

#define     LXi        (10.0)

#define     TBSWITCH            (1)

// Collision operator: 0=BGK/SRT, 1=MRT (Multi-Relaxation-Time)
#define     USE_MRT             1

//#define     Re         300
//#define     U_0        0.1018591

//#define     Retau       180
//#define     Ma          0.1
//#define     Umax        18.4824

#define     Re          700

//steps to end simulation
#define     loop      500000
//how many time steps to output val of monitor point(NX/2, NY/2, NZ/2)
#define		NDTMIT	   50
//how many time steps to modify the forcing term
#define     NDTFRC     1000 //每一萬步驟-更新外力
#define     force_alpha 3 //瑋傑學長的論文:alpha = 3~14 
//Re=100  , alpha = 10 
//Re=2800 , alpha = 3 or 14 
//After a few transients (∼ 200 ﬂow-throughtime), the velocity is time-averaged. As show
//無因次化時間步 0.67L/U_reference 
//whether to initial from the backup file
//0 : from initialization
//1 : from backup file
//2 : from merged VTK file (specify RESTART_VTK_FILE below)
#define     INIT    (0)   //2代表使用初始化資料 
#define     TBINIT  (1)
#define     RESTART_VTK_FILE  "result/velocity_merged_005001.vtk"
/****************** SECONDARY PARAMETER ******************/
#define     cs          (1.0/1.732050807568877)
#define     dt          minSize //因為直角坐標系中，c=1
#define     Uref        0.0583  //不可以任意提高，否則變向雷諾數降低
//[Senior Data] : Re700:0.0583 , Re14002800:0.0776 , Re5600:0.0464 , Re10595:0.0878 <=0.17320508075 //<= 0.17320508075
#define     niu         Uref/Re
// Flow-through time: T_FT = L / Uref (lattice time units)
// 論文 Fig.5 x軸: T*Uref/L, 其中 L = LY = 9h (hill-to-hill streamwise periodic length)
// 一個 FTT = LY/Uref 個 lattice time steps
// 第 n 步的 FTT 數 = n * dt_global / (LY / Uref) = n * dt_global * Uref / LY
// 注意: 曲線坐標系用 dt_global (runtime), 非 dt=minSize (直角坐標 compile-time)
#define     flow_through_time  (LY / Uref)
//block size of each direction
#define     NT          32     //block size x-dir threadnum
#endif
/*
直角坐標系：tau1 = 3*niu/dt + 0.5 ; dt = minSize  //tau1在本程式碼沒有使用到因此不需要定義
曲線座標系：omega_global = 3*niu/dt_global + 0.5 ;
曲線座標系(local time step) : omega_local(j,k) = 3*niu/dt_local(j,k) + 0.5 ;
*/





















