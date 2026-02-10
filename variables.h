#ifndef VARIABLES_FILE
#define VARIABLES_FILE

//stream wise : LX, for lid-driven cavity flow
#define     pi     3.14159265358979323846264338327950
#define     LX     (4.5)
#define     LY     (9.0)
#define     LZ     (3.036)
//global grid numbers of each direction
#define     NX      32
#define     NY      128
#define     NZ      128     // Doubled from 64 for Re=700 stability

#define     jp      4

#define     NX6    (NX+7)
#define     NYD6   (NY/jp+7)
#define     NY6    (NY+7)
#define     NZ6    (NZ+6)

//coefficient for non-uniform grid
#define     CFL                 0.6
#define     minSize             ((LZ-1.0)/(NZ6-6)*CFL)
//#define     minSize             (LZ/(NZ6-6)*CFL)
//1 : Yes,  0 : No
#define     Uniform_In_Xdir     1
#define     Uniform_In_Ydir     1
#define     Uniform_In_Zdir     0

#define     LXi        (10.0)

#define     TBSWITCH            (0)

//#define     Re         300
//#define     U_0        0.1018591

//#define     Retau       180
//#define     Ma          0.1
//#define     Umax        18.4824

#define     Re          700

//steps to end simulation
#define     loop      1000000
//how many time steps to output val of monitor point(NX/2, NY/2, NZ/2)
#define		NDTMIT	   50
//how many time steps to modify the forcing term
#define     NDTFRC     10000

//whether to initial from the backup file
//0 : from initialization
//1 : from backup file
#define     INIT    (0)
#define     TBINIT  (0)




















/****************** SECDONARY PARAMETER ******************/
#define     dt                  (minSize)
#define     cs                  (1.0/1.732050807568877)

//Parameters of 3-D Taylor-Green vortex
//#define     niu                 (U_0*LX/2.0/pi/Re)
//#define     tau                 (niu*3.0/dt+0.5)

//Parameters of test case
//#define     Force               0.0001
//#define     tau                 1.0
//#define     niu                 ((tau-0.5)/3.0*dt)

//Parameters of turbulent channel flow
//#define     niu         (Ma*cs*LZ/Umax/2.0/Retau)
//#define     tauw        (Ma*Ma/Umax/Umax/3.0)
//#define     Force       (tauw*2.0/LZ)
//#define     tau         (3.0*niu/dt+0.5)

//Parameters of periodic hills
#define     tau          0.6833
//#define     alpha       10.0
#define     niu         ((tau-0.5)/3.0*dt)

//Smagorinsky LES subgrid model parameters
#define     SMAGORINSKY  1          // 1=enable LES, 0=disable
#define     C_Smag       0.2        // Smagorinsky constant (typical: 0.1-0.2)
#define     DELTA        (1.0)      // Filter width in lattice units
//#define     Uref        (Re*niu/LZ)
#define     Uref        (Re*niu)

//block size of each direction
#define     NT          32     //block size x-dir threadnum


#endif
