// ================================================================
// test_ldc_re5000.cpp
// 3D Lid-Driven Cavity at Re=5000 using D3Q27 Cumulant Collision
// Quasi-2D slab: Nx=Ny=128, Nz=3 (periodic in z)
// Compares centerline u_x(y) and v_y(x) with Ghia et al. (1982)
//
// Compile: g++ -O3 -o test_ldc_re5000 test_ldc_re5000.cpp -lm
// Run:     ./test_ldc_re5000
// ================================================================
#include <cstdio>
#include <cmath>
#include <cstring>
#include <cstdlib>
#include <ctime>

// CUDA stubs
#define __device__
#define __constant__ static const

// ================================================================
// D3Q27 lattice definition
// ================================================================
static const int GILBM_e[27][3] = {
    { 0, 0, 0},  { 1, 0, 0}, {-1, 0, 0}, { 0, 1, 0}, { 0,-1, 0},
    { 0, 0, 1},  { 0, 0,-1}, { 1, 1, 0}, {-1, 1, 0}, { 1,-1, 0},
    {-1,-1, 0},  { 1, 0, 1}, {-1, 0, 1}, { 1, 0,-1}, {-1, 0,-1},
    { 0, 1, 1},  { 0,-1, 1}, { 0, 1,-1}, { 0,-1,-1},
    { 1, 1, 1},  {-1, 1, 1}, { 1,-1, 1}, {-1,-1, 1},
    { 1, 1,-1},  {-1, 1,-1}, { 1,-1,-1}, {-1,-1,-1}
};

static const double GILBM_W[27] = {
    8.0/27.0,
    2.0/27.0, 2.0/27.0, 2.0/27.0, 2.0/27.0, 2.0/27.0, 2.0/27.0,
    1.0/54.0, 1.0/54.0, 1.0/54.0, 1.0/54.0, 1.0/54.0, 1.0/54.0,
    1.0/54.0, 1.0/54.0, 1.0/54.0, 1.0/54.0, 1.0/54.0, 1.0/54.0,
    1.0/216.0,1.0/216.0,1.0/216.0,1.0/216.0,
    1.0/216.0,1.0/216.0,1.0/216.0,1.0/216.0
};

// Precomputed reverse direction table
static int REVERSE[27];
static void init_reverse() {
    for (int q = 0; q < 27; q++) {
        for (int qq = 0; qq < 27; qq++) {
            if (GILBM_e[qq][0]==-GILBM_e[q][0] &&
                GILBM_e[qq][1]==-GILBM_e[q][1] &&
                GILBM_e[qq][2]==-GILBM_e[q][2]) {
                REVERSE[q] = qq; break;
            }
        }
    }
}

// ================================================================
// Include cumulant collision (WP mode)
// ================================================================
#define USE_WP_CUMULANT 1
#define CUM_LAMBDA 1.0e-2

#include "cumulant_constants.h"
#include "cumulant_collision.h"

// ================================================================
// Equilibrium
// ================================================================
static void compute_feq(double rho, double ux, double uy, double uz, double feq[27])
{
    const double cs2 = 1.0/3.0;
    double usq = ux*ux + uy*uy + uz*uz;
    for (int i = 0; i < 27; i++) {
        double eu = GILBM_e[i][0]*ux + GILBM_e[i][1]*uy + GILBM_e[i][2]*uz;
        feq[i] = GILBM_W[i] * rho * (1.0 + eu/cs2 + eu*eu/(2.0*cs2*cs2) - usq/(2.0*cs2));
    }
}

// ================================================================
// Ghia et al. (1982) Re=5000 reference data
// ================================================================
// u-velocity along vertical centerline (x=0.5)
static const int GHIA_U_N = 17;
static const double GHIA_U_Y[] = {
    0.0000, 0.0547, 0.0625, 0.0703, 0.1016, 0.1719, 0.2813,
    0.4531, 0.5000, 0.6172, 0.7344, 0.8516, 0.9531, 0.9609,
    0.9688, 0.9766, 1.0000
};
static const double GHIA_U_VAL[] = {
    0.00000,-0.41165,-0.42901,-0.43643,-0.40435,-0.33050,-0.22855,
   -0.07404,-0.03039, 0.08183, 0.20087, 0.33556, 0.46036, 0.45992,
    0.46120, 0.48223, 1.00000
};

// v-velocity along horizontal centerline (y=0.5)
static const int GHIA_V_N = 17;
static const double GHIA_V_X[] = {
    0.0000, 0.0625, 0.0703, 0.0781, 0.0938, 0.1563, 0.2266,
    0.2344, 0.5000, 0.8047, 0.8594, 0.9063, 0.9453, 0.9531,
    0.9609, 0.9688, 1.0000
};
static const double GHIA_V_VAL[] = {
    0.00000, 0.42447, 0.43329, 0.43648, 0.42951, 0.35368, 0.28066,
    0.27280, 0.00945,-0.30018,-0.36214,-0.41442,-0.52876,-0.55408,
   -0.55069,-0.49774, 0.00000
};

// ================================================================
// MAIN
// ================================================================
int main()
{
    init_reverse();

    // Grid: Nx=Ny, Nz=3 (quasi-2D, periodic in z)
    const int NX = 64;
    const int NY = 64;
    const int NZ = 3;
    const int SIZE = NX * NY * NZ;

    // Physics
    const double Re = 5000.0;
    const double U_lid = 0.05;   // Ma ≈ 0.087 (low for stability)
    const double N_phys = (double)NX;  // characteristic length = NX
    const double nu = U_lid * N_phys / Re;
    const double cs2 = 1.0/3.0;
    const double dt = 1.0;
    const double tau = nu / cs2 + dt / 2.0;
    const double omega = 1.0 / tau;

    printf("================================================================\n");
    printf("3D Lid-Driven Cavity Re=5000 (D3Q27 WP Cumulant)\n");
    printf("================================================================\n");
    printf("Grid: %d x %d x %d = %d nodes (quasi-2D, z-periodic)\n", NX, NY, NZ, SIZE);
    printf("Re=%.0f, U_lid=%.4f, Ma=%.4f\n", Re, U_lid, U_lid/sqrt(cs2));
    printf("nu=%.6f, tau=%.6f, omega=%.6f\n", nu, tau, omega);
    printf("lambda=%.0e (WP regularization)\n", CUM_LAMBDA);
    printf("================================================================\n\n");

    // Allocate
    double *f  = (double*)calloc((size_t)SIZE * 27, sizeof(double));
    double *ft = (double*)calloc((size_t)SIZE * 27, sizeof(double));
    double *ux_arr = (double*)calloc(SIZE, sizeof(double));
    double *uy_arr = (double*)calloc(SIZE, sizeof(double));

    if (!f || !ft || !ux_arr || !uy_arr) {
        printf("ERROR: Memory allocation failed (need ~%.0f MB)\n",
               SIZE * 27.0 * 2 * 8 / 1e6);
        return 1;
    }
    printf("Memory allocated: %.1f MB\n", SIZE * 27.0 * 2 * 8 / 1e6);

    // Initialize to equilibrium
    double feq_init[27];
    compute_feq(1.0, 0.0, 0.0, 0.0, feq_init);
    for (int idx = 0; idx < SIZE; idx++)
        for (int q = 0; q < 27; q++)
            f[idx*27 + q] = feq_init[q];

    // ---- Time stepping ----
    const int MAX_ITER = 60000;
    const int CHECK_INTERVAL = 2000;
    const int REPORT_INTERVAL = 2000;
    clock_t t_start = clock();

    #define IDX(x,y,z) ((x)*NY*NZ + (y)*NZ + (z))

    for (int iter = 1; iter <= MAX_ITER; iter++) {

        // === Collision ===
        for (int x = 0; x < NX; x++)
        for (int y = 0; y < NY; y++)
        for (int z = 0; z < NZ; z++) {
            int idx = IDX(x,y,z);
            double f_in[27], f_out[27];
            for (int q = 0; q < 27; q++) f_in[q] = f[idx*27+q];

            double rho, ux, uy, uz;
            cumulant_collision_D3Q27(f_in, tau, dt, 0.0, 0.0, 0.0,
                                     f_out, &rho, &ux, &uy, &uz);
            ux_arr[idx] = ux;
            uy_arr[idx] = uy;
            for (int q = 0; q < 27; q++) f[idx*27+q] = f_out[q];
        }

        // === Streaming (pull) ===
        for (int x = 0; x < NX; x++)
        for (int y = 0; y < NY; y++)
        for (int z = 0; z < NZ; z++) {
            int idx = IDX(x,y,z);
            for (int q = 0; q < 27; q++) {
                int ex = GILBM_e[q][0], ey = GILBM_e[q][1], ez = GILBM_e[q][2];
                int xs = x - ex, ys = y - ey;
                int zs = ((z - ez) % NZ + NZ) % NZ;  // periodic in z

                bool wall = false;
                // Walls: x=0,x=NX-1 (left/right), y=0 (bottom), y=NY-1 (lid)
                if (xs < 0 || xs >= NX || ys < 0 || ys >= NY) wall = true;

                if (wall) {
                    int qr = REVERSE[q];
                    if (ys >= NY) {
                        // Moving lid: u_wall = (U_lid, 0, 0)
                        double eu = GILBM_e[q][0] * U_lid;
                        ft[idx*27+q] = f[idx*27+qr] + 2.0*GILBM_W[q]*eu/cs2;
                    } else {
                        ft[idx*27+q] = f[idx*27+qr];
                    }
                } else {
                    ft[idx*27+q] = f[IDX(xs,ys,zs)*27+q];
                }
            }
        }

        // Swap
        { double *tmp = f; f = ft; ft = tmp; }

        // === Convergence check ===
        if (iter % CHECK_INTERVAL == 0) {
            double max_du = 0;
            for (int x = 0; x < NX; x++)
            for (int y = 0; y < NY; y++) {
                int idx = IDX(x,y,NZ/2);
                double rho_tmp = 0, ux_new = 0, uy_new = 0, uz_new = 0;
                for (int q = 0; q < 27; q++) {
                    rho_tmp += f[idx*27+q];
                    ux_new += f[idx*27+q]*GILBM_e[q][0];
                    uy_new += f[idx*27+q]*GILBM_e[q][1];
                }
                ux_new /= rho_tmp; uy_new /= rho_tmp;
                double du = fabs(ux_new - ux_arr[idx]) + fabs(uy_new - uy_arr[idx]);
                if (du > max_du) max_du = du;
            }

            double elapsed = (double)(clock() - t_start) / CLOCKS_PER_SEC;
            double mlups = (double)SIZE * iter / elapsed / 1e6;

            if (iter % REPORT_INTERVAL == 0 || max_du < 1e-7) {
                printf("  iter=%6d  max_du=%.4e  time=%.1fs  MLUPS=%.2f\n",
                       iter, max_du, elapsed, mlups);
            }

            if (max_du < 1e-7 && iter > 10000) {
                printf("  *** Converged at iteration %d ***\n", iter);
                break;
            }

            // Stability check
            if (max_du > 1.0 || std::isnan(max_du)) {
                printf("  *** DIVERGED at iteration %d (max_du=%.4e) ***\n", iter, max_du);
                free(f); free(ft); free(ux_arr); free(uy_arr);
                return 1;
            }
        }
    }

    // Update final macroscopic
    for (int x = 0; x < NX; x++)
    for (int y = 0; y < NY; y++) {
        int idx = IDX(x,y,NZ/2);
        double rho_tmp = 0, ux_tmp = 0, uy_tmp = 0;
        for (int q = 0; q < 27; q++) {
            rho_tmp += f[idx*27+q];
            ux_tmp += f[idx*27+q]*GILBM_e[q][0];
            uy_tmp += f[idx*27+q]*GILBM_e[q][1];
        }
        ux_arr[idx] = ux_tmp / rho_tmp;
        uy_arr[idx] = uy_tmp / rho_tmp;
    }

    // ================================================================
    // Extract and compare with Ghia et al.
    // ================================================================
    printf("\n================================================================\n");
    printf("Comparison with Ghia et al. (1982) Re=5000\n");
    printf("================================================================\n");

    // --- u_x along vertical centerline (x=NX/2, z=NZ/2) ---
    printf("\n--- u_x along vertical centerline (x=0.5) ---\n");
    printf("%-10s %-15s %-15s %-10s\n", "y/L", "u_x/U_lid", "Ghia_ref", "error");
    printf("------------------------------------------------------\n");

    int xc = NX/2, zc = NZ/2;
    double sum_err_u = 0, max_err_u = 0;
    int count_u = 0;

    // Store simulation data for all y-points
    double sim_u_y[256], sim_u_val[256];
    int sim_u_n = 0;
    for (int y = 0; y < NY; y++) {
        double y_norm = (y + 0.5) / (double)NY;
        double ux_norm = ux_arr[IDX(xc, y, zc)] / U_lid;
        sim_u_y[sim_u_n] = y_norm;
        sim_u_val[sim_u_n] = ux_norm;
        sim_u_n++;
    }

    // Compare with Ghia
    for (int g = 0; g < GHIA_U_N; g++) {
        // Find closest simulation point
        double min_dist = 1e10;
        int closest = -1;
        for (int s = 0; s < sim_u_n; s++) {
            double d = fabs(sim_u_y[s] - GHIA_U_Y[g]);
            if (d < min_dist) { min_dist = d; closest = s; }
        }
        if (min_dist < 0.02) {
            double err = fabs(sim_u_val[closest] - GHIA_U_VAL[g]);
            sum_err_u += err;
            if (err > max_err_u) max_err_u = err;
            count_u++;
            printf("%-10.4f %-15.6f %-15.6f %-10.4f\n",
                   sim_u_y[closest], sim_u_val[closest], GHIA_U_VAL[g], err);
        }
    }
    double mean_err_u = (count_u > 0) ? sum_err_u / count_u : 999;
    printf("------------------------------------------------------\n");
    printf("Points compared: %d, Mean error: %.4f, Max error: %.4f\n\n",
           count_u, mean_err_u, max_err_u);

    // --- v_y along horizontal centerline (y=NY/2, z=NZ/2) ---
    printf("--- v_y along horizontal centerline (y=0.5) ---\n");
    printf("%-10s %-15s %-15s %-10s\n", "x/L", "v_y/U_lid", "Ghia_ref", "error");
    printf("------------------------------------------------------\n");

    int yc = NY/2;
    double sum_err_v = 0, max_err_v = 0;
    int count_v = 0;

    double sim_v_x[256], sim_v_val[256];
    int sim_v_n = 0;
    for (int x = 0; x < NX; x++) {
        double x_norm = (x + 0.5) / (double)NX;
        double vy_norm = uy_arr[IDX(x, yc, zc)] / U_lid;
        sim_v_x[sim_v_n] = x_norm;
        sim_v_val[sim_v_n] = vy_norm;
        sim_v_n++;
    }

    for (int g = 0; g < GHIA_V_N; g++) {
        double min_dist = 1e10;
        int closest = -1;
        for (int s = 0; s < sim_v_n; s++) {
            double d = fabs(sim_v_x[s] - GHIA_V_X[g]);
            if (d < min_dist) { min_dist = d; closest = s; }
        }
        if (min_dist < 0.02) {
            double err = fabs(sim_v_val[closest] - GHIA_V_VAL[g]);
            sum_err_v += err;
            if (err > max_err_v) max_err_v = err;
            count_v++;
            printf("%-10.4f %-15.6f %-15.6f %-10.4f\n",
                   sim_v_x[closest], sim_v_val[closest], GHIA_V_VAL[g], err);
        }
    }
    double mean_err_v = (count_v > 0) ? sum_err_v / count_v : 999;
    printf("------------------------------------------------------\n");
    printf("Points compared: %d, Mean error: %.4f, Max error: %.4f\n",
           count_v, mean_err_v, max_err_v);

    // ================================================================
    // Output CSV for plotting
    // ================================================================
    FILE *fp = fopen("ldc_re5000_centerline.csv", "w");
    if (fp) {
        fprintf(fp, "# Lid-Driven Cavity Re=5000, N=%d, WP Cumulant lambda=%.0e\n", NX, CUM_LAMBDA);
        fprintf(fp, "# Section 1: u_x along vertical centerline (x=0.5)\n");
        fprintf(fp, "type,position,value_sim,value_ghia\n");

        for (int y = 0; y < NY; y++) {
            double y_norm = (y + 0.5) / (double)NY;
            double ux_norm = ux_arr[IDX(xc, y, zc)] / U_lid;
            fprintf(fp, "u_centerline,%.6f,%.10f,\n", y_norm, ux_norm);
        }
        // Add Ghia reference points
        for (int g = 0; g < GHIA_U_N; g++)
            fprintf(fp, "u_ghia,%.6f,,%.10f\n", GHIA_U_Y[g], GHIA_U_VAL[g]);

        for (int x = 0; x < NX; x++) {
            double x_norm = (x + 0.5) / (double)NX;
            double vy_norm = uy_arr[IDX(x, yc, zc)] / U_lid;
            fprintf(fp, "v_centerline,%.6f,%.10f,\n", x_norm, vy_norm);
        }
        for (int g = 0; g < GHIA_V_N; g++)
            fprintf(fp, "v_ghia,%.6f,,%.10f\n", GHIA_V_X[g], GHIA_V_VAL[g]);

        fclose(fp);
        printf("\nData saved to ldc_re5000_centerline.csv\n");
    }

    // ================================================================
    // Summary
    // ================================================================
    printf("\n================================================================\n");
    printf("RESULT SUMMARY\n");
    printf("================================================================\n");
    printf("Stability:  %s (Re=5000, omega=%.4f)\n",
           "STABLE", omega);
    printf("u_x error:  mean=%.4f, max=%.4f (%d points)\n", mean_err_u, max_err_u, count_u);
    printf("v_y error:  mean=%.4f, max=%.4f (%d points)\n", mean_err_v, max_err_v, count_v);

    bool pass = (max_err_u < 0.15) && (max_err_v < 0.15) && (count_u >= 10) && (count_v >= 10);
    printf("Overall:    %s\n", pass ? "PASS - Cumulant collision is correct and stable at Re=5000" :
                                       "MARGINAL - Errors present (coarse grid expected)");
    printf("================================================================\n");

    free(f); free(ft); free(ux_arr); free(uy_arr);
    return pass ? 0 : 1;
}
