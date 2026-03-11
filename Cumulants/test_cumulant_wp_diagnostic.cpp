// ================================================================
// test_cumulant_wp_diagnostic.cpp
// Standalone CLI tool: Cumulant-WP parameter singularity diagnostic
//
// Two modes:
//   (1) Interactive:  ./test_cumulant_wp_diagnostic
//       Program prompts for Re, Uref, omega2, dt_global
//
//   (2) Parameter:    ./test_cumulant_wp_diagnostic <Re> <Uref> [omega2] [dt_global]
//       Re, Uref required; omega2 default 1.0; dt_global default estimated from grid
//
// dt_global note:
//   dt_global is computed by Imamura GTS from Jacobian metric terms.
//   If not provided, the program estimates from variables.h grid parameters (minSize).
//   If you have run a simulation, find "dt_global = X.XXXe-XX" in the log.
//
// Compile: g++ -O2 -std=c++17 -o test_cumulant_wp_diagnostic test_cumulant_wp_diagnostic.cpp -lm
// ================================================================

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <iostream>
#include <string>

// Include the diagnostic header (pure C++, no CUDA dependencies)
#include "cumulant_wp_diagnostic.h"

// ================================================================
// Grid constants from variables.h (for estimating dt_global)
// If you change variables.h, update these accordingly
// ================================================================
static const double GRID_LZ     = 3.036;
static const double GRID_H_HILL = 1.0;
static const int    GRID_NZ     = 192;
static const int    GRID_NZ6    = GRID_NZ + 6;
static const double GRID_CFL    = 0.5;

static double estimate_dt_global()
{
    // minSize = (LZ - H_HILL) / (NZ6 - 6) * CFL
    // dt_global ~ minSize (Cartesian) or smaller (curvilinear due to Jacobian)
    double minSize = (GRID_LZ - GRID_H_HILL) / (double)(GRID_NZ6 - 6) * GRID_CFL;
    // For curvilinear coordinates, dt_global is typically 2~5x smaller than minSize
    // Return minSize as upper-bound estimate; actual value from simulation log
    return minSize;
}

// ================================================================
// omega2 scan: find the best omega2 that maximizes min singularity distance
// ================================================================
static void scan_best_omega2(int Re, double Uref, double dt_global)
{
    printf("\n  -- omega2 scan: finding best omega2 ----------------------------------\n");
    printf("  (maximize min singularity distance from your omega1)\n\n");
    printf("  %6s | %8s | %8s | %8s | %8s | %8s | %s\n",
           "omega2", "w4_sing", "w4_dist", "AB_dist", "min_dist", "w4_used", "verdict");
    printf("  %6s-+-%8s-+-%8s-+-%8s-+-%8s-+-%8s-+-%s\n",
           "------", "--------", "--------", "--------", "--------", "--------", "-------");

    double best_w2 = 1.0, best_min = 0.0;

    for (int i = 5; i <= 18; i++) {
        double w2 = 0.1 * i;
        CumWP_OmegaReport rep = CumulantWP_ComputeReport(Re, Uref, dt_global, w2);

        // min distance (including w3 and w5)
        double min_d = rep.w4_sing_dist;
        if (rep.AB_min_dist < min_d) min_d = rep.AB_min_dist;
        if (rep.w3_sing_dist < min_d) min_d = rep.w3_sing_dist;
        if (rep.w5_sing_dist < min_d) min_d = rep.w5_sing_dist;

        const char *verdict = (min_d > 0.3)  ? "SAFE"
                            : (min_d > 0.15) ? "OK"
                            : "DANGER";

        printf("  %6.2f | %8.4f | %8.4f | %8.4f | %8.4f | %8.4f | %s",
               w2,
               (rep.w4_sing > 0.0) ? rep.w4_sing : -1.0,
               rep.w4_sing_dist < 90 ? rep.w4_sing_dist : -1.0,
               rep.AB_min_dist < 90 ? rep.AB_min_dist : -1.0,
               min_d < 90 ? min_d : -1.0,
               rep.w4_used,
               verdict);

        if (fabs(w2 - 1.0) < 0.01)  printf("  <-- current");
        printf("\n");

        if (min_d > best_min) {
            best_min = min_d;
            best_w2 = w2;
        }
    }

    printf("\n  * Best omega2 = %.2f (min singularity distance = %.4f)\n\n", best_w2, best_min);

    // Print full report with best omega2
    printf("  -- Full report with omega2 = %.2f: --\n", best_w2);
    CumWP_OmegaReport best_rep = CumulantWP_ComputeReport(Re, Uref, dt_global, best_w2);
    CumulantWP_PrintReport(best_rep);
}

// ================================================================
// main
// ================================================================
int main(int argc, char *argv[])
{
    printf("================================================================\n");
    printf("  Cumulant-WP Singularity Diagnostic Tool\n");
    printf("  D3Q27 Periodic Hill -- Pre-simulation parameter check\n");
    printf("================================================================\n");

    int Re = 0;
    double Uref = 0.0;
    double omega2 = 1.0;
    double dt_global = 0.0;
    bool dt_from_user = false;

    if (argc >= 3) {
        // Parameter mode
        Re = atoi(argv[1]);
        Uref = atof(argv[2]);
        if (argc >= 4) omega2 = atof(argv[3]);
        if (argc >= 5) { dt_global = atof(argv[4]); dt_from_user = true; }
    } else {
        // Interactive mode
        printf("\n  Enter Re (e.g. 50): ");
        std::cin >> Re;
        printf("  Enter Uref (e.g. 0.0583): ");
        std::cin >> Uref;
        printf("  Enter omega2 [default=1.0]: ");
        std::string line;
        std::getline(std::cin >> std::ws, line);
        if (!line.empty()) omega2 = atof(line.c_str());
        printf("  Enter dt_global [press Enter to estimate from grid]: ");
        std::getline(std::cin, line);
        if (!line.empty()) { dt_global = atof(line.c_str()); dt_from_user = true; }
    }

    if (Re <= 0 || Uref <= 0.0) {
        fprintf(stderr, "ERROR: Re and Uref must be positive.\n");
        return 1;
    }

    // dt_global estimation
    if (!dt_from_user || dt_global <= 0.0) {
        dt_global = estimate_dt_global();
        printf("\n  [NOTE] dt_global not provided. Using grid estimate: %.6e\n", dt_global);
        printf("         (= minSize from variables.h; actual dt_global from Imamura GTS\n");
        printf("          may be smaller due to curvilinear Jacobian.)\n");
        printf("         For exact value, check simulation log: 'dt_global = X.XXXe-XX'\n");
    }

    // -- Main report (current omega2) --
    printf("\n  ====== Report with omega2 = %.4f ======\n", omega2);
    CumWP_OmegaReport rep = CumulantWP_ComputeReport(Re, Uref, dt_global, omega2);
    CumulantWP_PrintReport(rep);

    // -- omega2 scan --
    scan_best_omega2(Re, Uref, dt_global);

    // -- Quick Uref scan (fixed Re, omega2, dt) --
    printf("  -- Uref scan (omega2=%.2f, Re=%d, dt=%.4e): ---------------------------\n\n", omega2, Re, dt_global);
    printf("  %8s | %8s | %8s | %8s | %8s | %s\n",
           "Uref", "tau", "omega1", "w4_raw", "w4_dist", "verdict");
    printf("  %8s-+-%8s-+-%8s-+-%8s-+-%8s-+-%s\n",
           "--------", "--------", "--------", "--------", "--------", "-------");

    // Scan Uref from 0.01 to 0.17
    for (int u = 1; u <= 17; u++) {
        double Ut = 0.01 * u;
        CumWP_OmegaReport rt = CumulantWP_ComputeReport(Re, Ut, dt_global, omega2);
        const char *v = (rt.w4_sing_dist > 0.3) ? "SAFE" : (rt.w4_sing_dist > 0.15) ? "OK" : "DANGER";
        printf("  %8.4f | %8.4f | %8.4f | %+8.4f | %8.4f | %s",
               Ut, rt.tau, rt.w1, rt.w4_raw, rt.w4_sing_dist < 90 ? rt.w4_sing_dist : -1.0, v);
        if (fabs(Ut - Uref) < 0.005) printf("  <-- current");
        printf("\n");
    }

    printf("\n  ================================================================\n");
    printf("  Diagnostic complete.\n");
    printf("  ================================================================\n\n");

    return 0;
}
