#ifndef FILEIO_FILE
#define FILEIO_FILE

#include <unistd.h>//用到access
#include <sys/types.h>
#include <sys/stat.h>//用mkdir
#include <fstream>
#include <sstream>
#include <string>
#include <iostream>
#include <iomanip>  // setprecision, fixed
using namespace std ;
void wirte_ASCII_of_str(char * str, FILE *file);

/*第一段:創建資料夾*/
void ExistOrCreateDir(const char* doc) {
	std::string path(doc);
	path = "./" + path;
	if (access(path.c_str(), F_OK) != 0) {
        if (mkdir(path.c_str(), S_IRWXU) == 0)
			std::cout << "folder " << path << " not exist, created"<< std::endl;
	}
}

void PreCheckDir() {
	ExistOrCreateDir("result");
	ExistOrCreateDir("monitor");
	// statistics directories for merged bin (35 fields)
	ExistOrCreateDir("statistics");
	const int num_files = 35;
	std::string name[num_files] = {
        "U","V","W","P",
        "OMEGA_U","OMEGA_V","OMEGA_W",
        "UU","UV","UW","VV","VW","WW",
        "PU","PV","PW",
        "DUDX2","DUDY2","DUDZ2","DVDX2","DVDY2","DVDZ2","DWDX2","DWDY2","DWDZ2",
        "UUU","UUV","UUW","VVU","UVW","WWU","VVV","VVW","WWV","WWW"};
	for( int i = 0; i < num_files; i++ ) {
		std::string fname = "./statistics/" + name[i];
		ExistOrCreateDir(fname.c_str());
	}
}

/*第二段:輸出速度場與分佈函數 (per-rank legacy format)*/
void result_writebin(double* arr_h, const char *fname, const int myid){
    ostringstream oss;
    oss << "./result/" << fname << "_" << myid << ".bin";
    string path = oss.str();
    ofstream file(path.c_str(), ios::binary);
    if (!file) {
        cout << "Output data error, exit..." << endl;
        CHECK_MPI( MPI_Abort(MPI_COMM_WORLD, 1) );
    }
    file.write(reinterpret_cast<char*>(arr_h), sizeof(double) * NX6 * NZ6 * NYD6);
    file.close();
}

void result_readbin(double *arr_h, const char *folder, const char *fname, const int myid){
    ostringstream oss;
    oss << "./" << folder << "/" << fname << "_" << myid << ".bin";
    string path = oss.str();
    ifstream file(path.c_str(), ios::binary);
    if (!file) {
        cout << "Read data error: " << path << ", exit...\n";
        CHECK_MPI( MPI_Abort(MPI_COMM_WORLD, 1) );
    }
    file.read(reinterpret_cast<char*>(arr_h), sizeof(double) * NX6 * NZ6 * NYD6);
    file.close();
}

// Legacy per-rank velocity+f write (kept for backward compatibility)
void result_writebin_velocityandf() {
    ostringstream oss;
    oss << "./result/velocity_" << myid << "_Final.vtk";
    ofstream out(oss.str().c_str());
    out << "# vtk DataFile Version 3.0\n";
    out << "LBM Velocity Field\n";
    out << "ASCII\n";
    out << "DATASET STRUCTURED_GRID\n";
    out << "DIMENSIONS " << NX6-6 << " " << NYD6-6 << " " << NZ6-6 << "\n";
    int nPoints = (NX6-6) * (NYD6-6) * (NZ6-6);
    out << "POINTS " << nPoints << " double\n";
    out << fixed << setprecision(6);
    for( int k = 3; k < NZ6-3; k++ ){
    for( int j = 3; j < NYD6-3; j++ ){
    for( int i = 3; i < NX6-3; i++ ){
        out << x_h[i] << " " << y_h[j] << " " << z_h[j*NZ6+k] << "\n";
    }}}
    out << "\nPOINT_DATA " << nPoints << "\n";
    out << "VECTORS velocity double\n";
    out << setprecision(15);
    for( int k = 3; k < NZ6-3; k++ ){
    for( int j = 3; j < NYD6-3; j++ ){
    for( int i = 3; i < NX6-3; i++ ){
        int index = j*NZ6*NX6 + k*NX6 + i;
        out << u_h_p[index] << " " << v_h_p[index] << " " << w_h_p[index] << "\n";
    }}}
    out.close();

    cout << "\n----------- Start Output, myid = " << myid << " ----------\n";
    if( myid == 0 ) {
        ofstream fp_gg("./result/0_force.dat");
        fp_gg << fixed << setprecision(15) << Force_h[0];
        fp_gg.close();
    }
    result_writebin(rho_h_p, "rho", myid);
    result_writebin(u_h_p,   "u",   myid);
    result_writebin(v_h_p,   "v",   myid);
    result_writebin(w_h_p,   "w",   myid);
    for( int q = 0; q < 19; q++ ) {
        ostringstream fname;
        fname << "f" << q;
        result_writebin(fh_p[q], fname.str().c_str(), myid);
    }
}

// Legacy per-rank velocity+f read
void result_readbin_velocityandf()
{
    PreCheckDir();
    const char* result = "result";
    ifstream fp_gg("./result/0_force.dat");
    fp_gg >> Force_h[0];
    fp_gg.close();
    CHECK_CUDA( cudaMemcpy(Force_d, Force_h, sizeof(double), cudaMemcpyHostToDevice) );
    result_readbin(rho_h_p, result, "rho", myid);
    result_readbin(u_h_p,   result, "u",   myid);
    result_readbin(v_h_p,   result, "v",   myid);
    result_readbin(w_h_p,   result, "w",   myid);
    for( int q = 0; q < 19; q++ ) {
        ostringstream fname;
        fname << "f" << q;
        result_readbin(fh_p[q], result, fname.str().c_str(), myid);
    }
}

/*第三段:統計量 merged I/O*/
// GPU-count independent (merged) binary statistics I/O
// File format: raw double[(NZ6-6) × NY × (NX6-6)] in k→j_global→i order
void statistics_writebin_merged(double *arr_d, const char *fname) {
    CHECK_MPI( MPI_Barrier(MPI_COMM_WORLD) );
    const size_t nBytes = (size_t)NX6 * NYD6 * NZ6 * sizeof(double);
    double *arr_h = (double*)malloc(nBytes);
    CHECK_CUDA( cudaMemcpy(arr_h, arr_d, nBytes, cudaMemcpyDeviceToHost) );

    const int nx = NX6 - 6;
    const int ny = NY;
    const int nz = NZ6 - 6;
    const int stride = NY / jp;

    const int local_count = nz * stride * nx;
    double *send_buf = (double*)malloc(local_count * sizeof(double));
    int idx = 0;
    for (int k = 3; k < NZ6 - 3; k++)
        for (int jl = 3; jl < 3 + stride; jl++)
            for (int i = 3; i < NX6 - 3; i++)
                send_buf[idx++] = arr_h[jl * NX6 * NZ6 + k * NX6 + i];

    double *recv_buf = NULL;
    if (myid == 0) recv_buf = (double*)malloc((size_t)local_count * jp * sizeof(double));
    MPI_Gather(send_buf, local_count, MPI_DOUBLE,
               recv_buf, local_count, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if (myid == 0) {
        double *global_buf = (double*)malloc((size_t)nz * ny * nx * sizeof(double));
        for (int r = 0; r < jp; r++) {
            int j_offset = r * stride;
            double *rank_data = recv_buf + (size_t)r * local_count;
            int ridx = 0;
            for (int kk = 0; kk < nz; kk++)
                for (int jl = 0; jl < stride; jl++)
                    for (int ii = 0; ii < nx; ii++)
                        global_buf[(size_t)kk * ny * nx + (j_offset + jl) * nx + ii] = rank_data[ridx++];
        }
        ostringstream oss;
        oss << "./statistics/" << fname << "/" << fname << "_merged.bin";
        ofstream file(oss.str().c_str(), ios::binary);
        file.write(reinterpret_cast<char*>(global_buf), (size_t)nz * ny * nx * sizeof(double));
        file.close();
        free(global_buf);
        free(recv_buf);
    }
    free(send_buf);
    free(arr_h);
}

void statistics_readbin_merged(double *arr_d, const char *fname) {
    const int nx = NX6 - 6;
    const int ny = NY;
    const int nz = NZ6 - 6;
    const int stride = NY / jp;

    ostringstream oss;
    oss << "./statistics/" << fname << "/" << fname << "_merged.bin";
    ifstream file(oss.str().c_str(), ios::binary);
    if (!file.is_open()) {
        if (myid == 0) printf("[WARNING] statistics_readbin_merged: %s not found, skipping.\n", oss.str().c_str());
        return;
    }
    double *global_buf = (double*)malloc((size_t)nz * ny * nx * sizeof(double));
    file.read(reinterpret_cast<char*>(global_buf), (size_t)nz * ny * nx * sizeof(double));
    file.close();

    const size_t nBytes = (size_t)NX6 * NYD6 * NZ6 * sizeof(double);
    double *arr_h = (double*)calloc(NX6 * NYD6 * NZ6, sizeof(double));
    int j_start = myid * stride;

    for (int kk = 0; kk < nz; kk++) {
        int k = kk + 3;
        for (int jl = 0; jl < stride; jl++) {
            int j_local = jl + 3;
            int j_global = j_start + jl;
            for (int ii = 0; ii < nx; ii++) {
                int i = ii + 3;
                arr_h[j_local * NX6 * NZ6 + k * NX6 + i] =
                    global_buf[(size_t)kk * ny * nx + j_global * nx + ii];
            }
        }
        {
            int j_local = 3 + stride;
            int j_global = (j_start + stride) % ny;
            for (int ii = 0; ii < nx; ii++) {
                int i = ii + 3;
                arr_h[j_local * NX6 * NZ6 + k * NX6 + i] =
                    global_buf[(size_t)kk * ny * nx + j_global * nx + ii];
            }
        }
    }

    CHECK_CUDA( cudaMemcpy(arr_d, arr_h, nBytes, cudaMemcpyHostToDevice) );
    free(arr_h);
    free(global_buf);
    CHECK_MPI( MPI_Barrier(MPI_COMM_WORLD) );
}

// Write all 35 statistics as merged binary
void statistics_writebin_merged_stress() {
    if (myid == 0) {
        ofstream fp_accu("./statistics/accu.dat");
        fp_accu << accu_count << " " << step;
        fp_accu.close();
    }
    CHECK_MPI( MPI_Barrier(MPI_COMM_WORLD) );
    // 一階矩 (4+3=7)
    statistics_writebin_merged(U, "U");
    statistics_writebin_merged(V, "V");
    statistics_writebin_merged(W, "W");
    statistics_writebin_merged(P, "P");
    statistics_writebin_merged(OMEGA_U_SUM, "OMEGA_U");
    statistics_writebin_merged(OMEGA_V_SUM, "OMEGA_V");
    statistics_writebin_merged(OMEGA_W_SUM, "OMEGA_W");
    // 二階矩 (6) + 壓力交叉 (3) = 9
    statistics_writebin_merged(UU, "UU");
    statistics_writebin_merged(UV, "UV");
    statistics_writebin_merged(UW, "UW");
    statistics_writebin_merged(VV, "VV");
    statistics_writebin_merged(VW, "VW");
    statistics_writebin_merged(WW, "WW");
    statistics_writebin_merged(PU, "PU");
    statistics_writebin_merged(PV, "PV");
    statistics_writebin_merged(PW, "PW");
    // 速度梯度平方 (9)
    statistics_writebin_merged(DUDX2, "DUDX2");
    statistics_writebin_merged(DUDY2, "DUDY2");
    statistics_writebin_merged(DUDZ2, "DUDZ2");
    statistics_writebin_merged(DVDX2, "DVDX2");
    statistics_writebin_merged(DVDY2, "DVDY2");
    statistics_writebin_merged(DVDZ2, "DVDZ2");
    statistics_writebin_merged(DWDX2, "DWDX2");
    statistics_writebin_merged(DWDY2, "DWDY2");
    statistics_writebin_merged(DWDZ2, "DWDZ2");
    // 三階矩 (10)
    statistics_writebin_merged(UUU, "UUU");
    statistics_writebin_merged(UUV, "UUV");
    statistics_writebin_merged(UUW, "UUW");
    statistics_writebin_merged(VVU, "VVU");
    statistics_writebin_merged(UVW, "UVW");
    statistics_writebin_merged(WWU, "WWU");
    statistics_writebin_merged(VVV, "VVV");
    statistics_writebin_merged(VVW, "VVW");
    statistics_writebin_merged(WWV, "WWV");
    statistics_writebin_merged(WWW, "WWW");
    if (myid == 0) printf("  statistics_writebin_merged_stress: 35 merged .bin files written (accu=%d, step=%d)\n",
                          accu_count, step);
}

// Read all 35 statistics from merged binary
void statistics_readbin_merged_stress() {
    ifstream fp_accu("./statistics/accu.dat");
    if (!fp_accu.is_open()) {
        if (myid == 0) printf("[WARNING] statistics_readbin_merged_stress: accu.dat not found, accu_count unchanged.\n");
        return;
    }
    int bin_step = -1;
    fp_accu >> accu_count;
    fp_accu >> bin_step;
    fp_accu.close();
    if (myid == 0)
        printf("  statistics_readbin_merged_stress: accu_count=%d, step=%d from accu.dat\n",
               accu_count, bin_step);

    statistics_readbin_merged(U, "U");
    statistics_readbin_merged(V, "V");
    statistics_readbin_merged(W, "W");
    statistics_readbin_merged(P, "P");
    statistics_readbin_merged(OMEGA_U_SUM, "OMEGA_U");
    statistics_readbin_merged(OMEGA_V_SUM, "OMEGA_V");
    statistics_readbin_merged(OMEGA_W_SUM, "OMEGA_W");
    statistics_readbin_merged(UU, "UU");
    statistics_readbin_merged(UV, "UV");
    statistics_readbin_merged(UW, "UW");
    statistics_readbin_merged(VV, "VV");
    statistics_readbin_merged(VW, "VW");
    statistics_readbin_merged(WW, "WW");
    statistics_readbin_merged(PU, "PU");
    statistics_readbin_merged(PV, "PV");
    statistics_readbin_merged(PW, "PW");
    statistics_readbin_merged(DUDX2, "DUDX2");
    statistics_readbin_merged(DUDY2, "DUDY2");
    statistics_readbin_merged(DUDZ2, "DUDZ2");
    statistics_readbin_merged(DVDX2, "DVDX2");
    statistics_readbin_merged(DVDY2, "DVDY2");
    statistics_readbin_merged(DVDZ2, "DVDZ2");
    statistics_readbin_merged(DWDX2, "DWDX2");
    statistics_readbin_merged(DWDY2, "DWDY2");
    statistics_readbin_merged(DWDZ2, "DWDZ2");
    statistics_readbin_merged(UUU, "UUU");
    statistics_readbin_merged(UUV, "UUV");
    statistics_readbin_merged(UUW, "UUW");
    statistics_readbin_merged(VVU, "VVU");
    statistics_readbin_merged(UVW, "UVW");
    statistics_readbin_merged(WWU, "WWU");
    statistics_readbin_merged(VVV, "VVV");
    statistics_readbin_merged(VVW, "VVW");
    statistics_readbin_merged(WWV, "WWV");
    statistics_readbin_merged(WWW, "WWW");
    CHECK_MPI( MPI_Barrier(MPI_COMM_WORLD) );
}

// ============================================================================
// Checkpoint write/read (binary, merged format, for exact restart)
// ============================================================================
// Directory structure: checkpoint_{step}/meta.dat + f00~f18.bin + rho.bin + u,v,w.bin
//                      + (if accu_count>0) 35 sum_*.bin

// Write merged binary to checkpoint directory
void checkpoint_writebin_merged(double *arr_h, const char *name, const char *dir) {
    CHECK_MPI( MPI_Barrier(MPI_COMM_WORLD) );
    const int nx = NX6 - 6;
    const int ny = NY;
    const int nz = NZ6 - 6;
    const int stride = NY / jp;
    const int local_count = nz * stride * nx;

    double *send_buf = (double*)malloc(local_count * sizeof(double));
    int idx = 0;
    for (int k = 3; k < NZ6 - 3; k++)
        for (int jl = 3; jl < 3 + stride; jl++)
            for (int i = 3; i < NX6 - 3; i++)
                send_buf[idx++] = arr_h[jl * NX6 * NZ6 + k * NX6 + i];

    double *recv_buf = NULL;
    if (myid == 0) recv_buf = (double*)malloc((size_t)local_count * jp * sizeof(double));
    MPI_Gather(send_buf, local_count, MPI_DOUBLE,
               recv_buf, local_count, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if (myid == 0) {
        double *global_buf = (double*)malloc((size_t)nz * ny * nx * sizeof(double));
        for (int r = 0; r < jp; r++) {
            int j_offset = r * stride;
            double *rank_data = recv_buf + (size_t)r * local_count;
            int ridx = 0;
            for (int kk = 0; kk < nz; kk++)
                for (int jl = 0; jl < stride; jl++)
                    for (int ii = 0; ii < nx; ii++)
                        global_buf[(size_t)kk * ny * nx + (j_offset + jl) * nx + ii] = rank_data[ridx++];
        }
        ostringstream oss;
        oss << "./" << dir << "/" << name << ".bin";
        ofstream file(oss.str().c_str(), ios::binary);
        file.write(reinterpret_cast<char*>(global_buf), (size_t)nz * ny * nx * sizeof(double));
        file.close();
        free(global_buf);
        free(recv_buf);
    }
    free(send_buf);
}

// Read merged binary from checkpoint directory
void checkpoint_readbin_merged(double *arr_h, const char *name, const char *dir) {
    const int nx = NX6 - 6;
    const int ny = NY;
    const int nz = NZ6 - 6;
    const int stride = NY / jp;

    ostringstream oss;
    oss << "./" << dir << "/" << name << ".bin";
    ifstream file(oss.str().c_str(), ios::binary);
    if (!file.is_open()) {
        if (myid == 0) printf("[ERROR] checkpoint_readbin_merged: %s not found!\n", oss.str().c_str());
        CHECK_MPI( MPI_Abort(MPI_COMM_WORLD, 1) );
    }
    double *global_buf = (double*)malloc((size_t)nz * ny * nx * sizeof(double));
    file.read(reinterpret_cast<char*>(global_buf), (size_t)nz * ny * nx * sizeof(double));
    file.close();

    int j_start = myid * stride;
    for (int kk = 0; kk < nz; kk++) {
        int k = kk + 3;
        for (int jl = 0; jl < stride; jl++) {
            int j_local = jl + 3;
            int j_global = j_start + jl;
            for (int ii = 0; ii < nx; ii++) {
                int i = ii + 3;
                arr_h[j_local * NX6 * NZ6 + k * NX6 + i] =
                    global_buf[(size_t)kk * ny * nx + j_global * nx + ii];
            }
        }
        // Overlap point
        {
            int j_local = 3 + stride;
            int j_global = (j_start + stride) % ny;
            for (int ii = 0; ii < nx; ii++) {
                int i = ii + 3;
                arr_h[j_local * NX6 * NZ6 + k * NX6 + i] =
                    global_buf[(size_t)kk * ny * nx + j_global * nx + ii];
            }
        }
    }
    free(global_buf);
    CHECK_MPI( MPI_Barrier(MPI_COMM_WORLD) );
}

// Read merged checkpoint to GPU array
void checkpoint_readbin_merged_to_gpu(double *arr_d, const char *name, const char *dir) {
    const size_t nBytes = (size_t)NX6 * NYD6 * NZ6 * sizeof(double);
    double *arr_h = (double*)calloc(NX6 * NYD6 * NZ6, sizeof(double));
    checkpoint_readbin_merged(arr_h, name, dir);
    CHECK_CUDA( cudaMemcpy(arr_d, arr_h, nBytes, cudaMemcpyHostToDevice) );
    free(arr_h);
}

// Write full checkpoint
void WriteCheckpoint(int step_num) {
    double FTT_now = (double)step_num * dt_global / (double)flow_through_time;

    // Create checkpoint directory
    ostringstream dir_oss;
    dir_oss << "checkpoint_" << step_num;
    string dir_name = dir_oss.str();
    if (myid == 0) ExistOrCreateDir(dir_name.c_str());
    CHECK_MPI( MPI_Barrier(MPI_COMM_WORLD) );

    // Write meta.dat
    if (myid == 0) {
        ostringstream meta_oss;
        meta_oss << "./" << dir_name << "/meta.dat";
        ofstream meta(meta_oss.str().c_str());
        meta << "step " << step_num << "\n";
        meta << "FTT " << fixed << setprecision(6) << FTT_now << "\n";
        meta << "accu_count " << accu_count << "\n";
        meta << scientific << setprecision(15) << "Force " << Force_h[0] << "\n";
        meta.close();
    }

    // Write velocity + rho (always, merged format using host arrays)
    checkpoint_writebin_merged(rho_h_p, "rho", dir_name.c_str());
    checkpoint_writebin_merged(u_h_p,   "u",   dir_name.c_str());
    checkpoint_writebin_merged(v_h_p,   "v",   dir_name.c_str());
    checkpoint_writebin_merged(w_h_p,   "w",   dir_name.c_str());

    // Write f distributions (merged format)
    for (int q = 0; q < 19; q++) {
        ostringstream fname;
        fname << "f" << setfill('0') << setw(2) << q;
        checkpoint_writebin_merged(fh_p[q], fname.str().c_str(), dir_name.c_str());
    }

    // Write 35 statistics (only if accu_count > 0, GPU → temp host → merged)
    if (accu_count > 0) {
        const size_t nBytes = (size_t)NX6 * NYD6 * NZ6 * sizeof(double);
        double *tmp_h = (double*)malloc(nBytes);

        // Helper: GPU → host → merged write
        #define WRITE_GPU_FIELD(gpu_ptr, name_str) \
            CHECK_CUDA( cudaMemcpy(tmp_h, gpu_ptr, nBytes, cudaMemcpyDeviceToHost) ); \
            checkpoint_writebin_merged(tmp_h, name_str, dir_name.c_str());

        WRITE_GPU_FIELD(U, "sum_u");
        WRITE_GPU_FIELD(V, "sum_v");
        WRITE_GPU_FIELD(W, "sum_w");
        WRITE_GPU_FIELD(P, "sum_P");
        WRITE_GPU_FIELD(OMEGA_U_SUM, "sum_omega_u");
        WRITE_GPU_FIELD(OMEGA_V_SUM, "sum_omega_v");
        WRITE_GPU_FIELD(OMEGA_W_SUM, "sum_omega_w");
        WRITE_GPU_FIELD(UU, "sum_uu");
        WRITE_GPU_FIELD(UV, "sum_uv");
        WRITE_GPU_FIELD(UW, "sum_uw");
        WRITE_GPU_FIELD(VV, "sum_vv");
        WRITE_GPU_FIELD(VW, "sum_vw");
        WRITE_GPU_FIELD(WW, "sum_ww");
        WRITE_GPU_FIELD(PU, "sum_Pu");
        WRITE_GPU_FIELD(PV, "sum_Pv");
        WRITE_GPU_FIELD(PW, "sum_Pw");
        WRITE_GPU_FIELD(UUU, "sum_uuu");
        WRITE_GPU_FIELD(UUV, "sum_uuv");
        WRITE_GPU_FIELD(UUW, "sum_uuw");
        WRITE_GPU_FIELD(VVU, "sum_uvv");
        WRITE_GPU_FIELD(UVW, "sum_uvw");
        WRITE_GPU_FIELD(WWU, "sum_uww");
        WRITE_GPU_FIELD(VVV, "sum_vvv");
        WRITE_GPU_FIELD(VVW, "sum_vvw");
        WRITE_GPU_FIELD(WWV, "sum_vww");
        WRITE_GPU_FIELD(WWW, "sum_www");
        WRITE_GPU_FIELD(DUDX2, "sum_dudx2");
        WRITE_GPU_FIELD(DUDY2, "sum_dudy2");
        WRITE_GPU_FIELD(DUDZ2, "sum_dudz2");
        WRITE_GPU_FIELD(DVDX2, "sum_dvdx2");
        WRITE_GPU_FIELD(DVDY2, "sum_dvdy2");
        WRITE_GPU_FIELD(DVDZ2, "sum_dvdz2");
        WRITE_GPU_FIELD(DWDX2, "sum_dwdx2");
        WRITE_GPU_FIELD(DWDY2, "sum_dwdy2");
        WRITE_GPU_FIELD(DWDZ2, "sum_dwdz2");

        #undef WRITE_GPU_FIELD
        free(tmp_h);
    }

    if (myid == 0)
        printf("[Checkpoint] %s written (accu=%d, FTT=%.2f)\n",
               dir_name.c_str(), accu_count, FTT_now);
}

// Read checkpoint for restart
void ReadCheckpoint(const char *dir, int init_level) {
    // Read meta.dat
    {
        ostringstream meta_oss;
        meta_oss << "./" << dir << "/meta.dat";
        ifstream meta(meta_oss.str().c_str());
        if (!meta.is_open()) {
            if (myid == 0) printf("[ERROR] Cannot open %s\n", meta_oss.str().c_str());
            CHECK_MPI( MPI_Abort(MPI_COMM_WORLD, 1) );
        }
        string key;
        while (meta >> key) {
            if (key == "step") meta >> restart_step;
            else if (key == "FTT") { double ftt_tmp; meta >> ftt_tmp; }
            else if (key == "accu_count") meta >> accu_count;
            else if (key == "Force") meta >> Force_h[0];
        }
        meta.close();
    }
    CHECK_CUDA( cudaMemcpy(Force_d, Force_h, sizeof(double), cudaMemcpyHostToDevice) );
    if (myid == 0)
        printf("  Checkpoint meta: step=%d, accu_count=%d, Force=%.5E\n",
               restart_step, accu_count, Force_h[0]);

    // INIT >= 1: read velocity + rho + f distributions
    checkpoint_readbin_merged(rho_h_p, "rho", dir);
    checkpoint_readbin_merged(u_h_p,   "u",   dir);
    checkpoint_readbin_merged(v_h_p,   "v",   dir);
    checkpoint_readbin_merged(w_h_p,   "w",   dir);
    for (int q = 0; q < 19; q++) {
        ostringstream fname;
        fname << "f" << setfill('0') << setw(2) << q;
        checkpoint_readbin_merged(fh_p[q], fname.str().c_str(), dir);
    }

    // x-periodic buffer fill
    {
        const int buffer = 3;
        const int shift = NX6 - 2*buffer - 1;
        for (int j = 3; j < NYD6-3; j++) {
        for (int k = 3; k < NZ6-3; k++) {
            for (int ib = 0; ib < buffer; ib++) {
                int idx_buf = j * NX6 * NZ6 + k * NX6 + ib;
                int idx_src = idx_buf + shift;
                rho_h_p[idx_buf] = rho_h_p[idx_src];
                u_h_p[idx_buf]   = u_h_p[idx_src];
                v_h_p[idx_buf]   = v_h_p[idx_src];
                w_h_p[idx_buf]   = w_h_p[idx_src];
                for (int q = 0; q < 19; q++) fh_p[q][idx_buf] = fh_p[q][idx_src];

                idx_buf = j * NX6 * NZ6 + k * NX6 + (NX6 - 1 - ib);
                idx_src = idx_buf - shift;
                rho_h_p[idx_buf] = rho_h_p[idx_src];
                u_h_p[idx_buf]   = u_h_p[idx_src];
                v_h_p[idx_buf]   = v_h_p[idx_src];
                w_h_p[idx_buf]   = w_h_p[idx_src];
                for (int q = 0; q < 19; q++) fh_p[q][idx_buf] = fh_p[q][idx_src];
            }
        }}
    }

    // MPI ghost zone exchange (velocity + rho + f)
    {
        const int slice_size = NX6 * NZ6;
        const int ghost_count = 3 * slice_size;

        double *fields_h[] = {u_h_p, v_h_p, w_h_p, rho_h_p};
        for (int f = 0; f < 4; f++) {
            MPI_Sendrecv(&fields_h[f][4 * slice_size],       ghost_count, MPI_DOUBLE, l_nbr, 600 + 2*f,
                         &fields_h[f][(NYD6-3) * slice_size], ghost_count, MPI_DOUBLE, r_nbr, 600 + 2*f,
                         MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Sendrecv(&fields_h[f][(NYD6-7) * slice_size], ghost_count, MPI_DOUBLE, r_nbr, 601 + 2*f,
                         &fields_h[f][0],                      ghost_count, MPI_DOUBLE, l_nbr, 601 + 2*f,
                         MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
        for (int q = 0; q < 19; q++) {
            MPI_Sendrecv(&fh_p[q][4 * slice_size],       ghost_count, MPI_DOUBLE, l_nbr, 700 + 2*q,
                         &fh_p[q][(NYD6-3) * slice_size], ghost_count, MPI_DOUBLE, r_nbr, 700 + 2*q,
                         MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Sendrecv(&fh_p[q][(NYD6-7) * slice_size], ghost_count, MPI_DOUBLE, r_nbr, 701 + 2*q,
                         &fh_p[q][0],                      ghost_count, MPI_DOUBLE, l_nbr, 701 + 2*q,
                         MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
    }

    // z-direction extrapolation
    {
        #define BUF_IDX(jj,kk,ii) ((jj)*NX6*NZ6 + (kk)*NX6 + (ii))
        for (int j = 0; j < NYD6; j++) {
        for (int i = 0; i < NX6; i++) {
            // Velocity + rho
            double *vel_fields[] = {u_h_p, v_h_p, w_h_p, rho_h_p};
            for (int f = 0; f < 4; f++) {
                double *F = vel_fields[f];
                F[BUF_IDX(j,2,i)]     = 2.0*F[BUF_IDX(j,3,i)]     - F[BUF_IDX(j,4,i)];
                F[BUF_IDX(j,1,i)]     = 2.0*F[BUF_IDX(j,2,i)]     - F[BUF_IDX(j,3,i)];
                F[BUF_IDX(j,0,i)]     = 2.0*F[BUF_IDX(j,1,i)]     - F[BUF_IDX(j,2,i)];
                F[BUF_IDX(j,NZ6-3,i)] = 2.0*F[BUF_IDX(j,NZ6-4,i)] - F[BUF_IDX(j,NZ6-5,i)];
                F[BUF_IDX(j,NZ6-2,i)] = 2.0*F[BUF_IDX(j,NZ6-3,i)] - F[BUF_IDX(j,NZ6-4,i)];
                F[BUF_IDX(j,NZ6-1,i)] = 2.0*F[BUF_IDX(j,NZ6-2,i)] - F[BUF_IDX(j,NZ6-3,i)];
            }
            // f distributions
            for (int q = 0; q < 19; q++) {
                double *F = fh_p[q];
                F[BUF_IDX(j,2,i)]     = 2.0*F[BUF_IDX(j,3,i)]     - F[BUF_IDX(j,4,i)];
                F[BUF_IDX(j,1,i)]     = 2.0*F[BUF_IDX(j,2,i)]     - F[BUF_IDX(j,3,i)];
                F[BUF_IDX(j,0,i)]     = 2.0*F[BUF_IDX(j,1,i)]     - F[BUF_IDX(j,2,i)];
                F[BUF_IDX(j,NZ6-3,i)] = 2.0*F[BUF_IDX(j,NZ6-4,i)] - F[BUF_IDX(j,NZ6-5,i)];
                F[BUF_IDX(j,NZ6-2,i)] = 2.0*F[BUF_IDX(j,NZ6-3,i)] - F[BUF_IDX(j,NZ6-4,i)];
                F[BUF_IDX(j,NZ6-1,i)] = 2.0*F[BUF_IDX(j,NZ6-2,i)] - F[BUF_IDX(j,NZ6-3,i)];
            }
        }}
        #undef BUF_IDX
    }

    // INIT >= 2: read 35 statistics arrays to GPU
    if (init_level >= 2 && accu_count > 0) {
        checkpoint_readbin_merged_to_gpu(U, "sum_u", dir);
        checkpoint_readbin_merged_to_gpu(V, "sum_v", dir);
        checkpoint_readbin_merged_to_gpu(W, "sum_w", dir);
        checkpoint_readbin_merged_to_gpu(P, "sum_P", dir);
        checkpoint_readbin_merged_to_gpu(OMEGA_U_SUM, "sum_omega_u", dir);
        checkpoint_readbin_merged_to_gpu(OMEGA_V_SUM, "sum_omega_v", dir);
        checkpoint_readbin_merged_to_gpu(OMEGA_W_SUM, "sum_omega_w", dir);
        checkpoint_readbin_merged_to_gpu(UU, "sum_uu", dir);
        checkpoint_readbin_merged_to_gpu(UV, "sum_uv", dir);
        checkpoint_readbin_merged_to_gpu(UW, "sum_uw", dir);
        checkpoint_readbin_merged_to_gpu(VV, "sum_vv", dir);
        checkpoint_readbin_merged_to_gpu(VW, "sum_vw", dir);
        checkpoint_readbin_merged_to_gpu(WW, "sum_ww", dir);
        checkpoint_readbin_merged_to_gpu(PU, "sum_Pu", dir);
        checkpoint_readbin_merged_to_gpu(PV, "sum_Pv", dir);
        checkpoint_readbin_merged_to_gpu(PW, "sum_Pw", dir);
        checkpoint_readbin_merged_to_gpu(UUU, "sum_uuu", dir);
        checkpoint_readbin_merged_to_gpu(UUV, "sum_uuv", dir);
        checkpoint_readbin_merged_to_gpu(UUW, "sum_uuw", dir);
        checkpoint_readbin_merged_to_gpu(VVU, "sum_uvv", dir);
        checkpoint_readbin_merged_to_gpu(UVW, "sum_uvw", dir);
        checkpoint_readbin_merged_to_gpu(WWU, "sum_uww", dir);
        checkpoint_readbin_merged_to_gpu(VVV, "sum_vvv", dir);
        checkpoint_readbin_merged_to_gpu(VVW, "sum_vvw", dir);
        checkpoint_readbin_merged_to_gpu(WWV, "sum_vww", dir);
        checkpoint_readbin_merged_to_gpu(WWW, "sum_www", dir);
        checkpoint_readbin_merged_to_gpu(DUDX2, "sum_dudx2", dir);
        checkpoint_readbin_merged_to_gpu(DUDY2, "sum_dudy2", dir);
        checkpoint_readbin_merged_to_gpu(DUDZ2, "sum_dudz2", dir);
        checkpoint_readbin_merged_to_gpu(DVDX2, "sum_dvdx2", dir);
        checkpoint_readbin_merged_to_gpu(DVDY2, "sum_dvdy2", dir);
        checkpoint_readbin_merged_to_gpu(DVDZ2, "sum_dvdz2", dir);
        checkpoint_readbin_merged_to_gpu(DWDX2, "sum_dwdx2", dir);
        checkpoint_readbin_merged_to_gpu(DWDY2, "sum_dwdy2", dir);
        checkpoint_readbin_merged_to_gpu(DWDZ2, "sum_dwdz2", dir);
        stats_announced = true;
        if (myid == 0) printf("  Checkpoint: 35 statistics arrays loaded to GPU (accu_count=%d)\n", accu_count);
    } else if (init_level >= 2 && accu_count == 0) {
        if (myid == 0) printf("  Checkpoint: accu_count=0, no statistics to load\n");
    }

    if (myid == 0)
        printf("  ReadCheckpoint: %s complete (INIT=%d)\n", dir, init_level);
}

/*第三.5段:從合併VTK檔案讀取初始場 (INIT=1 + VTK)*/
void InitFromMergedVTK(const char* vtk_path) {
    const int nyLocal  = NYD6 - 6;
    const int nxLocal  = NX6  - 6;
    const int nzLocal  = NZ6  - 6;
    const int nyGlobal = NY6  - 6;

    double e_loc[19][3] = {
        {0,0,0},{1,0,0},{-1,0,0},{0,1,0},{0,-1,0},{0,0,1},{0,0,-1},
        {1,1,0},{-1,1,0},{1,-1,0},{-1,-1,0},
        {1,0,1},{-1,0,1},{1,0,-1},{-1,0,-1},
        {0,1,1},{0,-1,1},{0,1,-1},{0,-1,-1}
    };
    double W_loc[19] = {
        1.0/3.0,
        1.0/18.0, 1.0/18.0, 1.0/18.0, 1.0/18.0, 1.0/18.0, 1.0/18.0,
        1.0/36.0, 1.0/36.0, 1.0/36.0, 1.0/36.0,
        1.0/36.0, 1.0/36.0, 1.0/36.0, 1.0/36.0,
        1.0/36.0, 1.0/36.0, 1.0/36.0, 1.0/36.0
    };

    for (int idx = 0; idx < NX6 * NYD6 * NZ6; idx++) {
        rho_h_p[idx] = 1.0;
        u_h_p[idx]   = 0.0;
        v_h_p[idx]   = 0.0;
        w_h_p[idx]   = 0.0;
    }

    ifstream vtk_in(vtk_path);
    if (!vtk_in.is_open()) {
        cout << "ERROR: Cannot open VTK file: " << vtk_path << endl;
        CHECK_MPI( MPI_Abort(MPI_COMM_WORLD, 1) );
    }

    // Parse header: step, Force, FTT
    double force_from_vtk = -1.0;
    double ftt_from_vtk = -1.0;
    int step_from_vtk = -1;
    string vtk_line;
    while (getline(vtk_in, vtk_line)) {
        size_t spos = vtk_line.find("step=");
        if (spos != string::npos)
            sscanf(vtk_line.c_str() + spos + 5, "%d", &step_from_vtk);
        size_t fpos = vtk_line.find("Force=");
        if (fpos != string::npos)
            sscanf(vtk_line.c_str() + fpos + 6, "%lf", &force_from_vtk);
        // New header format: STEP, FTT, ACCU_COUNT, FORCE
        size_t ftpos = vtk_line.find("FTT=");
        if (ftpos == string::npos) ftpos = vtk_line.find("FTT ");
        if (ftpos != string::npos)
            sscanf(vtk_line.c_str() + ftpos + 4, "%lf", &ftt_from_vtk);
        if (vtk_line.find("VECTORS") != string::npos || vtk_line.find("SCALARS") != string::npos) break;
    }

    const int stride = nyLocal - 1;
    int jg_start = myid * stride;
    int jg_end   = jg_start + nyLocal - 1;
    if (jg_end > nyGlobal - 1) jg_end = nyGlobal - 1;

    // Read velocity data — detect format from header line that broke the loop
    // Writer uses SCALARS format: 3 separate blocks (u, v, w) each with LOOKUP_TABLE
    // Old format used VECTORS: 3 values per line
    bool is_scalars = (vtk_line.find("SCALARS") != string::npos);

    // Helper lambda: read one scalar field into the specified host array
    // Reads nzLocal × nyGlobal × nxLocal values, assigns to local rank's portion
    auto read_scalar_field = [&](double *field_h) {
        double val;
        for (int k = 0; k < nzLocal; k++) {
        for (int jg = 0; jg < nyGlobal; jg++) {
        for (int i = 0; i < nxLocal; i++) {
            vtk_in >> val;
            if (jg >= jg_start && jg <= jg_end) {
                int j_local = jg - jg_start;
                int j  = j_local + 3;
                int kk = k + 3;
                int ii = i + 3;
                int index = j * NX6 * NZ6 + kk * NX6 + ii;
                field_h[index] = val;
                rho_h_p[index] = 1.0;
            }
        }}}
    };

    // Helper: skip lines until "LOOKUP_TABLE" is found (consumes SCALARS header + LOOKUP_TABLE)
    auto skip_to_data = [&]() {
        while (getline(vtk_in, vtk_line)) {
            if (vtk_line.find("LOOKUP_TABLE") != string::npos) return;
        }
    };

    if (is_scalars) {
        // SCALARS format: "SCALARS u_inst/Uref double 1\nLOOKUP_TABLE default\n<data>"
        // vtk_line = "SCALARS u_inst/Uref ..." from header loop — skip to LOOKUP_TABLE
        skip_to_data();
        read_scalar_field(u_h_p);

        skip_to_data();  // skip blank lines + "SCALARS v_inst/Uref..." + "LOOKUP_TABLE default"
        read_scalar_field(v_h_p);

        skip_to_data();  // skip blank lines + "SCALARS w_inst/Uref..." + "LOOKUP_TABLE default"
        read_scalar_field(w_h_p);

        if (myid == 0) printf("  VTK read: SCALARS format (u/v/w as-is, no Uref multiply)\n");
    } else {
        // VECTORS format: "VECTORS velocity double\n<u v w per line>"
        double u_val, v_val, w_val;
        for (int k = 0; k < nzLocal; k++) {
        for (int jg = 0; jg < nyGlobal; jg++) {
        for (int i = 0; i < nxLocal; i++) {
            vtk_in >> u_val >> v_val >> w_val;
            if (jg >= jg_start && jg <= jg_end) {
                int j_local = jg - jg_start;
                int j  = j_local + 3;
                int kk = k + 3;
                int ii = i + 3;
                int index = j * NX6 * NZ6 + kk * NX6 + ii;
                u_h_p[index]   = u_val;
                v_h_p[index]   = v_val;
                w_h_p[index]   = w_val;
                rho_h_p[index] = 1.0;
            }
        }}}
        if (myid == 0) printf("  VTK read: VECTORS format (u/v/w as-is)\n");
    }
    vtk_in.close();

    // x-periodic buffer fill
    {
        const int buffer = 3;
        const int shift = NX6 - 2*buffer - 1;
        for (int j = 3; j < NYD6-3; j++) {
        for (int k = 3; k < NZ6-3; k++) {
            for (int ib = 0; ib < buffer; ib++) {
                int idx_buf = j * NX6 * NZ6 + k * NX6 + ib;
                int idx_src = idx_buf + shift;
                u_h_p[idx_buf]   = u_h_p[idx_src];
                v_h_p[idx_buf]   = v_h_p[idx_src];
                w_h_p[idx_buf]   = w_h_p[idx_src];
                rho_h_p[idx_buf] = rho_h_p[idx_src];

                idx_buf = j * NX6 * NZ6 + k * NX6 + (NX6 - 1 - ib);
                idx_src = idx_buf - shift;
                u_h_p[idx_buf]   = u_h_p[idx_src];
                v_h_p[idx_buf]   = v_h_p[idx_src];
                w_h_p[idx_buf]   = w_h_p[idx_src];
                rho_h_p[idx_buf] = rho_h_p[idx_src];
            }
        }}
    }

    // MPI ghost zone exchange
    {
        const int slice_size = NX6 * NZ6;
        const int ghost_count = 3 * slice_size;
        double *fields_h[] = {u_h_p, v_h_p, w_h_p, rho_h_p};
        for (int f = 0; f < 4; f++) {
            MPI_Sendrecv(&fields_h[f][4 * slice_size],       ghost_count, MPI_DOUBLE, l_nbr, 600 + 2*f,
                         &fields_h[f][(NYD6-3) * slice_size], ghost_count, MPI_DOUBLE, r_nbr, 600 + 2*f,
                         MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Sendrecv(&fields_h[f][(NYD6-7) * slice_size], ghost_count, MPI_DOUBLE, r_nbr, 601 + 2*f,
                         &fields_h[f][0],                      ghost_count, MPI_DOUBLE, l_nbr, 601 + 2*f,
                         MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
    }

    // z-direction extrapolation
    {
        #define BUF_IDX(jj,kk,ii) ((jj)*NX6*NZ6 + (kk)*NX6 + (ii))
        for (int j = 0; j < NYD6; j++) {
        for (int i = 0; i < NX6; i++) {
            double *fields[] = {u_h_p, v_h_p, w_h_p, rho_h_p};
            for (int f = 0; f < 4; f++) {
                double *F = fields[f];
                F[BUF_IDX(j,2,i)]     = 2.0 * F[BUF_IDX(j,3,i)]     - F[BUF_IDX(j,4,i)];
                F[BUF_IDX(j,1,i)]     = 2.0 * F[BUF_IDX(j,2,i)]     - F[BUF_IDX(j,3,i)];
                F[BUF_IDX(j,0,i)]     = 2.0 * F[BUF_IDX(j,1,i)]     - F[BUF_IDX(j,2,i)];
                F[BUF_IDX(j,NZ6-3,i)] = 2.0 * F[BUF_IDX(j,NZ6-4,i)] - F[BUF_IDX(j,NZ6-5,i)];
                F[BUF_IDX(j,NZ6-2,i)] = 2.0 * F[BUF_IDX(j,NZ6-3,i)] - F[BUF_IDX(j,NZ6-4,i)];
                F[BUF_IDX(j,NZ6-1,i)] = 2.0 * F[BUF_IDX(j,NZ6-2,i)] - F[BUF_IDX(j,NZ6-3,i)];
            }
        }}
        #undef BUF_IDX
    }

    // f = feq (approximate restart)
    for (int k = 0; k < NZ6; k++) {
    for (int j = 0; j < NYD6; j++) {
    for (int i = 0; i < NX6; i++) {
        int index = j * NX6 * NZ6 + k * NX6 + i;
        double rho = rho_h_p[index];
        double uu = u_h_p[index], vv = v_h_p[index], ww = w_h_p[index];
        double udot = uu * uu + vv * vv + ww * ww;
        fh_p[0][index] = W_loc[0] * rho * (1.0 - 1.5 * udot);
        for (int dir = 1; dir <= 18; dir++) {
            double eu = e_loc[dir][0] * uu + e_loc[dir][1] * vv + e_loc[dir][2] * ww;
            fh_p[dir][index] = W_loc[dir] * rho * (1.0 + 3.0 * eu + 4.5 * eu * eu - 1.5 * udot);
        }
    }}}

    // Force from VTK header
    if (force_from_vtk > 0.0) {
        Force_h[0] = force_from_vtk;
        if (myid == 0) printf("  Force restored from VTK header: %.5E\n", force_from_vtk);
    } else {
        if (myid == 0) fprintf(stderr, "ERROR: Force= not found in VTK header [%s].\n", vtk_path);
        CHECK_MPI( MPI_Abort(MPI_COMM_WORLD, 1) );
    }
    CHECK_CUDA( cudaMemcpy(Force_d, Force_h, sizeof(double), cudaMemcpyHostToDevice) );

    // VTK restart: FTT continues from header, accu_count forced to 0
    accu_count = 0;
    if (step_from_vtk > 0) {
        restart_step = step_from_vtk;
        if (myid == 0) {
            double FTT_restart = (ftt_from_vtk > 0.0) ? ftt_from_vtk
                                : (double)restart_step * dt_global / (double)flow_through_time;
            printf("  [VTK RESTART] step=%d, FTT=%.4f (continues), accu_count=0 (reset)\n",
                   restart_step, FTT_restart);
            if (FTT_restart >= FTT_STATS_START)
                printf("  FTT >= %.0f: statistics accumulation starts immediately\n", FTT_STATS_START);
        }
    } else {
        restart_step = 0;
        if (myid == 0) printf("  WARNING: step= not found in VTK header, restarting from step 0.\n");
    }

    printf("Rank %d: Initialized from VTK [%s], jg=%d..%d\n", myid, vtk_path, jg_start, jg_end);
}

/*第四段:每 OUTVTK 步輸出 VTK 檔案*/
// 合併所有 GPU 結果，輸出單一 VTK 檔案
// VTK field list depends on FTT stage and VTK_OUTPUT_LEVEL:
//   FTT < 20: 6 fields (u_inst/Uref, v_inst/Uref, w_inst/Uref, omega_x/y/z_inst)
//   FTT >= 20, LEVEL 0: +14 = 20 fields (<u/v/w>/Uref, <omega_x/y/z>, <u'u'..>/Uref^2 ×6, k/Uref^2, <p>)
//   FTT >= 20, LEVEL 1: +1 = 21 fields (LEVEL 0 + epsilon*h/Uref^3)
void fileIO_velocity_vtk_merged(int step) {
    const int nyLocal = NYD6 - 6;
    const int nxLocal = NX6 - 6;
    const int nzLocal = NZ6 - 6;
    const int localPoints = nxLocal * nyLocal * nzLocal;
    const int zLocalSize = nyLocal * nzLocal;
    const int nyGlobal = NY6 - 6;
    const int globalPoints = nxLocal * nyGlobal * nzLocal;
    const int gatherPoints = localPoints * nProcs;

    double FTT_now = step * dt_global / (double)flow_through_time;
    double inv_Uref = 1.0 / (double)Uref;

    // Prepare local instantaneous velocity (normalized by Uref)
    double *u_local = (double*)malloc(localPoints * sizeof(double));
    double *v_local = (double*)malloc(localPoints * sizeof(double));
    double *w_local = (double*)malloc(localPoints * sizeof(double));
    double *z_local = (double*)malloc(zLocalSize * sizeof(double));

    int idx = 0;
    for( int k = 3; k < NZ6-3; k++ ){
    for( int j = 3; j < NYD6-3; j++ ){
    for( int i = 3; i < NX6-3; i++ ){
        int index = j*NZ6*NX6 + k*NX6 + i;
        u_local[idx] = u_h_p[index] * inv_Uref;
        v_local[idx] = v_h_p[index] * inv_Uref;
        w_local[idx] = w_h_p[index] * inv_Uref;
        idx++;
    }}}

    // Compute vorticity (instantaneous)
    double *ox_local = (double*)malloc(localPoints * sizeof(double));
    double *oy_local = (double*)malloc(localPoints * sizeof(double));
    double *oz_local = (double*)malloc(localPoints * sizeof(double));
    {
        double dx_val = (double)LX / (double)(NX6 - 7);
        double dx_inv = 1.0 / dx_val;
        double dy_val = (double)LY / (double)(NY6 - 7);
        double dy_inv = 1.0 / dy_val;
        const int nface = NX6 * NZ6;
        int oidx = 0;
        for (int k = 3; k < NZ6-3; k++) {
        for (int j = 3; j < NYD6-3; j++) {
        for (int i = 3; i < NX6-3; i++) {
            double dkdz = dk_dz_h[j * NZ6 + k];
            double dkdy = dk_dy_h[j * NZ6 + k];
            double du_dj = (u_h_p[(j+1)*nface + k*NX6 + i] - u_h_p[(j-1)*nface + k*NX6 + i]) * 0.5;
            double du_dk = (u_h_p[j*nface + (k+1)*NX6 + i] - u_h_p[j*nface + (k-1)*NX6 + i]) * 0.5;
            double dv_di = (v_h_p[j*nface + k*NX6 + (i+1)] - v_h_p[j*nface + k*NX6 + (i-1)]) * 0.5;
            double dv_dk = (v_h_p[j*nface + (k+1)*NX6 + i] - v_h_p[j*nface + (k-1)*NX6 + i]) * 0.5;
            double dw_di = (w_h_p[j*nface + k*NX6 + (i+1)] - w_h_p[j*nface + k*NX6 + (i-1)]) * 0.5;
            double dw_dj = (w_h_p[(j+1)*nface + k*NX6 + i] - w_h_p[(j-1)*nface + k*NX6 + i]) * 0.5;
            double dw_dk = (w_h_p[j*nface + (k+1)*NX6 + i] - w_h_p[j*nface + (k-1)*NX6 + i]) * 0.5;
            ox_local[oidx] = dy_inv * dw_dj + dkdy * dw_dk - dkdz * dv_dk;
            oy_local[oidx] = dkdz * du_dk - dx_inv * dw_di;
            oz_local[oidx] = dx_inv * dv_di - dy_inv * du_dj - dkdy * du_dk;
            oidx++;
        }}}
    }

    // Prepare statistics local arrays (if accu_count > 0)
    // Download GPU stats to temp host buffers
    double *U_mean_local = NULL, *V_mean_local = NULL, *W_mean_local = NULL;
    double *ou_mean_local = NULL, *ov_mean_local = NULL, *ow_mean_local = NULL;
    double *uu_local = NULL, *uv_local = NULL, *uw_local = NULL;
    double *vv_local = NULL, *vw_local = NULL, *ww_local = NULL;
    double *k_local = NULL, *P_mean_local = NULL, *eps_local = NULL;

    if (accu_count > 0) {
        size_t grid_bytes_rs = (size_t)NX6 * NYD6 * NZ6 * sizeof(double);
        // Download all needed GPU arrays
        double *U_h = (double*)malloc(grid_bytes_rs);
        double *V_h = (double*)malloc(grid_bytes_rs);
        double *W_h = (double*)malloc(grid_bytes_rs);
        double *P_h = (double*)malloc(grid_bytes_rs);
        double *OU_h = (double*)malloc(grid_bytes_rs);
        double *OV_h = (double*)malloc(grid_bytes_rs);
        double *OW_h = (double*)malloc(grid_bytes_rs);
        double *UU_h = (double*)malloc(grid_bytes_rs);
        double *UV_h = (double*)malloc(grid_bytes_rs);
        double *UW_h = (double*)malloc(grid_bytes_rs);
        double *VV_h = (double*)malloc(grid_bytes_rs);
        double *VW_h = (double*)malloc(grid_bytes_rs);
        double *WW_h = (double*)malloc(grid_bytes_rs);
        CHECK_CUDA(cudaMemcpy(U_h, U, grid_bytes_rs, cudaMemcpyDeviceToHost));
        CHECK_CUDA(cudaMemcpy(V_h, V, grid_bytes_rs, cudaMemcpyDeviceToHost));
        CHECK_CUDA(cudaMemcpy(W_h, W, grid_bytes_rs, cudaMemcpyDeviceToHost));
        CHECK_CUDA(cudaMemcpy(P_h, P, grid_bytes_rs, cudaMemcpyDeviceToHost));
        CHECK_CUDA(cudaMemcpy(OU_h, OMEGA_U_SUM, grid_bytes_rs, cudaMemcpyDeviceToHost));
        CHECK_CUDA(cudaMemcpy(OV_h, OMEGA_V_SUM, grid_bytes_rs, cudaMemcpyDeviceToHost));
        CHECK_CUDA(cudaMemcpy(OW_h, OMEGA_W_SUM, grid_bytes_rs, cudaMemcpyDeviceToHost));
        CHECK_CUDA(cudaMemcpy(UU_h, UU, grid_bytes_rs, cudaMemcpyDeviceToHost));
        CHECK_CUDA(cudaMemcpy(UV_h, UV, grid_bytes_rs, cudaMemcpyDeviceToHost));
        CHECK_CUDA(cudaMemcpy(UW_h, UW, grid_bytes_rs, cudaMemcpyDeviceToHost));
        CHECK_CUDA(cudaMemcpy(VV_h, VV, grid_bytes_rs, cudaMemcpyDeviceToHost));
        CHECK_CUDA(cudaMemcpy(VW_h, VW, grid_bytes_rs, cudaMemcpyDeviceToHost));
        CHECK_CUDA(cudaMemcpy(WW_h, WW, grid_bytes_rs, cudaMemcpyDeviceToHost));

#if VTK_OUTPUT_LEVEL >= 1
        // Download gradient sums for epsilon (LEVEL >= 1 only)
        double *DX2_h[9];
        double *grad_ptrs[] = {DUDX2, DUDY2, DUDZ2, DVDX2, DVDY2, DVDZ2, DWDX2, DWDY2, DWDZ2};
        for (int g = 0; g < 9; g++) {
            DX2_h[g] = (double*)malloc(grid_bytes_rs);
            CHECK_CUDA(cudaMemcpy(DX2_h[g], grad_ptrs[g], grid_bytes_rs, cudaMemcpyDeviceToHost));
        }
#endif

        U_mean_local = (double*)malloc(localPoints * sizeof(double));
        V_mean_local = (double*)malloc(localPoints * sizeof(double));
        W_mean_local = (double*)malloc(localPoints * sizeof(double));
        ou_mean_local = (double*)malloc(localPoints * sizeof(double));
        ov_mean_local = (double*)malloc(localPoints * sizeof(double));
        ow_mean_local = (double*)malloc(localPoints * sizeof(double));
        uu_local = (double*)malloc(localPoints * sizeof(double));
        uv_local = (double*)malloc(localPoints * sizeof(double));
        uw_local = (double*)malloc(localPoints * sizeof(double));
        vv_local = (double*)malloc(localPoints * sizeof(double));
        vw_local = (double*)malloc(localPoints * sizeof(double));
        ww_local = (double*)malloc(localPoints * sizeof(double));
        k_local = (double*)malloc(localPoints * sizeof(double));
        P_mean_local = (double*)malloc(localPoints * sizeof(double));
#if VTK_OUTPUT_LEVEL >= 1
        eps_local = (double*)malloc(localPoints * sizeof(double));
#endif

        double inv_N = 1.0 / (double)accu_count;
        double Uref2 = (double)Uref * (double)Uref;
        int ridx = 0;
        for (int k = 3; k < NZ6-3; k++) {
        for (int j = 3; j < NYD6-3; j++) {
        for (int i = 3; i < NX6-3; i++) {
            int index = j*NZ6*NX6 + k*NX6 + i;
            double u_avg = U_h[index]*inv_N;
            double v_avg = V_h[index]*inv_N;
            double w_avg = W_h[index]*inv_N;

            // Mean velocity / Uref
            U_mean_local[ridx] = u_avg * inv_Uref;
            V_mean_local[ridx] = v_avg * inv_Uref;
            W_mean_local[ridx] = w_avg * inv_Uref;

            // Mean vorticity
            ou_mean_local[ridx] = OU_h[index]*inv_N;
            ov_mean_local[ridx] = OV_h[index]*inv_N;
            ow_mean_local[ridx] = OW_h[index]*inv_N;

            // Reynolds stress (fluctuation-based) / Uref²
            double uu_f = (UU_h[index]*inv_N - u_avg*u_avg) / Uref2;
            double uv_f = (UV_h[index]*inv_N - u_avg*v_avg) / Uref2;
            double uw_f = (UW_h[index]*inv_N - u_avg*w_avg) / Uref2;
            double vv_f = (VV_h[index]*inv_N - v_avg*v_avg) / Uref2;
            double vw_f = (VW_h[index]*inv_N - v_avg*w_avg) / Uref2;
            double ww_f = (WW_h[index]*inv_N - w_avg*w_avg) / Uref2;
            uu_local[ridx] = uu_f;
            uv_local[ridx] = uv_f;
            uw_local[ridx] = uw_f;
            vv_local[ridx] = vv_f;
            vw_local[ridx] = vw_f;
            ww_local[ridx] = ww_f;

            // TKE / Uref²
            k_local[ridx] = 0.5 * (uu_f + vv_f + ww_f);

            // P_mean
            P_mean_local[ridx] = P_h[index]*inv_N;

#if VTK_OUTPUT_LEVEL >= 1
            // epsilon (pseudo-dissipation): ν × Σ(∂u_i/∂x_j)²/N × h/Uref³
            double eps = 0.0;
            for (int g = 0; g < 9; g++)
                eps += DX2_h[g][index] * inv_N;
            eps_local[ridx] = (double)niu * eps * (double)H_HILL / ((double)Uref * Uref2);
#endif

            ridx++;
        }}}

        free(U_h); free(V_h); free(W_h); free(P_h);
        free(OU_h); free(OV_h); free(OW_h);
        free(UU_h); free(UV_h); free(UW_h);
        free(VV_h); free(VW_h); free(WW_h);
#if VTK_OUTPUT_LEVEL >= 1
        for (int g = 0; g < 9; g++) free(DX2_h[g]);
#endif
    }

    // Prepare local z coordinates
    int zidx = 0;
    for( int j = 3; j < NYD6-3; j++ ){
    for( int k = 3; k < NZ6-3; k++ ){
        z_local[zidx++] = z_h[j*NZ6 + k];
    }}

    // MPI_Gather all local arrays to rank 0
    double *u_global = NULL, *v_global = NULL, *w_global = NULL, *z_global = NULL;
    if( myid == 0 ) {
        u_global = (double*)malloc(gatherPoints * sizeof(double));
        v_global = (double*)malloc(gatherPoints * sizeof(double));
        w_global = (double*)malloc(gatherPoints * sizeof(double));
        z_global = (double*)malloc(zLocalSize * nProcs * sizeof(double));
    }
    MPI_Gather(u_local, localPoints, MPI_DOUBLE, u_global, localPoints, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Gather(v_local, localPoints, MPI_DOUBLE, v_global, localPoints, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Gather(w_local, localPoints, MPI_DOUBLE, w_global, localPoints, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Gather(z_local, zLocalSize,  MPI_DOUBLE, z_global, zLocalSize,  MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Vorticity gather
    double *ox_global = NULL, *oy_global = NULL, *oz_global = NULL;
    if( myid == 0 ) {
        ox_global = (double*)malloc(gatherPoints * sizeof(double));
        oy_global = (double*)malloc(gatherPoints * sizeof(double));
        oz_global = (double*)malloc(gatherPoints * sizeof(double));
    }
    MPI_Gather(ox_local, localPoints, MPI_DOUBLE, ox_global, localPoints, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Gather(oy_local, localPoints, MPI_DOUBLE, oy_global, localPoints, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Gather(oz_local, localPoints, MPI_DOUBLE, oz_global, localPoints, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Statistics gather (if accu_count > 0)
    // Allocate global arrays for stats on rank 0
    double *Um_g=NULL, *Vm_g=NULL, *Wm_g=NULL;
    double *oum_g=NULL, *ovm_g=NULL, *owm_g=NULL;
    double *uu_g=NULL, *uv_g=NULL, *uw_g=NULL, *vv_g=NULL, *vw_g=NULL, *ww_g=NULL;
    double *k_g=NULL, *Pm_g=NULL, *eps_g=NULL;

    if (accu_count > 0) {
        if (myid == 0) {
            Um_g = (double*)malloc(gatherPoints * sizeof(double));
            Vm_g = (double*)malloc(gatherPoints * sizeof(double));
            Wm_g = (double*)malloc(gatherPoints * sizeof(double));
            oum_g = (double*)malloc(gatherPoints * sizeof(double));
            ovm_g = (double*)malloc(gatherPoints * sizeof(double));
            owm_g = (double*)malloc(gatherPoints * sizeof(double));
            uu_g = (double*)malloc(gatherPoints * sizeof(double));
            uv_g = (double*)malloc(gatherPoints * sizeof(double));
            uw_g = (double*)malloc(gatherPoints * sizeof(double));
            vv_g = (double*)malloc(gatherPoints * sizeof(double));
            vw_g = (double*)malloc(gatherPoints * sizeof(double));
            ww_g = (double*)malloc(gatherPoints * sizeof(double));
            k_g  = (double*)malloc(gatherPoints * sizeof(double));
            Pm_g = (double*)malloc(gatherPoints * sizeof(double));
#if VTK_OUTPUT_LEVEL >= 1
            eps_g = (double*)malloc(gatherPoints * sizeof(double));
#endif
        }
        MPI_Gather(U_mean_local, localPoints, MPI_DOUBLE, Um_g, localPoints, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Gather(V_mean_local, localPoints, MPI_DOUBLE, Vm_g, localPoints, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Gather(W_mean_local, localPoints, MPI_DOUBLE, Wm_g, localPoints, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Gather(ou_mean_local, localPoints, MPI_DOUBLE, oum_g, localPoints, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Gather(ov_mean_local, localPoints, MPI_DOUBLE, ovm_g, localPoints, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Gather(ow_mean_local, localPoints, MPI_DOUBLE, owm_g, localPoints, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Gather(uu_local, localPoints, MPI_DOUBLE, uu_g, localPoints, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Gather(uv_local, localPoints, MPI_DOUBLE, uv_g, localPoints, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Gather(uw_local, localPoints, MPI_DOUBLE, uw_g, localPoints, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Gather(vv_local, localPoints, MPI_DOUBLE, vv_g, localPoints, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Gather(vw_local, localPoints, MPI_DOUBLE, vw_g, localPoints, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Gather(ww_local, localPoints, MPI_DOUBLE, ww_g, localPoints, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Gather(k_local,  localPoints, MPI_DOUBLE, k_g,  localPoints, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Gather(P_mean_local, localPoints, MPI_DOUBLE, Pm_g, localPoints, MPI_DOUBLE, 0, MPI_COMM_WORLD);
#if VTK_OUTPUT_LEVEL >= 1
        MPI_Gather(eps_local, localPoints, MPI_DOUBLE, eps_g, localPoints, MPI_DOUBLE, 0, MPI_COMM_WORLD);
#endif
    }

    // Rank 0: write VTK
    if( myid == 0 ) {
        double dy = LY / (double)(NY6 - 7);
        double *y_global_arr = (double*)malloc(NY6 * sizeof(double));
        for( int j = 0; j < NY6; j++ ) y_global_arr[j] = dy * (double)(j - 3);
        const int stride = nyLocal - 1;  // must be declared before any goto

        ostringstream oss;
        oss << "./result/Low_Order_field_" << step << ".vtk";
        ofstream out(oss.str().c_str());
        if( !out.is_open() ) {
            cerr << "ERROR: Cannot open VTK file: " << oss.str() << endl;
            goto vtk_cleanup;
        }

        // Header with metadata
        out << "# vtk DataFile Version 3.0\n";
        out << "STEP " << step << " FTT " << fixed << setprecision(4) << FTT_now
            << " ACCU_COUNT " << accu_count
            << " FORCE " << scientific << setprecision(8) << Force_h[0] << "\n";
        out << "ASCII\n";
        out << "DATASET STRUCTURED_GRID\n";
        out << "DIMENSIONS " << nxLocal << " " << nyGlobal << " " << nzLocal << "\n";

        out << "POINTS " << globalPoints << " double\n";
        out << fixed << setprecision(6);
        for( int k = 0; k < nzLocal; k++ ){
        for( int jg = 0; jg < nyGlobal; jg++ ){
        for( int i = 0; i < nxLocal; i++ ){
            int gpu_id = jg / stride;
            if( gpu_id >= jp ) gpu_id = jp - 1;
            int j_local = jg - gpu_id * stride;
            int z_gpu_offset = gpu_id * zLocalSize;
            int z_local_idx = j_local * nzLocal + k;
            out << x_h[i+3] << " " << y_global_arr[jg+3] << " " << z_global[z_gpu_offset + z_local_idx] << "\n";
        }}}

        out << "\nPOINT_DATA " << globalPoints << "\n";
        out << setprecision(15);

        // Helper macro: write one scalar field from gathered array
        #define WRITE_SCALAR(field_name, global_arr) \
        { \
            out << "\nSCALARS " << field_name << " double 1\n"; \
            out << "LOOKUP_TABLE default\n"; \
            for( int _k = 0; _k < nzLocal; _k++ ){ \
            for( int _jg = 0; _jg < nyGlobal; _jg++ ){ \
            for( int _i = 0; _i < nxLocal; _i++ ){ \
                int _gpu_id = _jg / stride; \
                if( _gpu_id >= jp ) _gpu_id = jp - 1; \
                int _j_local = _jg - _gpu_id * stride; \
                int _gidx = _gpu_id * localPoints + _k * nyLocal * nxLocal + _j_local * nxLocal + _i; \
                out << global_arr[_gidx] << "\n"; \
            }}} \
        }

        // === Always output: 6 instantaneous fields ===
        // Note: code u=spanwise(x), v=streamwise(y), w=wall-normal(z)
        WRITE_SCALAR("u_inst/Uref", u_global);
        WRITE_SCALAR("v_inst/Uref", v_global);
        WRITE_SCALAR("w_inst/Uref", w_global);
        WRITE_SCALAR("omega_x_inst", ox_global);
        WRITE_SCALAR("omega_y_inst", oy_global);
        WRITE_SCALAR("omega_z_inst", oz_global);

        // === FTT >= 20: statistics fields (LEVEL 0 = 20 fields) ===
        if (accu_count > 0) {
            WRITE_SCALAR("<u>/Uref", Um_g);
            WRITE_SCALAR("<v>/Uref", Vm_g);
            WRITE_SCALAR("<w>/Uref", Wm_g);
            WRITE_SCALAR("<omega_x>", oum_g);
            WRITE_SCALAR("<omega_y>", ovm_g);
            WRITE_SCALAR("<omega_z>", owm_g);
            WRITE_SCALAR("<u'u'>/Uref^2", uu_g);
            WRITE_SCALAR("<u'v'>/Uref^2", uv_g);
            WRITE_SCALAR("<u'w'>/Uref^2", uw_g);
            WRITE_SCALAR("<v'v'>/Uref^2", vv_g);
            WRITE_SCALAR("<v'w'>/Uref^2", vw_g);
            WRITE_SCALAR("<w'w'>/Uref^2", ww_g);
            WRITE_SCALAR("k/Uref^2", k_g);
            WRITE_SCALAR("<p>", Pm_g);
#if VTK_OUTPUT_LEVEL >= 1
            WRITE_SCALAR("epsilon*h/Uref^3", eps_g);
#endif
        }

        #undef WRITE_SCALAR

        out.close();
        cout << "VTK: Low_Order_field_" << step << ".vtk";
#if VTK_OUTPUT_LEVEL >= 1
        if (accu_count > 0) cout << " (accu=" << accu_count << ", 21 fields)";
#else
        if (accu_count > 0) cout << " (accu=" << accu_count << ", 20 fields)";
#endif
        else cout << " (6 fields)";
        cout << "\n";

vtk_cleanup:
        free(u_global); free(v_global); free(w_global); free(z_global);
        free(y_global_arr);
        free(ox_global); free(oy_global); free(oz_global);
        if (Um_g) free(Um_g);
        if (Vm_g) free(Vm_g);
        if (Wm_g) free(Wm_g);
        if (oum_g) free(oum_g);
        if (ovm_g) free(ovm_g);
        if (owm_g) free(owm_g);
        if (uu_g) free(uu_g);
        if (uv_g) free(uv_g);
        if (uw_g) free(uw_g);
        if (vv_g) free(vv_g);
        if (vw_g) free(vw_g);
        if (ww_g) free(ww_g);
        if (k_g)  free(k_g);
        if (Pm_g) free(Pm_g);
        if (eps_g) free(eps_g);
    }

    free(u_local); free(v_local); free(w_local); free(z_local);
    free(ox_local); free(oy_local); free(oz_local);
    if (U_mean_local) free(U_mean_local);
    if (V_mean_local) free(V_mean_local);
    if (W_mean_local) free(W_mean_local);
    if (ou_mean_local) free(ou_mean_local);
    if (ov_mean_local) free(ov_mean_local);
    if (ow_mean_local) free(ow_mean_local);
    if (uu_local) free(uu_local);
    if (uv_local) free(uv_local);
    if (uw_local) free(uw_local);
    if (vv_local) free(vv_local);
    if (vw_local) free(vw_local);
    if (ww_local) free(ww_local);
    if (k_local)  free(k_local);
    if (P_mean_local) free(P_mean_local);
    if (eps_local) free(eps_local);

    CHECK_MPI( MPI_Barrier(MPI_COMM_WORLD) );
}

#endif
