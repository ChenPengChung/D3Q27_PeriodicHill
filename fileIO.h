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
//PreCheckDir函數式的輔助函數1 
void ExistOrCreateDir(const char* doc) {
    //步驟一:創立資料夾 
	std::string path(doc);// path 是 C++ string 物件
	path = "./" + path;
    //檢查資料夾是否存在
	if (access(path.c_str(), F_OK) != 0) { //c++字串傳成對應的字元陣列 
        //不存在，用mkdir() 創建 
        if (mkdir(path.c_str(), S_IRWXU) == 0)
			std::cout << "folder " << path << " not exist, created"<< std::endl;// S_IRWXU = 擁有者可讀寫執行 
	}
}
//創立資料夾 "result" , "statistics" , "statistics/XXX" (XXX 為 32+3 個統計量子資料夾, 3 OMEGA 保留未使用)
void PreCheckDir() {
    //預先建立資料夾
	ExistOrCreateDir("result");//程式碼初始狀態使用
    //湍流統計//35 個子資料夾 (32 active + 3 OMEGA reserved)
	ExistOrCreateDir("statistics");
    if ( TBSWITCH ) {
		const int num_files = 36;
		std::string name[num_files] = {
        "U","V","W","P",//4
        "UU","UV","UW","VV","VW","WW",//6
        "PU","PV","PW","PP",//4
        "DUDX2","DUDY2","DUDZ2","DVDX2","DVDY2","DVDZ2","DWDX2","DWDY2","DWDZ2", //9
        "UUU","UUV","UUW","UVW","VVU","VVV","VVW","WWU","WWV","WWW",//10
        "OMEGA_X","OMEGA_Y","OMEGA_Z"};//3
		for( int i = 0; i < num_files; i++ ) {
			std::string fname = "./statistics/" + name[i];
			ExistOrCreateDir(fname.c_str());
		}
	}
    /*////////////////////////////////////////////*/
}
/*第二段:輸出速度場與分佈函數*/
//result系列輔助函數1.
void result_writebin(double* arr_h, const char *fname, const int myid){
    // 組合檔案路徑
    ostringstream oss;//輸出整數轉字串資料流
    oss << "./result/" << fname << "_" << myid << ".bin";
    string path = oss.str();

    // 用 C++ ofstream 開啟二進制檔案
    ofstream file(path.c_str(), ios::binary);
    if (!file) {
        cout << "Output data error, exit..." << endl;
        CHECK_MPI( MPI_Abort(MPI_COMM_WORLD, 1) );
    }

    // 寫入資料
    file.write(reinterpret_cast<char*>(arr_h), sizeof(double) * NX6 * NZ6 * NYD6);
    file.close();
}
// 寫入指定資料夾版本 (for binary checkpoint)
void result_writebin_to(double* arr_h, const char *folder, const char *fname, const int myid){
    ostringstream oss;
    oss << "./" << folder << "/" << fname << "_" << myid << ".bin";
    string path = oss.str();
    ofstream file(path.c_str(), ios::binary);
    if (!file) {
        cout << "Checkpoint write error: " << path << endl;
        CHECK_MPI( MPI_Abort(MPI_COMM_WORLD, 1) );
    }
    file.write(reinterpret_cast<char*>(arr_h), sizeof(double) * NX6 * NZ6 * NYD6);
    file.close();
}
//result系列輔助函數2.
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
//result系列主函數1.(寫檔案)
void result_writebin_velocityandf() {
    // 輸出 Paraview VTK (最終結果，分 GPU 子域輸出)
    ostringstream oss;
    oss << "./result/velocity_" << myid << "_Final.vtk";
    ofstream out(oss.str().c_str());
    // VTK Header
    out << "# vtk DataFile Version 3.0\n";
    out << "LBM Velocity Field\n";
    out << "ASCII\n";
    out << "DATASET STRUCTURED_GRID\n";
    out << "DIMENSIONS " << NX6-6 << " " << NYD6-6 << " " << NZ6-6 << "\n";  // Z: 64 計算點 (k=3..NZ6-4)

    // 座標點
    int nPoints = (NX6-6) * (NYD6-6) * (NZ6-6);
    out << "POINTS " << nPoints << " double\n";
    out << fixed << setprecision(6);
    for( int k = 3; k < NZ6-3; k++ ){    // 包含壁面 k=3 和 k=NZ6-4
    for( int j = 3; j < NYD6-3; j++ ){
    for( int i = 3; i < NX6-3; i++ ){
        out << x_h[i] << " " << y_h[j] << " " << z_h[j*NZ6+k] << "\n";
    }}}

    // 速度向量
    out << "\nPOINT_DATA " << nPoints << "\n";
    out << "VECTORS velocity double\n";
    out << setprecision(15);
    for( int k = 3; k < NZ6-3; k++ ){    // 包含壁面 k=3 和 k=NZ6-4
    for( int j = 3; j < NYD6-3; j++ ){
    for( int i = 3; i < NX6-3; i++ ){
        int index = j*NZ6*NX6 + k*NX6 + i;
        out << u_h_p[index] << " " << v_h_p[index] << " " << w_h_p[index] << "\n";
    }}}
    out.close();
    ////////////////////////////////////////////////////////////////////////////
    
    cout << "\n----------- Start Output, myid = " << myid << " ----------\n";
    
    // 輸出力 (只有 rank 0)
    if( myid == 0 ) {
        ofstream fp_gg("./result/0_force.dat");
        fp_gg << fixed << setprecision(15) << Force_h[0];
        fp_gg.close();
    }
    
    // 輸出巨觀量 (rho, u, v, w)
    result_writebin(rho_h_p, "rho", myid);
    result_writebin(u_h_p,   "u",   myid);
    result_writebin(v_h_p,   "v",   myid);
    result_writebin(w_h_p,   "w",   myid);
    
    // 輸出分佈函數 (f0 ~ f18)
    for( int q = 0; q < 19; q++ ) {
        ostringstream fname;
        fname << "f" << q;
        result_writebin(fh_p[q], fname.str().c_str(), myid);
    }
}
//result系列主函數2.(讀檔案)
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

    result_readbin(fh_p[0],  result, "f0",  myid);
    result_readbin(fh_p[1],  result, "f1",  myid);
    result_readbin(fh_p[2],  result, "f2",  myid);
    result_readbin(fh_p[3],  result, "f3",  myid);
    result_readbin(fh_p[4],  result, "f4",  myid);
    result_readbin(fh_p[5],  result, "f5",  myid);
    result_readbin(fh_p[6],  result, "f6",  myid);
    result_readbin(fh_p[7],  result, "f7",  myid);
    result_readbin(fh_p[8],  result, "f8",  myid);
    result_readbin(fh_p[9],  result, "f9",  myid);
    result_readbin(fh_p[10], result, "f10", myid);
    result_readbin(fh_p[11], result, "f11", myid);
    result_readbin(fh_p[12], result, "f12", myid);
    result_readbin(fh_p[13], result, "f13", myid);
    result_readbin(fh_p[14], result, "f14", myid);
    result_readbin(fh_p[15], result, "f15", myid);
    result_readbin(fh_p[16], result, "f16", myid);
    result_readbin(fh_p[17], result, "f17", myid);
    result_readbin(fh_p[18], result, "f18", myid);
}

//=============================================================================
// Binary Checkpoint I/O — 保留完整 f 分佈函數 (含 f^neq)
//=============================================================================

// 週期性二進制 checkpoint (新格式, single accu_count):
//   永遠寫: f00~f18 + rho + meta.dat (20 files + metadata)
//   FTT >= 20 (accu_count > 0): + 33 統計量累加器
//   合計: 19(f) + 1(rho) + 33(stats) = 53 .bin + meta.dat
void SaveBinaryCheckpoint(int ckpt_step) {
    // 1. 建立資料夾 ./checkpoint/step_XXXXXX/
    ostringstream dir_oss;
    dir_oss << "checkpoint/step_" << ckpt_step;
    string dir_name = dir_oss.str();
    if (myid == 0) {
        ExistOrCreateDir("checkpoint");
        ExistOrCreateDir(dir_name.c_str());
    }
    CHECK_MPI( MPI_Barrier(MPI_COMM_WORLD) );

    // 2. 寫入 f00~f18 (分佈函數，含 f^neq)
    for (int q = 0; q < 19; q++) {
        char fname[8];
        sprintf(fname, "f%02d", q);
        result_writebin_to(fh_p[q], dir_name.c_str(), fname, myid);
    }

    // 3. 寫入 rho (瞬時密度)
    result_writebin_to(rho_h_p, dir_name.c_str(), "rho", myid);

    // 4. 寫入 32 統計量累加器 (FTT >= 20, accu_count > 0)
    if (accu_count > 0 && (int)TBSWITCH) {
        const size_t rs_bytes = (size_t)NX6 * NYD6 * NZ6 * sizeof(double);
        double *rs_tmp = (double*)malloc(rs_bytes);

        #define SAVE_STAT(gpu_ptr, name) \
            CHECK_CUDA(cudaMemcpy(rs_tmp, gpu_ptr, rs_bytes, cudaMemcpyDeviceToHost)); \
            result_writebin_to(rs_tmp, dir_name.c_str(), name, myid)

        // 一階矩 (3)
        SAVE_STAT(U, "sum_u");     SAVE_STAT(V, "sum_v");     SAVE_STAT(W, "sum_w");
        // 二階矩 (6)
        SAVE_STAT(UU, "sum_uu");   SAVE_STAT(UV, "sum_uv");   SAVE_STAT(UW, "sum_uw");
        SAVE_STAT(VV, "sum_vv");   SAVE_STAT(VW, "sum_vw");   SAVE_STAT(WW, "sum_ww");
        // 三階矩 (10)
        SAVE_STAT(UUU, "sum_uuu"); SAVE_STAT(UUV, "sum_uuv"); SAVE_STAT(UUW, "sum_uuw");
        SAVE_STAT(VVU, "sum_uvv"); SAVE_STAT(UVW, "sum_uvw"); SAVE_STAT(WWU, "sum_uww");
        SAVE_STAT(VVV, "sum_vvv"); SAVE_STAT(VVW, "sum_vvw"); SAVE_STAT(WWV, "sum_vww");
        SAVE_STAT(WWW, "sum_www");
        // 壓力 (5)
        SAVE_STAT(P, "sum_P");     SAVE_STAT(PP, "sum_PP");
        SAVE_STAT(PU, "sum_Pu");   SAVE_STAT(PV, "sum_Pv");   SAVE_STAT(PW, "sum_Pw");
        // 梯度平方 (9)
        SAVE_STAT(DUDX2, "sum_dudx2"); SAVE_STAT(DUDY2, "sum_dudy2"); SAVE_STAT(DUDZ2, "sum_dudz2");
        SAVE_STAT(DVDX2, "sum_dvdx2"); SAVE_STAT(DVDY2, "sum_dvdy2"); SAVE_STAT(DVDZ2, "sum_dvdz2");
        SAVE_STAT(DWDX2, "sum_dwdx2"); SAVE_STAT(DWDY2, "sum_dwdy2"); SAVE_STAT(DWDZ2, "sum_dwdz2");

        #undef SAVE_STAT
        free(rs_tmp);
    }

    // 5. meta.dat (rank 0 only)
    if (myid == 0) {
        double FTT_ckpt = (double)ckpt_step * dt_global / (double)flow_through_time;
        ostringstream meta_path;
        meta_path << "./" << dir_name << "/metadata.dat";
        ofstream meta(meta_path.str().c_str());
        meta << "step=" << ckpt_step << "\n";
        meta << fixed << setprecision(15);
        meta << "FTT=" << FTT_ckpt << "\n";
        meta << "accu_count=" << accu_count << "\n";
        meta << "Force=" << Force_h[0] << "\n";
        meta << "dt_global=" << dt_global << "\n";
        meta.close();
        printf("[CHECKPOINT] Saved: %s/ (FTT=%.2f, accu=%d, %d files)\n",
               dir_name.c_str(), FTT_ckpt, accu_count,
               20 + ((accu_count > 0 && (int)TBSWITCH) ? 33 : 0));
    }
}

// 從 binary checkpoint 讀取 (新格式):
//   永遠讀: f00~f18 + rho + meta.dat
//   accu_count > 0: + 33 統計量累加器
//   u,v,w 宏觀量從 f0~f18 即時計算 (D3Q19 moments)
void LoadBinaryCheckpoint(const char* checkpoint_dir) {
    PreCheckDir();

    // D3Q19 速度向量 (for computing macroscopic from f)
    const int e_loc[19][3] = {
        {0,0,0},{1,0,0},{-1,0,0},{0,1,0},{0,-1,0},{0,0,1},{0,0,-1},
        {1,1,0},{-1,1,0},{1,-1,0},{-1,-1,0},
        {1,0,1},{-1,0,1},{1,0,-1},{-1,0,-1},
        {0,1,1},{0,-1,1},{0,1,-1},{0,-1,-1}
    };

    // 1. 讀取 metadata
    {
        ostringstream meta_path;
        meta_path << "./" << checkpoint_dir << "/metadata.dat";
        ifstream meta(meta_path.str().c_str());
        if (!meta.is_open()) {
            if (myid == 0) cout << "ERROR: Cannot open " << meta_path.str() << endl;
            CHECK_MPI( MPI_Abort(MPI_COMM_WORLD, 1) );
        }
        string line;
        while (getline(meta, line)) {
            if (line.find("step=") == 0)
                sscanf(line.c_str() + 5, "%d", &restart_step);
            else if (line.find("Force=") == 0)
                sscanf(line.c_str() + 6, "%lf", &Force_h[0]);
            else if (line.find("accu_count=") == 0)
                sscanf(line.c_str() + 11, "%d", &accu_count);
            // Backward compat: old format had vel_avg_count/rey_avg_count
            else if (line.find("vel_avg_count=") == 0) {
                int old_val = 0;
                sscanf(line.c_str() + 14, "%d", &old_val);
                if (accu_count == 0) accu_count = old_val;  // use vel as fallback
            }
            else if (line.find("rey_avg_count=") == 0) {
                int old_val = 0;
                sscanf(line.c_str() + 14, "%d", &old_val);
                if (old_val > 0 && accu_count == 0) accu_count = old_val;
            }
        }
        meta.close();
    }
    CHECK_CUDA( cudaMemcpy(Force_d, Force_h, sizeof(double), cudaMemcpyHostToDevice) );

    // 2. 讀取 f00~f18 (含完整 f^neq!)
    for (int q = 0; q < 19; q++) {
        char fname[8];
        sprintf(fname, "f%02d", q);
        // Try new format first (f00), fall back to old format (f0)
        ostringstream probe;
        probe << "./" << checkpoint_dir << "/" << fname << "_" << myid << ".bin";
        ifstream test_file(probe.str().c_str(), ios::binary);
        if (test_file.is_open()) {
            test_file.close();
            result_readbin(fh_p[q], checkpoint_dir, fname, myid);
        } else {
            // Old format: f0, f1, ..., f18
            ostringstream old_fname;
            old_fname << "f" << q;
            result_readbin(fh_p[q], checkpoint_dir, old_fname.str().c_str(), myid);
        }
    }

    // 3. 讀取 rho
    result_readbin(rho_h_p, checkpoint_dir, "rho", myid);

    // 4. 從 f0~f18 計算 u, v, w (不再從檔案讀取)
    {
        const size_t nTotal = (size_t)NX6 * NYD6 * NZ6;
        for (size_t idx = 0; idx < nTotal; idx++) {
            double rho_loc = rho_h_p[idx];
            double mx = 0.0, my = 0.0, mz = 0.0;
            for (int q = 0; q < 19; q++) {
                double fq = fh_p[q][idx];
                mx += e_loc[q][0] * fq;
                my += e_loc[q][1] * fq;
                mz += e_loc[q][2] * fq;
            }
            if (rho_loc > 1e-15) {
                u_h_p[idx] = mx / rho_loc;
                v_h_p[idx] = my / rho_loc;
                w_h_p[idx] = mz / rho_loc;
            } else {
                u_h_p[idx] = 0.0;
                v_h_p[idx] = 0.0;
                w_h_p[idx] = 0.0;
            }
        }
        if (myid == 0) printf("[CHECKPOINT] u,v,w computed from f00~f18 (D3Q19 moments)\n");
    }

    // 5. 讀取 32 統計量累加器 (if accu_count > 0)
    if (accu_count > 0 && (int)TBSWITCH) {
        const size_t rs_bytes = (size_t)NX6 * NYD6 * NZ6 * sizeof(double);
        double *rs_tmp = (double*)malloc(rs_bytes);

        #define LOAD_STAT(gpu_ptr, name, old_name) { \
            ostringstream _probe; \
            _probe << "./" << checkpoint_dir << "/" << name << "_" << myid << ".bin"; \
            ifstream _test(_probe.str().c_str(), ios::binary); \
            if (_test.is_open()) { _test.close(); \
                result_readbin(rs_tmp, checkpoint_dir, name, myid); \
            } else { \
                result_readbin(rs_tmp, checkpoint_dir, old_name, myid); \
            } \
            CHECK_CUDA(cudaMemcpy(gpu_ptr, rs_tmp, rs_bytes, cudaMemcpyHostToDevice)); \
        }

        // 一階矩 (3)
        LOAD_STAT(U, "sum_u", "RS_U");   LOAD_STAT(V, "sum_v", "RS_V");   LOAD_STAT(W, "sum_w", "RS_W");
        // 二階矩 (6)
        LOAD_STAT(UU, "sum_uu", "RS_UU"); LOAD_STAT(UV, "sum_uv", "RS_UV"); LOAD_STAT(UW, "sum_uw", "RS_UW");
        LOAD_STAT(VV, "sum_vv", "RS_VV"); LOAD_STAT(VW, "sum_vw", "RS_VW"); LOAD_STAT(WW, "sum_ww", "RS_WW");
        // 三階矩 (10)
        LOAD_STAT(UUU, "sum_uuu", "RS_UUU"); LOAD_STAT(UUV, "sum_uuv", "RS_UUV"); LOAD_STAT(UUW, "sum_uuw", "RS_UUW");
        LOAD_STAT(VVU, "sum_uvv", "RS_VVU"); LOAD_STAT(UVW, "sum_uvw", "RS_UVW"); LOAD_STAT(WWU, "sum_uww", "RS_WWU");
        LOAD_STAT(VVV, "sum_vvv", "RS_VVV"); LOAD_STAT(VVW, "sum_vvw", "RS_VVW"); LOAD_STAT(WWV, "sum_vww", "RS_WWV");
        LOAD_STAT(WWW, "sum_www", "RS_WWW");
        // 壓力 (5)
        LOAD_STAT(P, "sum_P", "RS_P");   LOAD_STAT(PP, "sum_PP", "RS_PP");
        LOAD_STAT(PU, "sum_Pu", "RS_PU"); LOAD_STAT(PV, "sum_Pv", "RS_PV"); LOAD_STAT(PW, "sum_Pw", "RS_PW");
        // 梯度平方 (9)
        LOAD_STAT(DUDX2, "sum_dudx2", "RS_DUDX2"); LOAD_STAT(DUDY2, "sum_dudy2", "RS_DUDY2"); LOAD_STAT(DUDZ2, "sum_dudz2", "RS_DUDZ2");
        LOAD_STAT(DVDX2, "sum_dvdx2", "RS_DVDX2"); LOAD_STAT(DVDY2, "sum_dvdy2", "RS_DVDY2"); LOAD_STAT(DVDZ2, "sum_dvdz2", "RS_DVDZ2");
        LOAD_STAT(DWDX2, "sum_dwdx2", "RS_DWDX2"); LOAD_STAT(DWDY2, "sum_dwdy2", "RS_DWDY2"); LOAD_STAT(DWDZ2, "sum_dwdz2", "RS_DWDZ2");

        #undef LOAD_STAT
        free(rs_tmp);

        // Copy sum_u/v/w → u_tavg_d (VTK backward compat: AccumulateTavg uses u_tavg)
        if (u_tavg_h != NULL) {
            CHECK_CUDA( cudaMemcpy(u_tavg_h, U, rs_bytes, cudaMemcpyDeviceToHost) );
            CHECK_CUDA( cudaMemcpy(v_tavg_h, V, rs_bytes, cudaMemcpyDeviceToHost) );
            CHECK_CUDA( cudaMemcpy(w_tavg_h, W, rs_bytes, cudaMemcpyDeviceToHost) );
            CHECK_CUDA( cudaMemcpy(u_tavg_d, U, rs_bytes, cudaMemcpyDeviceToDevice) );
            CHECK_CUDA( cudaMemcpy(v_tavg_d, V, rs_bytes, cudaMemcpyDeviceToDevice) );
            CHECK_CUDA( cudaMemcpy(w_tavg_d, W, rs_bytes, cudaMemcpyDeviceToDevice) );
        }

        if (myid == 0)
            printf("[CHECKPOINT] Statistics loaded: 33 arrays from %s/ (accu_count=%d)\n",
                   checkpoint_dir, accu_count);
    }

    if (myid == 0)
        printf("[CHECKPOINT] Loaded: %s/ → step=%d, Force=%.6e, accu=%d\n",
               checkpoint_dir, restart_step, Force_h[0], accu_count);
}

/*第三段:輸出湍統計量*/
//1.statistics系列輔助函數1.(寫主機端資料入bin)
void statistics_writebin(double *arr_d, const char *fname, const int myid){
    CHECK_MPI( MPI_Barrier(MPI_COMM_WORLD) );
    //先傳入裝置端
    const size_t nBytes = NX6 * NYD6 * NZ6 * sizeof(double);
    double *arr_h = (double*)malloc(nBytes);
    //複製回主機端
    CHECK_CUDA( cudaMemcpy(arr_h, arr_d, nBytes, cudaMemcpyDeviceToHost) );

    // 組合檔案路徑//利用主機端變數寫檔案
    ostringstream oss;
    oss << "./statistics/" << fname << "/" << fname << "_" << myid << ".bin";
    
    // C++ ofstream 寫入二進制
    ofstream file(oss.str().c_str(), ios::binary);
    for( int k = 3; k < NZ6-3;  k++ ){    // 包含壁面 k=3 和 k=NZ6-4
    for( int j = 3; j < NYD6-3; j++ ){
    for( int i = 3; i < NX6-3;  i++ ){
        int index = j*NX6*NZ6 + k*NX6 + i;
        file.write(reinterpret_cast<char*>(&arr_h[index]), sizeof(double));
    }}}
    file.close();
    free( arr_h );
}
//2.statistics系列輔助函數2.(讀bin到主機端資料)
void statistics_readbin(double * arr_d, const char *fname, const int myid){
    const size_t nBytes = NX6 * NYD6 * NZ6 * sizeof(double);
    double *arr_h = (double*)malloc(nBytes);

    // 組合檔案路徑
    ostringstream oss;
    oss << "./statistics/" << fname << "/" << fname << "_" << myid << ".bin";
    
    // C++ ifstream 讀取二進制
    ifstream file(oss.str().c_str(), ios::binary);
    for( int k = 3; k < NZ6-3;  k++ ){    // 包含壁面 k=3 和 k=NZ6-4
    for( int j = 3; j < NYD6-3; j++ ){
    for( int i = 3; i < NX6-3;  i++ ){
        const int index = j*NX6*NZ6 + k*NX6 + i;
        file.read(reinterpret_cast<char*>(&arr_h[index]), sizeof(double));
    }}}
    file.close();
    //從主機端複製資料到裝置端
    CHECK_CUDA( cudaMemcpy(arr_d, arr_h, nBytes, cudaMemcpyHostToDevice) );
    //釋放主機端
    free( arr_h );
    CHECK_MPI( MPI_Barrier(MPI_COMM_WORLD) );
}
//3.statistics系列主函數1.
void statistics_writebin_stress(){
    if( myid == 0 ) {
        ofstream fp_accu("./statistics/accu.dat");
        fp_accu << accu_count;
        fp_accu.close();
    }
    CHECK_MPI( MPI_Barrier(MPI_COMM_WORLD) );
    statistics_writebin(U, "U", myid);
	statistics_writebin(V, "V", myid);
	statistics_writebin(W, "W", myid);
	statistics_writebin(P, "P", myid);//4
	statistics_writebin(UU, "UU", myid);
	statistics_writebin(UV, "UV", myid);
	statistics_writebin(UW, "UW", myid);
	statistics_writebin(VV, "VV", myid);
	statistics_writebin(VW, "VW", myid);
	statistics_writebin(WW, "WW", myid);
	statistics_writebin(PU, "PU", myid);
	statistics_writebin(PV, "PV", myid);
	statistics_writebin(PW, "PW", myid);
	statistics_writebin(PP, "PP", myid);//5
	statistics_writebin(DUDX2, "DUDX2", myid);
	statistics_writebin(DUDY2, "DUDY2", myid);
	statistics_writebin(DUDZ2, "DUDZ2", myid);
	statistics_writebin(DVDX2, "DVDX2", myid);
	statistics_writebin(DVDY2, "DVDY2", myid);
	statistics_writebin(DVDZ2, "DVDZ2", myid);
	statistics_writebin(DWDX2, "DWDX2", myid);
	statistics_writebin(DWDY2, "DWDY2", myid);
	statistics_writebin(DWDZ2, "DWDZ2", myid);//9.
	statistics_writebin(UUU, "UUU", myid);
	statistics_writebin(UUV, "UUV", myid);
	statistics_writebin(UUW, "UUW", myid);
	statistics_writebin(UVW, "UVW", myid);
	statistics_writebin(VVU, "VVU", myid);
	statistics_writebin(VVV, "VVV", myid);
	statistics_writebin(VVW, "VVW", myid);
	statistics_writebin(WWU, "WWU", myid);
	statistics_writebin(WWV, "WWV", myid);
	statistics_writebin(WWW, "WWW", myid);//9.
	//statistics_writebin(OMEGA_X, "OMEGA_X", myid);
	//statistics_writebin(OMEGA_Y, "OMEGA_Y", myid);
	//statistics_writebin(OMEGA_Z, "OMEGA_Z", myid);//3.
}
//4.statistics系列主函數2.
void statistics_readbin_stress() {
    ifstream fp_accu("./statistics/accu.dat");
    fp_accu >> accu_count;
    fp_accu.close();
    if (myid == 0) printf("  statistics_readbin_stress: accu_count=%d loaded from accu.dat\n", accu_count);

    statistics_readbin(U, "U", myid);
	statistics_readbin(V, "V", myid);
	statistics_readbin(W, "W", myid);
	statistics_readbin(P, "P", myid);
	statistics_readbin(UU, "UU", myid);
	statistics_readbin(UV, "UV", myid);
	statistics_readbin(UW, "UW", myid);
	statistics_readbin(VV, "VV", myid);
	statistics_readbin(VW, "VW", myid);
	statistics_readbin(WW, "WW", myid);
	statistics_readbin(PU, "PU", myid);
	statistics_readbin(PV, "PV", myid);
	statistics_readbin(PW, "PW", myid);
	statistics_readbin(PP, "PP", myid);
	statistics_readbin(DUDX2, "DUDX2", myid);
	statistics_readbin(DUDY2, "DUDY2", myid);
	statistics_readbin(DUDZ2, "DUDZ2", myid);
	statistics_readbin(DVDX2, "DVDX2", myid);
	statistics_readbin(DVDY2, "DVDY2", myid);
	statistics_readbin(DVDZ2, "DVDZ2", myid);
	statistics_readbin(DWDX2, "DWDX2", myid);
	statistics_readbin(DWDY2, "DWDY2", myid);
	statistics_readbin(DWDZ2, "DWDZ2", myid);
	statistics_readbin(UUU, "UUU", myid);
	statistics_readbin(UUV, "UUV", myid);
	statistics_readbin(UUW, "UUW", myid);
	statistics_readbin(UVW, "UVW", myid);
	statistics_readbin(VVU, "VVU", myid);
	statistics_readbin(VVV, "VVV", myid);
	statistics_readbin(VVW, "VVW", myid);
	statistics_readbin(WWU, "WWU", myid);
	statistics_readbin(WWV, "WWV", myid);
	statistics_readbin(WWW, "WWW", myid);
	//statistics_readbin(OMEGA_X, "OMEGA_X", myid);
	//statistics_readbin(OMEGA_Y, "OMEGA_Y", myid);
	//statistics_readbin(OMEGA_Z, "OMEGA_Z", myid);
	CHECK_MPI( MPI_Barrier(MPI_COMM_WORLD) );
}
// ============================================================================
// GPU-count independent (merged) binary statistics I/O
// ============================================================================
// File format: raw double[(NZ6-6) × NY × (NX6-6)] in k→j_global→i order
// No header — dimensions implied by code's NX, NY, NZ defines.
// Only interior points stored (no ghost/buffer), same as per-rank version.
// j-mapping: j_global = myid * stride + (j_local - 3), stride = NY/jp
//
// Write: each rank packs stride unique j-points → MPI_Gather → rank 0 writes single file
// Read:  every rank reads full file → extracts stride+1 j-points (including overlap)

// Single-array merged write (GPU array → single merged .bin file)
void statistics_writebin_merged(double *arr_d, const char *fname) {
    CHECK_MPI( MPI_Barrier(MPI_COMM_WORLD) );
    const size_t nBytes = (size_t)NX6 * NYD6 * NZ6 * sizeof(double);
    double *arr_h = (double*)malloc(nBytes);
    CHECK_CUDA( cudaMemcpy(arr_h, arr_d, nBytes, cudaMemcpyDeviceToHost) );

    const int nx = NX6 - 6;       // interior x-points (i=3..NX6-4)
    const int ny = NY;             // total unique y-points (no overlap)
    const int nz = NZ6 - 6;       // interior z-points (k=3..NZ6-4)
    const int stride = NY / jp;    // unique j per rank

    // Pack local data: stride unique j-points (j_local = 3..3+stride-1, skip overlap at NYD6-4)
    const int local_count = nz * stride * nx;
    double *send_buf = (double*)malloc(local_count * sizeof(double));
    int idx = 0;
    for (int k = 3; k < NZ6 - 3; k++)
        for (int jl = 3; jl < 3 + stride; jl++)
            for (int i = 3; i < NX6 - 3; i++)
                send_buf[idx++] = arr_h[jl * NX6 * NZ6 + k * NX6 + i];

    // Gather to rank 0
    double *recv_buf = NULL;
    if (myid == 0) recv_buf = (double*)malloc((size_t)local_count * jp * sizeof(double));
    MPI_Gather(send_buf, local_count, MPI_DOUBLE,
               recv_buf, local_count, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Rank 0: reorder [rank][k][j_local][i] → [k][j_global][i] and write
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

// Single-array merged read (single merged .bin file → GPU array, any jp)
void statistics_readbin_merged(double *arr_d, const char *fname) {
    const int nx = NX6 - 6;
    const int ny = NY;
    const int nz = NZ6 - 6;
    const int stride = NY / jp;

    // Every rank reads the full merged file (small: ~4 MB per statistic)
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

    // Extract local portion (stride unique + 1 overlap point)
    const size_t nBytes = (size_t)NX6 * NYD6 * NZ6 * sizeof(double);
    double *arr_h = (double*)calloc(NX6 * NYD6 * NZ6, sizeof(double));
    int j_start = myid * stride;  // first global j for this rank

    for (int kk = 0; kk < nz; kk++) {
        int k = kk + 3;  // physical k index
        // Fill j_local = 3..3+stride-1 (stride unique points)
        for (int jl = 0; jl < stride; jl++) {
            int j_local = jl + 3;
            int j_global = j_start + jl;
            for (int ii = 0; ii < nx; ii++) {
                int i = ii + 3;
                arr_h[j_local * NX6 * NZ6 + k * NX6 + i] =
                    global_buf[(size_t)kk * ny * nx + j_global * nx + ii];
            }
        }
        // Fill overlap point: j_local = 3+stride = NYD6-4
        {
            int j_local = 3 + stride;  // = NYD6 - 4
            int j_global = (j_start + stride) % ny;  // wrap for periodic
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

// Master function: write all 33 statistics as merged binary (32 user-spec + UVW)
// All statistics share single accu_count (FTT >= FTT_STATS_START)
void statistics_writebin_merged_stress() {
    if (myid == 0) {
        ofstream fp_accu("./statistics/accu.dat");
        fp_accu << accu_count << " " << step;
        fp_accu.close();
    }
    CHECK_MPI( MPI_Barrier(MPI_COMM_WORLD) );
    // 一階矩 (3)
    statistics_writebin_merged(U, "U");
    statistics_writebin_merged(V, "V");
    statistics_writebin_merged(W, "W");
    // 壓力 (2: P, PP)
    statistics_writebin_merged(P, "P");
    statistics_writebin_merged(PP, "PP");
    // 二階矩 (6)
    statistics_writebin_merged(UU, "UU");
    statistics_writebin_merged(UV, "UV");
    statistics_writebin_merged(UW, "UW");
    statistics_writebin_merged(VV, "VV");
    statistics_writebin_merged(VW, "VW");
    statistics_writebin_merged(WW, "WW");
    // 壓力交叉項 (3)
    statistics_writebin_merged(PU, "PU");
    statistics_writebin_merged(PV, "PV");
    statistics_writebin_merged(PW, "PW");
    // 梯度平方 (9)
    statistics_writebin_merged(DUDX2, "DUDX2");
    statistics_writebin_merged(DUDY2, "DUDY2");
    statistics_writebin_merged(DUDZ2, "DUDZ2");
    statistics_writebin_merged(DVDX2, "DVDX2");
    statistics_writebin_merged(DVDY2, "DVDY2");
    statistics_writebin_merged(DVDZ2, "DVDZ2");
    statistics_writebin_merged(DWDX2, "DWDX2");
    statistics_writebin_merged(DWDY2, "DWDY2");
    statistics_writebin_merged(DWDZ2, "DWDZ2");
    // 三階矩 (10, includes UVW)
    statistics_writebin_merged(UUU, "UUU");
    statistics_writebin_merged(UUV, "UUV");
    statistics_writebin_merged(UUW, "UUW");
    statistics_writebin_merged(UVW, "UVW");
    statistics_writebin_merged(VVU, "VVU");
    statistics_writebin_merged(VVV, "VVV");
    statistics_writebin_merged(VVW, "VVW");
    statistics_writebin_merged(WWU, "WWU");
    statistics_writebin_merged(WWV, "WWV");
    statistics_writebin_merged(WWW, "WWW");
    if (myid == 0) printf("  statistics_writebin_merged_stress: 33 merged .bin files written (accu=%d, step=%d)\n",
                          accu_count, step);
}

// Master function: read all 33 statistics from merged binary (any jp)
void statistics_readbin_merged_stress() {
    // accu.dat format: "accu_count step" (backward compat: old "rey_avg_count [vel step]")
    ifstream fp_accu("./statistics/accu.dat");
    if (!fp_accu.is_open()) {
        if (myid == 0) printf("[WARNING] statistics_readbin_merged_stress: accu.dat not found, accu_count unchanged.\n");
        return;
    }
    int bin_step = -1;
    fp_accu >> accu_count;
    fp_accu >> bin_step;  // may fail silently if old format
    fp_accu.close();
    if (myid == 0)
        printf("  statistics_readbin_merged_stress: accu=%d, step=%d from accu.dat\n", accu_count, bin_step);

    statistics_readbin_merged(U, "U");
    statistics_readbin_merged(V, "V");
    statistics_readbin_merged(W, "W");
    statistics_readbin_merged(P, "P");
    statistics_readbin_merged(PP, "PP");
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
    statistics_readbin_merged(UVW, "UVW");
    statistics_readbin_merged(VVU, "VVU");
    statistics_readbin_merged(VVV, "VVV");
    statistics_readbin_merged(VVW, "VVW");
    statistics_readbin_merged(WWU, "WWU");
    statistics_readbin_merged(WWV, "WWV");
    statistics_readbin_merged(WWW, "WWW");
    CHECK_MPI( MPI_Barrier(MPI_COMM_WORLD) );
}

/*第三.5段:從合併VTK檔案讀取初始場 (INIT=2 restart)*/
// 從 merged VTK 讀取速度場，設 rho=1，f=feq，用於續跑
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

    // 初始化全場為 rho=1, u=v=w=0
    for (int idx = 0; idx < NX6 * NYD6 * NZ6; idx++) {
        rho_h_p[idx] = 1.0;
        u_h_p[idx]   = 0.0;
        v_h_p[idx]   = 0.0;
        w_h_p[idx]   = 0.0;
    }

    // 開啟 VTK 檔案
    ifstream vtk_in(vtk_path);
    if (!vtk_in.is_open()) {
        cout << "ERROR: Cannot open VTK file: " << vtk_path << endl;
        CHECK_MPI( MPI_Abort(MPI_COMM_WORLD, 1) );
    }

    // 解析 header，讀取 step, Force, accu_count 值
    double force_from_vtk = -1.0;
    int step_from_vtk = -1;
    int accu_count_from_vtk = 0;
    string vtk_line;
    while (getline(vtk_in, vtk_line)) {
        size_t spos = vtk_line.find("step=");
        if (spos != string::npos) {
            sscanf(vtk_line.c_str() + spos + 5, "%d", &step_from_vtk);
        }
        size_t fpos = vtk_line.find("Force=");
        if (fpos != string::npos) {
            sscanf(vtk_line.c_str() + fpos + 6, "%lf", &force_from_vtk);
        }
        // New format: accu_count=XXX
        size_t apos = vtk_line.find("accu_count=");
        if (apos != string::npos) {
            sscanf(vtk_line.c_str() + apos + 11, "%d", &accu_count_from_vtk);
        }
        // Backward compat: old "vel_avg_count=XXX" → treat as accu_count
        if (accu_count_from_vtk == 0) {
            size_t vpos = vtk_line.find("vel_avg_count=");
            if (vpos != string::npos) {
                sscanf(vtk_line.c_str() + vpos + 14, "%d", &accu_count_from_vtk);
            }
        }
        // Backward compat: old "tavg_count=" → treat as accu_count
        if (accu_count_from_vtk == 0) {
            size_t tpos = vtk_line.find("tavg_count=");
            if (tpos != string::npos) {
                sscanf(vtk_line.c_str() + tpos + 11, "%d", &accu_count_from_vtk);
            }
        }
        if (vtk_line.find("VECTORS") != string::npos) break;
    }

    // 計算本 rank 的 jg 範圍 (stride-based, 端點重疊)
    // rank0: 0~32, rank1: 32~64, rank2: 64~96, rank3: 96~128
    const int stride = nyLocal - 1;  // = 32 (unique y-points per rank)
    int jg_start = myid * stride;
    int jg_end   = jg_start + nyLocal - 1;  // = jg_start + 32
    if (jg_end > nyGlobal - 1) jg_end = nyGlobal - 1;

    // 按 VTK 寫入順序讀取: k(outer) → jg(middle) → i(inner)
    double u_val, v_val, w_val;
    for (int k = 0; k < nzLocal; k++) {
    for (int jg = 0; jg < nyGlobal; jg++) {
    for (int i = 0; i < nxLocal; i++) {
        vtk_in >> u_val >> v_val >> w_val;
        if (jg >= jg_start && jg <= jg_end) {
            int j_local = jg - jg_start;
            int j  = j_local + 3;   // buffer offset
            int kk = k + 3;
            int ii = i + 3;
            int index = j * NX6 * NZ6 + kk * NX6 + ii;
            u_h_p[index]   = u_val;
            v_h_p[index]   = v_val;
            w_h_p[index]   = w_val;
            rho_h_p[index] = 1.0;
        }
    }}}
    vtk_in.close();

    // ========== 讀取時間平均場 ==========
    // New format: U_mean (÷Uref), W_mean (÷Uref), V_mean (÷Uref)
    // Old format: v_time_avg, w_time_avg (lattice units, no Uref normalization)
    // VTK 中存的是已除以 count 的平均值; main.cu 稍後會乘回 accu_count 得到累加和
    if (accu_count_from_vtk > 0 && v_tavg_h != NULL && w_tavg_h != NULL && u_tavg_h != NULL) {
        ifstream vtk_tavg(vtk_path);
        string line_tavg;
        bool found_U_mean = false, found_W_mean = false, found_V_mean = false;
        bool found_vtavg = false, found_wtavg = false;  // old format fallback

        // Helper lambda-like: read one scalar field from VTK into tavg array
        // We search for new names first, then fall back to old names
        #define READ_SCALAR_FIELD(tavg_arr, search_str, found_flag, uref_scale) \
        { \
            vtk_tavg.clear(); vtk_tavg.seekg(0); \
            string _line; \
            while (getline(vtk_tavg, _line)) { \
                if (_line.find(search_str) != string::npos) { \
                    getline(vtk_tavg, _line); /* skip LOOKUP_TABLE */ \
                    found_flag = true; \
                    double _val; \
                    for (int _k = 0; _k < nzLocal; _k++) { \
                    for (int _jg = 0; _jg < nyGlobal; _jg++) { \
                    for (int _i = 0; _i < nxLocal; _i++) { \
                        vtk_tavg >> _val; \
                        if (_jg >= jg_start && _jg <= jg_end) { \
                            int _jl = _jg - jg_start; \
                            int _idx = (_jl+3)*NX6*NZ6 + (_k+3)*NX6 + (_i+3); \
                            tavg_arr[_idx] = _val * uref_scale; \
                        } \
                    }}} \
                    break; \
                } \
            } \
        }

        // Try new format first (normalized by Uref → multiply back to lattice units)
        READ_SCALAR_FIELD(v_tavg_h, "SCALARS U_mean", found_U_mean, (double)Uref);
        READ_SCALAR_FIELD(w_tavg_h, "SCALARS W_mean", found_W_mean, (double)Uref);
        READ_SCALAR_FIELD(u_tavg_h, "SCALARS V_mean", found_V_mean, (double)Uref);

        // Backward compat: fall back to old names (no Uref normalization)
        if (!found_U_mean) {
            READ_SCALAR_FIELD(v_tavg_h, "SCALARS v_time_avg", found_vtavg, 1.0);
        }
        if (!found_W_mean) {
            READ_SCALAR_FIELD(w_tavg_h, "SCALARS w_time_avg", found_wtavg, 1.0);
        }

        #undef READ_SCALAR_FIELD
        vtk_tavg.close();

        bool have_streamwise = found_U_mean || found_vtavg;
        bool have_wallnormal = found_W_mean || found_wtavg;

        if (have_streamwise && have_wallnormal) {
            accu_count = accu_count_from_vtk;
            if (myid == 0) {
                printf("  VTK restart: velocity time-average restored (accu_count=%d", accu_count);
                if (found_U_mean) printf(", U_mean format");
                else printf(", v_time_avg format");
                if (found_V_mean) printf(", V_mean");
                printf(")\n");
            }
        } else {
            accu_count = 0;
            if (myid == 0)
                printf("  VTK restart: time-average fields not found, starting fresh\n");
        }
    } else {
        if (myid == 0 && accu_count_from_vtk == 0)
            printf("  VTK restart: no accu_count in header, starting fresh.\n");
    }

    // ========== x-direction 週期性邊界填充 buffer layer ==========
    // periodicSW 邏輯: left i=0,1,2 ← i=32,33,34; right i=36,37,38 ← i=4,5,6
    {
        const int buffer = 3;
        const int shift = NX6 - 2*buffer - 1;  // = 32
        for (int j = 3; j < NYD6-3; j++) {
        for (int k = 3; k < NZ6-3; k++) {
            for (int ib = 0; ib < buffer; ib++) {
                // Left buffer: i=ib ← i=ib+shift
                int idx_buf = j * NX6 * NZ6 + k * NX6 + ib;
                int idx_src = idx_buf + shift;
                u_h_p[idx_buf]   = u_h_p[idx_src];
                v_h_p[idx_buf]   = v_h_p[idx_src];
                w_h_p[idx_buf]   = w_h_p[idx_src];
                rho_h_p[idx_buf] = rho_h_p[idx_src];

                // Right buffer: i=(NX6-1-ib) ← i=(NX6-1-ib)-shift
                idx_buf = j * NX6 * NZ6 + k * NX6 + (NX6 - 1 - ib);
                idx_src = idx_buf - shift;
                u_h_p[idx_buf]   = u_h_p[idx_src];
                v_h_p[idx_buf]   = v_h_p[idx_src];
                w_h_p[idx_buf]   = w_h_p[idx_src];
                rho_h_p[idx_buf] = rho_h_p[idx_src];
            }
        }}
        if (myid == 0) printf("  VTK restart: x-periodic buffer layers filled.\n");
    }

    // ========== MPI 交換 ghost zone (u, v, w, rho) ==========
    // VTK 只包含計算區域 j=3..NYD6-4，ghost zone j=0..2 和 j=NYD6-3..NYD6-1 需要 MPI 填充
    // 否則 Init_FPC_Kernel 會讀到錯誤的 stencil 值導致發散
    {
        const int slice_size = NX6 * NZ6;
        const int ghost_count = 3 * slice_size;  // 3 個 j-slices
        
        // 交換 u_h_p
        // 發送 j=4..6 到左鄰居的 j=NYD6-3..NYD6-1，接收從右鄰居的 j=4..6 到 j=NYD6-3..NYD6-1
        // (與 main loop iToLeft/iFromRight 相同 pattern: send j=Buffer+1, recv j=NYD6-Buffer)
        MPI_Sendrecv(&u_h_p[4 * slice_size],       ghost_count, MPI_DOUBLE, l_nbr, 600,
                     &u_h_p[(NYD6-3) * slice_size], ghost_count, MPI_DOUBLE, r_nbr, 600,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        // 發送 j=NYD6-7..NYD6-5 到右鄰居的 j=0..2，接收從左鄰居的 j=NYD6-7..NYD6-5 到 j=0..2
        MPI_Sendrecv(&u_h_p[(NYD6-7) * slice_size], ghost_count, MPI_DOUBLE, r_nbr, 601,
                     &u_h_p[0],                      ghost_count, MPI_DOUBLE, l_nbr, 601,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        // 交換 v_h_p
        MPI_Sendrecv(&v_h_p[4 * slice_size],       ghost_count, MPI_DOUBLE, l_nbr, 602,
                     &v_h_p[(NYD6-3) * slice_size], ghost_count, MPI_DOUBLE, r_nbr, 602,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Sendrecv(&v_h_p[(NYD6-7) * slice_size], ghost_count, MPI_DOUBLE, r_nbr, 603,
                     &v_h_p[0],                      ghost_count, MPI_DOUBLE, l_nbr, 603,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        // 交換 w_h_p
        MPI_Sendrecv(&w_h_p[4 * slice_size],       ghost_count, MPI_DOUBLE, l_nbr, 604,
                     &w_h_p[(NYD6-3) * slice_size], ghost_count, MPI_DOUBLE, r_nbr, 604,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Sendrecv(&w_h_p[(NYD6-7) * slice_size], ghost_count, MPI_DOUBLE, r_nbr, 605,
                     &w_h_p[0],                      ghost_count, MPI_DOUBLE, l_nbr, 605,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        // 交換 rho_h_p
        MPI_Sendrecv(&rho_h_p[4 * slice_size],       ghost_count, MPI_DOUBLE, l_nbr, 606,
                     &rho_h_p[(NYD6-3) * slice_size], ghost_count, MPI_DOUBLE, r_nbr, 606,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Sendrecv(&rho_h_p[(NYD6-7) * slice_size], ghost_count, MPI_DOUBLE, r_nbr, 607,
                     &rho_h_p[0],                      ghost_count, MPI_DOUBLE, l_nbr, 607,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        
        if (myid == 0) printf("  VTK restart: ghost zone (u,v,w,rho) exchanged between MPI ranks.\n");
    }

    // ========== z-direction 壁面外插 buffer/ghost layer ==========
    // 與 GenerateMesh_Z 相同邏輯: linear extrapolation
    // Bottom wall k=3, Top wall k=NZ6-4; buffer k=2,NZ6-3; ghost k=0,1,NZ6-2,NZ6-1
    {
        #define BUF_IDX(jj,kk,ii) ((jj)*NX6*NZ6 + (kk)*NX6 + (ii))
        for (int j = 0; j < NYD6; j++) {
        for (int i = 0; i < NX6; i++) {
            double *fields[] = {u_h_p, v_h_p, w_h_p, rho_h_p};
            for (int f = 0; f < 4; f++) {
                double *F = fields[f];
                // Bottom: k=2 (buffer), k=1, k=0 (ghost)
                F[BUF_IDX(j,2,i)]     = 2.0 * F[BUF_IDX(j,3,i)]     - F[BUF_IDX(j,4,i)];
                F[BUF_IDX(j,1,i)]     = 2.0 * F[BUF_IDX(j,2,i)]     - F[BUF_IDX(j,3,i)];
                F[BUF_IDX(j,0,i)]     = 2.0 * F[BUF_IDX(j,1,i)]     - F[BUF_IDX(j,2,i)];
                // Top: k=NZ6-3 (buffer), k=NZ6-2, k=NZ6-1 (ghost)
                F[BUF_IDX(j,NZ6-3,i)] = 2.0 * F[BUF_IDX(j,NZ6-4,i)] - F[BUF_IDX(j,NZ6-5,i)];
                F[BUF_IDX(j,NZ6-2,i)] = 2.0 * F[BUF_IDX(j,NZ6-3,i)] - F[BUF_IDX(j,NZ6-4,i)];
                F[BUF_IDX(j,NZ6-1,i)] = 2.0 * F[BUF_IDX(j,NZ6-2,i)] - F[BUF_IDX(j,NZ6-3,i)];
            }
        }}
        #undef BUF_IDX
        if (myid == 0) printf("  VTK restart: z-direction buffer/ghost layers extrapolated.\n");
    }

    // 從 (rho, u, v, w) 計算 f = feq (現在包含正確的 ghost zone 值)
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

    // 設定 Force: 必須從 VTK header 讀取 (續跑不能重新初始化外力)
    if (force_from_vtk > 0.0) {
        Force_h[0] = force_from_vtk;
        if (myid == 0) printf("  Force restored from VTK header: %.5E\n", force_from_vtk);
    } else {
        if (myid == 0) {
            fprintf(stderr, "ERROR: Force= not found in VTK header [%s].\n", vtk_path);
            fprintf(stderr, "  Restart VTK must contain Force value. Re-run the simulation to generate a new VTK with Force.\n");
        }
        CHECK_MPI( MPI_Abort(MPI_COMM_WORLD, 1) );
    }

    // Force cap 移至 main.cu 初始狀態顯示之後 (先顯示 VTK 原始值)
    CHECK_CUDA( cudaMemcpy(Force_d, Force_h, sizeof(double), cudaMemcpyHostToDevice) );

    // 設定續跑起始步 = VTK step (用於顯示和 FTT 計算)
    // 注意: for-loop 實際從 restart_step+1 (偶數) 開始，以對齊 step%N==1 監測
    if (step_from_vtk > 0) {
        restart_step = step_from_vtk;
        if (myid == 0) {
            double FTT_restart = (double)restart_step * dt_global / (double)flow_through_time;
            printf("  Restart step = %d, FTT = %.4f\n", restart_step, FTT_restart);
        }
    } else {
        if (myid == 0)
            printf("  WARNING: step= not found in VTK header, restarting from step 0.\n");
        restart_step = 0;
    }

    printf("Rank %d: Initialized from VTK [%s], jg=%d..%d -> local j=%d..%d\n",
           myid, vtk_path, jg_start, jg_end, 3, 3 + (jg_end - jg_start));
}

/*第四段:每1000步輸出可視化VTK檔案*/
/*
 * ═══════════════════════════════════════════════════════════
 * 座標映射：Code ↔ VTK (Benchmark) ↔ ERCOFTAC
 * ═══════════════════════════════════════════════════════════
 *
 *   Code    物理方向      VTK 名稱    ERCOFTAC 名稱
 *   ─────────────────────────────────────────────────
 *   x (i)   展向 span     v           w
 *   y (j)   流向 stream   u           u
 *   z (k)   法向 normal   w           v
 *
 * 關鍵映射（ERCOFTAC 7 欄比較時）：
 *   ERCOFTAC col 2: <U>/Ub    = VTK U_mean  (streamwise = code v)
 *   ERCOFTAC col 3: <V>/Ub    = VTK W_mean  (wall-normal = code w, 不是 V_mean！)
 *   ERCOFTAC col 4: <u'u'>/Ub²= VTK uu_RS   (stream×stream)
 *   ERCOFTAC col 5: <v'v'>/Ub²= VTK ww_RS   (wallnorm×wallnorm, 不是 vv_RS！)
 *   ERCOFTAC col 6: <u'v'>/Ub²= VTK uw_RS   (stream×wallnorm, 不是 uv_RS！)
 *   ERCOFTAC col 7: k/Ub²     = VTK k_TKE
 *
 * 渦度映射（與速度 SCALAR 一致的 benchmark frame）：
 *   VTK omega_u (流向渦度) = code ω_y (oy)
 *   VTK omega_v (展向渦度) = code ω_x (ox)
 *   VTK omega_w (法向渦度) = code ω_z (oz)
 * ═══════════════════════════════════════════════════════════
 */
// 合併所有 GPU 結果，輸出單一 VTK 檔案 (Paraview)
void fileIO_velocity_vtk_merged(int step) {
    // 每個 GPU 內部有效區域的 y 層數 (不含 ghost)
    const int nyLocal = NYD6 - 6;  // 去除上下各3層ghost
    const int nxLocal = NX6 - 6;
    const int nzLocal = NZ6 - 6;  // 64 個 k 計算點 (k=3..NZ6-4)
    
    // 每個 GPU 發送的點數
    const int localPoints = nxLocal * nyLocal * nzLocal;
    const int zLocalSize = nyLocal * nzLocal;
    
    // 全域 y 層數
    const int nyGlobal = NY6 - 6;
    const int globalPoints = nxLocal * nyGlobal * nzLocal;  // VTK 輸出用
    // MPI_Gather 需要的緩衝區大小 = localPoints * nProcs
    const int gatherPoints = localPoints * nProcs;
    
    // 準備本地速度資料 (去除 ghost cells, 只取內部)
    double *u_local = (double*)malloc(localPoints * sizeof(double));
    double *v_local = (double*)malloc(localPoints * sizeof(double));
    double *w_local = (double*)malloc(localPoints * sizeof(double));
    double *z_local = (double*)malloc(zLocalSize * sizeof(double));
    
    int idx = 0;
    for( int k = 3; k < NZ6-3; k++ ){    // 包含壁面
    for( int j = 3; j < NYD6-3; j++ ){
    for( int i = 3; i < NX6-3; i++ ){
        int index = j*NZ6*NX6 + k*NX6 + i;
        u_local[idx] = u_h_p[index];
        v_local[idx] = v_h_p[index];
        w_local[idx] = w_h_p[index];
        idx++;
    }}}

    // 準備本地時間平均資料 (若有累積), 正規化: ÷accu_count÷Uref
    // 只輸出 U_mean (streamwise=code v) 和 W_mean (wall-normal=code w)
    double *vt_local = NULL, *wt_local = NULL;
    if (accu_count > 0) {
        vt_local = (double*)malloc(localPoints * sizeof(double));
        wt_local = (double*)malloc(localPoints * sizeof(double));
        double inv_count_uref = 1.0 / ((double)accu_count * (double)Uref);
        int tidx = 0;
        for( int k = 3; k < NZ6-3; k++ ){
        for( int j = 3; j < NYD6-3; j++ ){
        for( int i = 3; i < NX6-3; i++ ){
            int index = j*NZ6*NX6 + k*NX6 + i;
            vt_local[tidx] = v_tavg_h[index] * inv_count_uref;  // U_mean (streamwise=code v)
            wt_local[tidx] = w_tavg_h[index] * inv_count_uref;  // W_mean (wall-normal=code w)
            tidx++;
        }}}
    }

    // 準備 Reynolds stress + P_mean (若有累積)
    // Level 0: 只輸出 ERCOFTAC 7 欄對應的 3 個 RS + k_TKE + P_mean
    //   VTK uu_RS  ← code (Σvv/N − v̄²) / Uref²   = ERCOFTAC <u'u'>/Ub²  [GPU: VV, V]
    //   VTK uw_RS  ← code (Σvw/N − v̄·w̄) / Uref²  = ERCOFTAC <u'v'>/Ub²  [GPU: VW, V, W]
    //   VTK ww_RS  ← code (Σww/N − w̄²) / Uref²    = ERCOFTAC <v'v'>/Ub²  [GPU: WW, W]
    //   VTK k_TKE  ← 0.5*(uu + vv_inline + ww)     = ERCOFTAC k/Ub²
    //   vv (展向 RS) 只用於 k_TKE 計算，不單獨輸出
    double *uu_local = NULL, *uw_local = NULL, *ww_local = NULL;
    double *k_local = NULL, *p_mean_local = NULL;
    if (accu_count > 0 && (int)TBSWITCH) {
        uu_local = (double*)malloc(localPoints * sizeof(double));
        uw_local = (double*)malloc(localPoints * sizeof(double));
        ww_local = (double*)malloc(localPoints * sizeof(double));
        k_local  = (double*)malloc(localPoints * sizeof(double));
        p_mean_local = (double*)malloc(localPoints * sizeof(double));

        // Copy 8 MeanVars arrays from GPU → temporary host buffers
        // (UV, UW not needed: uv_RS/vw_RS removed from Level 0)
        size_t grid_bytes_rs = (size_t)NX6 * NYD6 * NZ6 * sizeof(double);
        double *U_h_rs = (double*)malloc(grid_bytes_rs);
        double *V_h_rs = (double*)malloc(grid_bytes_rs);
        double *W_h_rs = (double*)malloc(grid_bytes_rs);
        double *P_h_rs = (double*)malloc(grid_bytes_rs);
        double *UU_h_rs = (double*)malloc(grid_bytes_rs);
        double *VV_h_rs = (double*)malloc(grid_bytes_rs);
        double *WW_h_rs = (double*)malloc(grid_bytes_rs);
        double *VW_h_rs = (double*)malloc(grid_bytes_rs);
        CHECK_CUDA(cudaMemcpy(U_h_rs,  U,  grid_bytes_rs, cudaMemcpyDeviceToHost));
        CHECK_CUDA(cudaMemcpy(V_h_rs,  V,  grid_bytes_rs, cudaMemcpyDeviceToHost));
        CHECK_CUDA(cudaMemcpy(W_h_rs,  W,  grid_bytes_rs, cudaMemcpyDeviceToHost));
        CHECK_CUDA(cudaMemcpy(P_h_rs,  P,  grid_bytes_rs, cudaMemcpyDeviceToHost));
        CHECK_CUDA(cudaMemcpy(UU_h_rs, UU, grid_bytes_rs, cudaMemcpyDeviceToHost));
        CHECK_CUDA(cudaMemcpy(VV_h_rs, VV, grid_bytes_rs, cudaMemcpyDeviceToHost));
        CHECK_CUDA(cudaMemcpy(WW_h_rs, WW, grid_bytes_rs, cudaMemcpyDeviceToHost));
        CHECK_CUDA(cudaMemcpy(VW_h_rs, VW, grid_bytes_rs, cudaMemcpyDeviceToHost));

        double inv_N = 1.0 / (double)accu_count;
        double inv_Uref2 = 1.0 / ((double)Uref * (double)Uref);
        int ridx = 0;
        for( int k = 3; k < NZ6-3; k++ ){
        for( int j = 3; j < NYD6-3; j++ ){
        for( int i = 3; i < NX6-3; i++ ){
            int index = j*NZ6*NX6 + k*NX6 + i;
            double u_m = U_h_rs[index]*inv_N;  // <u_code> (spanwise)
            double v_m = V_h_rs[index]*inv_N;  // <v_code> (streamwise)
            double w_m = W_h_rs[index]*inv_N;  // <w_code> (wall-normal)
            uu_local[ridx] = (VV_h_rs[index]*inv_N - v_m*v_m) * inv_Uref2;  // ERCOFTAC <u'u'>/Ub²
            uw_local[ridx] = (VW_h_rs[index]*inv_N - v_m*w_m) * inv_Uref2;  // ERCOFTAC <u'v'>/Ub²
            ww_local[ridx] = (WW_h_rs[index]*inv_N - w_m*w_m) * inv_Uref2;  // ERCOFTAC <v'v'>/Ub²
            double vv_val  = (UU_h_rs[index]*inv_N - u_m*u_m) * inv_Uref2;  // 展向 RS (k_TKE 用)
            k_local[ridx]  = 0.5 * (uu_local[ridx] + vv_val + ww_local[ridx]);  // k/Ub²
            p_mean_local[ridx] = P_h_rs[index] * inv_N;  // <P>
            ridx++;
        }}}
        free(U_h_rs); free(V_h_rs); free(W_h_rs); free(P_h_rs);
        free(UU_h_rs); free(VV_h_rs); free(WW_h_rs); free(VW_h_rs);
    }

    // Compute instantaneous vorticity (omega_u, omega_v, omega_w) — benchmark coordinates
    // Curvilinear coordinate derivatives:
    //   ∂φ/∂x = (1/dx) ∂φ/∂i
    //   ∂φ/∂y = (1/dy) ∂φ/∂j + dk_dy ∂φ/∂k
    //   ∂φ/∂z = dk_dz ∂φ/∂k
    double *ox_local  = (double*)malloc(localPoints * sizeof(double));
    double *oy_local  = (double*)malloc(localPoints * sizeof(double));
    double *oz_local  = (double*)malloc(localPoints * sizeof(double));
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

            // omega_x = ∂w/∂y - ∂v/∂z
            ox_local[oidx] = dy_inv * dw_dj + dkdy * dw_dk - dkdz * dv_dk;
            // omega_y = ∂u/∂z - ∂w/∂x
            oy_local[oidx] = dkdz * du_dk - dx_inv * dw_di;
            // omega_z = ∂v/∂x - ∂u/∂y
            oz_local[oidx] = dx_inv * dv_di - dy_inv * du_dj - dkdy * du_dk;

            oidx++;
        }}}
    }

    // 準備本地 z 座標
    int zidx = 0;
    for( int j = 3; j < NYD6-3; j++ ){
    for( int k = 3; k < NZ6-3; k++ ){    // 包含壁面
        z_local[zidx++] = z_h[j*NZ6 + k];
    }}
    
    // rank 0 分配接收緩衝區
    double *u_global = NULL;
    double *v_global = NULL;
    double *w_global = NULL;
    double *z_global = NULL;
    
    if( myid == 0 ) {
        u_global = (double*)malloc(gatherPoints * sizeof(double));
        v_global = (double*)malloc(gatherPoints * sizeof(double));
        w_global = (double*)malloc(gatherPoints * sizeof(double));
        z_global = (double*)malloc(zLocalSize * nProcs * sizeof(double));
    }

    double *vt_global = NULL, *wt_global = NULL;
    if( myid == 0 && accu_count > 0 ) {
        vt_global = (double*)malloc(gatherPoints * sizeof(double));
        wt_global = (double*)malloc(gatherPoints * sizeof(double));
    }

    double *uu_global = NULL, *uw_global = NULL, *ww_global = NULL;
    double *k_global = NULL, *p_mean_global = NULL;
    if( myid == 0 && accu_count > 0 && (int)TBSWITCH ) {
        uu_global = (double*)malloc(gatherPoints * sizeof(double));
        uw_global = (double*)malloc(gatherPoints * sizeof(double));
        ww_global = (double*)malloc(gatherPoints * sizeof(double));
        k_global  = (double*)malloc(gatherPoints * sizeof(double));
        p_mean_global = (double*)malloc(gatherPoints * sizeof(double));
    }

    double *ox_global = NULL, *oy_global = NULL, *oz_global = NULL;
    if( myid == 0 ) {
        ox_global  = (double*)malloc(gatherPoints * sizeof(double));
        oy_global  = (double*)malloc(gatherPoints * sizeof(double));
        oz_global  = (double*)malloc(gatherPoints * sizeof(double));
    }

    // 所有 rank 一起呼叫 MPI_Gather
    MPI_Gather(u_local, localPoints, MPI_DOUBLE, u_global, localPoints, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Gather(v_local, localPoints, MPI_DOUBLE, v_global, localPoints, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Gather(w_local, localPoints, MPI_DOUBLE, w_global, localPoints, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Gather(z_local, zLocalSize, MPI_DOUBLE, z_global, zLocalSize, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if (accu_count > 0) {
        MPI_Gather(vt_local, localPoints, MPI_DOUBLE, vt_global, localPoints, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Gather(wt_local, localPoints, MPI_DOUBLE, wt_global, localPoints, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }
    if (accu_count > 0 && (int)TBSWITCH) {
        MPI_Gather(uu_local,     localPoints, MPI_DOUBLE, uu_global,     localPoints, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Gather(uw_local,     localPoints, MPI_DOUBLE, uw_global,     localPoints, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Gather(ww_local,     localPoints, MPI_DOUBLE, ww_global,     localPoints, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Gather(k_local,      localPoints, MPI_DOUBLE, k_global,      localPoints, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Gather(p_mean_local, localPoints, MPI_DOUBLE, p_mean_global, localPoints, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }
    MPI_Gather(ox_local,  localPoints, MPI_DOUBLE, ox_global,  localPoints, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Gather(oy_local,  localPoints, MPI_DOUBLE, oy_global,  localPoints, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Gather(oz_local,  localPoints, MPI_DOUBLE, oz_global,  localPoints, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // rank 0 輸出合併的 VTK
    if( myid == 0 ) {
        // 計算全域 y 座標 (uniform grid)
        double dy = LY / (double)(NY6 - 7);
        double *y_global_arr = (double*)malloc(NY6 * sizeof(double));
        for( int j = 0; j < NY6; j++ ) {
            y_global_arr[j] = dy * (double)(j - 3);
        }
        
        ostringstream oss;
        oss << "./result/velocity_merged_" << setfill('0') << setw(6) << step << ".vtk";
        ofstream out(oss.str().c_str());
        
        if( !out.is_open() ) {
            cerr << "ERROR: Cannot open VTK file: " << oss.str() << endl;
            free(u_global); free(v_global); free(w_global); free(z_global);
            free(u_local); free(v_local); free(w_local); free(z_local);
            return;
        }
        
        out << "# vtk DataFile Version 3.0\n";
        out << "LBM Velocity Field (merged) step=" << step << " Force=" << scientific << setprecision(8) << Force_h[0] << " accu_count=" << accu_count << "\n";
        out << "ASCII\n";
        out << "DATASET STRUCTURED_GRID\n";
        out << "DIMENSIONS " << nxLocal << " " << nyGlobal << " " << nzLocal << "\n";
        
        // 輸出座標點
        out << "POINTS " << globalPoints << " double\n";
        out << fixed << setprecision(6);
        const int stride = nyLocal - 1;  // = 32 (unique y-points per rank, overlap=1)
        for( int k = 0; k < nzLocal; k++ ){
        for( int jg = 0; jg < nyGlobal; jg++ ){
        for( int i = 0; i < nxLocal; i++ ){
            int gpu_id = jg / stride;
            if( gpu_id >= jp ) gpu_id = jp - 1;
            int j_local = jg - gpu_id * stride;

            // z 座標在 gather 後的位置
            int z_gpu_offset = gpu_id * zLocalSize;
            int z_local_idx = j_local * nzLocal + k;
            double z_val = z_global[z_gpu_offset + z_local_idx];

            out << x_h[i+3] << " " << y_global_arr[jg+3] << " " << z_val << "\n";
        }}}
        
        // 輸出速度向量
        out << "\nPOINT_DATA " << globalPoints << "\n";
        out << "VECTORS velocity double\n";
        out << setprecision(15);
        
        for( int k = 0; k < nzLocal; k++ ){
        for( int jg = 0; jg < nyGlobal; jg++ ){
        for( int i = 0; i < nxLocal; i++ ){
            int gpu_id = jg / stride;
            if( gpu_id >= jp ) gpu_id = jp - 1;
            int j_local = jg - gpu_id * stride;

            int gpu_offset = gpu_id * localPoints;
            int local_idx = k * nyLocal * nxLocal + j_local * nxLocal + i;
            int global_idx = gpu_offset + local_idx;

            out << u_global[global_idx] << " " << v_global[global_idx] << " " << w_global[global_idx] << "\n";
        }}}

        // ================================================================
        // VTK Level 0: 13 SCALARS + 1 VECTORS
        // Coordinate mapping: code_u=spanwise(V), code_v=streamwise(U), code_w=wall-normal(W)
        // See coordinate mapping comment at top of this function
        // ================================================================

        // Helper macro: write one SCALAR field over the global grid
        #define VTK_WRITE_SCALAR(name, arr_global) \
            out << "\nSCALARS " << (name) << " double 1\n"; \
            out << "LOOKUP_TABLE default\n"; \
            for( int kk = 0; kk < nzLocal; kk++ ){ \
            for( int jg = 0; jg < nyGlobal; jg++ ){ \
            for( int ii = 0; ii < nxLocal; ii++ ){ \
                int gid = jg / stride; \
                if( gid >= jp ) gid = jp - 1; \
                int jl = jg - gid * stride; \
                int gidx = gid * localPoints + kk * nyLocal * nxLocal + jl * nxLocal + ii; \
                out << (arr_global)[gidx] << "\n"; \
            }}}

        // [A] 瞬時速度 (3 SCALARS): u_inst=v_code/Uref, v_inst=u_code/Uref, w_inst=w_code/Uref
        {
            double inv_Uref = 1.0 / (double)Uref;
            out << "\nSCALARS u_inst double 1\n";
            out << "LOOKUP_TABLE default\n";
            for( int kk = 0; kk < nzLocal; kk++ ){
            for( int jg = 0; jg < nyGlobal; jg++ ){
            for( int ii = 0; ii < nxLocal; ii++ ){
                int gid = jg / stride;
                if( gid >= jp ) gid = jp - 1;
                int jl = jg - gid * stride;
                int gidx = gid * localPoints + kk * nyLocal * nxLocal + jl * nxLocal + ii;
                out << v_global[gidx] * inv_Uref << "\n";  // streamwise = code v
            }}}
            out << "\nSCALARS v_inst double 1\n";
            out << "LOOKUP_TABLE default\n";
            for( int kk = 0; kk < nzLocal; kk++ ){
            for( int jg = 0; jg < nyGlobal; jg++ ){
            for( int ii = 0; ii < nxLocal; ii++ ){
                int gid = jg / stride;
                if( gid >= jp ) gid = jp - 1;
                int jl = jg - gid * stride;
                int gidx = gid * localPoints + kk * nyLocal * nxLocal + jl * nxLocal + ii;
                out << u_global[gidx] * inv_Uref << "\n";  // spanwise = code u
            }}}
            out << "\nSCALARS w_inst double 1\n";
            out << "LOOKUP_TABLE default\n";
            for( int kk = 0; kk < nzLocal; kk++ ){
            for( int jg = 0; jg < nyGlobal; jg++ ){
            for( int ii = 0; ii < nxLocal; ii++ ){
                int gid = jg / stride;
                if( gid >= jp ) gid = jp - 1;
                int jl = jg - gid * stride;
                int gidx = gid * localPoints + kk * nyLocal * nxLocal + jl * nxLocal + ii;
                out << w_global[gidx] * inv_Uref << "\n";  // wall-normal = code w
            }}}
        }

        // [B] 瞬時渦度 (3 SCALARS): benchmark frame (omega_u=流向, omega_v=展向, omega_w=法向)
        VTK_WRITE_SCALAR("omega_u", oy_global);  // 流向渦度 = code ω_y
        VTK_WRITE_SCALAR("omega_v", ox_global);  // 展向渦度 = code ω_x
        VTK_WRITE_SCALAR("omega_w", oz_global);  // 法向渦度 = code ω_z

        // [C] 平均速度 (2 SCALARS): U_mean, W_mean (÷count÷Uref)
        if (accu_count > 0) {
            VTK_WRITE_SCALAR("U_mean", vt_global);  // ERCOFTAC <U>/Ub = streamwise = code v
            VTK_WRITE_SCALAR("W_mean", wt_global);  // ERCOFTAC <V>/Ub = wall-normal = code w
        }

        // [D] Reynolds Stress (3 SCALARS) + k_TKE (1) + P_mean (1) — ERCOFTAC col 4-7
        if (accu_count > 0 && (int)TBSWITCH) {
            VTK_WRITE_SCALAR("uu_RS", uu_global);   // ERCOFTAC col 4: <u'u'>/Ub²
            VTK_WRITE_SCALAR("uw_RS", uw_global);   // ERCOFTAC col 6: <u'v'>/Ub²
            VTK_WRITE_SCALAR("ww_RS", ww_global);   // ERCOFTAC col 5: <v'v'>/Ub²
            VTK_WRITE_SCALAR("k_TKE", k_global);    // ERCOFTAC col 7: k/Ub²
            VTK_WRITE_SCALAR("P_mean", p_mean_global);
        }

        // [E] VTK Level 1 placeholder (TODO: omega_mean, epsilon, Tturb, Pdiff, PP_RS)
        #if VTK_OUTPUT_LEVEL >= 1
        // Not yet implemented
        #endif

        #undef VTK_WRITE_SCALAR

        out.close();
        cout << "Merged VTK output: velocity_merged_" << setfill('0') << setw(6) << step << ".vtk";
        if (accu_count > 0) cout << " (accu=" << accu_count << ")";
        cout << "\n";

        free(u_global);
        free(v_global);
        free(w_global);
        free(z_global);
        free(y_global_arr);
        if (vt_global) free(vt_global);
        if (wt_global) free(wt_global);
        if (ox_global)  free(ox_global);
        if (oy_global)  free(oy_global);
        if (oz_global)  free(oz_global);
        if (uu_global)     free(uu_global);
        if (uw_global)     free(uw_global);
        if (ww_global)     free(ww_global);
        if (k_global)      free(k_global);
        if (p_mean_global) free(p_mean_global);
    }

    free(u_local);
    free(v_local);
    free(w_local);
    free(z_local);
    if (vt_local) free(vt_local);
    if (wt_local) free(wt_local);
    free(ox_local);
    free(oy_local);
    free(oz_local);
    if (uu_local)     free(uu_local);
    if (uw_local)     free(uw_local);
    if (ww_local)     free(ww_local);
    if (k_local)      free(k_local);
    if (p_mean_local) free(p_mean_local);
    
    CHECK_MPI( MPI_Barrier(MPI_COMM_WORLD) );
}

#endif