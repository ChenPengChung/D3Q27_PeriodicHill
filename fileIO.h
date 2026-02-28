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
//創立資料夾 "result" , "statistics" , "statistics/XXX" (XXX 為 35 個統計量子資料夾)
void PreCheckDir() {
    //預先建立資料夾
	ExistOrCreateDir("result");//程式碼初始狀態使用
    //湍流統計//35 個統計量子資料夾
	ExistOrCreateDir("statistics");
    if ( TBSWITCH ) {
		const int num_files = 35;
		std::string name[num_files] = {
        "U","V","W","P",//4
        "UU","UV","UW","VV","VW","WW",//6
        "PU","PV","PW",//3
        "KT",//1
        "DUDX2","DUDY2","DUDZ2","DVDX2","DVDY2","DVDZ2","DWDX2","DWDY2","DWDZ2", //9
        "UUU","UUV","UUW","VVU","VVV","VVW","WWU","WWV","WWW",//9
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
        fp_accu << accu_num;
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
	statistics_writebin(KT, "KT", myid);//10.
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
    int accu_num = 0;
    ifstream fp_accu("./statistics/accu.dat");
    fp_accu >> accu_num;
    fp_accu.close();

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
	statistics_readbin(KT, "KT", myid);
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

    // 解析 header，讀取 step 和 Force 值
    double force_from_vtk = -1.0;
    int step_from_vtk = -1;
    string vtk_line;
    while (getline(vtk_in, vtk_line)) {
        // 嘗試從 header 讀取 step (格式: "... step=50001 ...")
        size_t spos = vtk_line.find("step=");
        if (spos != string::npos) {
            sscanf(vtk_line.c_str() + spos + 5, "%d", &step_from_vtk);
        }
        // 嘗試從 header 讀取 Force (格式: "... Force=1.23456E-04")
        size_t fpos = vtk_line.find("Force=");
        if (fpos != string::npos) {
            sscanf(vtk_line.c_str() + fpos + 6, "%lf", &force_from_vtk);
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
        // 發送 j=3..5 到左鄰居，接收從右鄰居到 j=NYD6-3..NYD6-1
        MPI_Sendrecv(&u_h_p[3 * slice_size],       ghost_count, MPI_DOUBLE, l_nbr, 600,
                     &u_h_p[(NYD6-3) * slice_size], ghost_count, MPI_DOUBLE, r_nbr, 600,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        // 發送 j=NYD6-6..NYD6-4 到右鄰居，接收從左鄰居到 j=0..2
        MPI_Sendrecv(&u_h_p[(NYD6-6) * slice_size], ghost_count, MPI_DOUBLE, r_nbr, 601,
                     &u_h_p[0],                      ghost_count, MPI_DOUBLE, l_nbr, 601,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        
        // 交換 v_h_p
        MPI_Sendrecv(&v_h_p[3 * slice_size],       ghost_count, MPI_DOUBLE, l_nbr, 602,
                     &v_h_p[(NYD6-3) * slice_size], ghost_count, MPI_DOUBLE, r_nbr, 602,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Sendrecv(&v_h_p[(NYD6-6) * slice_size], ghost_count, MPI_DOUBLE, r_nbr, 603,
                     &v_h_p[0],                      ghost_count, MPI_DOUBLE, l_nbr, 603,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        
        // 交換 w_h_p
        MPI_Sendrecv(&w_h_p[3 * slice_size],       ghost_count, MPI_DOUBLE, l_nbr, 604,
                     &w_h_p[(NYD6-3) * slice_size], ghost_count, MPI_DOUBLE, r_nbr, 604,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Sendrecv(&w_h_p[(NYD6-6) * slice_size], ghost_count, MPI_DOUBLE, r_nbr, 605,
                     &w_h_p[0],                      ghost_count, MPI_DOUBLE, l_nbr, 605,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        
        // 交換 rho_h_p
        MPI_Sendrecv(&rho_h_p[3 * slice_size],       ghost_count, MPI_DOUBLE, l_nbr, 606,
                     &rho_h_p[(NYD6-3) * slice_size], ghost_count, MPI_DOUBLE, r_nbr, 606,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Sendrecv(&rho_h_p[(NYD6-6) * slice_size], ghost_count, MPI_DOUBLE, r_nbr, 607,
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

    // 準備本地時間平均資料 (若有累積)
    double *vt_local = NULL, *wt_local = NULL;
    if (time_avg_count > 0) {
        vt_local = (double*)malloc(localPoints * sizeof(double));
        wt_local = (double*)malloc(localPoints * sizeof(double));
        double inv_count = 1.0 / (double)time_avg_count;
        int tidx = 0;
        for( int k = 3; k < NZ6-3; k++ ){
        for( int j = 3; j < NYD6-3; j++ ){
        for( int i = 3; i < NX6-3; i++ ){
            int index = j*NZ6*NX6 + k*NX6 + i;
            vt_local[tidx] = v_tavg_h[index] * inv_count;
            wt_local[tidx] = w_tavg_h[index] * inv_count;
            tidx++;
        }}}
    }

    // Compute instantaneous omega_x (spanwise vorticity) = dw/dy - dv/dz
    // Curvilinear:
    //   dw/dy = (1/dy)(dw/dj) + dk_dy(dw/dk)
    //   dv/dz = dk_dz(dv/dk)
    double *ox_local = (double*)malloc(localPoints * sizeof(double));
    {
        double dy_val = (double)LY / (double)(NY6 - 7);
        double dy_inv = 1.0 / dy_val;
        const int nface = NX6 * NZ6;
        int oidx = 0;
        for (int k = 3; k < NZ6-3; k++) {
        for (int j = 3; j < NYD6-3; j++) {
        for (int i = 3; i < NX6-3; i++) {
            double dkdz = dk_dz_h[j * NZ6 + k];
            double dkdy = dk_dy_h[j * NZ6 + k];
            double dw_dj = (w_h_p[(j+1)*nface + k*NX6 + i] - w_h_p[(j-1)*nface + k*NX6 + i]) * 0.5;
            double dw_dk = (w_h_p[j*nface + (k+1)*NX6 + i] - w_h_p[j*nface + (k-1)*NX6 + i]) * 0.5;
            double dv_dk = (v_h_p[j*nface + (k+1)*NX6 + i] - v_h_p[j*nface + (k-1)*NX6 + i]) * 0.5;
            ox_local[oidx++] = dy_inv * dw_dj + dkdy * dw_dk - dkdz * dv_dk;
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
    if( myid == 0 && time_avg_count > 0 ) {
        vt_global = (double*)malloc(gatherPoints * sizeof(double));
        wt_global = (double*)malloc(gatherPoints * sizeof(double));
    }

    double *ox_global = NULL;
    if( myid == 0 ) {
        ox_global = (double*)malloc(gatherPoints * sizeof(double));
    }

    // 所有 rank 一起呼叫 MPI_Gather
    MPI_Gather(u_local, localPoints, MPI_DOUBLE, u_global, localPoints, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Gather(v_local, localPoints, MPI_DOUBLE, v_global, localPoints, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Gather(w_local, localPoints, MPI_DOUBLE, w_global, localPoints, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Gather(z_local, zLocalSize, MPI_DOUBLE, z_global, zLocalSize, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if (time_avg_count > 0) {
        MPI_Gather(vt_local, localPoints, MPI_DOUBLE, vt_global, localPoints, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Gather(wt_local, localPoints, MPI_DOUBLE, wt_global, localPoints, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }
    MPI_Gather(ox_local, localPoints, MPI_DOUBLE, ox_global, localPoints, MPI_DOUBLE, 0, MPI_COMM_WORLD);

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
        out << "LBM Velocity Field (merged) step=" << step << " Force=" << scientific << setprecision(8) << Force_h[0] << " tavg_count=" << time_avg_count << "\n";
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

        // 輸出時間平均場 (若有累積)
        if (time_avg_count > 0) {
            // v_time_avg (streamwise, y-direction)
            out << "\nSCALARS v_time_avg double 1\n";
            out << "LOOKUP_TABLE default\n";
            for( int k = 0; k < nzLocal; k++ ){
            for( int jg = 0; jg < nyGlobal; jg++ ){
            for( int i = 0; i < nxLocal; i++ ){
                int gpu_id = jg / stride;
                if( gpu_id >= jp ) gpu_id = jp - 1;
                int j_local = jg - gpu_id * stride;
                int gpu_offset = gpu_id * localPoints;
                int local_idx = k * nyLocal * nxLocal + j_local * nxLocal + i;
                out << vt_global[gpu_offset + local_idx] << "\n";
            }}}

            // w_time_avg (wall-normal, z-direction)
            out << "\nSCALARS w_time_avg double 1\n";
            out << "LOOKUP_TABLE default\n";
            for( int k = 0; k < nzLocal; k++ ){
            for( int jg = 0; jg < nyGlobal; jg++ ){
            for( int i = 0; i < nxLocal; i++ ){
                int gpu_id = jg / stride;
                if( gpu_id >= jp ) gpu_id = jp - 1;
                int j_local = jg - gpu_id * stride;
                int gpu_offset = gpu_id * localPoints;
                int local_idx = k * nyLocal * nxLocal + j_local * nxLocal + i;
                out << wt_global[gpu_offset + local_idx] << "\n";
            }}}
        }

        // omega_x: instantaneous spanwise vorticity (dw/dy - dv/dz)
        out << "\nSCALARS omega_x double 1\n";
        out << "LOOKUP_TABLE default\n";
        for( int k = 0; k < nzLocal; k++ ){
        for( int jg = 0; jg < nyGlobal; jg++ ){
        for( int i = 0; i < nxLocal; i++ ){
            int gpu_id = jg / stride;
            if( gpu_id >= jp ) gpu_id = jp - 1;
            int j_local = jg - gpu_id * stride;
            int gpu_offset = gpu_id * localPoints;
            int local_idx = k * nyLocal * nxLocal + j_local * nxLocal + i;
            out << ox_global[gpu_offset + local_idx] << "\n";
        }}}

        out.close();
        cout << "Merged VTK output: velocity_merged_" << setfill('0') << setw(6) << step << ".vtk";
        if (time_avg_count > 0) cout << " (tavg_count=" << time_avg_count << ")";
        cout << "\n";
        
        free(u_global);
        free(v_global);
        free(w_global);
        free(z_global);
        free(y_global_arr);
        if (vt_global) free(vt_global);
        if (wt_global) free(wt_global);
        if (ox_global) free(ox_global);
    }

    free(u_local);
    free(v_local);
    free(w_local);
    free(z_local);
    if (vt_local) free(vt_local);
    if (wt_local) free(wt_local);
    free(ox_local);
    
    CHECK_MPI( MPI_Barrier(MPI_COMM_WORLD) );
}

#endif