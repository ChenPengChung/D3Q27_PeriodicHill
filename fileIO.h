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
//PreCheckDir輔助檔案1 
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
    ostringstream oss;
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
//result系列主函數1.
void result_writebin_velocityandf() {
    ///////////////////////////////////////////////////////////////////////////////
    // 輸出 Paraview VTK (最終結果，分 GPU 子域輸出)
    ostringstream oss;
    oss << "./result/velocity_" << myid << "_Final.vtk";
    ofstream out(oss.str().c_str());
    // VTK Header
    out << "# vtk DataFile Version 3.0\n";
    out << "LBM Velocity Field\n";
    out << "ASCII\n";
    out << "DATASET STRUCTURED_GRID\n";
    out << "DIMENSIONS " << NX6-6 << " " << NYD6-6 << " " << NZ6-6 << "\n";

    // 座標點
    int nPoints = (NX6-6) * (NYD6-6) * (NZ6-6);
    out << "POINTS " << nPoints << " double\n";
    out << fixed << setprecision(6);
    for( int k = 3; k < NZ6-3; k++ ){
    for( int j = 3; j < NYD6-3; j++ ){
    for( int i = 3; i < NX6-3; i++ ){
        out << x_h[i] << " " << y_h[j] << " " << z_h[j*NZ6+k] << "\n";
    }}}

    // 速度向量
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
//result系列主函數2.
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
    for( int k = 3; k < NZ6-3;  k++ ){
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
    for( int k = 3; k < NZ6-3;  k++ ){
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
/*第四段:逐步輸出可視化VTK檔案*/
// 合併所有 GPU 結果，輸出單一 VTK 檔案 (Paraview)
void fileIO_velocity_vtk_merged(int step) {
    // 每個 GPU 內部有效區域的 y 層數 (不含 ghost)
    const int nyLocal = NYD6 - 6;  // 去除上下各3層ghost
    const int nxLocal = NX6 - 6;
    const int nzLocal = NZ6 - 6;
    
    // 每個 GPU 發送的點數
    const int localPoints = nxLocal * nyLocal * nzLocal;
    const int zLocalSize = nyLocal * nzLocal;
    
    // 全域 y 層數
    const int nyGlobal = NY6 - 6;
    const int globalPoints = nxLocal * nyGlobal * nzLocal;
    
    // 準備本地速度資料 (去除 ghost cells, 只取內部)
    double *u_local = (double*)malloc(localPoints * sizeof(double));
    double *v_local = (double*)malloc(localPoints * sizeof(double));
    double *w_local = (double*)malloc(localPoints * sizeof(double));
    double *z_local = (double*)malloc(zLocalSize * sizeof(double));
    
    int idx = 0;
    for( int k = 3; k < NZ6-3; k++ ){
    for( int j = 3; j < NYD6-3; j++ ){
    for( int i = 3; i < NX6-3; i++ ){
        int index = j*NZ6*NX6 + k*NX6 + i;
        u_local[idx] = u_h_p[index];
        v_local[idx] = v_h_p[index];
        w_local[idx] = w_h_p[index];
        idx++;
    }}}
    
    // 準備本地 z 座標
    int zidx = 0;
    for( int j = 3; j < NYD6-3; j++ ){
    for( int k = 3; k < NZ6-3; k++ ){
        z_local[zidx++] = z_h[j*NZ6 + k];
    }}
    
    // rank 0 分配接收緩衝區
    double *u_global = NULL;
    double *v_global = NULL;
    double *w_global = NULL;
    double *z_global = NULL;
    
    if( myid == 0 ) {
        u_global = (double*)malloc(globalPoints * sizeof(double));
        v_global = (double*)malloc(globalPoints * sizeof(double));
        w_global = (double*)malloc(globalPoints * sizeof(double));
        z_global = (double*)malloc(nyGlobal * nzLocal * sizeof(double));
    }
    
    // 所有 rank 一起呼叫 MPI_Gather
    MPI_Gather(u_local, localPoints, MPI_DOUBLE, u_global, localPoints, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Gather(v_local, localPoints, MPI_DOUBLE, v_global, localPoints, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Gather(w_local, localPoints, MPI_DOUBLE, w_global, localPoints, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Gather(z_local, zLocalSize, MPI_DOUBLE, z_global, zLocalSize, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    
    // rank 0 輸出合併的 VTK
    if( myid == 0 ) {
        // 計算全域 y 座標 (uniform grid)
        double dy = LY / (double)(NY6 - 7);
        double *y_global_arr = (double*)malloc(NY6 * sizeof(double));
        for( int j = 0; j < NY6; j++ ) {
            y_global_arr[j] = dy * (double)(j - 3);
        }
        
        ostringstream oss;
        oss << "./result/velocity_merged_" << step << ".vtk";
        ofstream out(oss.str().c_str());
        
        if( !out.is_open() ) {
            cerr << "ERROR: Cannot open VTK file: " << oss.str() << endl;
            free(u_global); free(v_global); free(w_global); free(z_global);
            free(u_local); free(v_local); free(w_local); free(z_local);
            return;
        }
        
        out << "# vtk DataFile Version 3.0\n";
        out << "LBM Velocity Field (merged) step=" << step << "\n";
        out << "ASCII\n";
        out << "DATASET STRUCTURED_GRID\n";
        out << "DIMENSIONS " << nxLocal << " " << nyGlobal << " " << nzLocal << "\n";
        
        // 輸出座標點
        out << "POINTS " << globalPoints << " double\n";
        out << fixed << setprecision(6);
        for( int k = 0; k < nzLocal; k++ ){
        for( int jg = 0; jg < nyGlobal; jg++ ){
        for( int i = 0; i < nxLocal; i++ ){
            int gpu_id = jg / nyLocal;
            if( gpu_id >= jp ) gpu_id = jp - 1;
            int j_local = jg % nyLocal;
            
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
            int gpu_id = jg / nyLocal;
            if( gpu_id >= jp ) gpu_id = jp - 1;
            int j_local = jg % nyLocal;
            
            int gpu_offset = gpu_id * localPoints;
            int local_idx = k * nyLocal * nxLocal + j_local * nxLocal + i;
            int global_idx = gpu_offset + local_idx;
            
            out << u_global[global_idx] << " " << v_global[global_idx] << " " << w_global[global_idx] << "\n";
        }}}
        
        out.close();
        cout << "Merged VTK output: velocity_merged_" << step << ".vtk\n";
        
        free(u_global);
        free(v_global);
        free(w_global);
        free(z_global);
        free(y_global_arr);
    }
    
    free(u_local);
    free(v_local);
    free(w_local);
    free(z_local);
    
    CHECK_MPI( MPI_Barrier(MPI_COMM_WORLD) );
}


#endif