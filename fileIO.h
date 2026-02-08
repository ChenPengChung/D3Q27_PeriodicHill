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


bool FileExist(const char *fileName) {
    //function: 確認檔案是否存在
    //input: 	file relative path. ex: "./backup/u0.bkp"
    //return: 	如果存在回傳1，如果不存在回傳0
    fstream file;
    file.open(fileName);
    return file.good(); //檔案開啟成功
}

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




void OutputData( double* arr_h, const char *fname, const int myid ){
    // 組合檔案路徑
    ostringstream oss;
    oss << "./result/" << fname << "_" << myid << ".bin";
    string path = oss.str();

    // 用 C++ ofstream 開啟二進制檔案
    ofstream file(path, ios::binary);
    if (!file) {
        cout << "Output data error, exit..." << endl;
        CHECK_MPI( MPI_Abort(MPI_COMM_WORLD, 1) );
    }

    // 寫入資料
    file.write(reinterpret_cast<char*>(arr_h), sizeof(double) * NX6 * NZ6 * NYD6);
    file.close();
}




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
        ofstream out(oss.str());
        
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



void fileIO_velocity() {
   ///////////////////////////////////////////////////////////////////////////////
    //輸出paraview (最終結果)//分ㄎGPU子域輸出
    ostringstream oss;
    oss << "./result/velocity_" << myid << "_Final.vtk";
    ofstream out(oss.str());
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
    // 輸出力
    if( myid == 0 ) {
        ofstream fp_gg("./result/0_force.dat");
        fp_gg << fixed << setprecision(15) << Force_h[0];
        fp_gg.close();
    }
    OutputData(rho_h_p, "rho", myid);
    OutputData(u_h_p,   "u",   myid);
    OutputData(v_h_p,   "v",   myid);
    OutputData(w_h_p,   "w",   myid);
}


















void fileIO_PDF()
{
    OutputData(fh_p[0],  "f0",  myid);
    OutputData(fh_p[1],  "f1",  myid);
    OutputData(fh_p[2],  "f2",  myid);
    OutputData(fh_p[3],  "f3",  myid);
    OutputData(fh_p[4],  "f4",  myid);
    OutputData(fh_p[5],  "f5",  myid);
    OutputData(fh_p[6],  "f6",  myid);
    OutputData(fh_p[7],  "f7",  myid);
    OutputData(fh_p[8],  "f8",  myid);
    OutputData(fh_p[9],  "f9",  myid);
    OutputData(fh_p[10], "f10", myid);
    OutputData(fh_p[11], "f11", myid);
    OutputData(fh_p[12], "f12", myid);
    OutputData(fh_p[13], "f13", myid);
    OutputData(fh_p[14], "f14", myid);
    OutputData(fh_p[15], "f15", myid);
    OutputData(fh_p[16], "f16", myid);
    OutputData(fh_p[17], "f17", myid);
    OutputData(fh_p[18], "f18", myid);
}


void ReadData(
    double *arr_h,
    const char *folder,     const char *fname,      const int myid  )
{
    ostringstream oss;
    oss << "./" << folder << "/" << fname << "_" << myid << ".bin";
    string path = oss.str();

    ifstream file(path, ios::binary);
    if (!file) {
        cout << "Read data error: " << path << ", exit...\n";
        CHECK_MPI( MPI_Abort(MPI_COMM_WORLD, 1) );
    }

    file.read(reinterpret_cast<char*>(arr_h), sizeof(double) * NX6 * NZ6 * NYD6);
    file.close();
}


void InitialUsingBkpData()
{
    PreCheckDir();

    char result[30];

    sprintf( result, "result" );

    FILE *fp_gg;
    fp_gg = fopen("./result/0_force.dat","r");
    fscanf( fp_gg, "%lf", &Force_h[0] );
    fclose( fp_gg );

    CHECK_CUDA( cudaMemcpy(Force_d, Force_h, sizeof(double), cudaMemcpyHostToDevice) );

    ReadData(rho_h_p, result, "rho", myid);
    ReadData(u_h_p,   result, "u",   myid);
    ReadData(v_h_p,   result, "v",   myid);
    ReadData(w_h_p,   result, "w",   myid);

    ReadData(fh_p[0],  result, "f0",  myid);
    ReadData(fh_p[1],  result, "f1",  myid);
    ReadData(fh_p[2],  result, "f2",  myid);
    ReadData(fh_p[3],  result, "f3",  myid);
    ReadData(fh_p[4],  result, "f4",  myid);
    ReadData(fh_p[5],  result, "f5",  myid);
    ReadData(fh_p[6],  result, "f6",  myid);
    ReadData(fh_p[7],  result, "f7",  myid);
    ReadData(fh_p[8],  result, "f8",  myid);
    ReadData(fh_p[9],  result, "f9",  myid);
    ReadData(fh_p[10], result, "f10", myid);
    ReadData(fh_p[11], result, "f11", myid);
    ReadData(fh_p[12], result, "f12", myid);
    ReadData(fh_p[13], result, "f13", myid);
    ReadData(fh_p[14], result, "f14", myid);
    ReadData(fh_p[15], result, "f15", myid);
    ReadData(fh_p[16], result, "f16", myid);
    ReadData(fh_p[17], result, "f17", myid);
    ReadData(fh_p[18], result, "f18", myid);
}
void OutputTBData(
    double *arr_d,
    const char *fname,      const int myid  )
{
    CHECK_MPI( MPI_Barrier(MPI_COMM_WORLD) );

    const size_t nBytes = NX6 * NYD6 * NZ6 * sizeof(double);
    double *arr_h = (double*)malloc(nBytes);

    CHECK_CUDA( cudaMemcpy(arr_h, arr_d, nBytes, cudaMemcpyDeviceToHost) );

    char path[100];
    sprintf( path, "./statistics/%s/%s_%d.bin", fname, fname, myid );

    FILE *fp = NULL;
    fp = fopen( path, "wb" );

    for( int k = 3; k < NZ6-3;  k++ ){
    for( int j = 3; j < NYD6-3; j++ ){
    for( int i = 3; i < NX6-3;  i++ ){
        int index = j*NX6*NZ6 + k*NX6 + i;
        fwrite( &arr_h[index], sizeof(double), 1, fp );
    }}}

    fclose( fp );
    free( arr_h );
}
void Launch_OutputTB()
{
    if( myid == 0 ) {
        FILE *fp_accu;
        fp_accu = fopen("./statistics/accu.dat","w");
        fprintf( fp_accu, "%d", accu_num );
        fclose( fp_accu );
    }
    CHECK_MPI( MPI_Barrier(MPI_COMM_WORLD) );

    OutputTBData(U, "U", myid);
	OutputTBData(V, "V", myid);
	OutputTBData(W, "W", myid);
	OutputTBData(P, "P", myid);
	OutputTBData(UU, "UU", myid);
	OutputTBData(UV, "UV", myid);
	OutputTBData(UW, "UW", myid);
	OutputTBData(VV, "VV", myid);
	OutputTBData(VW, "VW", myid);
	OutputTBData(WW, "WW", myid);
	OutputTBData(PU, "PU", myid);
	OutputTBData(PV, "PV", myid);
	OutputTBData(PW, "PW", myid);
	OutputTBData(KT, "KT", myid);
	OutputTBData(DUDX2, "DUDX2", myid);
	OutputTBData(DUDY2, "DUDY2", myid);
	OutputTBData(DUDZ2, "DUDZ2", myid);
	OutputTBData(DVDX2, "DVDX2", myid);
	OutputTBData(DVDY2, "DVDY2", myid);
	OutputTBData(DVDZ2, "DVDZ2", myid);
	OutputTBData(DWDX2, "DWDX2", myid);
	OutputTBData(DWDY2, "DWDY2", myid);
	OutputTBData(DWDZ2, "DWDZ2", myid);
	OutputTBData(UUU, "UUU", myid);
	OutputTBData(UUV, "UUV", myid);
	OutputTBData(UUW, "UUW", myid);
	OutputTBData(VVU, "VVU", myid);
	OutputTBData(VVV, "VVV", myid);
	OutputTBData(VVW, "VVW", myid);
	OutputTBData(WWU, "WWU", myid);
	OutputTBData(WWV, "WWV", myid);
	OutputTBData(WWW, "WWW", myid);
	//OutputTBData(OMEGA_X, "OMEGA_X", myid);
	//OutputTBData(OMEGA_Y, "OMEGA_Y", myid);
	//OutputTBData(OMEGA_Z, "OMEGA_Z", myid);
}
void ReadTBData(
    double * arr_d,
    const char *fname,      const int myid  )
{
    const size_t nBytes = NX6 * NYD6 * NZ6 * sizeof(double);
    double *arr_h = (double*)malloc(nBytes);

    char result[100];
    sprintf( result, "./statistics/%s/%s_%d.bin", fname, fname, myid );
    FILE *fp = NULL;
    fp = fopen(result, "rb");

    for( int k = 3; k < NZ6-3;  k++ ){
    for( int j = 3; j < NYD6-3; j++ ){
    for( int i = 3; i < NX6-3;  i++ ){
        const int index = j*NX6*NZ6 + k*NX6 + i;
        fread( &arr_h[index], sizeof(double), 1, fp );
    }}}

    fclose( fp );
    CHECK_CUDA( cudaMemcpy(arr_d, arr_h, nBytes, cudaMemcpyHostToDevice) );
    free( arr_h );
    CHECK_MPI( MPI_Barrier(MPI_COMM_WORLD) );
}
void InitialTBUsingBkpData() {

    FILE *fp_accu;
    fp_accu = fopen("./statistics/accu.dat","r");
    fscanf( fp_accu, "%d", &accu_num );
    fclose( fp_accu );

    ReadTBData(U, "U", myid);
	ReadTBData(V, "V", myid);
	ReadTBData(W, "W", myid);
	ReadTBData(P, "P", myid);
	ReadTBData(UU, "UU", myid);
	ReadTBData(UV, "UV", myid);
	ReadTBData(UW, "UW", myid);
	ReadTBData(VV, "VV", myid);
	ReadTBData(VW, "VW", myid);
	ReadTBData(WW, "WW", myid);
	ReadTBData(PU, "PU", myid);
	ReadTBData(PV, "PV", myid);
	ReadTBData(PW, "PW", myid);
	ReadTBData(KT, "KT", myid);
	ReadTBData(DUDX2, "DUDX2", myid);
	ReadTBData(DUDY2, "DUDY2", myid);
	ReadTBData(DUDZ2, "DUDZ2", myid);
	ReadTBData(DVDX2, "DVDX2", myid);
	ReadTBData(DVDY2, "DVDY2", myid);
	ReadTBData(DVDZ2, "DVDZ2", myid);
	ReadTBData(DWDX2, "DWDX2", myid);
	ReadTBData(DWDY2, "DWDY2", myid);
	ReadTBData(DWDZ2, "DWDZ2", myid);
	ReadTBData(UUU, "UUU", myid);
	ReadTBData(UUV, "UUV", myid);
	ReadTBData(UUW, "UUW", myid);
	ReadTBData(VVU, "VVU", myid);
	ReadTBData(VVV, "VVV", myid);
	ReadTBData(VVW, "VVW", myid);
	ReadTBData(WWU, "WWU", myid);
	ReadTBData(WWV, "WWV", myid);
	ReadTBData(WWW, "WWW", myid);
	//ReadTBData(OMEGA_X, "OMEGA_X", myid);
	//ReadTBData(OMEGA_Y, "OMEGA_Y", myid);
	//ReadTBData(OMEGA_Z, "OMEGA_Z", myid);

	CHECK_MPI( MPI_Barrier(MPI_COMM_WORLD) );
}
void Output3Dvelocity(){

    char filename_E2[300];

    sprintf(filename_E2, "test_%d.plt",myid);
    printf("%s\n", filename_E2);

    FILE *fpE3;

    fpE3 = fopen(filename_E2, "wb");

    int IMax = NX6-6;

    int JMax = NYD6-6;

    int KMax = NZ6-6;

    char Title[] = "Particle intensity";

    char Varname1[] = "X";

    char Varname2[] = "Y";

    char Varname3[] = "Z";

    //char Varname4[] = "Intensity";
    
    char Varname5[] = "U";
	
	char Varname6[] = "V"; 
	
	char Varname7[] = "W";

    char Zonename1[] = "Zone 001";

    float ZONEMARKER = 299.0;

    float EOHMARKER = 357.0;

    //==============Header Secontion =================//
    //------1.1 Magic number, Version number
    char MagicNumber[] = "#!TDV101";
    //cout << "watchout" << sizeof(MagicNumber) << endl;
    fwrite(MagicNumber, 8, 1, fpE3);

    //---- - 1.2.Integer value of 1.----------------------------------------------------------
    int IntegerValue = 1;
    fwrite(&IntegerValue, sizeof(IntegerValue), 1, fpE3);

    //---- - 1.3.Title and variable names.------------------------------------------------ -
    //---- - 1.3.1.The TITLE.
    wirte_ASCII_of_str(Title, fpE3);

    //---- - 1.3.2 Number of variables(NumVar) in the c_strfile.
    int NumVar = 6;
    fwrite(&NumVar, sizeof(NumVar), 1, fpE3);

    //------1.3.3 Variable names.N = L[1] + L[2] + ....L[NumVar]
    wirte_ASCII_of_str(Varname1, fpE3);
    wirte_ASCII_of_str(Varname2, fpE3);
    wirte_ASCII_of_str(Varname3, fpE3);
    //wirte_ASCII_of_str(Varname4, fpE3);
    wirte_ASCII_of_str(Varname5, fpE3);
    wirte_ASCII_of_str(Varname6, fpE3);
	wirte_ASCII_of_str(Varname7, fpE3);
    //---- - 1.4.Zones------------------------------------------------------------------ -
    //--------Zone marker.Value = 299.0
    fwrite(&ZONEMARKER, 1, sizeof(ZONEMARKER), fpE3);

    //--------Zone name.
    wirte_ASCII_of_str(Zonename1, fpE3);

    //--------Zone color
    int ZoneColor = -1;
    fwrite(&ZoneColor, sizeof(ZoneColor), 1, fpE3);

    //--------ZoneType
    int ZoneType = 0;
    fwrite(&ZoneType, sizeof(ZoneType), 1, fpE3);

    //--------DaraPacking 0=Block, 1=Point
    int DaraPacking = 1;
    fwrite(&DaraPacking, sizeof(DaraPacking), 1, fpE3);

    //--------Specify Var Location. 0 = Don't specify, all c_str is located at the nodes. 1 = Specify
    int SpecifyVarLocation = 0;
    fwrite(&SpecifyVarLocation, sizeof(SpecifyVarLocation), 1, fpE3);

    //--------Number of user defined face neighbor connections(value >= 0)
    int NumOfNeighbor = 0;
    fwrite(&NumOfNeighbor, sizeof(NumOfNeighbor), 1, fpE3);

    //-------- - IMax, JMax, KMax
    fwrite(&IMax, sizeof(IMax), 1, fpE3);
    fwrite(&JMax, sizeof(JMax), 1, fpE3);
    fwrite(&KMax, sizeof(KMax), 1, fpE3);

    //----------// -1 = Auxiliary name / value pair to follow 0 = No more Auxiliar name / value pairs.
    int AuxiliaryName = 0;
    fwrite(&AuxiliaryName, sizeof(AuxiliaryName), 1, fpE3);

    //----I HEADER OVER--------------------------------------------------------------------------------------------

    //=============================Geometries section=======================
    //=============================Text section======================
    // EOHMARKER, value = 357.0
    fwrite(&EOHMARKER, sizeof(EOHMARKER), 1, fpE3);

    //================II.Data section===============//
    //------ 2.1 zone---------------------------------------------------------------------- -
    fwrite(&ZONEMARKER, sizeof(ZONEMARKER), 1, fpE3);

    //--------variable c_str format, 1 = Float, 2 = Double, 3 = LongInt, 4 = ShortInt, 5 = Byte, 6 = Bit
    int fomat1 = 2;
    int fomat2 = 2;
    int fomat3 = 2;
    int fomat5 = 2;
    int fomat6 = 2;
	int fomat7 = 2;
    fwrite(&fomat1, sizeof(fomat1), 1, fpE3);
    fwrite(&fomat2, sizeof(fomat2), 1, fpE3);
    fwrite(&fomat3, sizeof(fomat3), 1, fpE3);
    fwrite(&fomat5, sizeof(fomat5), 1, fpE3);
    fwrite(&fomat6, sizeof(fomat6), 1, fpE3);
	fwrite(&fomat7, sizeof(fomat7), 1, fpE3);
    //--------Has variable sharing 0 = no, 1 = yes.
    int HasVarSharing = 0;
    fwrite(&HasVarSharing, sizeof(HasVarSharing), 1, fpE3);

    //----------Zone number to share connectivity list with(-1 = no sharing).
    int ZoneNumToShareConnectivity = -1;
    fwrite(&ZoneNumToShareConnectivity, sizeof(ZoneNumToShareConnectivity), 1, fpE3);

    //----------Zone c_str.Each variable is in c_str format asspecified above.
    for (int k = 3; k < NZ6-3; k++){
    for (int j = 3; j < NYD6-3; j++){
    for (int i = 3; i < NX6-3; i++){

    double VarToWrite1 = x_h[i];
    double VarToWrite2 = y_h[j];
    double VarToWrite3 = z_h[j*NZ6+k];

    int index = j*NZ6*NX6 + k*NX6 + i;
	double Uvelocity = u_h_p[index]/Uref;
	double Vvelocity = v_h_p[index]/Uref;
	double Wvelocity = w_h_p[index]/Uref; 
    fwrite(&VarToWrite1, sizeof(VarToWrite1), 1, fpE3);
    fwrite(&VarToWrite2, sizeof(VarToWrite2), 1, fpE3);
    fwrite(&VarToWrite3, sizeof(VarToWrite3), 1, fpE3);
	fwrite(&Uvelocity, sizeof(Uvelocity), 1, fpE3);
	fwrite(&Vvelocity, sizeof(Vvelocity), 1, fpE3);
	fwrite(&Wvelocity, sizeof(Wvelocity), 1, fpE3);
    }}}

    fclose(fpE3);

}
void wirte_ASCII_of_str(char * str, FILE * file)
{
    int value = 0;

    while ((*str) != '\0'){
        value = (int)*str;
        fwrite(&value, sizeof(int), 1, file);
        str++;
    }

    char null_char[] = "";

    value = (int)*null_char;

    fwrite(&value, sizeof(int), 1, file);
}






#endif