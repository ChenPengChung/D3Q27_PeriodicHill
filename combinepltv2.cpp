// combinepltv2.cpp - 後處理程式 (C++ 版本)
// 編譯: g++ -std=c++17 -O3 combinepltv2.cpp -o combineplt
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <cmath>
#include <iomanip>

#include "variables.h"

using namespace std;

// 全域資料
vector<double> x, y, z;
vector<double> u_global, v_global, w_global, rho_global;

void ReadData(vector<double>& arr, const string& folder, const string& fname, int myid) {
    ostringstream oss;
    oss << "./" << folder << "/" << fname << "_" << myid << ".bin";
    
    ifstream file(oss.str(), ios::binary);
    if (!file) {
        cerr << "Read data error: " << oss.str() << ", exit...\n";
        exit(1);
    }
    
    file.read(reinterpret_cast<char*>(arr.data()), sizeof(double) * NX6 * NZ6 * NYD6);
    file.close();
}

void Mesh_scan() {
    // 讀取 Z 網格
    ifstream meshZ("meshZ.DAT");
    if (!meshZ) { cerr << "Cannot open meshZ.DAT\n"; exit(1); }
    for (int k = 0; k < NZ6; k++) {
        for (int j = 0; j < NY6; j++) {
            meshZ >> z[j * NZ6 + k];
        }
    }
    meshZ.close();

    // 讀取 X 網格
    ifstream meshX("meshX.DAT");
    if (!meshX) { cerr << "Cannot open meshX.DAT\n"; exit(1); }
    for (int i = 0; i < NX6; i++) {
        meshX >> x[i];
    }
    meshX.close();

    // 讀取 Y 網格
    ifstream meshY("meshY.DAT");
    if (!meshY) { cerr << "Cannot open meshY.DAT\n"; exit(1); }
    for (int j = 0; j < NY6; j++) {
        meshY >> y[j];
    }
    meshY.close();
}

void printutau() {
    ofstream utau("utau.dat");
    utau << fixed << setprecision(15);
    
    for (int j = 3; j < NY6 - 3; j++) {
        int idx = j * NX6 * NZ6 + 3 * NX6 + NX6 / 2;
        double tau_wall = niu * ((-0 + 9 * v_global[idx] - v_global[idx + NX6]) / (3 * minSize));
        double Utau = sqrt(tau_wall);
        utau << tau_wall << "\t " << Utau << "\n";
    }
    utau.close();
}

void write_ASCII_of_str(const string& str, ofstream& file) {
    for (char c : str) {
        int value = static_cast<int>(c);
        file.write(reinterpret_cast<char*>(&value), sizeof(int));
    }
    int null_val = 0;
    file.write(reinterpret_cast<char*>(&null_val), sizeof(int));
}

void Output3Dvelocity() {
    ostringstream oss;
    oss << (NX6 - 6) << "x" << (NY6 - 6) << "x" << (NZ6 - 4) << ".plt";
    cout << oss.str() << "\n";

    ofstream fpE3(oss.str(), ios::binary);

    int IMax = NX6 - 6;
    int JMax = NY6 - 6;
    int KMax = NZ6 - 4;

    string Title = "Particle intensity";
    string Varname1 = "X", Varname2 = "Y", Varname3 = "Z";
    string Varname4 = "Rho", Varname5 = "U", Varname6 = "V", Varname7 = "W";
    string Zonename1 = "Zone 001";

    float ZONEMARKER = 299.0f;
    float EOHMARKER = 357.0f;

    // Header Section
    // 1.1 Magic number
    const char MagicNumber[] = "#!TDV101";
    fpE3.write(MagicNumber, 8);

    // 1.2 Integer value of 1
    int IntegerValue = 1;
    fpE3.write(reinterpret_cast<char*>(&IntegerValue), sizeof(int));

    // 1.3 Title and variable names
    write_ASCII_of_str(Title, fpE3);

    int NumVar = 7;
    fpE3.write(reinterpret_cast<char*>(&NumVar), sizeof(int));

    write_ASCII_of_str(Varname1, fpE3);
    write_ASCII_of_str(Varname2, fpE3);
    write_ASCII_of_str(Varname3, fpE3);
    write_ASCII_of_str(Varname4, fpE3);
    write_ASCII_of_str(Varname5, fpE3);
    write_ASCII_of_str(Varname6, fpE3);
    write_ASCII_of_str(Varname7, fpE3);

    // 1.4 Zones
    fpE3.write(reinterpret_cast<char*>(&ZONEMARKER), sizeof(float));
    write_ASCII_of_str(Zonename1, fpE3);

    int ZoneColor = -1;
    fpE3.write(reinterpret_cast<char*>(&ZoneColor), sizeof(int));

    int ZoneType = 0;
    fpE3.write(reinterpret_cast<char*>(&ZoneType), sizeof(int));

    int DataPacking = 1;
    fpE3.write(reinterpret_cast<char*>(&DataPacking), sizeof(int));

    int SpecifyVarLocation = 0;
    fpE3.write(reinterpret_cast<char*>(&SpecifyVarLocation), sizeof(int));

    int NumOfNeighbor = 0;
    fpE3.write(reinterpret_cast<char*>(&NumOfNeighbor), sizeof(int));

    fpE3.write(reinterpret_cast<char*>(&IMax), sizeof(int));
    fpE3.write(reinterpret_cast<char*>(&JMax), sizeof(int));
    fpE3.write(reinterpret_cast<char*>(&KMax), sizeof(int));

    int AuxiliaryName = 0;
    fpE3.write(reinterpret_cast<char*>(&AuxiliaryName), sizeof(int));

    // EOHMARKER
    fpE3.write(reinterpret_cast<char*>(&EOHMARKER), sizeof(float));

    // Data section
    fpE3.write(reinterpret_cast<char*>(&ZONEMARKER), sizeof(float));

    // Variable format (2 = Double)
    int format = 2;
    for (int n = 0; n < 7; n++) {
        fpE3.write(reinterpret_cast<char*>(&format), sizeof(int));
    }

    int HasVarSharing = 0;
    fpE3.write(reinterpret_cast<char*>(&HasVarSharing), sizeof(int));

    int ZoneNumToShareConnectivity = -1;
    fpE3.write(reinterpret_cast<char*>(&ZoneNumToShareConnectivity), sizeof(int));

    // Write data
    for (int k = 2; k < NZ6 - 2; k++) {
    for (int j = 3; j < NY6 - 3; j++) {
    for (int i = 3; i < NX6 - 3; i++) {
        double X = x[i];
        double Y = y[j];
        double Z = z[j * NZ6 + k];

        int idx_global = j * NX6 * NZ6 + k * NX6 + i;
        double Rho = rho_global[idx_global];
        double U = u_global[idx_global] / Uref;
        double V = v_global[idx_global] / Uref;
        double W = w_global[idx_global] / Uref;

        fpE3.write(reinterpret_cast<char*>(&X), sizeof(double));
        fpE3.write(reinterpret_cast<char*>(&Y), sizeof(double));
        fpE3.write(reinterpret_cast<char*>(&Z), sizeof(double));
        fpE3.write(reinterpret_cast<char*>(&Rho), sizeof(double));
        fpE3.write(reinterpret_cast<char*>(&U), sizeof(double));
        fpE3.write(reinterpret_cast<char*>(&V), sizeof(double));
        fpE3.write(reinterpret_cast<char*>(&W), sizeof(double));
    }}}

    fpE3.close();
}

// 輸出 VTK 格式給 Paraview
void Output3Dvelocity_VTK() {
    ostringstream oss;
    oss << (NX6 - 6) << "x" << (NY6 - 6) << "x" << (NZ6 - 4) << ".vtk";
    cout << "VTK output: " << oss.str() << "\n";

    ofstream out(oss.str());
    out << "# vtk DataFile Version 3.0\n";
    out << "LBM Velocity Field (combined)\n";
    out << "ASCII\n";
    out << "DATASET STRUCTURED_GRID\n";
    out << "DIMENSIONS " << (NX6 - 6) << " " << (NY6 - 6) << " " << (NZ6 - 4) << "\n";

    int nPoints = (NX6 - 6) * (NY6 - 6) * (NZ6 - 4);
    out << "POINTS " << nPoints << " double\n";
    out << fixed << setprecision(6);

    for (int k = 2; k < NZ6 - 2; k++) {
    for (int j = 3; j < NY6 - 3; j++) {
    for (int i = 3; i < NX6 - 3; i++) {
        out << x[i] << " " << y[j] << " " << z[j * NZ6 + k] << "\n";
    }}}

    out << "\nPOINT_DATA " << nPoints << "\n";
    out << "SCALARS rho double 1\n";
    out << "LOOKUP_TABLE default\n";
    out << setprecision(15);
    for (int k = 2; k < NZ6 - 2; k++) {
    for (int j = 3; j < NY6 - 3; j++) {
    for (int i = 3; i < NX6 - 3; i++) {
        int idx = j * NX6 * NZ6 + k * NX6 + i;
        out << rho_global[idx] << "\n";
    }}}

    out << "\nVECTORS velocity double\n";
    for (int k = 2; k < NZ6 - 2; k++) {
    for (int j = 3; j < NY6 - 3; j++) {
    for (int i = 3; i < NX6 - 3; i++) {
        int idx = j * NX6 * NZ6 + k * NX6 + i;
        out << u_global[idx] / Uref << " " 
            << v_global[idx] / Uref << " " 
            << w_global[idx] / Uref << "\n";
    }}}

    out.close();
}

void Outputstreamwise() {
    for (int n = 0; n <= 8; n++) {
        ostringstream fname;
        fname << "velocity_y=" << n << ".DAT";
        
        ofstream fout(fname.str());
        fout << "VARIABLES=\"z\",\"uavg\",\"vavg\",\"wavg\"\n";
        fout << "ZONE T=\"y=" << n << "\", F=POINT\n";
        fout << "K=" << (NZ6 - 4) << "\n";
        fout << fixed;

        for (int k = 2; k < NZ6 - 2; k++) {
            int j = static_cast<int>(n / 9.0 * (NY6 - 6)) + 3;
            int idx = j * NX6 * NZ6 + k * NX6 + NX6 / 2;

            fout << setprecision(5) << z[j * NZ6 + k] << "\t"
                 << setprecision(15) << (u_global[idx] / Uref) << "\t"
                 << (n + v_global[idx] / Uref) << "\t"
                 << (n + 4.0 * w_global[idx] / Uref) << "\n";
        }
        fout.close();
    }
}

void OutputMiddlePlane() {
    ofstream middle("middleplane.dat");
    middle << "VARIABLES=\"y\",\"z\",\"vavg\",\"wavg\"\n";
    middle << "ZONE T=\"x=6\", F=POINT\n";
    middle << "j = " << (NY6 - 6) << " k = " << (NZ6 - 4) << "\n";
    middle << fixed;

    for (int k = 2; k < NZ6 - 2; k++) {
        for (int j = 3; j < NY6 - 3; j++) {
            int index = j * NX6 * NZ6 + k * NX6 + NX6 / 2;

            middle << setprecision(5) << y[j] << "\t" << z[j * NZ6 + k] << "\t"
                   << setprecision(15) << v_global[index] / Uref << "\t"
                   << w_global[index] / Uref << "\n";
        }
    }
    middle.close();
}

int main(int argc, char* argv[]) {
    cout << "=== Periodic Hill Post-Processor (C++) ===\n";
    cout << "Grid: " << NX6 << " x " << NY6 << " x " << NZ6 << "\n";
    cout << "GPU partitions: " << jp << "\n\n";

    // 分配記憶體 (使用 vector)
    u_global.resize(NX6 * NY6 * NZ6);
    v_global.resize(NX6 * NY6 * NZ6);
    w_global.resize(NX6 * NY6 * NZ6);
    rho_global.resize(NX6 * NY6 * NZ6);

    x.resize(NX6);
    y.resize(NY6);
    z.resize(NY6 * NZ6);

    // 暫存各 GPU 資料
    vector<double> u_local(NX6 * NYD6 * NZ6);
    vector<double> v_local(NX6 * NYD6 * NZ6);
    vector<double> w_local(NX6 * NYD6 * NZ6);
    vector<double> rho_local(NX6 * NYD6 * NZ6);

    // 讀取網格
    Mesh_scan();
    cout << "Mesh loaded.\n";

    // 讀取各 GPU 資料並合併
    for (int myid = 0; myid < jp; myid++) {
        cout << "Reading data... myid = " << myid << "\n";

        ReadData(u_local, "result", "u", myid);
        ReadData(v_local, "result", "v", myid);
        ReadData(w_local, "result", "w", myid);
        ReadData(rho_local, "result", "rho", myid);

        // 合併到全域陣列
        for (int k = 2; k < NZ6 - 2; k++) {
        for (int j = 3; j < NYD6 - 3; j++) {
        for (int i = 3; i < NX6 - 3; i++) {
            int j_global = myid * (NYD6 - 7) + j;
            int idx_local = j * NX6 * NZ6 + k * NX6 + i;
            int idx_global = j_global * NX6 * NZ6 + k * NX6 + i;

            u_global[idx_global] = u_local[idx_local];
            v_global[idx_global] = v_local[idx_local];
            w_global[idx_global] = w_local[idx_local];
            rho_global[idx_global] = rho_local[idx_local];
        }}}
    }
    cout << "Data merged.\n\n";

    // 輸出各種格式
    cout << "Outputting utau...\n";
    printutau();

    cout << "Outputting streamwise profiles...\n";
    Outputstreamwise();

    cout << "Outputting middle plane...\n";
    OutputMiddlePlane();

    cout << "Outputting 3D velocity (Tecplot)...\n";
    Output3Dvelocity();

    cout << "Outputting 3D velocity (VTK for Paraview)...\n";
    Output3Dvelocity_VTK();

    cout << "\n=== Post-processing complete! ===\n";
    return 0;
}
