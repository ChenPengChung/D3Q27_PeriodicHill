#ifndef INITIALIZATIONTOOL_FILE
#define INITIALIZATIONTOOL_FILE

#define tanhFunction( L, LatticeSize, a, j, N )     \
(           \
    L/2.0 + LatticeSize/2.0 + ((L/2.0)/a)*tanh((-1.0+2.0*(double)(j)/(double)(N))/2.0*log((1.0+a)/(1.0-a)))     \
)

double GetNonuniParameter() {
    double total = LZ - HillFunction( 0.0 ) - minSize;
    double a_temp[2] = {0.1, 1.0};
    double a_mid;

    double x_temp[2], dx;
    do{
        a_mid = (a_temp[0]+a_temp[1]) / 2.0;
        x_temp[0] = tanhFunction(total, minSize, a_mid, 0, (NZ6-7));
        x_temp[1] = tanhFunction(total, minSize, a_mid, 1, (NZ6-7));
        dx = x_temp[1] - x_temp[0];
        if( dx - minSize >= 0.0 ){
            a_temp[0] = a_mid;
        } else {
            a_temp[1] = a_mid;
        }
    } while ( fabs( dx - minSize ) > 1e-14 );
    

    /*if( myid == 0 ){
        printf("a = %lf\n", a_mid);
    }*/

    return a_mid;
}

double Lagrange_2nd(
    double x,   double x_i,
    double x1,  double x2  )
{
    double Para = (x - x1)/(x_i - x1)*(x - x2)/(x_i - x2);

    return Para;
}

void GetParameter_2nd(
    double *Para_h[7],      double Position,
    double *Pos,            int i,              int n  )
{
    Para_h[0][i] = Lagrange_2nd(Position, Pos[n],   Pos[n+1], Pos[n+2]);
    Para_h[1][i] = Lagrange_2nd(Position, Pos[n+1], Pos[n]  , Pos[n+2]);
    Para_h[2][i] = Lagrange_2nd(Position, Pos[n+2], Pos[n]  , Pos[n+1]);
}

double Lagrange_6th(
    double x,   double x_i,
    double x1,  double x2,  double x3,  double x4,  double x5,  double x6)
{
    double Para = (x - x1)/(x_i - x1)*(x - x2)/(x_i - x2)*(x - x3)/(x_i - x3)*(x - x4)/(x_i - x4)*(x - x5)/(x_i - x5)*(x - x6)/(x_i - x6);

    return Para;
}

void GetParameter_6th(
    double *Para_h[7],      double Position,
    double *Pos,            int i,              int n  )
{
    Para_h[0][i] = Lagrange_6th(Position, Pos[n],   Pos[n+1], Pos[n+2], Pos[n+3], Pos[n+4], Pos[n+5], Pos[n+6]);
    Para_h[1][i] = Lagrange_6th(Position, Pos[n+1], Pos[n],   Pos[n+2], Pos[n+3], Pos[n+4], Pos[n+5], Pos[n+6]);
    Para_h[2][i] = Lagrange_6th(Position, Pos[n+2], Pos[n],   Pos[n+1], Pos[n+3], Pos[n+4], Pos[n+5], Pos[n+6]);
    Para_h[3][i] = Lagrange_6th(Position, Pos[n+3], Pos[n],   Pos[n+1], Pos[n+2], Pos[n+4], Pos[n+5], Pos[n+6]);
    Para_h[4][i] = Lagrange_6th(Position, Pos[n+4], Pos[n],   Pos[n+1], Pos[n+2], Pos[n+3], Pos[n+5], Pos[n+6]);
    Para_h[5][i] = Lagrange_6th(Position, Pos[n+5], Pos[n],   Pos[n+1], Pos[n+2], Pos[n+3], Pos[n+4], Pos[n+6]);
    Para_h[6][i] = Lagrange_6th(Position, Pos[n+6], Pos[n],   Pos[n+1], Pos[n+2], Pos[n+3], Pos[n+4], Pos[n+5]);
}

int IsBFLBCNeeded(const double Y, const double Z) {
    const double hill = HillFunction( Y );
    if( hill > Z ){
        return 1;
    } else {
        return 0;
    }
}

double GetDeltaHorizontal(
    const double z_target,
    const double y_large,       const double y_small,       const double y_point )
{
    double y_temp[2] = {y_large, y_small};
    double y_mid;
    do{
        y_mid = (y_temp[0]+y_temp[1]) / 2.0;
        if( HillFunction(y_mid) >= z_target ){
            y_temp[0] = y_mid;
        } else {
            y_temp[1] = y_mid;
        }
    } while ( fabs( HillFunction(y_mid) - z_target ) > 1e-13 );

    double d = minSize - 2.0*fabs(y_point-y_mid);
    return d;
}

double GetDelta45Degree(
    const double z_pnt,      const double y_pnt,
    const double y_zdominate,const double y_ydominate  )
{
    double y_temp[2] = {y_zdominate, y_ydominate};
    double y_mid;
    int a = 0;
    do{
        y_mid = (y_temp[0]+y_temp[1]) / 2.0;
        if( fabs( fabs(HillFunction(y_mid) - z_pnt) > fabs(y_mid-y_pnt) ) ){
            y_temp[0] = y_mid;
        } else {
            y_temp[1] = y_mid;
        }
        a++;
    } while ( fabs( fabs(HillFunction(y_mid)-z_pnt) - fabs(y_mid-y_pnt) ) > 1e-13 );
    double d = minSize - 2.0*fabs(y_pnt-y_mid);
    return d;
}

void GetBFLXiParameter(
    double *XiPara_h[7],    double pos_z,       double pos_y,
    double *Pos_xi,         int IdxToStore,     int k  )
{
    double L = LZ - HillFunction(pos_y) - minSize;
    double pos_xi = LXi * (pos_z - (HillFunction(pos_y)+minSize/2.0)) / L;
    double a = GetNonuniParameter();
    //double pos_xi = atanh((pos_z-HillFunction(pos_y)-minSize/2.0-L/2.0)/((L/2.0)/a))/log((1.0+a)/(1.0-a))*2.0;

    if( k >= 3 && k <= 4 ){
        GetParameter_6th( XiPara_h, pos_xi, Pos_xi, IdxToStore, 3 );
    }
}


#endif
