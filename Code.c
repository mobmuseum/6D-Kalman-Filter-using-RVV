#include <stdio.h>
#include <math.h>

// Define constants
#define DELTA_T 1.0    // Time step (1 second)
#define SIGMA_A 0.2    // Standard deviation for acceleration noise
#define SIGMA_M 3.0    // Standard deviation for measurement noise (position)
// #define DEBUG         // Uncomment to print intermediate matrices

// Define state vector (6D: [x, vx, ax, y, vy, ay])
double x[6] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};

// Covariance matrix (6x6)
double P[6][6] = {
    {500, 0, 0, 0, 0, 0},
    {0, 500, 0, 0, 0, 0},
    {0, 0, 500, 0, 0, 0},
    {0, 0, 0, 500, 0, 0},
    {0, 0, 0, 0, 500, 0},
    {0, 0, 0, 0, 0, 500}
};

// State transition matrix (6x6)
double F[6][6] = {
    {1, DELTA_T, 0.5*DELTA_T*DELTA_T, 0, 0, 0},
    {0, 1, DELTA_T, 0, 0, 0},
    {0, 0, 1, 0, 0, 0},
    {0, 0, 0, 1, DELTA_T, 0.5*DELTA_T*DELTA_T},
    {0, 0, 0, 0, 1, DELTA_T},
    {0, 0, 0, 0, 0, 1}
};

// Process noise covariance (6x6)
double Q[6][6] = {0};

// Measurement matrix (2x6)
double H[2][6] = {
    {1, 0, 0, 0, 0, 0},
    {0, 0, 0, 1, 0, 0}
};

// Measurement noise covariance (2x2)
double R[2][2] = {
    {SIGMA_M*SIGMA_M, 0},
    {0, SIGMA_M*SIGMA_M}
};

// Helper function to enforce matrix symmetry
void enforce_symmetry(double matrix[6][6]) {
    for(int i=0; i<6; i++) {
        for(int j=0; j<i; j++) {
            matrix[i][j] = matrix[j][i] = (matrix[i][j] + matrix[j][i])/2.0;
        }
    }
}

// Initialize process noise covariance matrix
void initialize_Q() {
    double q = SIGMA_A * SIGMA_A;
    // X-components
    Q[0][0] = q * pow(DELTA_T,4)/4;
    Q[0][1] = Q[1][0] = q * pow(DELTA_T,3)/2;
    Q[0][2] = Q[2][0] = q * pow(DELTA_T,2)/2;
    Q[1][1] = q * pow(DELTA_T,2);
    Q[1][2] = Q[2][1] = q * DELTA_T;
    Q[2][2] = q;
    
    // Y-components (same structure as X)
    Q[3][3] = Q[0][0]; Q[3][4] = Q[0][1]; Q[3][5] = Q[0][2];
    Q[4][3] = Q[1][0]; Q[4][4] = Q[1][1]; Q[4][5] = Q[1][2];
    Q[5][3] = Q[2][0]; Q[5][4] = Q[2][1]; Q[5][5] = Q[2][2];
}

// Matrix multiplication: 6x6 * 6x6
void multiply_6x6_6x6(double A[6][6], double B[6][6], double result[6][6]) {
    for(int i=0; i<6; i++) {
        for(int j=0; j<6; j++) {
            result[i][j] = 0;
            for(int k=0; k<6; k++) {
                result[i][j] += A[i][k] * B[k][j];
            }
        }
    }
}

// Matrix multiplication: 6x6 * 6x2
void multiply_6x6_6x2(double A[6][6], double B[6][2], double result[6][2]) {
    for(int i=0; i<6; i++) {
        for(int j=0; j<2; j++) {
            result[i][j] = 0;
            for(int k=0; k<6; k++) {
                result[i][j] += A[i][k] * B[k][j];
            }
        }
    }
}

// Matrix multiplication: 2x6 * 6x6
void multiply_2x6_6x6(double A[2][6], double B[6][6], double result[2][6]) {
    for(int i=0; i<2; i++) {
        for(int j=0; j<6; j++) {
            result[i][j] = 0;
            for(int k=0; k<6; k++) {
                result[i][j] += A[i][k] * B[k][j];
            }
        }
    }
}

// Matrix multiplication: 2x6 * 6x2
void multiply_2x6_6x2(double A[2][6], double B[6][2], double result[2][2]) {
    for(int i=0; i<2; i++) {
        for(int j=0; j<2; j++) {
            result[i][j] = 0;
            for(int k=0; k<6; k++) {
                result[i][j] += A[i][k] * B[k][j];
            }
        }
    }
}

// Matrix multiplication: 6x2 * 2x6
void multiply_6x2_2x6(double A[6][2], double B[2][6], double result[6][6]) {
    for(int i=0; i<6; i++) {
        for(int j=0; j<6; j++) {
            result[i][j] = 0;
            for(int k=0; k<2; k++) {
                result[i][j] += A[i][k] * B[k][j];
            }
        }
    }
}

// Matrix transpose: 6x6
void transpose_6x6(double A[6][6], double result[6][6]) {
    for(int i=0; i<6; i++) {
        for(int j=0; j<6; j++) {
            result[j][i] = A[i][j];
        }
    }
}

// Matrix transpose: 2x6 -> 6x2
void transpose_2x6_to_6x2(double A[2][6], double result[6][2]) {
    for(int i=0; i<2; i++) {
        for(int j=0; j<6; j++) {
            result[j][i] = A[i][j];
        }
    }
}

// 2x2 matrix inversion with regularization
void invert_2x2(double A[2][2], double A_inv[2][2]) {
    double det = A[0][0]*A[1][1] - A[0][1]*A[1][0];
    if(fabs(det) < 1e-7) {  // Increased stability threshold
        det = copysign(fmax(fabs(det), 1e-7), det);
    }
    A_inv[0][0] = A[1][1]/det;
    A_inv[0][1] = -A[0][1]/det;
    A_inv[1][0] = -A[1][0]/det;
    A_inv[1][1] = A[0][0]/det;
}

// Debugging function to print 6x6 matrix
#ifdef DEBUG
void print_matrix_6x6(double mat[6][6], const char* name) {
    printf("\n%s:\n", name);
    for(int i=0; i<6; i++) {
        for(int j=0; j<6; j++) {
            printf("%10.2f ", mat[i][j]);
        }
        printf("\n");
    }
}

// Debugging function to print 6x2 matrix
void print_matrix_6x2(double mat[6][2], const char* name) {
    printf("\n%s:\n", name);
    for(int i=0; i<6; i++) {
        printf("%10.2f %10.2f\n", mat[i][0], mat[i][1]);
    }
}
#endif

// Kalman Filter update step
void kalman_filter_update(double z[2]) {
    // Temporary matrices
    double F_T[6][6], P_pred[6][6], FP[6][6], FPFT[6][6];
    double H_T[6][2], PHT[6][2], S[2][2], S_inv[2][2], K[6][2];
    double I[6][6] = {0}, KH[6][6], I_KH[6][6];
    
    // --- Prediction Step ---
    // x_pred = F * x
    double x_pred[6] = {0};
    for(int i=0; i<6; i++) {
        for(int j=0; j<6; j++) {
            x_pred[i] += F[i][j] * x[j];
        }
    }
    
    // P_pred = F*P*F^T + Q
    multiply_6x6_6x6(F, P, FP);
    transpose_6x6(F, F_T);
    multiply_6x6_6x6(FP, F_T, FPFT);
    for(int i=0; i<6; i++) {
        for(int j=0; j<6; j++) {
            P_pred[i][j] = FPFT[i][j] + Q[i][j];
        }
    }
    enforce_symmetry(P_pred);

    #ifdef DEBUG
    print_matrix_6x6(P_pred, "Predicted Covariance P_pred");
    #endif

    // --- Update Step ---
    // Compute Kalman Gain
    transpose_2x6_to_6x2(H, H_T);
    multiply_6x6_6x2(P_pred, H_T, PHT);
    
    // S = H*P_pred*H^T + R
    for(int i=0; i<2; i++) {
        for(int j=0; j<2; j++) {
            S[i][j] = R[i][j];
            for(int k=0; k<6; k++) {
                for(int l=0; l<6; l++) {
                    S[i][j] += H[i][k] * P_pred[k][l] * H[j][l];
                }
            }
        }
    }
    invert_2x2(S, S_inv);
    
    // K = PHT * S_inv
    for(int i=0; i<6; i++) {
        for(int j=0; j<2; j++) {
            K[i][j] = 0;
            for(int k=0; k<2; k++) {
                K[i][j] += PHT[i][k] * S_inv[k][j];
            }
        }
    }

    #ifdef DEBUG
    printf("\nKalman Gain K:\n");
    for(int i=0; i<6; i++) {
        printf("[%6.4f, %6.4f]\n", K[i][0], K[i][1]);
    }
    #endif

    // Update state estimate
    double y[2] = {z[0] - x_pred[0], z[1] - x_pred[3]};
    for(int i=0; i<6; i++) {
        x[i] = x_pred[i] + K[i][0]*y[0] + K[i][1]*y[1];
    }

    // Update covariance: P = (I - KH)P_pred
    for(int i=0; i<6; i++) I[i][i] = 1.0;
    multiply_6x2_2x6(K, H, KH);
    for(int i=0; i<6; i++) {
        for(int j=0; j<6; j++) {
            I_KH[i][j] = I[i][j] - KH[i][j];
        }
    }
    multiply_6x6_6x6(I_KH, P_pred, P);
    enforce_symmetry(P);

    #ifdef DEBUG
    print_matrix_6x6(P, "Updated Covariance P");
    #endif
}

int main() {
    initialize_Q();
    
    double measurements[35][2] = {
        {301.5, -401.46}, {298.23, -375.44}, {297.83, -346.15},
        {302.45, -318.22}, {305.76, -292.78}, {309.75, -265.35},
        {315.84, -241.79}, {319.34, -214.69}, {326.99, -189.3},
        {333.35, -162.54}, {337.93, -138.52}, {342.61, -111.92},
        {347.87, -86.49},  {354.04, -59.87},  {359.77, -34.62},
        {365.13, -9.18},   {371.32, 17.95},   {376.91, 41.44},
        {383.45, 67.89},   {390.37, 92.39},   {397.11, 117.95},
        {401.55, 145.27},  {407.62, 167.23},  {410.51, 194.33},
        {417.67, 218.61},  {422.51, 242.91},  {426.47, 271.52},
        {429.89, 295.57},  {431.6, 322.47},   {435.9, 350.73},
        {440.2, 374.52},   {441.75, 401.7},   {444.73, 428.39},
        {448.56, 451.5},   {450.73, 476.4}
    };

    printf("Initial State: [%.2f, %.2f]\n", x[0], x[3]);
    
    for(int i=0; i<35; i++) {
        kalman_filter_update(measurements[i]);
        printf("Estimate %2d: X=%7.2f, Y=%7.2f, Vx=%6.2f, Vy=%6.2f\n",
               i+1, x[0], x[3], x[1], x[4]);
    }
    
    return 0;
}
