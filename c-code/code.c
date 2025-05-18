#include <stdio.h>
#include <math.h> 

// Define constants
#define DELTA_T 1.0f    // Time step (1 second)
#define SIGMA_A 0.2f    // Standard deviation for acceleration noise
#define SIGMA_M 3.0f    // Standard deviation for measurement noise (position)
// #define DEBUG         // Uncomment to print intermediate matrices

// Define state vector (6D: [x, vx, ax, y, vy, ay])
float x[6] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};

// Covariance matrix (6x6)
float P[6][6] = {
    {500.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f},
    {0.0f, 500.0f, 0.0f, 0.0f, 0.0f, 0.0f},
    {0.0f, 0.0f, 500.0f, 0.0f, 0.0f, 0.0f},
    {0.0f, 0.0f, 0.0f, 500.0f, 0.0f, 0.0f},
    {0.0f, 0.0f, 0.0f, 0.0f, 500.0f, 0.0f},
    {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 500.0f}
};

// State transition matrix (6x6)
float F[6][6] = {
    {1.0f, DELTA_T, 0.5f*DELTA_T*DELTA_T, 0.0f, 0.0f, 0.0f},
    {0.0f, 1.0f, DELTA_T, 0.0f, 0.0f, 0.0f},
    {0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f},
    {0.0f, 0.0f, 0.0f, 1.0f, DELTA_T, 0.5f*DELTA_T*DELTA_T},
    {0.0f, 0.0f, 0.0f, 0.0f, 1.0f, DELTA_T},
    {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f}
};

// Process noise covariance (6x6) - will be initialized by initialize_Q
float Q[6][6]; 

// Measurement matrix (2x6)
float H[2][6] = {
    {1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f},
    {0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f}
};

// Measurement noise covariance (2x2)
float R[2][2] = {
    {SIGMA_M*SIGMA_M, 0.0f},
    {0.0f, SIGMA_M*SIGMA_M}
};

// Helper function to enforce matrix symmetry for 6x6 float matrix
void enforce_symmetry(float matrix[6][6]) {
    for(int i=0; i<6; i++) {
        for(int j=0; j<i; j++) { // Iterate only through lower/upper triangle
            matrix[i][j] = matrix[j][i] = (matrix[i][j] + matrix[j][i])/2.0f;
        }
    }
}

// Initialize process noise covariance matrix Q
void initialize_Q() {
    // First, zero out Q in case it wasn't zero-initialized by static storage
    for(int i=0; i<6; ++i) {
        for(int j=0; j<6; ++j) {
            Q[i][j] = 0.0f;
        }
    }

    float q_noise = SIGMA_A * SIGMA_A;
    float dt = DELTA_T;
    float dt2 = dt * dt;
    float dt3 = dt2 * dt;
    float dt4 = dt3 * dt;

    // X-components
    Q[0][0] = q_noise * dt4 / 4.0f;
    Q[0][1] = q_noise * dt3 / 2.0f;
    Q[0][2] = q_noise * dt2 / 2.0f;
    Q[1][0] = Q[0][1]; // Symmetry
    Q[1][1] = q_noise * dt2;
    Q[1][2] = q_noise * dt;
    Q[2][0] = Q[0][2]; // Symmetry
    Q[2][1] = Q[1][2]; // Symmetry
    Q[2][2] = q_noise;
    
    // Y-components (same structure as X)
    Q[3][3] = Q[0][0]; Q[3][4] = Q[0][1]; Q[3][5] = Q[0][2];
    Q[4][3] = Q[1][0]; Q[4][4] = Q[1][1]; Q[4][5] = Q[1][2];
    Q[5][3] = Q[2][0]; Q[5][4] = Q[2][1]; Q[5][5] = Q[2][2];
}

// Matrix multiplication: 6x6 * 6x6 (float C[6][6], float A[6][6], float B[6][6])
void multiply_6x6_6x6(float A[6][6], float B[6][6], float result[6][6]) {
    for(int i=0; i<6; i++) {
        for(int j=0; j<6; j++) {
            result[i][j] = 0.0f;
            for(int k=0; k<6; k++) {
                result[i][j] += A[i][k] * B[k][j];
            }
        }
    }
}

// Matrix multiplication: 6x6 * 6x2 (float C[6][2], float A[6][6], float B[6][2])
void multiply_6x6_6x2(float A[6][6], float B[6][2], float result[6][2]) {
    for(int i=0; i<6; i++) {
        for(int j=0; j<2; j++) {
            result[i][j] = 0.0f;
            for(int k=0; k<6; k++) {
                result[i][j] += A[i][k] * B[k][j];
            }
        }
    }
}

// Matrix multiplication: 2x6 * 6x6 (float C[2][6], float A[2][6], float B[6][6])
void multiply_2x6_6x6(float A[2][6], float B[6][6], float result[2][6]) {
    for(int i=0; i<2; i++) {
        for(int j=0; j<6; j++) {
            result[i][j] = 0.0f;
            for(int k=0; k<6; k++) {
                result[i][j] += A[i][k] * B[k][j];
            }
        }
    }
}

// Matrix multiplication: 2x6 * 6x2 (float C[2][2], float A[2][6], float B[6][2])
void multiply_2x6_6x2(float A[2][6], float B[6][2], float result[2][2]) {
    for(int i=0; i<2; i++) {
        for(int j=0; j<2; j++) {
            result[i][j] = 0.0f;
            for(int k=0; k<6; k++) {
                result[i][j] += A[i][k] * B[k][j];
            }
        }
    }
}

// Matrix multiplication: 6x2 * 2x6 (float C[6][6], float A[6][2], float B[2][6])
void multiply_6x2_2x6(float A[6][2], float B[2][6], float result[6][6]) {
    for(int i=0; i<6; i++) {
        for(int j=0; j<6; j++) {
            result[i][j] = 0.0f;
            for(int k=0; k<2; k++) {
                result[i][j] += A[i][k] * B[k][j];
            }
        }
    }
}

// Matrix transpose: 6x6 (float B[6][6], float A[6][6])
void transpose_6x6(float A[6][6], float result[6][6]) {
    for(int i=0; i<6; i++) {
        for(int j=0; j<6; j++) {
            result[j][i] = A[i][j];
        }
    }
}

// Matrix transpose: 2x6 -> 6x2 (float B[6][2], float A[2][6])
void transpose_2x6_to_6x2(float A[2][6], float result[6][2]) {
    for(int i=0; i<2; i++) {
        for(int j=0; j<6; j++) {
            result[j][i] = A[i][j];
        }
    }
}

// 2x2 matrix inversion with regularization (float A_inv[2][2], float A[2][2])
void invert_2x2(float A[2][2], float A_inv[2][2]) {
    float det = A[0][0]*A[1][1] - A[0][1]*A[1][0];
    if(fabsf(det) < 1e-7f) {
        det = copysignf(fmaxf(fabsf(det), 1e-7f), det);
    }
    A_inv[0][0] = A[1][1]/det;
    A_inv[0][1] = -A[0][1]/det;
    A_inv[1][0] = -A[1][0]/det;
    A_inv[1][1] = A[0][0]/det;
}

// Function to initialize a 6x6 matrix to identity
void init_identity_matrix(float matrix[6][6]) {
    for (int i = 0; i < 6; ++i) {
        for (int j = 0; j < 6; ++j) {
            matrix[i][j] = (i == j) ? 1.0f : 0.0f;
        }
    }
}


// Kalman Filter update step
void kalman_filter_update(float z[2]) {
    // Temporary matrices
    float F_T[6][6], P_pred[6][6], FP[6][6], FPFT[6][6];
    float H_T[6][2], PHT[6][2], S[2][2], S_inv[2][2], K[6][2];
    float I[6][6], KH[6][6], I_KH[6][6]; // I is now initialized by function
    float temp_HP[2][6]; 

    // --- Prediction Step ---
    float x_pred[6] = {0.0f};
    for(int i=0; i<6; i++) {
        for(int j=0; j<6; j++) {
            x_pred[i] += F[i][j] * x[j];
        }
    }
    
    multiply_6x6_6x6(F, P, FP);
    transpose_6x6(F, F_T);
    multiply_6x6_6x6(FP, F_T, FPFT);
    for(int i=0; i<6; i++) {
        for(int j=0; j<6; j++) {
            P_pred[i][j] = FPFT[i][j] + Q[i][j];
        }
    }
    enforce_symmetry(P_pred);

    // --- Update Step ---
    // S = H*P_pred*H^T + R
    transpose_2x6_to_6x2(H, H_T);               // H_T is 6x2
    multiply_2x6_6x6(H, P_pred, temp_HP);       // temp_HP (2x6) = H (2x6) * P_pred (6x6)
    multiply_2x6_6x2(temp_HP, H_T, S);          // S (2x2) = temp_HP (2x6) * H_T (6x2) -> this is H*P_pred*H_T
    
    for(int i=0; i<2; i++) { // Add R to S
        for(int j=0; j<2; j++) {
            S[i][j] += R[i][j];
        }
    }
    invert_2x2(S, S_inv);
    
    // K = P_pred*H^T*S_inv
    multiply_6x6_6x2(P_pred, H_T, PHT); 
    
    for(int i=0; i<6; i++) {
        for(int j=0; j<2; j++) {
            K[i][j] = 0.0f;
            for(int k_loop=0; k_loop<2; k_loop++) {
                K[i][j] += PHT[i][k_loop] * S_inv[k_loop][j];
            }
        }
    }

    float y[2] = {z[0] - x_pred[0], z[1] - x_pred[3]};
    for(int i=0; i<6; i++) {
        x[i] = x_pred[i] + K[i][0]*y[0] + K[i][1]*y[1];
    }

    // P = (I - KH)P_pred
    init_identity_matrix(I); // Initialize I to identity
    multiply_6x2_2x6(K, H, KH); 
    for(int i=0; i<6; i++) {
        for(int j=0; j<6; j++) {
            I_KH[i][j] = I[i][j] - KH[i][j];
        }
    }
    multiply_6x6_6x6(I_KH, P_pred, P);
    enforce_symmetry(P);
}

int main() {
    initialize_Q();
    
    float measurements[35][2] = {
        {301.5f, -401.46f}, {298.23f, -375.44f}, {297.83f, -346.15f}, {300.42f, -320.2f}, {301.94f, -300.08f}, {299.5f, -274.12f},
        {305.98f, -253.45f}, {301.25f, -226.4f}, {299.73f, -200.65f}, {299.2f, -171.62f}, {298.62f, -152.11f}, {301.84f, -125.19f},
        {299.6f, -93.4f}, {295.3f, -74.79f}, {299.3f, -49.12f}, {301.95f, -28.73f}, {296.3f, 2.99f}, {295.11f, 25.65f}, {295.12f, 49.86f},
        {289.9f, 72.87f}, {283.51f, 96.34f}, {276.42f, 120.4f}, {264.22f, 144.69f}, {250.25f, 168.06f}, {236.66f, 184.99f}, {217.47f, 205.11f},
        {199.75f, 221.82f}, {179.7f, 238.3f}, {160.0f, 253.02f}, {140.92f, 267.19f}, {113.53f, 270.71f}, {93.68f, 285.86f}, {69.71f, 288.48f}, 
        {45.93f, 292.9f}, {20.87f, 298.77f}
    };

    printf("Initial State: X=%.2f, Y=%.2f\n", x[0], x[3]);
    
    for(int i=0; i<35; i++) {
        kalman_filter_update(measurements[i]);
        printf("Estimate %2d: X=%7.2f, Y=%7.2f, Vx=%6.2f, Vy=%6.2f\n",
               i+1, x[0], x[3], x[1], x[4]);
    }
    
    return 0;
}
