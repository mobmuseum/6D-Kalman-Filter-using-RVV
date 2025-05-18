#define STDOUT 0xd0580000

.section .text
.global _start
_start:
#-------------------------------------------------------------------------
# Main function: Initializes the system and runs the Kalman filter
#-------------------------------------------------------------------------
main:
    # Setup the stack frame
    addi sp, sp, -8
    sw ra, 4(sp)
    sw s0, 0(sp)
    csrw frm, zero          
    
    # Initialize the process noise covariance matrix Q
    jal ra, initialize_Q
    
    # Run Kalman filter for all measurements
    la s0, measurements     # s0 will hold the measurement pointer for kalman_loop
    addi s1, zero, 0        # s1 is the measurement counter for kalman_loop

    # Explicitly call kalman_loop
    jal ra, kalman_loop


# Inputs: a0 = filename address, a1 = buffer address, a2 = length
# Return: Result of the 'close' call (0 on success, -1 on error)
# Clobbers: t0-t6, a0-a7 (standard caller-saved)
# Uses: s0 (for file descriptor), s1 (for buffer address), s2 (for length)
# Saves: ra, s0, s1, s2 on stack
write_to_file:
    addi sp, sp, -16     # Allocate stack space
    sw ra, 0(sp)         # Save return address
    sw s0, 4(sp)         # Save s0 (will store file descriptor)
    sw s1, 8(sp)         # Save s1 (will save original buffer address)
    sw s2, 12(sp)        # Save s2 (will save original length)

    # --- Save original inputs a1 (buffer address) and a2 (length) ---
    mv s1, a1            # Save original buffer address in s1
    mv s2, a2            # Save original length in s2

    # --- Open the file ---
    # Arguments for open:
    # a0 = filename address (input arg 1 - already correct)
    # a1 = flags (O_WRONLY | O_CREAT | O_TRUNC = 0x601)
    # a2 = mode (0666 = 0x1b6)

    # a0 is already the filename address
    li a1, 0x601         # Load flags directly into a1
    li a2, 0x1b6         # Load mode (0666) directly into a2

    call open            # Call the open function

    # File descriptor is now in a0. Save it.
    mv s0, a0            # Save the file descriptor in s0

    # --- Write to the file ---
    # Arguments for write:
    # a0 = file descriptor (from s0)
    # a1 = buffer address (from s1)
    # a2 = count (from s2)

    mv a0, s0            # Move the file descriptor from s0 to a0
    mv a1, s1            # Restore buffer address from s1 to a1
    mv a2, s2            # Restore length from s2 to a2

    call write           # Call the write function

    # --- Close the file ---
    # Arguments for close:
    # a0 = file descriptor (from s0)
    mv a0, s0            # Move the file descriptor from s0 to a0

    call close           # Call the close function

    # --- Function Epilogue ---
    # Restore saved registers
    lw ra, 0(sp)         # Restore return address
    lw s0, 4(sp)         # Restore s0
    lw s1, 8(sp)         # Restore s1
    lw s2, 12(sp)        # Restore s2
    addi sp, sp, 16      # Deallocate stack space

    ret                  # Return from the function (a0 contains close result)


#-------------------------------------------------------------------------
# Function: initialize_Q
# Initializes the process noise covariance matrix Q
#-------------------------------------------------------------------------
initialize_Q:
    addi sp, sp, -8
    sw ra, 4(sp)
    sw s0, 0(sp)
    
    # Calculate SIGMA_A^2
    la a0, SIGMA_A
    flw fs0, 0(a0)
    fmul.s fs0, fs0, fs0     # q = SIGMA_A * SIGMA_A
    
    # Calculate powers of DELTA_T
    la a0, DELTA_T
    flw fs1, 0(a0)          # DELTA_T
    
    # DELTA_T^2
    fmul.s fs2, fs1, fs1    # DELTA_T^2
    
    # DELTA_T^3
    fmul.s fs3, fs2, fs1    # DELTA_T^3
    
    # DELTA_T^4
    fmul.s fs4, fs3, fs1    # DELTA_T^4
    
    # Calculate q * DELTA_T^4 / 4
    fmul.s fs5, fs0, fs4
    li a0, 4
    fcvt.s.w fs6, a0
    fdiv.s fs5, fs5, fs6    # q * DELTA_T^4 / 4
    
    # Calculate q * DELTA_T^3 / 2
    fmul.s fs6, fs0, fs3
    li a0, 2
    fcvt.s.w fs7, a0
    fdiv.s fs6, fs6, fs7    # q * DELTA_T^3 / 2
    
    # Calculate q * DELTA_T^2 / 2
    fmul.s fs7, fs0, fs2    # q * DELTA_T^2
    li a0, 2
    fcvt.s.w fs8, a0
    fdiv.s fs7, fs7, fs8    # q * DELTA_T^2 / 2

    
    # Calculate q * DELTA_T^2
    fmul.s fs3, fs0, fs2    # q * DELTA_T^2
    
    # Calculate q * DELTA_T
    fmul.s fs4, fs0, fs1    # q * DELTA_T
    
    # Now set Q matrix values
    la a0, Q
    
    # X-components
    fsw fs5, 0(a0)          # Q[0][0] = q * DELTA_T^4 / 4
    fsw fs6, 4(a0)          # Q[0][1] = q * DELTA_T^3 / 2
    fsw fs7, 8(a0)         # Q[0][2] = q * DELTA_T^2 / 2
    fsw fs6, 12(a0)         # Q[1][0] = q * DELTA_T^3 / 2
    fsw fs3, 16(a0)         # Q[1][1] = q * DELTA_T^2
    fsw fs4, 20(a0)         # Q[1][2] = q * DELTA_T
    fsw fs7, 24(a0)         # Q[2][0] = q * DELTA_T^2 / 2
    fsw fs4, 28(a0)        # Q[2][1] = q * DELTA_T
    fsw fs0, 32(a0)        # Q[2][2] = q
    
    # Y-components (same structure as X)
    fsw fs5, 84(a0)        # Q[3][3] = q * DELTA_T^4 / 4
    fsw fs6, 88(a0)        # Q[3][4] = q * DELTA_T^3 / 2
    fsw fs7, 92(a0)        # Q[3][5] = q * DELTA_T^2 / 2
    fsw fs6, 108(a0)        # Q[4][3] = q * DELTA_T^3 / 2
    fsw fs3, 112(a0)        # Q[4][4] = q * DELTA_T^2
    fsw fs4, 116(a0)        # Q[4][5] = q * DELTA_T
    fsw fs7, 132(a0)        # Q[5][3] = q * DELTA_T^2 / 2
    fsw fs4, 136(a0)        # Q[5][4] = q * DELTA_T
    fsw fs0, 140(a0)        # Q[5][5] = q
    
    lw ra, 4(sp)
    lw s0, 0(sp)
    addi sp, sp, 8
    ret

#-------------------------------------------------------------------------
# Function: enforce_symmetry
# Enforces symmetry in a 6x6 matrix
# a0: Matrix address
#-------------------------------------------------------------------------
enforce_symmetry:
    addi sp, sp, -8
    sw ra, 4(sp)
    sw s0, 0(sp)
    
    mv s0, a0      # Save matrix address
    
    # For each i < j, set matrix[i][j] = matrix[j][i] = (matrix[i][j] + matrix[j][i])/2
    # We only need to process the upper triangular portion
    li t0, 0       # i = 0
    
i_loop:
    li t1, 0       # j = 0
    
j_loop:
    # Check if j < i
    bge t1, t0, j_loop_end   # skip upper triangle & diagonal
    li  t2, 6                # hard upper bound
    bge t1, t2, j_loop_end   # j >= 6  → done

    # Calculate address for matrix[i][j]
    li t2, 6       # Matrix width
    mul t3, t0, t2 # i * width
    add t3, t3, t1 # i * width + j
    slli t3, t3, 2 # (i * width + j) * 4
    add t3, s0, t3 # matrix + (i * width + j) * 4
    
    # Calculate address for matrix[j][i]
    mul t4, t1, t2 # j * width
    add t4, t4, t0 # j * width + i
    slli t4, t4, 2 # (j * width + i) * 4
    add t4, s0, t4 # matrix + (j * width + i) * 4
    
    # Load values
    flw fs0, 0(t3) # matrix[i][j]
    flw fs1, 0(t4) # matrix[j][i]
    
    # Calculate average
    fadd.s fs2, fs0, fs1
    li t5, 2
    fcvt.s.w fs3, t5
    fdiv.s fs2, fs2, fs3    # (matrix[i][j] + matrix[j][i])/2
    
    # Store back to both positions
    fsw fs2, 0(t3)
    fsw fs2, 0(t4)
    
    addi t1, t1, 1 # j++
    j j_loop
    
j_loop_end:
    addi t0, t0, 1 # i++
    li t2, 6
    blt t0, t2, i_loop
    
    lw ra, 4(sp)
    lw s0, 0(sp)
    addi sp, sp, 8
    ret

#-------------------------------------------------------------------------
# Function: multiply_6x6_6x2
# Matrix multiplication C = A * B
# A is 6x6, B is 6x2, C is 6x2
# Computes one row of C at a time using vector instructions.
# a0: Address of matrix A (6x6)
# a1: Address of matrix B (6x2)
# a2: Address of result matrix C (6x2)
# Assumptions: Matrices are stored in row-major order. Elements are single-precision floats (4 bytes).
#-------------------------------------------------------------------------
multiply_6x6_6x2:
    addi sp, sp, -40      # Adjust stack pointer for saved registers
    sw ra, 36(sp)         # Save return address
    sw s0, 32(sp)         # Base address of A
    sw s1, 28(sp)         # Base address of B
    sw s2, 24(sp)         # Base address of C
    sw s3, 20(sp)         # Outer loop limit M (6 for A rows, C rows)
    sw s4, 16(sp)         # Inner loop limit K (6 for A cols / B rows)
    sw s5, 12(sp)         # Columns of B / C (N = 2)
    sw s6, 8(sp)          # Row stride for A and B in bytes (A: 6c*4B=24, B: 2c*4B=8)
                          # s6 will be specifically for A_row_stride (24)
    sw s7, 4(sp)          # Row stride for C in bytes (2c * 4B = 8)
                          # s7 will be specifically for B_row_stride (8)
    sw s8, 0(sp)          # Pointer to current row of C (&C[i][0])

    mv s0, a0             # s0 = Address of A
    mv s1, a1             # s1 = Address of B
    mv s2, a2             # s2 = Address of C

    li s3, 6              # M = 6 (rows of A, rows of C)
    li s4, 6              # K = 6 (cols of A, rows of B) - Inner loop limit
    li s5, 2              # N = 2 (cols of B, cols of C) - Vector Length for loads/stores of B and C rows

    li s6, 24             # Row stride for A = K_A * sizeof(float) = 6 * 4 = 24 bytes
    li t4, 8              # Temp register for B_row_stride = N_B * sizeof(float) = 2 * 4 = 8 bytes
    sw t4, 4(sp)          # Store B_row_stride to s7's slot
    li s7, 8              # Row stride for C = N_C * sizeof(float) = 2 * 4 = 8 bytes (re-using s7 for C_row_stride now)
                          # We'll load B_row_stride from stack when needed or use t4 if available.

    # Set vector configuration for rows of B and C:
    # VL = N (2), SEW = 32 bits (e32), LMUL = 1 (m1)
    vsetvli t0, s5, e32, m1, ta, ma # t0 will hold actual VL (should be 2)

    # Outer loop: Iterate through rows of A and C (i from 0 to M-1)
    li t1, 0              # t1 = row counter 'i'
i_loop_6x6_6x2:
    # Calculate &C[i][0] and store in s8
    mul t2, t1, s7        # t2 = i * row_stride_C (8 bytes)
    add s8, s2, t2        # s8 = &C[i][0]

    # Calculate &A[i][0]
    mul t2, t1, s6        # t2 = i * row_stride_A (24 bytes)
    add t6, s0, t2        # t6 = &A[i][0] (pointer to current row of A)

    # Initialize vector accumulator v_acc (e.g., v24) to zeros for C[i][*]
    # This will hold the 2 elements of a row of C.
    vmv.v.i v24, 0        # v24 = {0.0, 0.0}

    # Inner loop: Iterate for k (summation dimension from 0 to K-1)
    li t3, 0              # t3 = 'k' counter (cols of A / rows of B)
k_loop_6x6_6x2:
    # Load A[i][k] (scalar)
    slli t2, t3, 2        # t2 = k * sizeof(float) = k * 4
    add t2, t6, t2        # t2 = &A[i][k]
    flw fs0, 0(t2)        # fs0 = A[i][k]

    # Broadcast A[i][k] (fs0) to a vector register (e.g., v0)
    # VL is already set to 2.
    vfmv.v.f v0, fs0      # v0 = {A[i][k], A[i][k]}

    # Calculate &B[k][0]
    lw t5, 4(sp)          # Load B_row_stride (8 bytes) into t5
    mul t2, t3, t5        # t2 = k * row_stride_B (8 bytes)
    add t2, s1, t2        # t2 = &B[k][0]

    # Load B[k][*] (vector B[k][0...N-1]) into vector register (e.g., v1)
    # VL is 2, SEW=32.
    vle32.v v1, (t2)      # v1 = {B[k][0], B[k][1]}

    # Vector multiply-accumulate: v24 = v24 + (v0 * v1)
    vfmacc.vv v24, v0, v1

    addi t3, t3, 1        # k++
    blt t3, s4, k_loop_6x6_6x2 # Loop if k < K (6)

    # Store the resulting row vector C[i][*] from v24 to memory at &C[i][0] (s8)
    # VL is 2, SEW=32.
    vse32.v v24, (s8)

    addi t1, t1, 1        # i++
    blt t1, s3, i_loop_6x6_6x2 # Loop if i < M (6)

    # Epilogue
    lw s8, 0(sp)
    lw s7, 4(sp)          # This will be B_row_stride that was saved
    lw s6, 8(sp)
    lw s5, 12(sp)
    lw s4, 16(sp)
    lw s3, 20(sp)
    lw s2, 24(sp)
    lw s1, 28(sp)
    lw s0, 32(sp)
    lw ra, 36(sp)
    addi sp, sp, 40       # Restore stack pointer
    ret

# Matrix A: 6x2, Matrix B: 2x6, Matrix C: 6x6 
#-------------------------------------------------------------------------
# Function: multiply_6x2_2x6
# Matrix multiplication C = A * B
# A is 6x2, B is 2x6, C is 6x6
# Computes one row of C at a time using vector instructions.
# a0: Address of matrix A (6x2)
# a1: Address of matrix B (2x6)
# a2: Address of result matrix C (6x6)
# Assumptions: Matrices are stored in row-major order. Elements are single-precision floats (4 bytes).
#-------------------------------------------------------------------------
multiply_6x2_2x6:
    addi sp, sp, -40      # Adjust stack pointer for saved registers
    sw ra, 36(sp)         # Save return address
    sw s0, 32(sp)         # Base address of A
    sw s1, 28(sp)         # Base address of B
    sw s2, 24(sp)         # Base address of C
    sw s3, 20(sp)         # Outer loop limit M (6 for A rows, C rows)
    sw s4, 16(sp)         # Inner loop limit K (2 for A cols / B rows)
    sw s5, 12(sp)         # Columns of B / C (N = 6)
    sw s6, 8(sp)          # Row stride for A in bytes (2 cols * 4 bytes/col = 8)
    sw s7, 4(sp)          # Row stride for B and C in bytes (6 cols * 4 bytes/col = 24)
    sw s8, 0(sp)          # Pointer to current row of C (&C[i][0])

    mv s0, a0             # s0 = Address of A
    mv s1, a1             # s1 = Address of B
    mv s2, a2             # s2 = Address of C

    li s3, 6              # M = 6 (rows of A, rows of C)
    li s4, 2              # K = 2 (cols of A, rows of B) - Inner loop limit
    li s5, 6              # N = 6 (cols of B, cols of C) - Vector Length for loads/stores of B and C rows

    li s6, 8              # Row stride for A = K * sizeof(float) = 2 * 4 = 8 bytes
    li s7, 24             # Row stride for B and C = N * sizeof(float) = 6 * 4 = 24 bytes

    # Set vector configuration for rows of B and C:
    # VL = N (6), SEW = 32 bits (e32), LMUL = 1 (m1)
    # vsetvli rd, avl, sew, lmul, ta, ma
    vsetvli t0, s5, e32, m1, ta, ma # t0 will hold actual VL (should be 6)

    # Outer loop: Iterate through rows of A and C (i from 0 to M-1)
    li t1, 0              # t1 = row counter 'i'
i_loop_6x2_2x6:
    # Calculate &C[i][0] and store in s8
    mul t2, t1, s7        # t2 = i * row_stride_C (24 bytes)
    add s8, s2, t2        # s8 = &C[i][0]

    # Calculate &A[i][0]
    mul t2, t1, s6        # t2 = i * row_stride_A (8 bytes)
    add t6, s0, t2        # t6 = &A[i][0] (pointer to current row of A)

    # Initialize vector accumulator v_acc (e.g., v24) to zeros for C[i][*]
    # This will hold the 6 elements of a row of C.
    vmv.v.i v24, 0        # v24 = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0}

    # Inner loop: Iterate for k (summation dimension from 0 to K-1)
    li t3, 0              # t3 = 'k' counter (cols of A / rows of B)
k_loop_6x2_2x6:
    # Load A[i][k] (scalar)
    slli t4, t3, 2        # t4 = k * sizeof(float) = k * 4
    add t4, t6, t4        # t4 = &A[i][k]
    flw fs0, 0(t4)        # fs0 = A[i][k]

    # Broadcast A[i][k] (fs0) to a vector register (e.g., v0)
    # VL is already set to 6.
    vfmv.v.f v0, fs0      # v0 = {A[i][k], A[i][k], ..., A[i][k]} (6 times)

    # Calculate &B[k][0]
    mul t5, t3, s7        # t5 = k * row_stride_B (24 bytes)
    add t5, s1, t5        # t5 = &B[k][0]

    # Load B[k][*] (vector B[k][0...N-1]) into vector register (e.g., v1)
    # VL is 6, SEW=32.
    vle32.v v1, (t5)      # v1 = {B[k][0], B[k][1], ..., B[k][5]}

    # Vector multiply-accumulate: v24 = v24 + (v0 * v1)
    # vfmacc.vv vd, vs1, vs2  (vd = vd + vs1*vs2)
    # vs1 is v0 (A_broadcast), vs2 is v1 (B_row)
    vfmacc.vv v24, v0, v1

    addi t3, t3, 1        # k++
    blt t3, s4, k_loop_6x2_2x6 # Loop if k < K (2)

    # Store the resulting row vector C[i][*] from v24 to memory at &C[i][0] (s8)
    # VL is 6, SEW=32.
    vse32.v v24, (s8)

    addi t1, t1, 1        # i++
    blt t1, s3, i_loop_6x2_2x6 # Loop if i < M (6)

    # Epilogue
    lw s8, 0(sp)
    lw s7, 4(sp)
    lw s6, 8(sp)
    lw s5, 12(sp)
    lw s4, 16(sp)
    lw s3, 20(sp)
    lw s2, 24(sp)
    lw s1, 28(sp)
    lw s0, 32(sp)
    lw ra, 36(sp)
    addi sp, sp, 40       # Restore stack pointer
    ret

# Fully optimized version for multiply_6x6_6x6 using vector instructions
#-------------------------------------------------------------------------
# Function: multiply_6x6_6x6
# Matrix multiplication C = A * B for 6x6 matrices using vector instructions.
# Computes one row of C at a time.
# a0: Address of matrix A (6x6)
# a1: Address of matrix B (6x6)
# a2: Address of result matrix C (6x6)
#-------------------------------------------------------------------------
multiply_6x6_6x6:
    addi sp, sp, -32
    sw ra, 28(sp)
    sw s0, 24(sp)         # Base address of A
    sw s1, 20(sp)         # Base address of B
    sw s2, 16(sp)         # Base address of C
    sw s3, 12(sp)         # Loop limit N (6)
    sw s4, 8(sp)          # Row stride in bytes (24)
    sw s5, 4(sp)          # Pointer to current row of A (&A[i][0])
    sw s6, 0(sp)          # Pointer to current row of C (&C[i][0])

    mv s0, a0                # s0 = Address of A
    mv s1, a1                # s1 = Address of B
    mv s2, a2                # s2 = Address of C
    
    li s3, 6                 # Matrix dimension N = 6
    li s4, 24                # Row stride = N * sizeof(float) = 6 * 4 = 24 bytes

    # Set vector configuration:
    # VL = 6 (as s3 is 6), SEW = 32 bits (e32), LMUL = 1 (m1)
    # Tail agnostic (ta), Mask agnostic (ma) - typical defaults
    # rd for vsetvli is t0, which will hold the actual VL.
    # rs1 is the requested vector length (AVL), here we want to process N elements.
    vsetvli t0, s3, e32, m1, ta, ma

    # Outer loop: Iterate through rows of A and C (i)
    li t1, 0                 # t1 will be our row counter 'i'
i_loop_vec:
    # Calculate &C[i][0] and store in s6
    mul t2, t1, s4           # t2 = i * row_stride
    add s6, s2, t2           # s6 = &C[i][0]

    # Calculate &A[i][0] and store in s5
    #mul t2, t1, s4          # t2 = i * row_stride (already calculated for C, can reuse if t1 is 'i')
    add s5, s0, t2           # s5 = &A[i][0]

    # Initialize vector accumulator v_acc (e.g., v24) to zeros for C[i][*]
    # The vsetvli above already set VL for 6 elements, SEW=32.
    # vmv.v.i instruction splats an immediate into a vector register.
    vmv.v.i v24, 0           # v24 = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0}

    # Inner loop: Iterate for k (summation dimension from 0 to N-1)
    li t3, 0                 # t3 will be our 'k' counter
k_loop_vec:
    # Load A[i][k] (scalar)
    slli t4, t3, 2           # t4 = k * sizeof(float) = k * 4
    add t4, s5, t4           # t4 = &A[i][k]
    flw fs0, 0(t4)           # fs0 = A[i][k]

    # Broadcast A[i][k] (fs0) to a vector register (e.g., v0)
    vfmv.v.f v0, fs0         # v0 = {A[i][k], A[i][k], ..., A[i][k]}

    # Calculate &B[k][0]
    mul t5, t3, s4           # t5 = k * row_stride
    add t5, s1, t5           # t5 = &B[k][0]

    # Load B[k][*] (vector B[k][0...5]) into vector register (e.g., v1)
    # vle32.v loads SEW=32 bit elements. Our VL is 6.
    vle32.v v1, (t5)         # v1 = {B[k][0], B[k][1], ..., B[k][5]}

    # Vector multiply-accumulate: v24 = v24 + (v0 * v1)
    # vfmacc.vv vd, vs1, vs2 means vd[i] = +(vs1[i] * vs2[i]) + vd[i]
    # Here, vs1 is v0 (A_broadcast), vs2 is v1 (B_row)
    vfmacc.vv v24, v0, v1

    addi t3, t3, 1           # k++
    blt t3, s3, k_loop_vec   # Loop if k < N

    # Store the resulting row vector C[i][*] from v24 to memory at &C[i][0]
    # vse32.v stores SEW=32 bit elements. Our VL is 6.
    vse32.v v24, (s6)

    addi t1, t1, 1           # i++
    blt t1, s3, i_loop_vec   # Loop if i < N

    # Epilogue
    lw s6, 0(sp)
    lw s5, 4(sp)
    lw s4, 8(sp)
    lw s3, 12(sp)
    lw s2, 16(sp)
    lw s1, 20(sp)
    lw s0, 24(sp)
    lw ra, 28(sp)
    addi sp, sp, 32
    ret

#-------------------------------------------------------------------------
# Function: transpose_6x6
# Transposes a 6x6 matrix: B = A^T
# a0: Address of matrix A
# a1: Address of result matrix B
#-------------------------------------------------------------------------
transpose_6x6:
    addi sp, sp, -16
    sw ra, 12(sp)
    sw s0, 8(sp)
    sw s1, 4(sp)
    
    mv s0, a0      # Address of A
    mv s1, a1      # Address of B
    
    # For each element B[j][i] = A[i][j]
    addi t0, zero, 0       # i = 0
    
i_loop_trans:
    addi t1, zero, 0       # j = 0
    
j_loop_trans:
    # Calculate address for A[i][j]
    addi t2, zero, 6       # Matrix width
    mul t3, t0, t2 # i * width
    add t3, t3, t1 # i * width + j
    slli t3, t3, 2 # (i * width + j) * 4
    add t3, s0, t3 # A + (i * width + j) * 4
    
    # Calculate address for B[j][i]
    mul t4, t1, t2 # j * width
    add t4, t4, t0 # j * width + i
    slli t4, t4, 2 # (j * width + i) * 4
    add t4, s1, t4 # B + (j * width + i) * 4
    
    # Load from A[i][j] and store to B[j][i]
    flw fs0, 0(t3)
    fsw fs0, 0(t4)
    
    addi t1, t1, 1 # j++
    addi t5, zero, 6
    blt t1, t5, j_loop_trans
    
    addi t0, t0, 1 # i++
    addi t5, zero, 6
    blt t0, t5, i_loop_trans
    
    lw ra, 12(sp)
    lw s0, 8(sp)
    lw s1, 4(sp)
    addi sp, sp, 16
    ret

#-------------------------------------------------------------------------
# Function: transpose_2x6_to_6x2
# Transposes a 2x6 matrix into a 6x2 matrix: B = A^T
# a0: Address of matrix A (2x6)
# a1: Address of result matrix B (6x2)
#-------------------------------------------------------------------------
transpose_2x6_to_6x2:
    addi sp, sp, -16
    sw ra, 12(sp)
    sw s0, 8(sp)
    sw s1, 4(sp)
    
    mv s0, a0      # Address of A
    mv s1, a1      # Address of B
    
    # For each element B[j][i] = A[i][j]
    addi t0, zero, 0       # i = 0
    
i_loop_trans_2x6:
    addi t1, zero, 0       # j = 0
    
j_loop_trans_2x6:
    # Calculate address for A[i][j]
    addi t2, zero, 6       # A matrix width
    mul t3, t0, t2 # i * A_width
    add t3, t3, t1 # i * A_width + j
    slli t3, t3, 2 # (i * A_width + j) * 4
    add t3, s0, t3 # A + (i * A_width + j) * 4
    
    # Calculate address for B[j][i]
    addi t4, zero, 2       # B matrix width
    mul t5, t1, t4 # j * B_width
    add t5, t5, t0 # j * B_width + i
    slli t5, t5, 2 # (j * B_width + i) * 4
    add t5, s1, t5 # B + (j * B_width + i) * 4
    
    # Load from A[i][j] and store to B[j][i]
    flw fs0, 0(t3)
    fsw fs0, 0(t5)
    
    addi t1, t1, 1 # j++
    addi t6, zero, 6
    blt t1, t6, j_loop_trans_2x6
    
    addi t0, t0, 1 # i++
    addi t6, zero, 2
    blt t0, t6, i_loop_trans_2x6
    
    lw ra, 12(sp)
    lw s0, 8(sp)
    lw s1, 4(sp)
    addi sp, sp, 16
    ret

#-------------------------------------------------------------------------
# Function: invert_2x2
# Inverts a 2x2 matrix with regularization
# a0: Address of matrix A (2x2)
# a1: Address of result matrix A_inv (2x2)
#-------------------------------------------------------------------------
invert_2x2:
    addi sp, sp, -16
    sw ra, 12(sp)
    sw s0, 8(sp)
    sw s1, 4(sp)
    
    mv s0, a0      # Address of A
    mv s1, a1      # Address of A_inv
    
    # Load the elements of matrix A
    flw fs0, 0(s0)     # A[0][0]
    flw fs1, 4(s0)     # A[0][1]
    flw fs2, 8(s0)    # A[1][0]
    flw fs3, 12(s0)    # A[1][1]
    
    # Calculate determinant: det = A[0][0]*A[1][1] - A[0][1]*A[1][0]
    fmul.s fs4, fs0, fs3
    fmul.s fs5, fs1, fs2
    fsub.s fs6, fs4, fs5    # fs6 = det
    
    # Handle singularity: if(fsbs(det) < 1e-7)
    fabs.s fs7, fs6        # fs6 is the determinant
    la t4, epsilon_val     
    flw fs5, 0(t4)
    flt.s a0, fs7, fs5     # if |det| < 1e-7
    beqz a0, invert_2x2_proceed
    
    # Set det to sign(det) * max(|det|, 1e-7)
    fcvt.s.w fs4, zero
    flt.s a0, fs6, fs4
    bnez a0, det_is_negative

det_is_positive:
    fmv.s fs6, fs5
    j invert_2x2_proceed
    
det_is_negative:
    fneg.s fs6, fs5
    
invert_2x2_proceed:
    # Calculate the inverse elements
    # A_inv[0][0] = A[1][1]/det
    fdiv.s fs0, fs3, fs6
    fsw fs0, 0(s1)
    
    # A_inv[0][1] = -A[0][1]/det
    fneg.s fs1, fs1
    fdiv.s fs1, fs1, fs6
    fsw fs1, 4(s1)
    
    # A_inv[1][0] = -A[1][0]/det
    fneg.s fs2, fs2
    fdiv.s fs2, fs2, fs6
    fsw fs2, 8(s1)
    
    # A_inv[1][1] = A[0][0]/det
    flw fs3, 0(s0)     # Reload A[0][0]
    fdiv.s fs3, fs3, fs6
    fsw fs3, 12(s1)
    
    lw ra, 12(sp)
    lw s0, 8(sp)
    lw s1, 4(sp)
    addi sp, sp, 16
    ret

#-------------------------------------------------------------------------
# Function: matrix_subtract_6x6
# Subtracts matrix B from matrix A element-wise: C = A - B
# a0: Address of matrix A (6x6)
# a1: Address of matrix B (6x6)
# a2: Address of result matrix C (6x6)
#-------------------------------------------------------------------------
matrix_subtract_6x6:
    addi sp, sp, -16
    sw ra, 12(sp)
    sw s0, 8(sp)
    sw s1, 4(sp)
    sw s2, 0(sp)
    
    mv s0, a0      # Address of A
    mv s1, a1      # Address of B
    mv s2, a2      # Address of C
    
    # For each element C[i][j] = A[i][j] - B[i][j]
    addi t0, zero, 0       # i = 0
    
i_loop_sub:
    addi t1, zero, 0       # j = 0
    
j_loop_sub:
    # Calculate addresses
    addi t2, zero, 6       # Matrix width
    mul t3, t0, t2 # i * width
    add t3, t3, t1 # i * width + j
    slli t3, t3, 2 # (i * width + j) * 4
    
    add t4, s0, t3 # A + (i * width + j) * 4
    add t5, s1, t3 # B + (i * width + j) * 4
    add t6, s2, t3 # C + (i * width + j) * 4
    
    # Load values
    flw fs0, 0(t4)
    flw fs1, 0(t5)
    
    # Subtract
    fsub.s fs2, fs0, fs1
    
    # Store result
    fsw fs2, 0(t6)
    
    addi t1, t1, 1 # j++
    addi s5, zero, 6
    blt t1, s5, j_loop_sub
    
    addi t0, t0, 1 # i++
    addi s5, zero, 6
    blt t0, s5, i_loop_sub
    
    lw ra, 12(sp)
    lw s0, 8(sp)
    lw s1, 4(sp)
    lw s2, 0(sp)
    addi sp, sp, 16
    ret

#-------------------------------------------------------------------------
# Function: matrix_add_6x6
# Adds matrices A and B element-wise: C = A + B
# a0: Address of matrix A (6x6)
# a1: Address of matrix B (6x6)
# a2: Address of result matrix C (6x6)
#-------------------------------------------------------------------------
matrix_add_6x6:
    addi sp, sp, -16
    sw ra, 12(sp)
    sw s0, 8(sp)
    sw s1, 4(sp)
    sw s2, 0(sp)
    
    mv s0, a0      # Address of A
    mv s1, a1      # Address of B
    mv s2, a2      # Address of C
    
    # For each element C[i][j] = A[i][j] + B[i][j]
    addi t0, zero, 0       # i = 0
    
i_loop_add:
    addi t1, zero, 0       # j = 0
    
j_loop_add:
    # Calculate addresses
    addi t2, zero, 6       # Matrix width
    mul t3, t0, t2 # i * width
    add t3, t3, t1 # i * width + j
    slli t3, t3, 2 # (i * width + j) * 4 (size of .float)
    
    add t4, s0, t3 # A + (i * width + j) * 4
    add t5, s1, t3 # B + (i * width + j) * 4
    add t6, s2, t3 # C + (i * width + j) * 4
    
    # Load values
    flw fs0, 0(t4)
    flw fs1, 0(t5)
    
    # Add
    fadd.s fs2, fs0, fs1
    
    # Store result
    fsw fs2, 0(t6)
    
    addi t1, t1, 1 # j++
    addi s5, zero, 6
    blt t1, s5, j_loop_add
    
    addi t0, t0, 1 # i++
    addi s5, zero, 6
    blt t0, s5, i_loop_add
    
    lw ra, 12(sp)
    lw s0, 8(sp)
    lw s1, 4(sp)
    lw s2, 0(sp)
    addi sp, sp, 16
    ret

#-------------------------------------------------------------------------
# Function: init_identity_matrix
# Initializes a 6x6 identity matrix
# a0: Address of matrix to initialize
#-------------------------------------------------------------------------
init_identity_matrix:
    addi sp, sp, -8
    sw ra, 4(sp)
    sw s0, 0(sp)
    
    mv s0, a0      # Matrix address
    
    # Zero out the matrix first
    addi t0, zero, 0       # i = 0
    
zero_loop:
    addi t1, zero, 0       # j = 0
    
zero_inner_loop:
    # Calculate address for matrix[i][j]
    addi t2, zero, 6       # Matrix width
    mul t3, t0, t2 # i * width
    add t3, t3, t1 # i * width + j
    slli t3, t3, 2 # (i * width + j) * 4 (size of .float)
    add t3, s0, t3 # matrix + (i * width + j) * 4
    
    # Store 0.0
    fcvt.s.w fs0, zero
    fsw fs0, 0(t3)
    
    addi t1, t1, 1 # j++
    addi t4, zero, 6
    blt t1, t4, zero_inner_loop
    
    addi t0, t0, 1 # i++
    addi t4, zero, 6
    blt t0, t4, zero_loop
    
    # Set the diagonal elements to 1.0
    addi t0, zero, 0       # i = 0
    
diag_loop:
    # Calculate address for matrix[i][i]
    addi t1, zero, 6       # Matrix width
    mul t2, t0, t1 # i * width
    add t2, t2, t0 # i * width + i
    slli t2, t2, 2 # (i * width + i) * 4 (size of .float)
    add t2, s0, t2 # matrix + (i * width + i) * 4
    
    # Store 1.0
    addi t3, zero, 1
    fcvt.s.w fs0, t3
    fsw fs0, 0(t2)
    
    addi t0, t0, 1 # i++
    addi t4, zero, 6
    blt t0, t4, diag_loop
    
    lw ra, 4(sp)
    lw s0, 0(sp)
    addi sp, sp, 8
    ret

#-------------------------------------------------------------------------
# Function: kalman_filter_update
# Performs one Kalman filter update step
# a0: Address of measurement z (2 elements)
#-------------------------------------------------------------------------
kalman_filter_update:
    addi sp, sp, -24
    sw ra, 20(sp)
    sw s0, 16(sp)
    sw s1, 12(sp)
    sw s2, 8(sp)
    sw s3, 4(sp)
    sw s4, 0(sp)
    
    mv s0, a0      # Address of measurement z
    
    # --- Prediction Step ---
    # x_pred = F * x
    la a0, F
    la a1, x
    la a2, x_pred
    
    # Multiply F * x for each element
    addi t0, zero, 0       # i = 0
    
x_pred_loop:
    # Initialize sum to 0
    fcvt.s.w fs0, zero
    
    # For each j, sum += F[i][j] * x[j]
    addi t1, zero, 0       # j = 0
    
x_pred_inner_loop:
    # Calculate address for F[i][j]
    addi t2, zero, 6       # Matrix width
    mul t3, t0, t2 # i * width
    add t3, t3, t1 # i * width + j
    slli t3, t3, 2 # (i * width + j) * 4 (size of .float)
    add t3, a0, t3 # F + (i * width + j) * 4
    
    # Calculate address for x[j]
    slli t4, t1, 2 # j * 4 (size of .float)
    add t4, a1, t4 # x + j * 4
    
    # Load values
    flw fs1, 0(t3) # F[i][j]
    flw fs2, 0(t4) # x[j]
    
    # Multiply and accumulate
    fmadd.s fs0, fs1, fs2, fs0
    
    addi t1, t1, 1 # j++
    addi t5, zero, 6
    blt t1, t5, x_pred_inner_loop
    
    # Store result to x_pred[i]
    slli t5, t0, 2 # i * 4 (size of .float)
    add t5, a2, t5 # x_pred + i * 4
    fsw fs0, 0(t5)
    
    addi t0, t0, 1 # i++
    addi t5, zero, 6
    blt t0, t5, x_pred_loop
    
    # P_pred = F*P*F^T + Q
    # 1. Calculate F*P
    la a0, F
    la a1, P
    la a2, FP
    jal ra, multiply_6x6_6x6
    
    # 2. Calculate F_T = F^T
    la a0, F
    la a1, F_T
    jal ra, transpose_6x6
    
    # 3. Calculate FP*F_T = F*P*F^T
    la a0, FP
    la a1, F_T
    la a2, FPFT
    jal ra, multiply_6x6_6x6
    
    # 4. Calculate P_pred = FPFT + Q
    la a0, FPFT
    la a1, Q
    la a2, P_pred
    jal ra, matrix_add_6x6
    
    # Ensure P_pred is symmetric
    la a0, P_pred
    jal ra, enforce_symmetry
    
    # --- Update Step ---
    # Compute Kalman Gain
    # 1. Calculate H_T = H^T
    la a0, H
    la a1, H_T
    jal ra, transpose_2x6_to_6x2
    
    # 2. Calculate PHT = P_pred * H_T
    la a0, P_pred
    la a1, H_T
    la a2, PHT
    jal ra, multiply_6x6_6x2
    
    # 3. Calculate S = H*P_pred*H^T + R
    # We'll first calculate H*P_pred = (P_pred^T * H^T)^T
    la a0, P_pred
    la a1, H_T
    la a2, PHT      # Reusing PHT as temporary storage
    jal ra, multiply_6x6_6x2
    
    # Now compute S = H*PHT + R
    # S[0][0] = R[0][0] + H[0][0]*PHT[0][0] + H[0][1]*PHT[1][0] + ... + H[0][5]*PHT[5][0]
    # S[0][1] = R[0][1] + H[0][0]*PHT[0][1] + H[0][1]*PHT[1][1] + ... + H[0][5]*PHT[5][1]
    # S[1][0] = R[1][0] + H[1][0]*PHT[0][0] + H[1][1]*PHT[1][0] + ... + H[1][5]*PHT[5][0]
    # S[1][1] = R[1][1] + H[1][0]*PHT[0][1] + H[1][1]*PHT[1][1] + ... + H[1][5]*PHT[5][1]
    
    la t0, R
    la t1, H
    la t2, PHT
    la t3, S
    
    # S[0][0]
    flw fs0, 0(t0)      # R[0][0]
    addi t4, zero, 0            # j = 0
S_00_loop:
    slli t5, t4, 3      # j * 8
    add t5, t2, t5      # PHT + j*4 = &PHT[j][0]
    flw fs1, 0(t5)      # PHT[j][0]
    slli t6, t4, 2      # j * 4
    add t6, t1, t6      # H + j*4 = &H[0][j] 
    flw fs2, 0(t6)      # H[0][j]
    fmadd.s fs0, fs1, fs2, fs0
    addi t4, t4, 1
    addi s5, zero, 6
    blt t4, s5, S_00_loop
    fsw fs0, 0(t3)      # S[0][0]
    
    # S[0][1]
    flw fs0, 4(t0)      # R[0][1]
    addi t4, zero, 0            # j = 0
S_01_loop:
    slli t5, t4, 3      # j * 8
    add t5, t2, t5      # PHT + j*4
    flw fs1, 4(t5)      # PHT[j][1]
    slli t6, t4, 2      # j * 4
    add t6, t1, t6      # H + j*4
    flw fs2, 0(t6)      # H[0][j]
    fmadd.s fs0, fs1, fs2, fs0
    addi t4, t4, 1
    addi s5, zero, 6
    blt t4, s5, S_01_loop
    fsw fs0, 4(t3)      # S[0][1]
    
    # S[1][0]
    flw fs0, 8(t0)     # R[1][0]
    addi t4, zero, 0            # j = 0
S_10_loop:
    slli t5, t4, 3      # j * 8
    add t5, t2, t5      # PHT + j*4
    flw fs1, 0(t5)      # PHT[j][0]
    slli t6, t4, 2      # j * 4
    add t6, t1, t6      # H + j*4
    flw fs2, 24(t6)     # H[1][j] (offset = 6*8 + j*4)
    fmadd.s fs0, fs1, fs2, fs0
    addi t4, t4, 1
    addi s5, zero, 6
    blt t4, s5, S_10_loop
    fsw fs0, 8(t3)     # S[1][0]
    
    # S[1][1]
    flw fs0, 12(t0)     # R[1][1]
    addi t4, zero, 0            # j = 0
S_11_loop:
    slli t5, t4, 3      # j * 8
    add t5, t2, t5      # PHT + j*4
    flw fs1, 4(t5)      # PHT[j][1]
    slli t6, t4, 2      # j * 4
    add t6, t1, t6      # H + j*4
    flw fs2, 24(t6)     # H[1][j] (offset = 6*8 + j*4)
    fmadd.s fs0, fs1, fs2, fs0
    addi t4, t4, 1
    addi s5, zero, 6
    blt t4, s5, S_11_loop
    fsw fs0, 12(t3)     # S[1][1]
    
    # 4. Calculate S_inv = S^-1
    la a0, S
    la a1, S_inv
    jal ra, invert_2x2
    
    # 5. Calculate K = PHT * S_inv
    la a0, PHT
    la a1, S_inv
    la a2, K
    
    # Manually compute K = PHT * S_inv for 6x2 * 2x2 = 6x2 matrix
    addi t0, zero, 0       # i = 0
K_loop:
    addi t1, zero, 0       # j = 0
K_inner_loop:
    # Initialize sum to 0
    fcvt.s.w fs0, zero
    
    # For each k, sum += PHT[i][k] * S_inv[k][j]
    addi t2, zero, 0       # k = 0
K_sum_loop:
    # Calculate address for PHT[i][k]
    addi t3, zero, 2       # PHT width
    mul t4, t0, t3 # i * width
    add t4, t4, t2 # i * width + k
    slli t4, t4, 2 # (i * width + k) * 4
    add t4, a0, t4 # PHT + (i * width + k) * 4
    
    # Calculate address for S_inv[k][j]
    mul t5, t2, t3 # k * width
    add t5, t5, t1 # k * width + j
    slli t5, t5, 2 # (k * width + j) * 4
    add t5, a1, t5 # S_inv + (k * width + j) * 4
    
    # Load values
    flw fs1, 0(t4) # PHT[i][k]
    flw fs2, 0(t5) # S_inv[k][j]
    
    # Multiply and accumulate
    fmadd.s fs0, fs1, fs2, fs0
    
    addi t2, t2, 1 # k++
    addi t6, zero, 2
    blt t2, t6, K_sum_loop
    
    # Store result to K[i][j]
    mul t6, t0, t3 # i * width
    add t6, t6, t1 # i * width + j
    slli t6, t6, 2 # (i * width + j) * 4
    add t6, a2, t6 # K + (i * width + j) * 4
    fsw fs0, 0(t6)
    
    addi t1, t1, 1 # j++
    addi s5, zero, 2
    blt t1, s5, K_inner_loop
    
    addi t0, t0, 1 # i++
    addi s5, zero, 6
    blt t0, s5, K_loop
    
    # Update state estimate
    # 1. Calculate innovation y = z - H*x_pred
    # y[0] = z[0] - x_pred[0] (as H[0][0]=1, all other H[0][j]=0)
    # y[1] = z[1] - x_pred[3] (as H[1][3]=1, all other H[1][j]=0)
    
    # Load z[0] and z[1]
    flw fs0, 0(s0)     # z[0]
    flw fs1, 4(s0)     # z[1]
    
    # Load x_pred[0] and x_pred[3]
    la t0, x_pred
    flw fs2, 0(t0)     # x_pred[0]
    flw fs3, 12(t0)    # x_pred[3]
    
    # Calculate y[0] = z[0] - x_pred[0] and y[1] = z[1] - x_pred[3]
    fsub.s fs4, fs0, fs2  # y[0]
    fsub.s fs5, fs1, fs3  # y[1]
    
    # 2. Update state: x = x_pred + K*y
    la t0, x_pred          # Address of x_pred
    la t1, K               # Address of K
    la t2, x               # Address of x (state)
    
    addi t3, zero, 0              # i = 0
x_update_loop:
    # Calculate K[i][0]*y[0] + K[i][1]*y[1]
    addi t4, zero, 2              # K width
    mul t5, t3, t4        # i * width
    slli t5, t5, 2        # (i * width) * 4
    add t5, t1, t5        # K + (i * width) * 4
    
    flw fs6, 0(t5)        # K[i][0]
    flw fs7, 4(t5)        # K[i][1]
    
    fmul.s fs6, fs6, fs4  # K[i][0] * y[0]
    fmul.s fs7, fs7, fs5  # K[i][1] * y[1]
    
    # Load x_pred[i]
    slli t6, t3, 2        # i * 4
    add t6, t0, t6        # x_pred + i * 4
    flw fs0, 0(t6)        # x_pred[i]
    
    # Calculate x[i] = x_pred[i] + K[i][0]*y[0] + K[i][1]*y[1]
    fadd.s fs6, fs6, fs7  # K[i][0]*y[0] + K[i][1]*y[1]
    fadd.s fs0, fs0, fs6  # x_pred[i] + (K[i][0]*y[0] + K[i][1]*y[1])
    
    # Store result in x[i]
    slli t6, t3, 2        # i * 4
    add t6, t2, t6        # x + i * 4
    fsw fs0, 0(t6)        # x[i]
    
    addi t3, t3, 1        # i++
    addi s5, zero, 6
    blt t3, s5, x_update_loop
    
    # 3. Update covariance: P = (I - KH)P_pred
    # a. Initialize I (identity matrix)
    la a0, I
    jal ra, init_identity_matrix
    
    # b. Calculate KH
    la a0, K
    la a1, H
    la a2, KH
    jal ra, multiply_6x2_2x6
    
    # c. Calculate I_KH = I - KH
    la a0, I
    la a1, KH
    la a2, I_KH
    jal ra, matrix_subtract_6x6
    
    # d. Calculate P = I_KH * P_pred
    la a0, I_KH
    la a1, P_pred
    la a2, P
    jal ra, multiply_6x6_6x6
    
    # Ensure P is symmetric
    la a0, P
    jal ra, enforce_symmetry
    
    lw ra, 20(sp)
    lw s0, 16(sp)
    lw s1, 12(sp)
    lw s2, 8(sp)
    lw s3, 4(sp)
    lw s4, 0(sp)
    addi sp, sp, 24
    ret
    
kalman_loop:
    # Call Kalman filter update with current measurement
    mv a0, s0
    jal ra, kalman_filter_update
    
    # Move to next measurement
    addi s0, s0, 8     # Each measurement is 2 .floats = 8 bytes
    addi s1, s1, 1      # Increment counter
    addi t0, zero, 35
    blt s1, t0, kalman_loop
    
    # Exit
    lw ra, 4(sp)
    lw s0, 0(sp)
    addi sp, sp, 8


    # Create file
    la a0, filename
    la a1, x
    li a2, 24  # 24 because 6 floats, each of size 4 byes
    call write_to_file

    j _finish
## END YOU CODE HERE


# Function: _finish
# VeeR Related function which writes to to_host which stops the simulator
_finish:
    li x3, 0xd0580000
    addi x5, x0, 0xff
    sb x5, 0(x3)
    beq x0, x0, _finish

    .rept 100
        nop
    .endr


.data
## ALL DATA IS DEFINED HERE LIKE MATRIX, CONSTANTS ETC

epsilon_val: .float 1.0e-7

filename: .string "output.hex"

# Constants
.align 3
DELTA_T:        .float 1.0       # Time step (1 second)
SIGMA_A:        .float 0.2       # Standard deviation for acceleration noise
SIGMA_M:        .float 3.0       # Standard deviation for measurement noise

# State vector (6D: [x, vx, ax, y, vy, ay])
.align 3
x:              .float 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

# Covariance matrix (6x6)
.align 3
P:              .float 500.0, 0.0, 0.0, 0.0, 0.0, 0.0
                .float 0.0, 500.0, 0.0, 0.0, 0.0, 0.0
                .float 0.0, 0.0, 500.0, 0.0, 0.0, 0.0
                .float 0.0, 0.0, 0.0, 500.0, 0.0, 0.0
                .float 0.0, 0.0, 0.0, 0.0, 500.0, 0.0
                .float 0.0, 0.0, 0.0, 0.0, 0.0, 500.0

# State transition matrix (6x6)
.align 3
F:              .float 1.0, 1.0, 0.5, 0.0, 0.0, 0.0       # Using DELTA_T=1.0
                .float 0.0, 1.0, 1.0, 0.0, 0.0, 0.0
                .float 0.0, 0.0, 1.0, 0.0, 0.0, 0.0
                .float 0.0, 0.0, 0.0, 1.0, 1.0, 0.5
                .float 0.0, 0.0, 0.0, 0.0, 1.0, 1.0
                .float 0.0, 0.0, 0.0, 0.0, 0.0, 1.0

# Process noise covariance (6x6)
.align 3
Q:              .space 144        # 6x6 matrix, 4 bytes per .float = 144 bytes

# Measurement matrix (2x6)
.align 3
H:              .float 1.0, 0.0, 0.0, 0.0, 0.0, 0.0
                .float 0.0, 0.0, 0.0, 1.0, 0.0, 0.0

# Measurement noise covariance (2x2)
.align 3
R:              .float 9.0, 0.0  # SIGMA_M^2 = 3.0^2 = 9.0
                .float 0.0, 9.0

# Temporary matrices used in calculations
.align 3
F_T:            .space 144        # 6x6 matrix
P_pred:         .space 144        # 6x6 matrix
FP:             .space 144        # 6x6 matrix
FPFT:           .space 144        # 6x6 matrix
H_T:            .space 48         # 6x2 matrix
PHT:            .space 48         # 6x2 matrix
S:              .space 16         # 2x2 matrix
S_inv:          .space 16         # 2x2 matrix
K:              .space 48         # 6x2 matrix
I:              .space 144        # 6x6 identity matrix
KH:             .space 144        # 6x6 matrix
I_KH:           .space 144        # 6x6 matrix
x_pred:         .space 24         # 6-element vector

# For measurements
.align 3
measurements:   .float 301.5, -401.46
                .float 298.23, -375.44
                .float 297.83, -346.15
                .float 300.42, -320.2
                .float 301.94, -300.08
                .float 299.5, -274.12
                .float 305.98, -253.45
                .float 301.25, -226.4
                .float 299.73, -200.65
                .float 299.2, -171.62
                .float 298.62, -152.11
                .float 301.84, -125.19
                .float 299.6, -93.4
                .float 295.3, -74.79
                .float 299.3, -49.12
                .float 301.95, -28.73
                .float 296.3, 2.99
                .float 295.11, 25.65
                .float 295.12, 49.86
                .float 289.9, 72.87
                .float 283.51, 96.34
                .float 276.42, 120.4
                .float 264.22, 144.69
                .float 250.25, 168.06
                .float 236.66, 184.99
                .float 217.47, 205.11
                .float 199.75, 221.82
                .float 179.7, 238.3
                .float 160, 253.02
                .float 140.92, 267.19
                .float 113.53, 270.71
                .float 93.68, 285.86
                .float 69.71, 288.48
                .float 45.93, 292.9
                .float 20.87, 298.77
## DATA DEFINE END
