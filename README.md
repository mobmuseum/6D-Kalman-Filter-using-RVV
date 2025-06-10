# 6D Kalman Filter using RISC-V Vector Extensions
This project implements a 6D Kalman Filter for estimating an object's position, velocity, and acceleration in 2D space using noisy position data. The core matrix computations are optimized using RISC-V Vector Extensions (RVV) to achieve significant speed-ups over scalar implementations.
The work is based on Alex Becker's foundational Kalman Filter tutorial and has been expanded and vectorized for embedded real-time applications.


## 📁 Project Structure


   - `assembly/`: Contains the RISC-V assembly code.
        - `Nonvectorized.s`: Non-vectorized implementation of Kalman filter in RISC-V assembly.
        - `Vectorized.s`:  Vectorized implementation of Kalman filter in RISC-V assembly.


   - `c/`: C code for Kalman filter.
        - `code.c`: Kalman filter implementation in C; base code for our conversion.

    
   - `python/`: Useful Python code(s).
        - `Hex-viewer.py`: The final estimate is stored in Output.hex. This code converts the .hex file to float.


- `veer/`: Contains necessary files for running the Veer simulator.
    - `link.ld`: Linker script for compilation.
    - `whisper.json`: Configuration file for the Veer simulator.
    - `tempFiles/`: Output files are stored in this folder after execution.


- `Makefile`: Script to compile and run simulations on the Veer simulator.


- `Report`: IEEE-standard project report.


## 🚀 Features


Refer to the report for an extensive overview of the features of the 6D Kalman filter.


## 🛠 Prerequisites


Before setting up and running the simulation, ensure you meet the following prerequisites:

- Linux (tested on Ubuntu 20.04)
- [RISC-V GNU Toolchain](https://github.com/riscv-collab/riscv-gnu-toolchain)
- [VeeR-ISS RISC-V Simulator](https://github.com/chipsalliance/VeeR-ISS)
- [Verilator](https://verilator.org/guide/latest/install.html) (Tested on Version 5.006)


## 🧪 Simulation Instructions


1. Clone this repository to your local machine:

```bash
git clone https://github.com/mobmuseum/6D-Kalman-Filter-using-RVV.git
cd 6D-Kalman-Filter-using-RVV
```


2. At the end of the assembly file, there should be a .data section. In the .data section, modify the measurements array as per your noisy dataset. You may also adjust the number of iterations based on your requirements.
3.  To run the simulation on the VeeR simulator, execute the following command:

```bash
make cleanV      (or make cleanNV for non-vectorized code)
make allV        (or make allNV for non-vectorized code)
```


This will compile the code, run it on the VeeR simulator, generate a log file with the results, and run **Hex-viewer.py** to convert the Hex values into floating point numbers. Ensure that you have Python 3.0+ installed on your system.


## 🤝 Contributions

- Aadesh Panjwani
- Muhammad Rayyan Khan
- Syed Ahmed Farrukh
- Mustafa Mehmood

We implemented and benchmarked both scalar and RVV-optimized Kalman Filters as part of our Computer Architecture and Assembly Language coursework at IBA Karachi.

## 🙏 Acknowledgements
- Special thanks to Teaching Assistant Abdul Wasay Imran & Dr. Salman Zaffar for technical guidance. 
- Inspired by Alex Becker’s Kalman Filter tutorial (2023).
- Based on the foundational work by Rudolf E. Kalman (1960).
