# GPU Project - Parallelization of Ford Fulkerson Algorithm

Serial and parallel versions of the code have been attached along with the files used to test correctness.

parallel_code_v3.cu code is best optimised and gives good speedup.

### To Generate Testcases
g++ generate_test_cases.cpp -std=c++11
./a.out > tc1

Modify the generate_test_cases.cpp to generate testcases with different number of vertices and edges.

### To Execute the Serial Code
g++ serial_code.cpp -std=c++11
./a.out < <input_file>

### To Execute the Parallel Codes
nvcc parallel_code_v1.cu
nvcc parallel_code_v2.cu
nvcc -arch=sm_35 -rdc=true parallel_code_v3.cu
./a.out < <input_file>
