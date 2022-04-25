/* This program will load images (uint16 buffers) from the sun (SDO AIA images)
* and runs a blur and edge detection kernel/stencil on it.
*
* BUT the created output array is smaller than the input.
* The outer 3 lines and columns on each side are cropped.
* This helps you with the border conditions of the shared memory loads.
* It is the same for all thread blocks.
*
*/

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
#include <fstream>
#include <assert.h>
#include <cmath>
#include <stdio.h>
#include <chrono>
#include <omp.h>
#include <immintrin.h>
#include <emmintrin.h>
#include <numeric>

using namespace std;

#define DIM_SIZE 4096
#define BLUR_KERNEL_SIZE 7
#define EDGE_KERNEL_SIZE 3
#define HALO_SIZE 3
#define NUMBER_FILES 6

__constant__ int BLUR_KERNEL[BLUR_KERNEL_SIZE * BLUR_KERNEL_SIZE];
__constant__ int EDGE_KERNEL[EDGE_KERNEL_SIZE * EDGE_KERNEL_SIZE];


// CUDA macro wrapper for checking errors
#define gpuErrCheck(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char* file, int line, bool abort = true)
{
    if (code != cudaSuccess)
    {
        std::cout << "GPUassert: " << cudaGetErrorString(code) << " " << file << " " << line << std::endl;
        if (abort)
        {
            exit(code);
        }
    }
}


// Compare result arrays CPU vs GPU result. If no diff, the result pass.
int compareResultVec(uint16_t* vectorCPU, uint16_t* vectorGPU, int size)
{
    int error = 0;
    for (int i = 0; i < size; i++)
    {
        error += abs(vectorCPU[i] - vectorGPU[i]);
    }
    if (error == 0)
    {
        cout << "Test passed." << endl;
        return 0;
    }
    else
    {
        cout << "Accumulated error: " << error << endl;
        return -1;
    }
}


void readFileIntoBuffer(std::string path, uint16_t* buffer)
{
    ifstream inp(path.c_str(), ios::binary);
    if (!inp) {
        assert(0, "could not open file");
        exit(-1);
    }

    inp.read((char*)buffer, DIM_SIZE * DIM_SIZE * sizeof(uint16_t));
    inp.close();
}


uint16_t* devInput() {
    uint16_t* raw = new uint16_t[DIM_SIZE * DIM_SIZE];
    for (int y = 0; y < DIM_SIZE; ++y) {
        for (int x = 0; x < DIM_SIZE; ++x) {
            raw[y * DIM_SIZE + x] = y * DIM_SIZE + x;
        }
    }
    return raw;
}


void host_processing(uint16_t* input, uint16_t* output, int* blur_kernel, int* edge_kernel) {

    // After applying the blur stencil, we have to divide by the
    // total of the blur stencil
    int blurFactor = 0;
    for (int z = 0; z < BLUR_KERNEL_SIZE * BLUR_KERNEL_SIZE; ++z)
        blurFactor += blur_kernel[z];

    // In order to keep the input buffer clean, we work with an intermediate
    // result buffer of the same size as the origin
    uint16_t* blurResult = new uint16_t[DIM_SIZE * DIM_SIZE];
    for (int i = 0; i < DIM_SIZE * DIM_SIZE; ++i)
        blurResult[i] = input[i];

    // Apply blur kernel to the image
    // We do not start at pixel (0,0) as we use the most outer pixels as halo elements
    for (int j = BLUR_KERNEL_SIZE / 2; j < DIM_SIZE - BLUR_KERNEL_SIZE / 2; ++j) {
        for (int i = BLUR_KERNEL_SIZE / 2; i < DIM_SIZE - BLUR_KERNEL_SIZE / 2; ++i) {

            int tmp = 0;

            // Iterate the blur stencil/kernel
            for (int k = -(BLUR_KERNEL_SIZE / 2); k <= BLUR_KERNEL_SIZE / 2; ++k) {
                for (int l = -(BLUR_KERNEL_SIZE / 2); l <= BLUR_KERNEL_SIZE / 2; ++l) {
                    auto kidx = (k + BLUR_KERNEL_SIZE / 2) * BLUR_KERNEL_SIZE + (l + BLUR_KERNEL_SIZE / 2);
                    tmp += input[(j + k) * DIM_SIZE + (i + l)] * blur_kernel[kidx];
                }
            }
            blurResult[j * DIM_SIZE + i] = (uint16_t)(tmp / blurFactor);
        }
    }

    // Apply edge kernel to the image
    for (int j = BLUR_KERNEL_SIZE / 2; j < DIM_SIZE - BLUR_KERNEL_SIZE / 2; ++j) {
        for (int i = BLUR_KERNEL_SIZE / 2; i < DIM_SIZE - BLUR_KERNEL_SIZE / 2; ++i) {

            int tmp = 0;

            // Iterate the edge stencil/kernel
            for (int k = -(EDGE_KERNEL_SIZE / 2); k <= EDGE_KERNEL_SIZE / 2; ++k) {
                for (int l = -(EDGE_KERNEL_SIZE / 2); l <= EDGE_KERNEL_SIZE / 2; ++l) {
                    auto kidx = (k + EDGE_KERNEL_SIZE / 2) * EDGE_KERNEL_SIZE + (l + EDGE_KERNEL_SIZE / 2);
                    tmp += blurResult[(j + k) * DIM_SIZE + (i + l)] * edge_kernel[kidx];
                }
            }
            output[(j - BLUR_KERNEL_SIZE / 2) * (DIM_SIZE - 6) + i - (BLUR_KERNEL_SIZE / 2)] = tmp < 0 ? 0 : (uint16_t)tmp;
        }
    }
}


int main(void)
{
    // Init data
    // ToDo: Copy aia_data_uint16_0.raw 5 times to aia_data_uint16_[1-5].raw
    const char* files[NUMBER_FILES] = {
        "aia_data_uint16_0.raw", "aia_data_uint16_1.raw", "aia_data_uint16_2.raw",
        "aia_data_uint16_3.raw", "aia_data_uint16_4.raw", "aia_data_uint16_5.raw"
    };

    // You can also choose the devIput which creates a simple matrix,
    // where you easily know the results per row after the blur and edge kernel step.
    //uint16_t* data = devInput();

    int blurKernel[64] = {
        0, 0, 1, 2, 1, 0, 0,
        0, 3, 13, 22, 13, 3, 0,
        1, 13, 59, 97, 59, 13, 1,
        2, 22, 97, 159, 97, 22, 2,
        1, 13, 59, 97, 59, 13, 1,
        0, 3, 13, 22, 13, 3, 0,
        0, 0, 1, 2, 1, 0, 0,
    };
    int edgeKernel[9] = {
        -1, -1, -1,
        -1, 8, -1,
        -1, -1, -1
    };


    // Do 1 image on the CPU first, you can use hostProcessedImage to 
    // compare the results of the CPU vs the GPU implementation
    auto startTime = chrono::high_resolution_clock::now();
    uint16_t* data = new uint16_t[DIM_SIZE * DIM_SIZE];
    uint16_t* hostProcessedImage = new uint16_t[(DIM_SIZE - 6) * (DIM_SIZE - 6)];
    readFileIntoBuffer(files[0], data);
    host_processing(data, hostProcessedImage, blurKernel, edgeKernel);
    auto endTime = chrono::high_resolution_clock::now();
    cout << "CPU Time for 1 image [ms]: " << chrono::duration_cast<chrono::milliseconds>(endTime - startTime).count() << endl;

    // All images on CPU
    startTime = chrono::high_resolution_clock::now();
    uint16_t* dataBuffer = new uint16_t[DIM_SIZE * DIM_SIZE];
    uint16_t* resultBuffer = new uint16_t[(DIM_SIZE - 6) * (DIM_SIZE - 6)];
    for (int i = 0; i < NUMBER_FILES; ++i) {
        // Reuse the buffer as we do nothing with the result
        readFileIntoBuffer(files[i], data);
        host_processing(dataBuffer, resultBuffer, blurKernel, edgeKernel);
    }
    endTime = chrono::high_resolution_clock::now();
    cout << "CPU Time for all images [ms]: " << chrono::duration_cast<chrono::milliseconds>(endTime - startTime).count() << endl;

    // Do your GPU magic

    // Free memory on host
    delete[] hostProcessedImage;
    delete[] data;
    delete[] dataBuffer;
    delete[] resultBuffer;

    return 0;
}
