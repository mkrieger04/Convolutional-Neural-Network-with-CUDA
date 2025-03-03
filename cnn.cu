#include <cmath>
#include <iostream>
#include "gpu-new-forward.h"

#define TILE_WIDTH 20

__constant__ float const_mask[6000];

__global__ void conv_forward_kernel(float *output, const float *input, const float *mask, const int B, const int M, const int C, const int H, const int W, const int K, const int S)
{
    const int H_out = (H - K) / S + 1;
    const int W_out = (W - K) / S + 1;
    const int W_size = (W_out + TILE_WIDTH - 1) / TILE_WIDTH;

    #define out_4d(i3, i2, i1, i0) output[(i3) * (M * H_out * W_out) + (i2) * (H_out * W_out) + (i1) * (W_out) + i0]
    #define in_4d(i3, i2, i1, i0) input[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]
    #define mask_4d(i3, i2, i1, i0) const_mask[(i3) * (C * K * K) + (i2) * (K * K) + (i1) * (K) + i0]

    int m = blockIdx.x;
    int b = blockIdx.z;
    int h = (blockIdx.y / W_size) * TILE_WIDTH + threadIdx.y;
    int w = (blockIdx.y % W_size) * TILE_WIDTH + threadIdx.x;
    float acc = 0.0f;
    if((h < H_out) && (w < W_out)){
        for(int c = 0; c < C; c++){
            for(int p = 0; p < K; p++)
                for(int q = 0; q < K; q++)
                    acc += in_4d(b, c, (S * h) + p, q + (w * S)) * mask_4d(m, c, p, q);
        }
        out_4d(b, m, h, w) = acc;
    }

    #undef out_4d
    #undef in_4d
    #undef mask_4d
}
    
__host__ void GPUInterface::conv_forward_gpu_prolog(float *host_output, const float *host_input, const float *host_mask, float **device_output_ptr, float **device_input_ptr, float **device_mask_ptr, const int B, const int M, const int C, const int H, const int W, const int K, const int S)
{
    const int H_out = (H - K)/S + 1;
    const int W_out = (W - K)/S + 1;

    cudaMalloc((void**)device_output_ptr, B * M * H_out * W_out * sizeof(float));
    cudaMalloc((void**)device_input_ptr, B * C * H * W * sizeof(float));
    cudaMalloc((void**)device_mask_ptr, M * C * K * K * sizeof(float));

    cudaMemcpy(*device_output_ptr, host_output, B * M * H_out * W_out * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(*device_input_ptr, host_input, B * C * H * W * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(const_mask, host_mask, M * C * K * K * sizeof(float));
}


__host__ void GPUInterface::conv_forward_gpu(float *device_output, const float *device_input, const float *device_mask, const int B, const int M, const int C, const int H, const int W, const int K, const int S)
{
    const int H_out = (H - K) / S + 1;
    const int W_out = (W - K) / S + 1;
    
    int W_size = (W_out + TILE_WIDTH - 1) / TILE_WIDTH;
    int H_size = (H_out + TILE_WIDTH - 1) / TILE_WIDTH; 
    int Y = H_size * W_size; 
    
    dim3 blockDim(TILE_WIDTH, TILE_WIDTH, 1); 
    dim3 gridDim(M, Y, B); 

    conv_forward_kernel<<<gridDim, blockDim>>>(device_output, device_input, device_mask, B, M, C, H, W, K,S);
    cudaDeviceSynchronize();
}



__host__ void GPUInterface::conv_forward_gpu_epilog(float *host_output, float *device_output, float *device_input, float *device_mask, const int B, const int M, const int C, const int H, const int W, const int K, const int S)
{
     // Copy the output back to host
    const int H_out = (H - K) / S + 1;
    const int W_out = (W - K) / S + 1;
    cudaMemcpy(host_output, device_output, B * M * H_out * W_out * sizeof(float), cudaMemcpyDeviceToHost);
    // Free device memory
    cudaFree(device_output);
    cudaFree(device_input);
    cudaFree(device_mask);
}


__host__ void GPUInterface::get_device_properties()
{
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);

    for(int dev = 0; dev < deviceCount; dev++)
    {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, dev);

        std::cout<<"Device "<<dev<<" name: "<<deviceProp.name<<std::endl;
        std::cout<<"Computational capabilities: "<<deviceProp.major<<"."<<deviceProp.minor<<std::endl;
        std::cout<<"Max Global memory size: "<<deviceProp.totalGlobalMem<<std::endl;
        std::cout<<"Max Constant memory size: "<<deviceProp.totalConstMem<<std::endl;
        std::cout<<"Max Shared memory size per block: "<<deviceProp.sharedMemPerBlock<<std::endl;
        std::cout<<"Max threads per block: "<<deviceProp.maxThreadsPerBlock<<std::endl;
        std::cout<<"Max block dimensions: "<<deviceProp.maxThreadsDim[0]<<" x, "<<deviceProp.maxThreadsDim[1]<<" y, "<<deviceProp.maxThreadsDim[2]<<" z"<<std::endl;
        std::cout<<"Max grid dimensions: "<<deviceProp.maxGridSize[0]<<" x, "<<deviceProp.maxGridSize[1]<<" y, "<<deviceProp.maxGridSize[2]<<" z"<<std::endl;
        std::cout<<"Warp Size: "<<deviceProp.warpSize<<std::endl;
    }
}

// **************************************************************************************************************************************************************************
// **************************************************************** Input Channel Reduction: Atomics 2 Points ***************************************************************
// **************************************************************************************************************************************************************************
// #include <cmath>
// #include <iostream>
// #include "gpu-new-forward.h"

// #define TILE_WIDTH 16

// __global__ void conv_forward_kernel(float *output, const float *input, const float *mask, const int B, const int M, const int C, const int H, const int W, const int K, const int S)
// {
//     const int H_out = (H - K) / S + 1;
//     const int W_out = (W - K) / S + 1;
//     const int W_size = (W_out + TILE_WIDTH - 1) / TILE_WIDTH;

//     #define out_4d(i3, i2, i1, i0) output[(i3) * (M * H_out * W_out) + (i2) * (H_out * W_out) + (i1) * (W_out) + i0]
//     #define in_4d(i3, i2, i1, i0) input[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]
//     #define mask_4d(i3, i2, i1, i0) mask[(i3) * (C * K * K) + (i2) * (K * K) + (i1) * (K) + i0]

//     int m = blockIdx.x;
//     int b = blockIdx.z;
//     int h = (blockIdx.y / W_size) * TILE_WIDTH + threadIdx.y;
//     int w = (blockIdx.y % W_size) * TILE_WIDTH + threadIdx.x;
//     int c = threadIdx.z;
//     float acc = 0.0f;
    
//     if((h < H_out) && (w < W_out))
//     {
//             for(int p = 0; p < K; p++)
//                 for(int q = 0; q < K; q++)
//                     acc += in_4d(b, c, (S * h) + p, q + (w * S)) * mask_4d(m, c, p, q);
//         atomicAdd(&out_4d(b, m, h, w), acc);
//     }
//     #undef out_4d
//     #undef in_4d
//     #undef mask_4d
// }

    
// __host__ void GPUInterface::conv_forward_gpu_prolog( float *host_output, const float *host_input, const float *host_mask, float **device_output_ptr, float **device_input_ptr, float **device_mask_ptr, const int B, const int M, const int C, const int H, const int W, const int K, const int S)
// {
//     const int H_out = (H - K)/S + 1;
//     const int W_out = (W - K)/S + 1;

//     cudaMalloc((void**)device_output_ptr, B * M * H_out * W_out * sizeof(float));
//     cudaMalloc((void**)device_input_ptr, B * C * H * W * sizeof(float));
//     cudaMalloc((void**)device_mask_ptr, M * C * K * K * sizeof(float));

//     //cudaMemcpy(*device_output_ptr, host_output, B * M * H_out * W_out * sizeof(float), cudaMemcpyHostToDevice);
//     cudaMemcpy(*device_input_ptr, host_input, B * C * H * W * sizeof(float), cudaMemcpyHostToDevice);
//     cudaMemcpy(*device_mask_ptr, host_mask, M * C * K * K * sizeof(float), cudaMemcpyHostToDevice);
// }


// __host__ void GPUInterface::conv_forward_gpu(float *device_output, const float *device_input, const float *device_mask, const int B, const int M, const int C, const int H, const int W, const int K, const int S)
// {
//     const int H_out = (H - K) / S + 1;
//     const int W_out = (W - K) / S + 1;
    
//     int W_size = (W_out + TILE_WIDTH - 1) / TILE_WIDTH; // number of horizontal tiles per output map
//     int H_size = (H_out + TILE_WIDTH - 1) / TILE_WIDTH; // number of vertical tiles per output map
//     int Y = H_size * W_size; // total number of tiles per map
    
//     dim3 blockDim(TILE_WIDTH, TILE_WIDTH, C); // output tile for untiled code
//     dim3 gridDim(M, Y, B); //Changed one to batch size

//     conv_forward_kernel<<<gridDim, blockDim>>>(device_output, device_input, device_mask, B, M, C, H, W, K,S);
//     cudaDeviceSynchronize();
// }


// __host__ void GPUInterface::conv_forward_gpu_epilog(float *host_output, float *device_output, float *device_input, float *device_mask, const int B, const int M, const int C, const int H, const int W, const int K, const int S)
// {
//     const int H_out = (H - K) / S + 1;
//     const int W_out = (W - K) / S + 1;
//     cudaMemcpy(host_output, device_output, B * M * H_out * W_out * sizeof(float), cudaMemcpyDeviceToHost);
//     cudaFree(device_output);
//     cudaFree(device_input);
//     cudaFree(device_mask);
// }


// __host__ void GPUInterface::get_device_properties()
// {
//     int deviceCount;
//     cudaGetDeviceCount(&deviceCount);

//     for(int dev = 0; dev < deviceCount; dev++)
//     {
//         cudaDeviceProp deviceProp;
//         cudaGetDeviceProperties(&deviceProp, dev);

//         std::cout<<"Device "<<dev<<" name: "<<deviceProp.name<<std::endl;
//         std::cout<<"Computational capabilities: "<<deviceProp.major<<"."<<deviceProp.minor<<std::endl;
//         std::cout<<"Max Global memory size: "<<deviceProp.totalGlobalMem<<std::endl;
//         std::cout<<"Max Constant memory size: "<<deviceProp.totalConstMem<<std::endl;
//         std::cout<<"Max Shared memory size per block: "<<deviceProp.sharedMemPerBlock<<std::endl;
//         std::cout<<"Max threads per block: "<<deviceProp.maxThreadsPerBlock<<std::endl;
//         std::cout<<"Max block dimensions: "<<deviceProp.maxThreadsDim[0]<<" x, "<<deviceProp.maxThreadsDim[1]<<" y, "<<deviceProp.maxThreadsDim[2]<<" z"<<std::endl;
//         std::cout<<"Max grid dimensions: "<<deviceProp.maxGridSize[0]<<" x, "<<deviceProp.maxGridSize[1]<<" y, "<<deviceProp.maxGridSize[2]<<" z"<<std::endl;
//         std::cout<<"Warp Size: "<<deviceProp.warpSize<<std::endl;
//     }
// }
//
// **************************************************************************************************************************************************************************
// *********************************************************************** FP16 Optimization 4 Points ***********************************************************************
// **************************************************************************************************************************************************************************
// #include <cmath>
// #include <iostream>
// #include "gpu-new-forward.h"
// #include <cuda_fp16.h>

// #define TILE_WIDTH 16

// __global__ void conv_forward_kernel(float *output, const float *input, const float *mask, const int B, const int M, const int C, const int H, const int W, const int K, const int S)
// {
//     const int H_out = (H - K) / S + 1;
//     const int W_out = (W - K) / S + 1;
//     const int W_size = (W_out + TILE_WIDTH - 1) / TILE_WIDTH;

//     #define out_4d(i3, i2, i1, i0) output[(i3) * (M * H_out * W_out) + (i2) * (H_out * W_out) + (i1) * (W_out) + i0]
//     #define in_4d(i3, i2, i1, i0) input[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]
//     #define mask_4d(i3, i2, i1, i0) mask[(i3) * (C * K * K) + (i2) * (K * K) + (i1) * (K) + i0]

//     int m = blockIdx.x;
//     int b = blockIdx.z;
//     int h = (blockIdx.y / W_size) * TILE_WIDTH + threadIdx.y;
//     int w = (blockIdx.y % W_size) * TILE_WIDTH + threadIdx.x;
//     __half acc = 0.0f;

//     if ((h < H_out) && (w < W_out)) {
//         for (int c = 0; c < C; c++) {
//             for (int p = 0; p < K; p++) {
//                 for (int q = 0; q < K; q++) {
//                     // Convert input and mask to __half
//                     __half input_val = __float2half(in_4d(b, c, (S * h) + p, q + (w * S)));
//                     __half mask_val = __float2half(mask_4d(m, c, p, q));

//                     // Perform computation in __half
//                     acc += input_val * mask_val;
//                 }
//             }
//         }
//         // Store result back as float
//         out_4d(b, m, h, w) = __half2float(acc);
//     }

//     #undef out_4d
//     #undef in_4d
//     #undef mask_4d
// }
//   __host__ void GPUInterface::conv_forward_gpu_prolog(float *host_output, const float *host_input, const float *host_mask, float **device_output_ptr, float **device_input_ptr, float **device_mask_ptr, const int B, const int M, const int C, const int H, const int W, const int K, const int S)
// {
//     // Allocate memory and copy over the relevant data structures to the GPU
//     const int H_out = (H - K)/S + 1;
//     const int W_out = (W - K)/S + 1;

//     cudaMalloc((void**)device_output_ptr, B * M * H_out * W_out * sizeof(float));
//     cudaMalloc((void**)device_input_ptr, B * C * H * W * sizeof(float));
//     cudaMalloc((void**)device_mask_ptr, M * C * K * K * sizeof(float));

//     cudaMemcpy(*device_output_ptr, host_output, B * M * H_out * W_out * sizeof(float), cudaMemcpyHostToDevice);
//     cudaMemcpy(*device_input_ptr, host_input, B * C * H * W * sizeof(float), cudaMemcpyHostToDevice);
//     cudaMemcpy(*device_mask_ptr, host_mask, M * C * K * K * sizeof(float), cudaMemcpyHostToDevice);
// }


// __host__ void GPUInterface::conv_forward_gpu(float *device_output, const float *device_input, const float *device_mask, const int B, const int M, const int C, const int H, const int W, const int K, const int S)
// {
//     const int H_out = (H - K) / S + 1;
//     const int W_out = (W - K) / S + 1;
    
//     int W_size = (W_out + TILE_WIDTH - 1) / TILE_WIDTH; // number of horizontal tiles per output map
//     int H_size = (H_out + TILE_WIDTH - 1) / TILE_WIDTH; // number of vertical tiles per output map
//     int Y = H_size * W_size; // total number of tiles per map
    
//     dim3 blockDim(TILE_WIDTH, TILE_WIDTH, 1); // output tile for untiled code
//     dim3 gridDim(M, Y, B); //Changed one to batch size

//     conv_forward_kernel<<<gridDim, blockDim>>>(device_output, device_input, device_mask, B, M, C, H, W, K,S);
//     cudaDeviceSynchronize();
// }



// __host__ void GPUInterface::conv_forward_gpu_epilog(float *host_output, float *device_output, float *device_input, float *device_mask, const int B, const int M, const int C, const int H, const int W, const int K, const int S)
// {
//      // Copy the output back to host
//     const int H_out = (H - K) / S + 1;
//     const int W_out = (W - K) / S + 1;
//     cudaMemcpy(host_output, device_output, B * M * H_out * W_out * sizeof(float), cudaMemcpyDeviceToHost);
//     // Free device memory
//     cudaFree(device_output);
//     cudaFree(device_input);
//     cudaFree(device_mask);
// }


// __host__ void GPUInterface::get_device_properties()
// {
//     int deviceCount;
//     cudaGetDeviceCount(&deviceCount);

//     for(int dev = 0; dev < deviceCount; dev++)
//     {
//         cudaDeviceProp deviceProp;
//         cudaGetDeviceProperties(&deviceProp, dev);

//         std::cout<<"Device "<<dev<<" name: "<<deviceProp.name<<std::endl;
//         std::cout<<"Computational capabilities: "<<deviceProp.major<<"."<<deviceProp.minor<<std::endl;
//         std::cout<<"Max Global memory size: "<<deviceProp.totalGlobalMem<<std::endl;
//         std::cout<<"Max Constant memory size: "<<deviceProp.totalConstMem<<std::endl;
//         std::cout<<"Max Shared memory size per block: "<<deviceProp.sharedMemPerBlock<<std::endl;
//         std::cout<<"Max threads per block: "<<deviceProp.maxThreadsPerBlock<<std::endl;
//         std::cout<<"Max block dimensions: "<<deviceProp.maxThreadsDim[0]<<" x, "<<deviceProp.maxThreadsDim[1]<<" y, "<<deviceProp.maxThreadsDim[2]<<" z"<<std::endl;
//         std::cout<<"Max grid dimensions: "<<deviceProp.maxGridSize[0]<<" x, "<<deviceProp.maxGridSize[1]<<" y, "<<deviceProp.maxGridSize[2]<<" z"<<std::endl;
//         std::cout<<"Warp Size: "<<deviceProp.warpSize<<std::endl;
//     }
// }
//
// ***************************************************************************************************************************************************************************
// ***************************************************************** Constant Memory Optimization 0.5 Points *****************************************************************
// ***************************************************************************************************************************************************************************
// #include <cmath>
// #include <iostream>
// #include "gpu-new-forward.h"

// #define TILE_WIDTH 16

// __constant__ float const_mask[10000];

// __global__ void conv_forward_kernel(float *output, const float *input, const float *mask, const int B, const int M, const int C, const int H, const int W, const int K, const int S)
// {
//     const int H_out = (H - K) / S + 1;
//     const int W_out = (W - K) / S + 1;
//     const int W_size = (W_out + TILE_WIDTH - 1) / TILE_WIDTH;

//     #define out_4d(i3, i2, i1, i0) output[(i3) * (M * H_out * W_out) + (i2) * (H_out * W_out) + (i1) * (W_out) + i0]
//     #define in_4d(i3, i2, i1, i0) input[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]
//     #define mask_4d(i3, i2, i1, i0) const_mask[(i3) * (C * K * K) + (i2) * (K * K) + (i1) * (K) + i0]

//     int m = blockIdx.x;
//     int b = blockIdx.z;
//     int h = (blockIdx.y / W_size) * TILE_WIDTH + threadIdx.y;
//     int w = (blockIdx.y % W_size) * TILE_WIDTH + threadIdx.x;
//     float acc = 0.0f;
//     if((h < H_out) && (w < W_out)){
//         for(int c = 0; c < C; c++){
//             for(int p = 0; p < K; p++)
//                 for(int q = 0; q < K; q++)
//                     acc += in_4d(b, c, (S * h) + p, q + (w * S)) * mask_4d(m, c, p, q);
//         }
//         out_4d(b, m, h, w) = acc;
//     }
//     #undef out_4d
//     #undef in_4d
//     #undef mask_4d
// }
    
// __host__ void GPUInterface::conv_forward_gpu_prolog(float *host_output, const float *host_input, const float *host_mask, float **device_output_ptr, float **device_input_ptr, float **device_mask_ptr, const int B, const int M, const int C, const int H, const int W, const int K, const int S)
// {
//     const int H_out = (H - K)/S + 1;
//     const int W_out = (W - K)/S + 1;
//     cudaMalloc((void**)device_output_ptr, B * M * H_out * W_out * sizeof(float));
//     cudaMalloc((void**)device_input_ptr, B * C * H * W * sizeof(float));
//     cudaMalloc((void**)device_mask_ptr, M * C * K * K * sizeof(float));
//     cudaMemcpy(*device_output_ptr, host_output, B * M * H_out * W_out * sizeof(float), cudaMemcpyHostToDevice);
//     cudaMemcpy(*device_input_ptr, host_input, B * C * H * W * sizeof(float), cudaMemcpyHostToDevice);
//     cudaMemcpyToSymbol(const_mask, host_mask, M * C * K * K * sizeof(float));
// }
// __host__ void GPUInterface::conv_forward_gpu(float *device_output, const float *device_input, const float *device_mask, const int B, const int M, const int C, const int H, const int W, const int K, const int S)
// {
//     const int H_out = (H - K) / S + 1;
//     const int W_out = (W - K) / S + 1;
    
//     int W_size = (W_out + TILE_WIDTH - 1) / TILE_WIDTH;
//     int H_size = (H_out + TILE_WIDTH - 1) / TILE_WIDTH; 
//     int Y = H_size * W_size; 
    
//     dim3 blockDim(TILE_WIDTH, TILE_WIDTH, 1); 
//     dim3 gridDim(M, Y, B); 

//     conv_forward_kernel<<<gridDim, blockDim>>>(device_output, device_input, device_mask, B, M, C, H, W, K,S);
//     cudaDeviceSynchronize();
// }
// __host__ void GPUInterface::conv_forward_gpu_epilog(float *host_output, float *device_output, float *device_input, float *device_mask, const int B, const int M, const int C, const int H, const int W, const int K, const int S)
// {
//      // Copy the output back to host
//     const int H_out = (H - K) / S + 1;
//     const int W_out = (W - K) / S + 1;
//     cudaMemcpy(host_output, device_output, B * M * H_out * W_out * sizeof(float), cudaMemcpyDeviceToHost);
//     // Free device memory
//     cudaFree(device_output);
//     cudaFree(device_input);
//     cudaFree(device_mask);
// }


// __host__ void GPUInterface::get_device_properties()
// {
//     int deviceCount;
//     cudaGetDeviceCount(&deviceCount);

//     for(int dev = 0; dev < deviceCount; dev++)
//     {
//         cudaDeviceProp deviceProp;
//         cudaGetDeviceProperties(&deviceProp, dev);

//         std::cout<<"Device "<<dev<<" name: "<<deviceProp.name<<std::endl;
//         std::cout<<"Computational capabilities: "<<deviceProp.major<<"."<<deviceProp.minor<<std::endl;
//         std::cout<<"Max Global memory size: "<<deviceProp.totalGlobalMem<<std::endl;
//         std::cout<<"Max Constant memory size: "<<deviceProp.totalConstMem<<std::endl;
//         std::cout<<"Max Shared memory size per block: "<<deviceProp.sharedMemPerBlock<<std::endl;
//         std::cout<<"Max threads per block: "<<deviceProp.maxThreadsPerBlock<<std::endl;
//         std::cout<<"Max block dimensions: "<<deviceProp.maxThreadsDim[0]<<" x, "<<deviceProp.maxThreadsDim[1]<<" y, "<<deviceProp.maxThreadsDim[2]<<" z"<<std::endl;
//         std::cout<<"Max grid dimensions: "<<deviceProp.maxGridSize[0]<<" x, "<<deviceProp.maxGridSize[1]<<" y, "<<deviceProp.maxGridSize[2]<<" z"<<std::endl;
//         std::cout<<"Warp Size: "<<deviceProp.warpSize<<std::endl;
//     }
// }
//
// **************************************************************************************************************************************************************************
// ********************************************************************** STREAM OPTIMIZATION 4 Points **********************************************************************
// **************************************************************************************************************************************************************************
//
// #include <cmath>
// #include <iostream>
// #include "gpu-new-forward.h"

// #define TILE_WIDTH 16

// __global__ void conv_forward_kernel(float *output, const float *input, const float *mask, const int B, const int M, const int C, const int H, const int W, const int K, const int S)
// {
//     const int H_out = (H - K) / S + 1;
//     const int W_out = (W - K) / S + 1;
//     const int W_size = (W_out + TILE_WIDTH - 1) / TILE_WIDTH;

//     #define out_4d(i3, i2, i1, i0) output[(i3) * (M * H_out * W_out) + (i2) * (H_out * W_out) + (i1) * (W_out) + i0]
//     #define in_4d(i3, i2, i1, i0) input[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]
//     #define mask_4d(i3, i2, i1, i0) mask[(i3) * (C * K * K) + (i2) * (K * K) + (i1) * (K) + i0]

//     // Insert your GPU convolution kernel code here
//     int m = blockIdx.x;
//     int b = blockIdx.z;
//     int h = (blockIdx.y / W_size) * TILE_WIDTH + threadIdx.y;
//     int w = (blockIdx.y % W_size) * TILE_WIDTH + threadIdx.x;
//     float acc = 0.0f;
//     if((h < H_out) && (w < W_out)){
//         for(int c = 0; c < C; c++){
//             for(int p = 0; p < K; p++)
//                 for(int q = 0; q < K; q++)
//                     acc += in_4d(b, c, (S * h) + p, q + (w * S)) * mask_4d(m, c, p, q);
//         }
//         out_4d(b, m, h, w) = acc;
//     }

//     #undef out_4d
//     #undef in_4d
//     #undef mask_4d
// }
    
// __host__ void GPUInterface::conv_forward_gpu_prolog(float *host_output, const float *host_input, const float *host_mask, float **device_output_ptr, float **device_input_ptr, float **device_mask_ptr, const int B, const int M, const int C, const int H, const int W, const int K, const int S)
// {
//     int nStreams = B;

//     const int H_out = (H - K) / S + 1; //Output-Dim
//     const int W_out = (W - K) / S + 1;
    
//     int W_size = (W_out + TILE_WIDTH - 1) / TILE_WIDTH; // number of horizontal tiles per output map
//     int H_size = (H_out + TILE_WIDTH - 1) / TILE_WIDTH; // number of vertical tiles per output map
//     int Y = H_size * W_size; // total number of tiles per map
    
//     dim3 blockDim(TILE_WIDTH, TILE_WIDTH, 1); // output tile for untiled code
//     dim3 gridDim(M, Y, 1); //Changed one to batch size
    
//     //Used to distribute work across multiple streams
//     //Used to calculate offset of data across each stream
//     int input_size = C * H * W; // Size of input batch 
//     int output_size = M * H_out * W_out; // Size of output batch

//     cudaMalloc((void**)device_output_ptr, B * M * H_out * W_out * sizeof(float));
//     cudaMalloc((void**)device_input_ptr, B * C * H * W * sizeof(float));
//     cudaMalloc((void**)device_mask_ptr, M * C * K * K * sizeof(float));
    
//     cudaStream_t stream[nStreams]; // Declare Streams

//     for (int i = 0; i < nStreams; i++) // Create Streams
//         cudaStreamCreate(&stream[i]);
        
//     cudaMemcpy(*device_mask_ptr, host_mask, M * C * K * K * sizeof(float), cudaMemcpyHostToDevice);
       
//     for (int i = 0; i < nStreams; i++)
//     {
//         int input_offset = input_size * i; // Each batch of size total/nStreams processed per kernel call
//         int output_offset = output_size * i;

//         cudaMemcpyAsync((*device_input_ptr) + input_offset, (host_input) + input_offset, input_size * sizeof(float), cudaMemcpyHostToDevice, stream[i]);
//         conv_forward_kernel<<<gridDim, blockDim, 0, stream[i]>>>((*device_output_ptr) + output_offset, (*device_input_ptr) + input_offset, *device_mask_ptr, B, M, C, H, W, K, S);
//         cudaMemcpyAsync((host_output) + output_offset, (*device_output_ptr) + output_offset, output_size * sizeof(float), cudaMemcpyDeviceToHost, stream[i]);
//     }
//     cudaDeviceSynchronize();
        
//     for (int i = 0; i < nStreams; i++) // Destroy Streams
//         cudaStreamDestroy(stream[i]);

//     cudaFree(*device_output_ptr);
//     cudaFree(*device_input_ptr);
//     cudaFree(*device_mask_ptr);
// }


// __host__ void GPUInterface::conv_forward_gpu(float *device_output, const float *device_input, const float *device_mask, const int B, const int M, const int C, const int H, const int W, const int K, const int S)
// {
//     return;
// }



// __host__ void GPUInterface::conv_forward_gpu_epilog(float *host_output, float *device_output, float *device_input, float *device_mask, const int B, const int M, const int C, const int H, const int W, const int K, const int S)
// {
//     return;
// }


// __host__ void GPUInterface::get_device_properties()
// {
//     int deviceCount;
//     cudaGetDeviceCount(&deviceCount);

//     for(int dev = 0; dev < deviceCount; dev++)
//     {
//         cudaDeviceProp deviceProp;
//         cudaGetDeviceProperties(&deviceProp, dev);

//         std::cout<<"Device "<<dev<<" name: "<<deviceProp.name<<std::endl;
//         std::cout<<"Computational capabilities: "<<deviceProp.major<<"."<<deviceProp.minor<<std::endl;
//         std::cout<<"Max Global memory size: "<<deviceProp.totalGlobalMem<<std::endl;
//         std::cout<<"Max Constant memory size: "<<deviceProp.totalConstMem<<std::endl;
//         std::cout<<"Max Shared memory size per block: "<<deviceProp.sharedMemPerBlock<<std::endl;
//         std::cout<<"Max threads per block: "<<deviceProp.maxThreadsPerBlock<<std::endl;
//         std::cout<<"Max block dimensions: "<<deviceProp.maxThreadsDim[0]<<" x, "<<deviceProp.maxThreadsDim[1]<<" y, "<<deviceProp.maxThreadsDim[2]<<" z"<<std::endl;
//         std::cout<<"Max grid dimensions: "<<deviceProp.maxGridSize[0]<<" x, "<<deviceProp.maxGridSize[1]<<" y, "<<deviceProp.maxGridSize[2]<<" z"<<std::endl;
//         std::cout<<"Warp Size: "<<deviceProp.warpSize<<std::endl;
//     }
// }


// *******************************************************************************************************************************************************
// ********************************************************************** Base Code **********************************************************************
// *******************************************************************************************************************************************************
	// #include <cmath>
	// #include <iostream>
	// #include "gpu-new-forward.h"
	// #define TILE_WIDTH 16
	// __global__ void conv_forward_kernel(float *output, const float *input, const float *mask, const int B, const int M, const int C, const int H, const int W, const int K, const int S)
	// {
	//     const int H_out = (H - K) / S + 1;
	//     const int W_out = (W - K) / S + 1;
	//     const int W_size = (W_out + TILE_WIDTH - 1) / TILE_WIDTH;

	//     #define out_4d(i3, i2, i1, i0) output[(i3) * (M * H_out * W_out) + (i2) * (H_out * W_out) + (i1) * (W_out) + i0]
	//     #define in_4d(i3, i2, i1, i0) input[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]
	//     #define mask_4d(i3, i2, i1, i0) mask[(i3) * (C * K * K) + (i2) * (K * K) + (i1) * (K) + i0]

	//     // Insert your GPU convolution kernel code here
	//     int m = blockIdx.x;
	//     int b = blockIdx.z;
	//     int h = (blockIdx.y / W_size) * TILE_WIDTH + threadIdx.y;
	//     int w = (blockIdx.y % W_size) * TILE_WIDTH + threadIdx.x;
	//     float acc = 0.0f;
	//     if((h < H_out) && (w < W_out)){
	//         for(int c = 0; c < C; c++){
	//             for(int p = 0; p < K; p++)
	//                 for(int q = 0; q < K; q++)
	//                     acc += in_4d(b, c, (S * h) + p, q + (w * S)) * mask_4d(m, c, p, q);
	//         }
	//         out_4d(b, m, h, w) = acc;
	//     }

	//     #undef out_4d
	//     #undef in_4d
	//     #undef mask_4d
	// }

	    
	// __host__ void GPUInterface::conv_forward_gpu_prolog(float *host_output, const float *host_input, const float *host_mask, float **device_output_ptr, float **device_input_ptr, float **device_mask_ptr, const int B, const int M, const int C, const int H, const int W, const int K, const int S)
	// {
	//     // Allocate memory and copy over the relevant data structures to the GPU
	//     const int H_out = (H - K)/S + 1;
	//     const int W_out = (W - K)/S + 1;

	//     cudaMalloc((void**)device_output_ptr, B * M * H_out * W_out * sizeof(float));
	//     cudaMalloc((void**)device_input_ptr, B * C * H * W * sizeof(float));
	//     cudaMalloc((void**)device_mask_ptr, M * C * K * K * sizeof(float));

	//     cudaMemcpy(*device_output_ptr, host_output, B * M * H_out * W_out * sizeof(float), cudaMemcpyHostToDevice);
	//     cudaMemcpy(*device_input_ptr, host_input, B * C * H * W * sizeof(float), cudaMemcpyHostToDevice);
	//     cudaMemcpy(*device_mask_ptr, host_mask, M * C * K * K * sizeof(float), cudaMemcpyHostToDevice);   
	// }


	// __host__ void GPUInterface::conv_forward_gpu(float *device_output, const float *device_input, const float *device_mask, const int B, const int M, const int C, const int H, const int W, const int K, const int S)
	// {
	//     const int H_out = (H - K) / S + 1;
	//     const int W_out = (W - K) / S + 1;
	    
	//     int W_size = (W_out + TILE_WIDTH - 1) / TILE_WIDTH; // number of horizontal tiles per output map
	//     int H_size = (H_out + TILE_WIDTH - 1) / TILE_WIDTH; // number of vertical tiles per output map
	//     int Y = H_size * W_size; // total number of tiles per map
	    
	//     dim3 blockDim(TILE_WIDTH, TILE_WIDTH, 1); // output tile for untiled code
	//     dim3 gridDim(M, Y, B); //Changed one to batch size

	//     conv_forward_kernel<<<gridDim, blockDim>>>(device_output, device_input, device_mask, B, M, C, H, W, K,S);
	//     cudaDeviceSynchronize();
	// }



	// __host__ void GPUInterface::conv_forward_gpu_epilog(float *host_output, float *device_output, float *device_input, float *device_mask, const int B, const int M, const int C, const int H, const int W, const int K, const int S)
	// {
	//      // Copy the output back to host
	//     const int H_out = (H - K) / S + 1;
	//     const int W_out = (W - K) / S + 1;
	//     cudaMemcpy(host_output, device_output, B * M * H_out * W_out * sizeof(float), cudaMemcpyDeviceToHost);
	//     // Free device memory
	//     cudaFree(device_output);
	//     cudaFree(device_input);
	//     cudaFree(device_mask);
	// }


	// __host__ void GPUInterface::get_device_properties()
	// {
	//     int deviceCount;
	//     cudaGetDeviceCount(&deviceCount);

	//     for(int dev = 0; dev < deviceCount; dev++)
	//     {
	//         cudaDeviceProp deviceProp;
	//         cudaGetDeviceProperties(&deviceProp, dev);

	//         std::cout<<"Device "<<dev<<" name: "<<deviceProp.name<<std::endl;
	//         std::cout<<"Computational capabilities: "<<deviceProp.major<<"."<<deviceProp.minor<<std::endl;
	//         std::cout<<"Max Global memory size: "<<deviceProp.totalGlobalMem<<std::endl;
	//         std::cout<<"Max Constant memory size: "<<deviceProp.totalConstMem<<std::endl;
	//         std::cout<<"Max Shared memory size per block: "<<deviceProp.sharedMemPerBlock<<std::endl;
	//         std::cout<<"Max threads per block: "<<deviceProp.maxThreadsPerBlock<<std::endl;
	//         std::cout<<"Max block dimensions: "<<deviceProp.maxThreadsDim[0]<<" x, "<<deviceProp.maxThreadsDim[1]<<" y, "<<deviceProp.maxThreadsDim[2]<<" z"<<std::endl;
	//         std::cout<<"Max grid dimensions: "<<deviceProp.maxGridSize[0]<<" x, "<<deviceProp.maxGridSize[1]<<" y, "<<deviceProp.maxGridSize[2]<<" z"<<std::endl;
	//         std::cout<<"Warp Size: "<<deviceProp.warpSize<<std::endl;
	//     }
	// }

// #ifndef SRC_LAYER_GPU_NEW_FORWARD_H
// #define SRC_LAYER_GPU_NEW_FORWARD_H

// class GPUInterface
// {
//     public:
//     void get_device_properties();
//     void conv_forward_gpu_prolog(const float *host_output, const float *host_input, const float *host_mask, float **device_output_ptr, float **device_input_ptr, float **device_mask_ptr, const int B, const int M, const int C, const int H, const int W, const int K, const int S = 1);
//     void conv_forward_gpu(float *device_output, const float *device_input, const float *device_mask, const int B, const int M, const int C, const int H, const int W, const int K, const int S = 1);
//     void conv_forward_gpu_epilog(float *host_output, float *device_output, float *device_input, float *device_mask, const int B, const int M, const int C, const int H, const int W, const int K, const int S = 1);
// };
