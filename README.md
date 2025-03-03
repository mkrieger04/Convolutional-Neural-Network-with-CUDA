# Convolutional-Neural-Network-with-CUDA
Overview
This repository showcases my work on Milestone 3 of the ECE 408/CS483 course at the University of Illinois. The project involves the optimization of a convolutional neural network's forward pass using various techniques in CUDA. My goal was to improve the performance and efficiency of the forward convolution operation. The optimization strategies I implemented include CUDA streams, FP16 half precision, and constant memory. Below, you'll find a breakdown of the optimizations, their performance, and the profiling results.

Performance Metrics (Baseline)
Before implementing any optimizations, the following performance metrics were recorded for batch sizes of 100, 1k, and 5k images using the basic forward convolution kernel from Milestone 2:

Batch Size	Op Time 1	Op Time 2	Total Execution Time	Accuracy
100	0.19047 ms	0.668445 ms	0m1.716s	0.86
1000	1.72673 ms	6.42378 ms	0m11.377s	0.886
5000	8.48088 ms	36.7921 ms	0m51.027s	0.871
Optimization 1: CUDA Streams
Choice of Optimization
I chose to implement CUDA streams to optimize the host-side operations of the convolutional kernel. This optimization allows concurrent execution of data transfers and kernel launches, which maximizes GPU resource utilization and reduces execution time by overlapping computation with memory operations.

How the Optimization Works
By organizing the computation into separate CUDA streams for each batch, data transfers and kernel executions run concurrently. Asynchronous operations within each stream allow for overlap, reducing idle time and improving overall efficiency.

Profiling Results After Optimization
Batch Size	Op Time 1	Op Time 2	Total Execution Time	Accuracy
100	0.001216 ms	0.001395 ms	0m1.716s	0.86
1000	0.001872 ms	0.002327 ms	0m10.270s	0.886
5000	0.001987 ms	0.002455 ms	0m50.424s	0.871
Performance Comparison: This optimization led to a significant reduction in individual operation times. However, the total execution time showed little change, as the reduction in operation time was offset by the complexity of managing CUDA streams. Profiling results from Nsight show a 68.5% reduction in cudaMemcpy time.

Profiling Data (nsys)
cudaMemcpy: Reduced by 68.5%, down to 568,714,496 ns
conv_forward_kernel: Reduced to 41,206,285 ns, improving parallel processing efficiency
cudaMalloc: 17% reduction in time
cudaDeviceSynchronize: 3.9% reduction in time
References
CUDA Stream Optimization
Course Chatbot / Textbook
Optimization 2: FP16 Half Precision
Choice of Optimization
The second optimization employed was the use of FP16 half precision arithmetic to reduce memory usage and speed up computation. FP16 offers faster processing and reduces the overhead of handling 32-bit float data.

How the Optimization Works
The convolution kernel was modified to use FP16 arithmetic, converting the input data to half precision before computation and converting it back to float for output. This strategy aims to speed up convolution operations and improve memory efficiency without requiring additional data transfers.

Profiling Results After Optimization
Batch Size	Op Time 1	Op Time 2	Total Execution Time	Accuracy
100	11.7705 ms	10.0519 ms	0m1.664s	0.86
1000	2.13604 ms	8.26046 ms	0m10.718s	0.887
5000	36.9927 ms	9.54077 ms	0m55.143s	0.8712
Performance Comparison: Unfortunately, this optimization led to a slowdown in performance, as shown by profiling results. The cudaMemcpy time did not significantly improve, and memory throughput actually decreased from 77.37 GB/s to 59.8 GB/s.

Profiling Data (nsys and Nsight-Compute)
Memory Throughput: Dropped from 77.37 GB/s to 59.8 GB/s
conv_forward_kernel: Increased from 41,206,285 ns to 52,124,108 ns
cudaMemcpy: No significant improvement
References
FP16 Code for NVIDIA GPUs
Mixed Precision Training
Course Chatbot / Textbook
Optimization 3: Constant Memory
Choice of Optimization
The final optimization I implemented involves using constant memory to store the convolution mask, capitalizing on its read-only nature and faster access compared to global memory.

How the Optimization Works
Constant memory is utilized for the convolution filter, and the data is transferred using cudaMemcpyToSymbol. This allows the GPU to cache the data in a way that results in faster access during kernel execution.

Expected Performance Gain
By using constant memory, the convolutional kernel should experience faster data retrieval due to more efficient caching mechanisms, reducing latency and increasing bandwidth utilization.

Future Improvements
I plan to further optimize memory throughput by combining this technique with streaming and FP16 optimizations to reduce memory transfer times and improve overall performance.

Conclusion
The CUDA optimizations implemented in this project demonstrated mixed results. While streams showed considerable performance improvement, FP16 arithmetic did not yield the expected gains, and constant memory is still under investigation for further performance enhancements. The implementation of CUDA streams has already contributed to significant gains in parallelism, and I plan to further refine the FP16 and constant memory techniques to better suit the needs of convolutional neural networks.
