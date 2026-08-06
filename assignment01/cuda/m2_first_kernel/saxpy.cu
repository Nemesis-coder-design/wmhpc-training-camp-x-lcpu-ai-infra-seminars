#include<cuda_runtime.h>
#include<cstdlib>
#include<cstdint>
#include<iostream>
#include<cmath>
#include<iomanip>

//不允许 include common.h。错误检查宏和 cudaEvent 计时都要自己写一遍。
//命令行用法：./saxpy <n>，n是元素个数。输入数据按固定公式生成（都是 float）
// x[i]= ((i % 2048) - 1024) * 0.5f，y[i] = (i % 1024) - 512。
//kernel 算完把 y拷回 host，用 double 累加所有 y[i]，输出一行 SUM=< 总和 >（用printf("SUM=%.0f\n", s) 这样的格式，
// 同一行里可以再带上n和kernel 毫秒数），SUM结果将用于对拍检验程序正确性，exit code 应为 0。

//3 SIMT 执行 6• n= 0 时输出 SUM=0，exit code 为 0（0 个 block 的 kernel launch 是非法的，特判即可）。

#define CUDA_CHECK(call)                                      \
do{                                                           \
    cudaError_t err=(call);                                   \
    if(err!=cudaSuccess){                                     \
        fprintf(stderr,"CUDA error at %s:%d: %s\n",           \
                __FILE__,__LINE__,cudaGetErrorString(err));   \
        exit(1);                                              \
    }                                                         \
}while(0)



__global__ void saxpy(float *x,float *y,float *z,unsigned int N){
    unsigned int i=threadIdx.x+blockDim.x*blockIdx.x;
    if(i<N) z[i]=2.0f*x[i]+y[i];
}




int main(int argc,char **argv){
    if(argc!=2){
        std::cerr<<"Usage: "<<argv[0]<<" <n>"<<'\n';
        return 1;
    }
    size_t n=(size_t)strtoull(argv[1],NULL,10);
    if(n==0){
        std::cout<<"SUM=0\n";
        return 0;
    }
    size_t bytes=n*sizeof(float);

    float *x=new float[n];
    float *y=new float[n];

    for (size_t i=0;i<n;i++) {
        x[i]=((int)(i%2048)-1024)*0.5f;
        y[i]=(int)(i%1024)-512;
    }


    float *d_x, *d_y;
    CUDA_CHECK(cudaMalloc(&d_x, bytes));
    CUDA_CHECK(cudaMalloc(&d_y, bytes));
    CUDA_CHECK(cudaMemcpy(d_x, x, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_y, y, bytes, cudaMemcpyHostToDevice));
    
    cudaEvent_t start,stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    int threads=256;
    int blocks=ceil(float(n)/threads);

    CUDA_CHECK(cudaEventRecord(start));
    saxpy<<<blocks, threads>>>(d_x, d_y, d_y, n);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms=0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));

    CUDA_CHECK(cudaMemcpy(y, d_y, bytes, cudaMemcpyDeviceToHost));

    double sum=0.0;
    for (size_t i=0;i<n;i++) sum+=(double)y[i];
    
    std::cout<<std::fixed<<std::setprecision(0)<<"SUM="<<sum<<" n="<<n<<" kernel_ms="<<ms<<'\n';

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_x));
    CUDA_CHECK(cudaFree(d_y));
    delete[] x;
    delete[] y;

    return 0;
}
