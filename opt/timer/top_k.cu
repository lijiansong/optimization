/*
Members:
201618013229011	李坚松
201618013229015	刘刚
201618013229006	黄若然
201618013229012	李琨
201618013229014	刘伯然
*/
#include <vector>
#include <queue>
#include <limits.h>
#include <float.h>
#include <functional>
#include <fstream>

#include <assert.h>
#include <sys/time.h>
#include <unistd.h>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <cuda_runtime.h>
//#include "cuPrintf.cu"

using namespace std;

typedef vector<float> vf;
typedef priority_queue<float,vector<float>,greater<float> > pq;

#define MAX_THREAD_PER_BLOCK 1024

static void HandleError(cudaError_t err, const char *file, int line )
{
	if (err != cudaSuccess)
	{
		printf("%s in %s at line %d\n",cudaGetErrorString(err), file, line );
		exit(EXIT_FAILURE);
	}
}

#define HANDLE_ERROR( err ) (HandleError(err, __FILE__, __LINE__))
#define HANDLE_NULL( a ) {if (a == NULL) { \
                            printf( "Host memory failed in %s at line %d\n", \
                                    __FILE__, __LINE__ ); \
                            exit( EXIT_FAILURE );}}
		
void get_device_info()
{
	cudaDeviceProp prop;
	int count;
	HANDLE_ERROR(cudaGetDeviceCount(&count));
	for (int i = 0; i < count; ++i)
	{
		HANDLE_ERROR(cudaGetDeviceProperties(&prop, i));
		printf("  --- Info of device %d ---\n",i );
		printf("device name : %s.\n", prop.name);
		printf("total global mem : %d.\n", prop.totalGlobalMem);
		printf("shared mem per block : %d.\n", prop.sharedMemPerBlock);
		printf("registers per block : %d.\n", prop.regsPerBlock);
		printf("threrads in warp : %d.\n", prop.warpSize);
		printf("mem pitch : %d.\n", prop.memPitch);
		printf("max threads per block : %d.\n", prop.maxThreadsPerBlock);
		printf("max threads dimensions : (%d %d %d).\n", prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
		printf("max grid dimensions : (%d %d %d).\n", prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
		printf("total const Mem : %d.\n", prop.totalConstMem);
		printf("major.minor : %d.%d.\n", prop.major, prop.minor);
		printf("clock rate : %d.\n", prop.clockRate);
		printf("texture alignment : %d.\n", prop.textureAlignment);
		printf("device overlap : %d.\n", prop.deviceOverlap);
		printf("multiprocessor count : %d.\n", prop.multiProcessorCount);
		printf("\n");
	}
}

int partition(vf &data,int start, int end)
{
	int i=start,j=end;
	float tmp;
	if(start<end)
	{
		tmp=data[start];
		while(i!=j)
		{
			while(j>i&&data[j]-tmp<=1e-9)	--j;
			data[i]=data[j];
			while(j>i&&data[i]-tmp>1e-9)	++i;
			data[j]=data[i];
			
		}
		data[i]=tmp;
	}
	return i;
}

void _top_k(vf &data,int k)
{
	int start=0,end=data.size()-1;
	int index=partition(data,start,end);
	while(index!=k-1)
	{
		if(index > k-1)
		{
			end=index-1;
			index=partition(data,start,end);
		}
		else
		{
			start=index+1;
			index=partition(data,start,end);
		}
	}
}

void heap_init(vf &data, pq &heap, int k)
{
	if(k > data.size())	return ;
	for(int i=0;i<k;++i)
	{
		heap.push(data[i]);
	}
}

void top_k(vf &data, pq &heap, int k)
{
	int len=data.size();
	if(k > len)	return ;
	for(int i=k;i<len;++i)
	{
		if(heap.top()-data[i]<1e-9)
		{
			heap.pop();
			heap.push(data[i]);
		}
	}
}

__device__ void selection_sort(float *dev_data, int left, int right)
{
	for (int i = left; i <= right; ++i)
	{
		float max_val=dev_data[i];
		int max_idx=i;
		for (int j = i+1; j <= right; ++j)
		{
			float val_j=dev_data[j];
			if(val_j - max_val > 1e-9)
			{
				max_idx=j;
				max_val=val_j;
			}
		}
		if (i != max_idx)
		{
			dev_data[max_idx]=dev_data[i];
			dev_data[i]=max_val;
		}
	}
}

__global__ void topk_kernel(float *dev_data, float *dev_result,const int &k,const int &data_len, const int &width)
{
	int tx=threadIdx.x;
	//cuPrintf("-----------tx is:%d\n", tx);
	//printf("-----------tx is:%d\n", tx);
	//printf("-------------------%d",tx*width+width);
	if (tx*width+width <= data_len)
	{
		selection_sort(dev_data,tx*width,tx*width+width-1);
	}
	else
	{
		selection_sort(dev_data,tx*width,data_len-1);
	}
	
	__shared__ float heap[MAX_THREAD_PER_BLOCK];
	__shared__ int res_index;
	__shared__ int local_index[MAX_THREAD_PER_BLOCK];
	assert(tx!=0);
	if (tx==0)
	{
		res_index=0;
		for (int i = 0; i < MAX_THREAD_PER_BLOCK; ++i)
		{
			heap[i]=FLT_MIN;
			local_index[i]=0;
		}
	}
	__syncthreads();
	
	for (int i = 0; i < k; ++i)
	{
		if (local_index[tx] < width)
		{
			//printf("-------------%f",dev_data[tx*width+local_index[tx]]);
			heap[tx]=dev_data[tx*width+local_index[tx]];
		}
		else
		{
			heap[tx]=FLT_MIN;
		}
		__syncthreads();

		if(tx==0)
		{
			float max_val=heap[0];
			int max_index=0;
			for (int i = 1; i < MAX_THREAD_PER_BLOCK; ++i)
			{
				if (heap[i]-max_val > 1e-9)
				{
					max_val=heap[i];
					max_index=i;
				}
			}
			++local_index[max_index];
			//cuPrintf("-------------------------max_val is:%f\n", max_val);
			dev_result[res_index]=max_val;
			++res_index;
		}
		__syncthreads();
	}
}

void topk_device(const vf &data,const int &k,float *result)
{
	float *dev_data;
	size_t len=data.size();
	//printf("len--------------%d\n", len);
	HANDLE_ERROR(cudaMalloc((void**)&dev_data,len*sizeof(float)));
	HANDLE_ERROR(cudaMemcpy(dev_data,&data[0],len*sizeof(float),cudaMemcpyHostToDevice));

	float *dev_result;
	HANDLE_ERROR(cudaMalloc((void**)&dev_result,k*sizeof(float)));

	int width=len/MAX_THREAD_PER_BLOCK;
	//printf("-----------width %d\n", width);

	//cudaPrintfInit(); 

	dim3 dimBlock(MAX_THREAD_PER_BLOCK,1);
	dim3 dimGrid(1,1);
	topk_kernel<<<dimGrid,dimBlock>>>(dev_data,dev_result,k,len,width);

	//cudaPrintfDisplay(stdout, true); 
	//cudaPrintfEnd(); 

	cudaMemcpy(result,dev_result,k*sizeof(float),cudaMemcpyDeviceToHost);

	cudaFree(dev_data);
	cudaFree(dev_result);
}

void parse_data(const char* filename,vf &data)
{
	ifstream infile;
    string line;
    infile.open(filename);
    if(infile.is_open())
    {
    	while(getline(infile,line,'\n'))
    	{
    		data.push_back((float)atof(line.c_str()));
    	}
    }
    infile.close();
}

int main(int argc, char const *argv[])
{
	if(argc!=3)	return -1;
	
	//get_device_info();
	struct timeval _start,_end;
	double diff;
	
	vf data;
	pq heap;
	parse_data(argv[1],data);
	int k=atoi(argv[2]);

	float *result=(float*)malloc(k*sizeof(float));

	cudaEvent_t start,stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start,0);
	
	//cudaPrintfInit(); 
	
	topk_device(data,k,result);
	
	//cudaPrintfDisplay(stdout, true); 
	//cudaPrintfEnd(); 
	
	cudaEventRecord(stop,0);
	cudaEventSynchronize(stop);
	float elapsedTime;
	cudaEventElapsedTime(&elapsedTime,start,stop);
	
	gettimeofday(&_start,NULL);
	//heap_init(data,heap,k);
    //top_k(data,heap,k);
	_top_k(data,k);
	gettimeofday(&_end,NULL);
    diff = (1000000 * (_end.tv_sec-_start.tv_sec)+ _end.tv_usec-_start.tv_usec)/10000.0;
	
	printf("----------------top %d----------------\n",k);
	//for (int i = 0; i < k; ++i)
	//{
	//	printf("%f ", result[i]);
	//}
	//printf("\n");
	//printf("\n");
	//for(int i=0;i<k;++i)
	//{
        //printf("%f ",heap.top());
        //heap.pop();
   // }
   for(int i=0;i<k;++i)	printf("%f ",data[i]);
	printf("\n");
	printf("time consuming: %f ms\n", diff);
	//printf("time consuming: %f ms\n", elapsedTime);

	free(result);
	return 0;
}
