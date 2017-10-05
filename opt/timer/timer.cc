#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <ctime>
#include <vector>
#include <priority_queue>
using namespace std;
typedef vector<float> vf;
typedef priority_queue<float,vector<float>,greater<float> > pq;

float* get_rand(float *data, const int size)
{
    srand((unsigned)time(NULL));
    for(int i=0; i<size; ++i) data[i] = rand()%1000.0 + 1;
}

void show_data(float *data, int size)
{
    for(int i=0; i<size; ++i) printf("%f ", data[i]);
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

void partition_top_k(vf &data,int k)
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

//int main(int argc, const char *argv[])
//{
//    const int size = 1e4;
//    float *data = (float*)malloc(sizeof(float)*size);
//    get_rand(data, size);
//    unsigned __int64 tick_start, tick_end;
//
//    show_data(data, size);
//    return 0;
//}

int main(int argc, char const *argv[])
{
	if(argc!=3)	return -1;
	
	struct timeval _start,_end;
	double diff;
	
	vf data;
	pq heap;
	parse_data(argv[1],data);
	int k=atoi(argv[2]);

	float *result=(float*)malloc(k*sizeof(float));

	gettimeofday(&_start,NULL);
	//heap_init(data,heap,k);
    //top_k(data,heap,k);
	_top_k(data,k);
	gettimeofday(&_end,NULL);
    diff = (1000000 * (_end.tv_sec-_start.tv_sec)+ _end.tv_usec-_start.tv_usec)/1000.0;
	printf("time consuming: %f ms\n", diff);
	free(result);
	return 0;
}
