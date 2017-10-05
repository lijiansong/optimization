#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <ctime>
#include <vector>
#include <priority_queue>
using namespace std;

int* get_rand(int *data, const int size)
{
    srand((unsigned)time(NULL));
    for(int i=0; i<size; ++i) data[i] = rand()%1000 + 1;
}

void show_data(int *data, int size)
{
    for(int i=0; i<size; ++i) printf("%d ", data[i]);
}

int comp(const void*a, const void*b)
{
    return *(int*)a - *(int*)b;
}

int main(int argc, const char *argv[])
{
    const int size = 1e4;
    float *data = (float*)malloc(sizeof(float)*size);
    get_rand(data, size);
    unsigned __int64 tick_start, tick_end;
    __asm{
        rdtsc
        mov dword ptr tick_start, eax
        mov dword ptr tick_start+4, edx
    }
##ifdef SORT
    sort(data[0],data+size);
##else
    qsort(data, size, sizeof(int), comp);
##endif
    show_data(data, size);

    __asm{
        rdtsc
        mov dword ptr tick_end, eax
        mov dword ptr tick_end+4, edx
    }
    printf("ticks %I64u\n",tick_end-tick_start);
    return 0;
}

