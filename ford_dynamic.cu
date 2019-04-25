#include <math.h>
#include <limits.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <stdbool.h>
#define ll long long int
#define max_nodes 512
#include <time.h>

// __host__ __device__ int min(int a, int b)
// {
//     if(a <= b)
//     {
//         return a;
//     }
//     else
//     {
//         return b;
//     }
// }

__host__ __device__ ll hash(int potential, int parent, int labelled, int scanned, int direction)
{
    // potential can take any integer values
    ll hash_value = potential;
    // parent can only be in [0, 1024)
    hash_value <<= 10; 
    hash_value += parent;
    // Rest are boolean values
    // false is 0 and true is 1
    hash_value <<= 1;
    hash_value += labelled; 
    hash_value <<= 1; 
    hash_value += scanned;
    hash_value <<= 1;
    // '-' is 0 and '+' is 1
    hash_value += direction;

    return hash_value;
}

__host__ __device__ int potential(ll hash_value)
{
    return hash_value>>13;
}

__host__ __device__ int parent(ll hash_value)
{
    return (hash_value>>3)&(((1<<10)-1));
}

__host__ __device__ int labelled(ll hash_value)
{
    return (hash_value>>2)&1;
}

__host__ __device__ int scanned(ll hash_value)
{
    return (hash_value>>1)&1;
}

__host__ __device__ int direction(ll hash_value)
{
    return (hash_value)&1;
}



__global__ void parallel_traverse(int *d_flow, int *d_capacity, ll *d_label, bool *d_flag, int n)
{
    int u = blockIdx.x, v = threadIdx.x;
    if(u<n && v<n)
    {
        // printf("u: %d, v: %d, d_label[u]: %lld\n", u, v, d_label[u]);
        if(labelled(d_label[u]) == 1 && scanned(d_label[u]) == 0)
        {
            if(labelled(d_label[v]) == 0)
            {
                if(d_flow[u*n+v] < d_capacity[u*n+v])
                {
                    int v_potential = min(potential(d_label[u]), d_capacity[u*n+v]-d_flow[u*n+v]);
                    // labels[v] = {true, false, u, '+', v_potential};
                    d_label[v] = hash(v_potential, u, 1, 0, 1);
                    *d_flag = true;
                }

                if(d_flow[v*n+u] > 0)
                {
                    int v_potential = min(potential(d_label[u]), d_flow[v*n+u]);
                    // labels[v] = {true, false, u, '-', v_potential};
                    d_label[v] = hash(v_potential, u, 1, 0, 0); 
                    *d_flag = true;                   
                }
            }
        }
    }

    // Note __syncthreads() can be used only with the current kernel configuration as can only be performed on a single block
    __syncthreads();
    // scanned[u] = true;
    d_label[u] |= 1<<1;
}

__global__ void fordFulker(int *d_capacity, int *d_flow, ll *d_label, bool* d_augmentation_made, int n, int s, int t)
{
    d_label[s] = hash(INT_MAX, s, 1, 0, 0);

    *d_augmentation_made = true;
    while((*d_augmentation_made) && labelled(d_label[t])==0)
    {
        *d_augmentation_made = false;
        parallel_traverse<<<n, n>>>(d_flow, d_capacity, d_label, d_augmentation_made, n);
        cudaDeviceSynchronize();
    }

    if(*d_augmentation_made)
    {
        int x = t, y, t_potential = potential(d_label[t]);
        while( x != s)
        {
            y = parent(d_label[x]); 

            if(direction(d_label[x]) == 1)
            {
                d_flow[y*n+x] += t_potential;
            }
            else
            {
                d_flow[x*n+y] -= t_potential;
            }
            x = y;
        }
    }
}
int main()
{
    int n, m;
    scanf("%d %d", &n, &m);

    int *capacity = (int *)malloc(n*n*sizeof(int));
    memset(capacity, 0, n*n*sizeof(int)); 

    int u, v, w;
    for(int i=0;i<m;++i)
    {
        scanf("%d %d %d", &u, &v, &w);
        capacity[u*n+v] = w; 
    }

    clock_t t;
    t = clock(); 
    
    int *flow = (int *)malloc(n*n*sizeof(int));
    memset(flow, 0, n*n*sizeof(int)); 

    int *d_flag;
    cudaMalloc(&d_flag, sizeof(int));
    
    int *d_capacity;
    cudaMalloc(&d_capacity, n*n*sizeof(int));
    cudaMemcpy(d_capacity, capacity, sizeof(int)*n*n, cudaMemcpyHostToDevice);
    
    int *d_flow;
    cudaMalloc(&d_flow, n*n*sizeof(int));
    
    ll *d_label;
    cudaMalloc(&d_label, n*sizeof(ll)); 
    
    bool h_augmentation_made;
    bool *d_augmentation_made; cudaMalloc((void**)&d_augmentation_made, sizeof(bool));

    h_augmentation_made = true;
    
    float time1=0;
    while(h_augmentation_made)
    {
        cudaMemset(d_label, 0, n*sizeof(ll)); 
        h_augmentation_made = false;
        cudaMemset(d_augmentation_made, 0, sizeof(bool));

        float t;
		cudaEvent_t start, end;
		cudaEventCreate( &start); cudaEventCreate( &end);
        cudaEventRecord( start, 0);
            
        fordFulker<<<1,1>>>(d_capacity, d_flow, d_label, d_augmentation_made, n, 0, n-1);

        cudaEventRecord( end, 0);
		cudaEventSynchronize( end);
        cudaEventElapsedTime( &t, start, end);   // Returns time t in milliseconds.
        time1 += t; 
        cudaDeviceSynchronize();
        cudaMemcpy(&h_augmentation_made, d_augmentation_made, sizeof(bool), cudaMemcpyDeviceToHost);
    }
    
    cudaMemcpy(flow, d_flow, sizeof(int)*n*n, cudaMemcpyDeviceToHost);
    
    int net_flow = 0;
    for(int i=0;i<n;++i)
    {
        net_flow += flow[0*n+i];
    }
    t = clock() - t;
    double time_taken = ((double)t)/CLOCKS_PER_SEC; 
    
	printf("Answer: %d\n",net_flow);
    printf("GPU Time: %f\n", time1);
    printf("CPU Time: %f\n", time_taken);
    
	return 0;
}