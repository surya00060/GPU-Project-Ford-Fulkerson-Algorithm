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

// Note: Using the same u potential multiple nodes can be assigned potentials, but eventually no more than one of them will be in the path.
// There is frequent context switch between cpu and gpu, try improving that by moving the loop inside kernel
// Some of the device arrays are constants --- Check if improvement possible if they are changed accordingly

__global__ void parallel_traverse(int *d_flow, int *d_capacity, ll *d_label, int *d_flag, int n)
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
                    *d_flag += 1;
                }

                if(d_flow[v*n+u] > 0)
                {
                    int v_potential = min(potential(d_label[u]), d_flow[v*n+u]);
                    // labels[v] = {true, false, u, '-', v_potential};
                    d_label[v] = hash(v_potential, u, 1, 0, 0); 
                    *d_flag += 1;                   
                }
            }
        }
    }
    

    // Note __syncthreads() can be used only with the current kernel configuration as can only be performed on a single block
    __syncthreads();
    // scanned[u] = true;
    d_label[u] |= 1<<1;
}

int fordFulkerson(int *capacity, int n, int s, int t)
{
    int *flow = (int *)malloc(n*n*sizeof(int));
    memset(flow, 0, n*n*sizeof(int)); 
    // Memory allocated only once
    ll *label = (ll *)malloc(n*sizeof(ll));
    memset(label, 0, n*sizeof(ll)); 
    int *flag = (int *)malloc(sizeof(int));

    int *d_flag;
    cudaMalloc(&d_flag, sizeof(int));
    int *d_capacity;
    cudaMalloc(&d_capacity, n*n*sizeof(int));
    int *d_flow;
    cudaMalloc(&d_flow, n*n*sizeof(int));
    ll *d_label;
    cudaMalloc(&d_label, n*sizeof(ll));

    cudaMemcpy(d_capacity, capacity, n*n*sizeof(int), cudaMemcpyHostToDevice);
	
	float time = 0;
    while(true)
    {
        memset(label, 0, n*sizeof(ll)); 

        // labelled[s] = true;
        // direction[s] = '+';
        // potential[s] = INT_MAX;
        label[s] = hash(INT_MAX, s, 1, 0, 0);
        // printf("label[s]: %lld\n", label[s]);
        cudaMemcpy(d_flow, flow, n*n*sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_label, label, n*sizeof(ll), cudaMemcpyHostToDevice);

        // while sink is not yet labelled
        while(labelled(label[t]) == 0)
        {
            *flag = 0;
            cudaMemcpy(d_flag, flag, sizeof(int), cudaMemcpyHostToDevice);
			
			float t;
			cudaEvent_t start, end;
			cudaEventCreate( &start); cudaEventCreate( &end);
			cudaEventRecord( start, 0);
            
			parallel_traverse<<<n, n>>>(d_flow, d_capacity, d_label, d_flag, n);
			
			cudaEventRecord( end, 0);
			cudaEventSynchronize( end);
			cudaEventElapsedTime( &t, start, end);   // Returns time t in milliseconds.
			time += t;

            // Not needed to copy all labels back and forth - If changed then this is to be added outside loop
            cudaMemcpy(label, d_label, n*sizeof(ll), cudaMemcpyDeviceToHost);
            cudaMemcpy(flag, d_flag, sizeof(int), cudaMemcpyDeviceToHost);

            // printf("flag: %d\n", *flag);
            if(*flag == 0)
            {
                break;
            }
        }

        // Below line is not needed as it doesn't change in device
        // cudaMemcpy(flow, d_flow, sizeof(flow), cudaMemcpyDeviceToHost);

        if(*flag == 0)
        {
            int net_flow = 0;
            for(int i=0;i<n;++i)
            {
                net_flow += flow[s*n+i];
            }
			printf("GPU Kernel Execution Time: %f\n", time);
            return net_flow;
        }

        int x = t, y, t_potential = potential(label[t]);
        //printf("%d\n", t_potential);
        while(x != s)
        {
            // Getting bits from 3 to 13 - y = parent[x]
            y = parent(label[x]); 

            // if(labels[x].direction == '+')
            if(direction(label[x]) == 1)
            {
                flow[y*n+x] += t_potential;
            }
            else
            {
                flow[x*n+y] -= t_potential;
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
	int a = fordFulkerson(capacity, n, 0, n-1);
	t = clock() - t;
	double time_taken = ((double)t)/CLOCKS_PER_SEC; 
    
	printf("Answer: %d\n",a);
	printf("Total Time: %f\n",time_taken);

	return 0;
}