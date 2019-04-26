// Uses "two arrays for label" so that label of v won't get updated before it's block starts executing. 
// This is an improvement from the previous version as there aren't multiple kernel calls for each source,
// they have been made different blocks of the same kernel and hence can execute parallely.

#include <math.h>
#include <limits.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <stdbool.h>
#define ll long long int

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

__global__ void parallel_traverse(int *d_flow, int *d_capacity, ll *d_label, ll *d_new_label, int *d_flag, int n)
{
    int u = blockIdx.x, v = threadIdx.x;
    // printf("u: %d, v: %d, d_label[u]: %lld\n", u, v, d_label[u]);
    if(labelled(d_label[u]) == 1 && scanned(d_label[u]) == 0)
    {
        if(labelled(d_label[v]) == 0)
        {
            if(d_flow[u*n+v] < d_capacity[u*n+v])
            {
                int v_potential = min(potential(d_label[u]), d_capacity[u*n+v]-d_flow[u*n+v]);
                // labels[v] = {true, false, u, '+', v_potential};
                d_new_label[v] = hash(v_potential, u, 1, 0, 1);
                *d_flag += 1;
            }

            if(d_flow[v*n+u] > 0)
            {
                int v_potential = min(potential(d_label[u]), d_flow[v*n+u]);
                // labels[v] = {true, false, u, '-', v_potential};
                d_new_label[v] = hash(v_potential, u, 1, 0, 0); 
                *d_flag += 1;                   
            }
        }

        // scanned[u] = true;
        d_new_label[u] = d_label[u]|1<<1;
    }
}

int fordFulkerson(int *capacity, int n, int s, int t)
{
    int *flag = (int *)malloc(sizeof(int));
    int *flow = (int *)malloc(n*n*sizeof(int));
    memset(flow, 0, n*n*sizeof(int)); 
    // Memory allocated only once
    ll *label = (ll *)malloc(n*sizeof(ll));
    memset(label, 0, n*sizeof(ll)); 

    int *d_capacity;
    cudaMalloc(&d_capacity, n*n*sizeof(int));
    int *d_flag;
    cudaMalloc(&d_flag, sizeof(int));
    int *d_flow;
    cudaMalloc(&d_flow, n*n*sizeof(int));
    ll *d_label, *d_new_label;
    cudaMalloc(&d_label, n*sizeof(ll));
    cudaMalloc(&d_new_label, n*sizeof(ll));

    cudaMemcpy(d_capacity, capacity, n*n*sizeof(int), cudaMemcpyHostToDevice);

    while(true)
    {
        memset(label, 0, n*sizeof(ll)); 

        // labelled[s] = true;
        // direction[s] = '+';
        // potential[s] = INT_MAX;
        label[s] = hash(INT_MAX, s, 1, 0, 0);
        // printf("label[s]: %lld\n", label[s]);
        cudaMemcpy(d_flow, flow, n*n*sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_new_label, label, n*sizeof(ll), cudaMemcpyHostToDevice);

        // while sink is not yet labelled
        while(labelled(label[t]) == 0)
        {
            *flag = 0;
            cudaMemcpy(d_flag, flag, sizeof(int), cudaMemcpyHostToDevice);
            cudaMemcpy(d_label, label, n*sizeof(ll), cudaMemcpyHostToDevice);
            parallel_traverse<<<n, n>>>(d_flow, d_capacity, d_label, d_new_label, d_flag, n);
            // Not needed to copy all labels back and forth - If changed then this is to be added outside loop
            cudaMemcpy(label, d_new_label, n*sizeof(ll), cudaMemcpyDeviceToHost);
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
            return net_flow;
        }

        int x = t, y, t_potential = potential(label[t]);
        // printf("%d\n", t_potential);
        while(x != s)
        {
            // printf("%d ", x);
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
        // printf("%d\n", t_potential);
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

    printf("%d\n", fordFulkerson(capacity, n, 0, n-1));
}
