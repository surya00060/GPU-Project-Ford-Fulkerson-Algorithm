// Uses multiple kernel calls keeping each node as source so that label of v won't get updated before it's block starts executing. 
// As might lead to the another iteration in the search and later v's potential might again get changed 
// by some other u giving inconsistent potential values(paths).

#include <math.h>
#include <limits.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <stdbool.h>
#define ll long long int

// hashing used to update all of the attributes of a node at the same time
// This is to avoid race conditions as multiple u's could try to label a particular neighbour
// Using a mutex lock instead would severely affect performance as threads would be in busy wait
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


// The following functions return the corresponding field 
// in the hashed value using bitwise operations.
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
// There is frequent context switch between cpu and gpu, try improving that by moving the loop inside kernel -- dynamic parallelism performed.

// Kernel tries to expand the frontier by one step ... frontier is the collection of nodes who had just been labelled in the previous iteration.
__global__ void parallel_traverse(int u, int *d_flow, int *d_capacity, ll *d_label, int *d_flag, int n)
{
    int v = threadIdx.x;
    // printf("u: %d, v: %d, d_label[u]: %lld\n", u, v, d_label[u]);
    if(labelled(d_label[u]) == 1 && scanned(d_label[u]) == 0)
    {
        if(labelled(d_label[v]) == 0)
        {
            // v can be reached via forward edge
            if(d_flow[u*n+v] < d_capacity[u*n+v])
            {
                int v_potential = min(potential(d_label[u]), d_capacity[u*n+v]-d_flow[u*n+v]);
                // labels[v] = {true, false, u, '+', v_potential};
                d_label[v] = hash(v_potential, u, 1, 0, 1);
                *d_flag += 1;
            }

            // v can be reached by reducing the flow (backward edge)
            if(d_flow[v*n+u] > 0)
            {
                int v_potential = min(potential(d_label[u]), d_flow[v*n+u]);
                // labels[v] = {true, false, u, '-', v_potential};
                d_label[v] = hash(v_potential, u, 1, 0, 0); 
                *d_flag += 1;                   
            }
        }
        
        // Note __syncthreads() can be used only with the current kernel configuration as can only be performed on a single block
        __syncthreads();
        // scanned[u] = true;
        d_label[u] |= 1<<1;
        // printf("%d\n", scanned(d_label[u]));
    }
}

int fordFulkerson(int *capacity, int n, int s, int t)
{
    // flag variable is to show if any update is happening in an iteration
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
    ll *d_label;
    cudaMalloc(&d_label, n*sizeof(ll));

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
        cudaMemcpy(d_label, label, n*sizeof(ll), cudaMemcpyHostToDevice);

        // while sink is not yet labelled
        while(labelled(label[t]) == 0)
        {
            *flag = 0;
            cudaMemcpy(d_flag, flag, sizeof(int), cudaMemcpyHostToDevice);
            for(int i=0;i<n;++i)
            {
                parallel_traverse<<<1, n>>>(i, d_flow, d_capacity, d_label, d_flag, n);
            }
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
            // No more augmenting paths can be found
            int net_flow = 0;
            for(int i=0;i<n;++i)
            {
                net_flow += flow[s*n+i];
            }
            // Returning sum of flows originating from source
            return net_flow;
        }

        // Potential of sink is the bottleneck flow through the augmenting path
        int x = t, y, t_potential = potential(label[t]);
        // printf("%d\n", t_potential);
        while(x != s)
        {
            // printf("%d ", x);
            // Getting bits from 3 to 13 - y = parent[x]
            y = parent(label[x]); 

            // Updating all flow values on the path depending on the direction of the link
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

    // 1-D arrays to pass the same to the kernel
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
