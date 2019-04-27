#include <math.h>
#include <stdio.h>
#include <limits.h>
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

// Kernel tries to expand the frontier by one step ... frontier is the collection of nodes who had just been labelled in the previous iteration.
__global__ void parallel_traverse(int *d_flow, int *d_capacity, ll *d_label, ll *d_new_label,  bool *d_flag, int n)
{
    int u = blockIdx.x; int v = threadIdx.x;
    
	if(labelled(d_label[u]) == 1 && scanned(d_label[u]) == 0)
    {
        if(labelled(d_label[v]) == 0)
        {
            // v can be reached via forward edge
            if(d_flow[u*n+v] < d_capacity[u*n+v])
            {
                int v_potential = min(potential(d_label[u]), d_capacity[u*n+v]-d_flow[u*n+v]);
                // labels[v] = {true, false, u, '+', v_potential};
                d_new_label[v] = hash(v_potential, u, 1, 0, 1);
                *d_flag = true;
            }

            // v can be reached by reducing the flow (backward edge)
            if(d_flow[v*n+u] > 0)
            {
                int v_potential = min(potential(d_label[u]), d_flow[v*n+u]);
                // labels[v] = {true, false, u, '-', v_potential};
                d_new_label[v] = hash(v_potential, u, 1, 0, 0); 
                *d_flag = true;                   
            }
        }
        // scanned[u] = true;
        d_new_label[u] = d_label[u] | 1<<1;
    }
}

__global__ void fordFulkerson(int *d_capacity, int *d_flow, ll *d_label, ll *d_new_label, bool *d_augmentation_made, int n, int s, int t)
{
    // Initializing source
    // labelled[s] = true;
    // direction[s] = '+';
    // potential[s] = INT_MAX;
    d_label[s] = hash(INT_MAX, s, 1, 0, 0);
	d_new_label[s] = hash(INT_MAX, s, 1, 0, 0);

    bool even_iterations = true;
    *d_augmentation_made = true;
    // while sink is not yet labelled
    while((*d_augmentation_made) && (labelled(d_label[t])==0 && labelled(d_new_label[t]) == 0))
    {
        *d_augmentation_made = false;
        if (even_iterations)
        {
            parallel_traverse<<<n, n>>>(d_flow, d_capacity, d_label, d_new_label, d_augmentation_made, n);
            even_iterations = !even_iterations;
            cudaDeviceSynchronize();
        }
        else
        {
            parallel_traverse<<<n, n>>>(d_flow, d_capacity, d_new_label, d_label, d_augmentation_made, n);
            even_iterations = !even_iterations;
            cudaDeviceSynchronize();
        }
    }

    if(*d_augmentation_made)
    {
        if (even_iterations)
        {
            // Potential of sink is the bottleneck flow through the augmenting path
            int x = t, y, t_potential = potential(d_label[t]);
            while( x != s)
            {
                y = parent(d_label[x]); 

                // Updating all flow values on the path depending on the direction of the link
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
        else
        {
            // Potential of sink is the bottleneck flow through the augmenting path
            int x = t, y, t_potential = potential(d_new_label[t]);
            while( x != s)
            {
                y = parent(d_new_label[x]); 

                // Updating all flow values on the path depending on the direction of the link
                if(direction(d_new_label[x]) == 1)
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

    int *flow = (int *)malloc(n*n*sizeof(int));
    memset(flow, 0, n*n*sizeof(int)); 

    int *d_flag;
    cudaMalloc(&d_flag, sizeof(int));
    
    int *d_capacity;
    cudaMalloc(&d_capacity, n*n*sizeof(int));
    cudaMemcpy(d_capacity, capacity, sizeof(int)*n*n, cudaMemcpyHostToDevice);
    
    int *d_flow;
    cudaMalloc(&d_flow, n*n*sizeof(int));
    
    // Two labels interchangeably passed to the kernel depending on the iteration parity
    // Needed to avoid a memcpy to update the old label value
    ll *d_label;
    cudaMalloc(&d_label, n*sizeof(ll)); 
    
	ll *d_new_label;
	cudaMalloc(&d_new_label, n*sizeof(ll));

    bool h_augmentation_made;
    bool *d_augmentation_made; cudaMalloc((void**)&d_augmentation_made, sizeof(bool));

    h_augmentation_made = true;
    
    while(h_augmentation_made)
    {
        cudaMemset(d_label, 0, n*sizeof(ll)); 
		cudaMemset(d_new_label, 0, n*sizeof(ll));
        h_augmentation_made = false;
        cudaMemset(d_augmentation_made, 0, sizeof(bool));
            
        fordFulkerson<<<1,1>>>(d_capacity, d_flow, d_label, d_new_label, d_augmentation_made, n, 0, n-1);
        cudaMemcpy(&h_augmentation_made, d_augmentation_made, sizeof(bool), cudaMemcpyDeviceToHost);
    }
    
    cudaMemcpy(flow, d_flow, sizeof(int)*n*n, cudaMemcpyDeviceToHost);
    
    int net_flow = 0;
    for(int i=0;i<n;++i)
    {
        // No more augmenting paths can be found
        net_flow += flow[0*n+i];
    }
    
    // Printing sum of flows originating from source
    printf("%d\n", net_flow);
	return 0;
}