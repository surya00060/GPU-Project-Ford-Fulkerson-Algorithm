// C++ program for implementation of Ford Fulkerson algorithm 
// Reference: https://www.geeksforgeeks.org/ford-fulkerson-algorithm-for-maximum-flow-problem/
// Code modified as required in our input format

#include <iostream> 
#include <limits.h> 
#include <string.h> 
#include <queue> 
using namespace std; 
#define vb vector<bool>
#define vi vector<int>
#define vvi vector<vector<int>>

/* Returns true if there is a path from source 's' to sink 't' in 
residual graph. Also fills parent[] to store the path */
bool bfs(vvi &rGraph, int n, int s, int t, vi &parent) 
{ 
	// Create a visited array and mark all vertices as not visited 
	vb visited(n, false); 

	// Create a queue, enqueue source vertex and mark source vertex 
	// as visited 
	queue <int> q; 
	q.push(s); 
	visited[s] = true; 
	parent[s] = -1; 

	// Standard BFS Loop 
	while (!q.empty()) 
	{ 
		int u = q.front(); 
		q.pop(); 

		for (int v=0; v<n; v++) 
		{ 
			if (visited[v]==false && rGraph[u][v] > 0) 
			{ 
				q.push(v); 
				parent[v] = u; 
				visited[v] = true; 
			} 
		} 
	} 

	// If we reached sink in BFS starting from source, then return 
	// true, else false 
	return (visited[t] == true); 
} 

// Returns the maximum flow from s to t in the given graph 
int fordFulkerson(vvi &graph, int n, int s, int t) 
{ 
	int u, v; 

	// Create a residual graph and fill the residual graph with 
	// given capacities in the original graph as residual capacities 
	// in residual graph 
	vvi rGraph(n, vi(n)); // Residual graph where rGraph[i][j] indicates 
					// residual capacity of edge from i to j (if there 
					// is an edge. If rGraph[i][j] is 0, then there is not) 
	for (u = 0; u < n; u++) 
		for (v = 0; v < n; v++) 
			rGraph[u][v] = graph[u][v]; 

	vi parent(n); // This array is filled by BFS and to store path 

	int max_flow = 0; // There is no flow initially 

	// Augment the flow while tere is path from source to sink 
	while (bfs(rGraph, n, s, t, parent)) 
	{ 
		// Find minimum residual capacity of the edges along the 
		// path filled by BFS. Or we can say find the maximum flow 
		// through the path found. 
		int path_flow = INT_MAX; 
		for (v=t; v!=s; v=parent[v]) 
		{ 
			u = parent[v]; 
			path_flow = min(path_flow, rGraph[u][v]); 
		} 

		// update residual capacities of the edges and reverse edges 
		// along the path 
		for (v=t; v != s; v=parent[v]) 
		{ 
			u = parent[v]; 
			rGraph[u][v] -= path_flow; 
			rGraph[v][u] += path_flow; 
		} 

		// Add path flow to overall flow 
		max_flow += path_flow; 
	} 

	// Return the overall flow 
	return max_flow; 
} 

// Driver program to test above functions 
int main() 
{ 
    int n, m;
    cin >> n >> m;
	vvi graph(n, vi(n, 0)); 

    int u, v, w;
    for(int i=0;i<m;++i)
    {
        cin >> u >> v >> w;
        graph[u][v] = w; 
    }

	cout << fordFulkerson(graph, n, 0, n-1) << '\n'; 
	return 0; 
} 
