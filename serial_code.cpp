// Implementation of serial code in cpp used for profiling the performance on cpu 
// and for comparing the parallel implementations.

#include <bits/stdc++.h>
using namespace std;
#define vi vector<int>
#define vvi vector<vector<int>>

struct label
{
    bool labelled;
    int parent;
    char direction;
    int potential;
};

int fordFulkerson(vvi &capacity, int n, int s, int t)
{
    vvi flow(n, vi(n, 0));
    
    while(true)
    {
        queue<int> scan_queue;
        vector<label> labels(n, {false, 0, '0', 0});
        labels[s] = {true, s, '+', INT_MAX};
        scan_queue.push(s);
        bool flag = false;

        while(!scan_queue.empty())
        {
            int u = scan_queue.front();
            scan_queue.pop();
            for(int v=0;v<n;++v)
            {
                if(labels[v].labelled == false)
                {
                    if(flow[u][v] < capacity[u][v])
                    {
                        int v_potential = min(labels[u].potential, capacity[u][v]-flow[u][v]);
                        labels[v] = {true, u, '+', v_potential};
                        scan_queue.push(v);                    
                        if(v == t)
                        {
                            flag = true;
                            break;
                        }
                    }
                    
                    if(flow[v][u] > 0)
                    {
                        int v_potential = min(labels[u].potential, flow[v][u]);
                        labels[v] = {true, u, '-', v_potential};
                        scan_queue.push(v);
                        if(v == t)
                        {
                            flag = true;
                            break;
                        }
                    }
                }
            }

            if(flag == true)
            {
                break;
            }
        }

        // for(int i=0;i<n;++i)
        // {
        //     cout << labels[i].labelled << ' '
        //          << labels[i].parent << ' '
        //          << labels[i].direction << ' '
        //          << labels[i].potential << '\n';
        // }

        if(scan_queue.empty())
        {
            int net_flow = 0;
            for(int i=0;i<n;++i)
            {
                net_flow += flow[s][i];
            }

            return net_flow;
        }

        int x = t, y, t_potential = labels[t].potential;
        while(x != s)
        {
            y = labels[x].parent; 
            if(labels[x].direction == '+')
            {
                flow[y][x] += t_potential;
            }
            else
            {
                flow[x][y] -= t_potential;
            }

            x = y;
        }
    }
}

int main()
{
    int n, m;
    cin >> n >> m;
	vvi capacity(n, vi(n, 0)); 

    int u, v, w;
    for(int i=0;i<m;++i)
    {
        cin >> u >> v >> w;
        capacity[u][v] = w; 
    }

	cout << fordFulkerson(capacity, n, 0, n-1) << '\n'; 
}
