// Code to generate test cases, number of nodes to be changed in the program.
// Number of edges can also be changed or be kept random.
// Only one edge between pair of vertices.
// No incoming edge to source and no outgoing edge from sink.

#include <bits/stdc++.h>
using namespace std;

int main()
{
    srand(clock());
    int n = 100;
    int m = rand()%((n*(n-1))/2+1);
    cout << n << ' ' << m << '\n';

    int u, v, w, w_max = 100;
    set<pair<int, int>> edge_set;

    while(m > 0)
    {
        u = rand()%n;
        v = rand()%n;
        w = rand()%w_max+1;

        if(u == n-1 || v == 0 || u == v || edge_set.find(make_pair(u, v)) != edge_set.end())
        {
            continue;
        }
        else
        {
            edge_set.insert(make_pair(u, v));
            edge_set.insert(make_pair(v, u));
            cout << u << ' ' << v << ' ' << w << '\n';
            m--;
        }
    }
}