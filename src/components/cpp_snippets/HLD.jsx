 // Heavy Light Decomposition
  
const hld_code = `
struct HLD
{
    int n;
    vector<int> siz, top, dep, parent, in, out, seq;
    vector<vector<int>> adj;
    int cur;

    HLD() {}
    HLD(int n)
    {
        init(n);
    }
    void init(int n)
    {
        this->n = n;
        siz.resize(n);
        top.resize(n);
        dep.resize(n);
        parent.resize(n);
        in.resize(n);
        out.resize(n);
        seq.resize(n);
        cur = 0;
        adj.assign(n, {});
    }
    void addEdge(int u, int v)
    {
        adj[u].push_back(v);
        adj[v].push_back(u);
    }
    void work(int root = 0)
    {
        top[root] = root;
        dep[root] = 0;
        parent[root] = -1;
        dfs1(root);
        dfs2(root);
    }
    void dfs1(int u)
    {
        if (parent[u] != -1)
        {
            adj[u].erase(find(adj[u].begin(), adj[u].end(), parent[u]));
        }

        siz[u] = 1;
        for (auto &v : adj[u])
        {
            parent[v] = u;
            dep[v] = dep[u] + 1;
            dfs1(v);
            siz[u] += siz[v];
            if (siz[v] > siz[adj[u][0]])
            {
                swap(v, adj[u][0]);
            }
        }
    }
    void dfs2(int u)
    {
        in[u] = cur++;
        seq[in[u]] = u;
        for (auto v : adj[u])
        {
            top[v] = v == adj[u][0] ? top[u] : v;
            dfs2(v);
        }
        out[u] = cur;
    }
    int lca(int u, int v)
    {
        while (top[u] != top[v])
        {
            if (dep[top[u]] > dep[top[v]])
            {
                u = parent[top[u]];
            }
            else
            {
                v = parent[top[v]];
            }
        }
        return dep[u] < dep[v] ? u : v;
    }

    int dist(int u, int v)
    {
        return dep[u] + dep[v] - 2 * dep[lca(u, v)];
    }

    int jump(int u, int k)
    {
        if (dep[u] < k)
        {
            return -1;
        }

        int d = dep[u] - k;

        while (dep[top[u]] > d)
        {
            u = parent[top[u]];
        }

        return seq[in[u] - dep[u] + d];
    }

    bool isAncester(int u, int v)
    {
        return in[u] <= in[v] && in[v] < out[u];
    }

    int rootedParent(int u, int v)
    {
        swap(u, v);
        if (u == v)
        {
            return u;
        }
        if (!isAncester(u, v))
        {
            return parent[u];
        }
        auto it = upper_bound(
            adj[u].begin(),
            adj[u].end(),
            v,
            [&](int x, int y)
            {
                return in[x] < in[y];
            });
        --it;
        return *it;
    }
    int rootedSize(int u, int v)
    {
        if (u == v)
        {
            return n;
        }
        if (!isAncester(v, u))
        {
            return siz[v];
        }
        return n - siz[rootedParent(u, v)];
    }

    int rootedLca(int a, int b, int c)
    {
        return lca(a, b) ^ lca(b, c) ^ lca(c, a);
    }
};
/*
         1
       / | \\
      2  3  4
     /|    / \\
    5 6   7   8
         / \\
        9  10
hld_tree.rootedSize(1, 2) is 3 containing 2,5,6
rootedSize(1, 2):
hld_tree.work(5); process the tree considering the root as 5
rootedLca(1, 2, 3):

This function calculates the lowest common ancestor (LCA) of nodes 2, 3, and 4 along the path from node 1.
The LCA of nodes 2 and 3 is node 1.
The LCA of nodes 3 and 4 is node 1.
The LCA of nodes 2 and 4 is node 1.
Therefore, the rooted LCA of nodes 2, 3, and 4 along the path from node 1 is node 1.\

dist(5, 10):
This function calculates the distance between nodes 5 and 10.
The distance between two nodes in a tree is the sum of the distances from each node to their lowest common ancestor.

jump(8, 2):

This function jumps k steps from node 8 towards node 2.
Since node 8 is at a higher level than node 2, the function cannot reach node 2 in k steps.
Therefore, the function returns -1 to indicate that the destination node cannot be reached in k steps.

isAncester(2, 6):

This function checks whether node 2 is an ancestor of node 6.
Node 2 is indeed an ancestor of node 6.
Therefore, the function returns true.
rootedParent(1, 5):

This function finds the parent of node 5 along the path to node 1.
Since node 5 is already on the path to node 1, its parent along this path is node 2.
Therefore, the function returns 2.
lca(5, 9):

This function calculates the lowest common ancestor (LCA) of nodes 5 and 9.
Therefore, the function returns 1.
*/`;
export default hld_code