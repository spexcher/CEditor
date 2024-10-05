import {
  Box,
  Heading,
  Text,
  VStack,
  Code,
  Divider,
  Button,
  useClipboard,
  HStack,
} from "@chakra-ui/react";

const SnippetPage = () => {
  // Segment Tree snippet text
  const segmentTreeCode = `template <typename num_t>
struct segtree
{
    int n, depth;
    vector<num_t> tree, lazy;
    void init(const vector<long long> &arr)
    {
        n = arr.size();
        tree = vector<num_t>(4 * n, 0);
        lazy = vector<num_t>(4 * n, 0);
        init(0, 0, n - 1, arr);
    }
    num_t init(int i, int l, int r, const vector<long long> &arr)
    {
        if (l == r)
            return tree[i] = arr[l];
        int mid = (l + r) / 2;
        num_t a = init(2 * i + 1, l, mid, arr);
        num_t b = init(2 * i + 2, mid + 1, r, arr);
        return tree[i] = a.op(b);
    }
    void update(int l, int r, num_t v)
    {
        if (l > r)
            return;
        update(0, 0, n - 1, l, r, v);
    }
    num_t update(int i, int tl, int tr, int ql, int qr, num_t v)
    {
        eval_lazy(i, tl, tr);
        if (tr < ql || qr < tl)
            return tree[i];
        if (ql <= tl && tr <= qr)
        {
            lazy[i] = lazy[i].val + v.val;
            eval_lazy(i, tl, tr);
            return tree[i];
        }
        int mid = (tl + tr) / 2;
        num_t a = update(2 * i + 1, tl, mid, ql, qr, v), b = update(2 * i + 2, mid + 1, tr, ql, qr, v);
        return tree[i] = a.op(b);
    }
    num_t query(int l, int r)
    {
        if (l > r)
            return num_t::null_v;
        return query(0, 0, n - 1, l, r);
    }
    num_t query(int i, int tl, int tr, int ql, int qr)
    {
        eval_lazy(i, tl, tr);
        if (ql <= tl && tr <= qr)
            return tree[i];
        if (tr < ql || qr < tl)
            return num_t::null_v;
        int mid = (tl + tr) / 2;
        num_t a = query(2 * i + 1, tl, mid, ql, qr), b = query(2 * i + 2, mid + 1, tr, ql, qr);
        return a.op(b);
    }
    void eval_lazy(int i, int l, int r)
    {
        tree[i] = tree[i].lazy_op(lazy[i], (r - l + 1));
        if (l != r)
        {
            lazy[i * 2 + 1] = lazy[i].val + lazy[i * 2 + 1].val;
            lazy[i * 2 + 2] = lazy[i].val + lazy[i * 2 + 2].val;
        }
        lazy[i] = num_t();
    }
};
struct min_t
{
    long long val;
    static const long long null_v = 9223372036854775807LL;
    min_t() : val(0) {}
    min_t(long long v) : val(v) {}
    min_t op(min_t &other) { return min_t(min(val, other.val)); }
    min_t lazy_op(min_t &v, int size) { return min_t(val + v.val); }
};
struct sum_t
{
  long long val;
  static const long long null_v = 0;
  sum_t() : val(0) {}
  sum_t(long long v) : val(v) {}
  sum_t op(sum_t &other) { return sum_t(val + other.val); }
  sum_t lazy_op(sum_t &v, int size) { return sum_t(val + v.val * size); }
};
struct max_t
{
    long long val;
    static const long long null_v = -9223372036854775807LL;
    max_t() : val(0) {}
    max_t(long long v) : val(v) {}
    max_t op(max_t &other) { return max_t(max(val, other.val)); }
    max_t lazy_op(max_t &v, int size) { return max_t(val + v.val); }
};
`;
  const simple_segment_tree_code = `vector<int> segtree(4 * size);
function<void(int, int, int, int)> build = [&](int index, int l, int r)
{
    if (l == r)
    {
        segtree[index] = arr[l];
        return;
    }
    int mid = (l + r) / 2;
    build(2 * index + 1, l, mid);
    build(2 * index + 2, mid + 1, r);
    segtree[index] = 0; // combine logic
};
function<void(int, int, int, int, int, int)> update = [&](int index, int l, int r, int pos, int val)
{
    if (l == r)
    {
        segtree[index] = val;
        return;
    }
    int mid = (l + r) / 2;
    if (pos <= mid)
        update(2 * index + 1, l, mid, pos, val);
    else
        update(2 * index + 2, mid + 1, r, pos, val);
    segtree[index] = 0; // apply logic
};
function<int(int, int, int, int, int, int)> query = [&](int index, int l, int r, int lq, int rq)
{
    if (lq > r || rq < l)
    {
        // No overlap
        return 0; // or appropriate identity value (e.g., INT_MAX for min, 0 for sum)
    }
    if (lq <= l && rq >= r)
    {
        // Total overlap
        return segtree[index];
    }
    int mid = (l + r) / 2;
    int ans = 0;
    int left = query(2 * index + 1, l, mid, lq, rq);
    int right = query(2 * index + 2, mid + 1, r, lq, rq);
    segtree[index] = 0; // combine logic
};
`;

  // Matrix snippet text
  const matrix_code = `struct matrix
{
  using TYPE = ll;
  // TYPE v[n][n];
  ll n;
  vector<vector<TYPE>> v;
  matrix(ll n) : n(n), v(n, vector<TYPE>(n, 0)) {}

  // Matrix multiplication with modular arithmetic optimization
  matrix mul(matrix &b)
  {
    matrix res(n);
    static const ll msq = mod * mod;
    for (int i = 0; i < n; i++)
    {
      for (int k = 0; k < n; k++)
      {
        for (int j = 0; j < n; j++)
        {
          res.v[i][j] += v[i][k] * b.v[k][j];
          res.v[i][j] = (res.v[i][j] >= msq ? res.v[i][j] - msq : res.v[i][j]);
        }
      }
    }
    // Final modulo operation
    for (int i = 0; i < n; i++)
    {
      for (int j = 0; j < n; j++)
      {
        res.v[i][j] %= mod;
      }
    }
    return res;
  }

  // Matrix exponentiation by squaring
  matrix pow(matrix &a, long long x)
  {
    matrix res(n);
    for (int i = 0; i < n; i++)
      res.v[i][i] = 1; // Identity matrix

    while (x)
    {
      if (x & 1)
      {
        res = res.mul(a);
      }
      x /= 2;
      a = a.mul(a);
    }
    return res;
  }

  // Print matrix
  void pr()
  {
    cout << "------------\n";
    for (int i = 0; i < n; i++)
    {
      for (int j = 0; j < n; j++)
      {
        cout << v[i][j] << " ";
      }
      cout << '\n';
    }
    cout << "------------\n";
  }
};`;
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
  // Binary Search snippet text
  const binarySearchCode = `auto poss = [&](int mid) {

};
long long lo = 0;
long long hi = 2e18;
long long res = 2e18;
long long mid;
while (lo <= hi)
{
    mid = lo + ((hi - lo) >> 1);
    if (poss(mid))
    {
        hi = mid - 1;
        res = mid;
    }
    else
        lo = mid + 1;
}`;

  // Dijkstra's Algorithm snippet text
  const dijkstraCode = `
function<void(int)> dijkstra = [&](int S)
{
    set<pair<int, int>> st;

    st.insert({0, S});
    dist[S] = 0;
    //parent[S] = S;
    while (!st.empty())
    {
        auto it = *(st.begin());
        int node = it.second;
        int dis = it.first;
        st.erase(it);
        for (auto it : g[node])
        {
            int adjNode = it.ff;
            int edgW = it.ss;
            if (dis + edgW < dist[adjNode])
            {
                if (dist[adjNode] != 1e18)
                    st.erase({dist[adjNode], adjNode});
                dist[adjNode] = dis + edgW;
                st.insert({dist[adjNode], adjNode});
                //parent[adjNode] = node;
            }
        }
    }
};`;
  const substring_hash_code = `string s;
cin >> s;
int n = sz(s);
vector<int> hash(n);
vector<int> inverse(n);
int p = 1e9 + 7;
int power = 1;
hash[0] = (s[0] - 'a' + 1);
inverse[0] = 1;
FoF(i, 1, n - 1)
{
    power = (power * p) % mod;
    inverse[i] = inv(power);
    hash[i] = (hash[i - 1] + (s[i] - 'a' + 1) * power) % mod;
}
auto hash_val = [&](int l, int r)
{
    int res = hash[r];
    if (l > 0)
        res = (res - hash[l - 1] + mod) % mod;
    res = (res * inverse[l]) % mod;
    return res;
};`;

  const DSU_code = `struct DSU
{
    std::vector<int> f, siz;

    DSU() {}
    DSU(int n)
    {
        init(n);
    }

    void init(int n)
    {
        f.resize(n);
        std::iota(f.begin(), f.end(), 0);
        siz.assign(n, 1);
    }

    int find(int x)
    {
        while (x != f[x])
        {
            x = f[x] = f[f[x]];
        }
        return x;
    }

    bool same(int x, int y)
    {
        return find(x) == find(y);
    }

    bool merge(int x, int y)
    {
        x = find(x);
        y = find(y);
        if (x == y)
        {
            return false;
        }
        siz[x] += siz[y];
        f[y] = x;
        return true;
    }

    int size(int x)
    {
        return siz[find(x)];
    }
};`;
  const add_multiply_strings_code = `#define all(x) (x).begin(), (x).end()
#define rall(x) (x).rbegin(), (x).rend()
// Function to convert a string of digits into a vector of integers
vector<int> stringToVector(const string &s)
{
  vector<int> digits;
  for (char c : s)
  {
    digits.push_back(c - '0'); // Convert char to int
  }
  return digits;
}

// Function to convert a vector of integers into a string of digits
string vectorToString(const vector<int> &digits)
{
  string s;
  for (int digit : digits)
  {
    s.push_back(digit + '0'); // Convert int to char
  }
  return s;
}

// Function to add two vectors of digits
vector<int> add(const vector<int> &a, const vector<int> &b)
{
  vector<int> result;
  int carry = 0;
  int n = max(a.size(), b.size());
  for (int i = 0; i < n || carry; ++i)
  {
    if (i < a.size())
      carry += a[i];
    if (i < b.size())
      carry += b[i];
    result.push_back(carry % 10);
    carry /= 10;
  }
  return result;
}

// Function to multiply two vectors of digits using elementary multiplication
vector<int> multiply(const vector<int> &a, const vector<int> &b)
{
  vector<int> result(a.size() + b.size(), 0);
  for (int i = 0; i < a.size(); ++i)
  {
    int carry = 0;
    for (int j = 0; j < b.size() || carry; ++j)
    {
      long long current = result[i + j] + carry;
      if (j < b.size())
        current += (long long)a[i] * b[j];
      result[i + j] = current % 10;
      carry = current / 10;
    }
  }
  while (result.size() > 1 && result.back() == 0)
  {
    result.pop_back();
  }
  return result;
}

// Wrapper function to multiply two long integer strings
string multiplyStrings(string s1, string s2)
{
  reverse(all(s1));
  reverse(all(s2));
  vector<int> a = stringToVector(s1);
  vector<int> b = stringToVector(s2);
  if (a.size() == 1 && a[0] == 0 || b.size() == 1 && b[0] == 0)
    return "0";
  vector<int> result = multiply(a, b);
  reverse(result.begin(), result.end());
  return vectorToString(result);
}
string addStrings(string s1, string s2)
{
  reverse(all(s1));
  reverse(all(s2));
  vector<int> a = stringToVector(s1);
  vector<int> b = stringToVector(s2);
  if (a.size() == 1 && a[0] == 0 || b.size() == 1 && b[0] == 0)
    return "0";
  vector<int> result = add(a, b);
  reverse(result.begin(), result.end());
  return vectorToString(result);
}

// string s1 = "99";
// string s2 = "99";

// string result = addStrings(s1, s2);
// string result2 = multiplyStrings(s1, s2);

// cout << "Result of multiplying strings: " << result2 << endl;
// cout << "Result of adding strings: " << result << endl;
`;



  const multiply_polynomials_code = `vector<int> karatsuba_multiply_polynomials(const vector<int> &a, const vector<int> &b)
{
  int n = a.size();
  vector<int> res(2 * n, 0);

  if (n <= 32)
  { // Base case: switch to traditional multiplication for small polynomials
    for (int i = 0; i < n; i++)
    {
      for (int j = 0; j < n; j++)
      {
        res[i + j] += a[i] * b[j];
      }
    }
    return res;
  }

  int k = n / 2;

  // Split the polynomials into smaller ones
  vector<int> a1(a.begin(), a.begin() + k);
  vector<int> a0(a.begin() + k, a.end());
  vector<int> b1(b.begin(), b.begin() + k);
  vector<int> b0(b.begin() + k, b.end());

  // Recursively compute the three multiplications
  vector<int> p1 = karatsuba_multiply_polynomials(a1, b1);
  vector<int> p2 = karatsuba_multiply_polynomials(a0, b0);

  for (int i = 0; i < k; i++)
  {
    a0[i] += a1[i];
    b0[i] += b1[i];
  }

  vector<int> p3 = karatsuba_multiply_polynomials(a0, b0);

  // Combine the results of the multiplications
  for (int i = 0; i < n; i++)
  {
    p3[i] -= p1[i] + p2[i];
  }

  for (int i = 0; i < 2 * k; i++)
  {
    res[i + k] += p3[i];
    res[i] += p1[i];
  }

  for (int i = 0; i < 2 * k; i++)
  {
    res[i + 2 * k] += p2[i];
  }

  return res;
}

// Wrapper function to multiply two polynomials
vector<int> multiply_polynomials(const vector<int> &a, const vector<int> &b)
{
  int n = max(a.size(), b.size());
  vector<int> paddedA(a), paddedB(b);

  while (paddedA.size() < n * 2)
    paddedA.push_back(0);
  while (paddedB.size() < n * 2)
    paddedB.push_back(0);

  return karatsuba_multiply_polynomials(paddedA, paddedB);
}

int main()
{
  vector<int> a = {5, 2, 6, 8};
  vector<int> b = {0, 6, 1, 8};

  vector<int> result = multiply_polynomials(a, b);
  while (result.back() == 0)
    result.pop_back();

  cout << "Result of multiplication: ";
  for (int coeff : result)
  {
    cout << coeff << " ";
  }
  cout << endl;

  return 0;
}
`;

  const KMP_code = `// KMP Algorithm template --------------------------------
vector<int> computeLPS(const string &pattern)
{
    int m = pattern.size();
    vector<int> lps(m, 0);
    int len = 0;
    int i = 1;
    while (i < m)
    {
        if (pattern[i] == pattern[len])
        {
            len++;
            lps[i] = len;
            i++;
        }
        else
        {
            if (len != 0)
                len = lps[len - 1];
            else
            {
                lps[i] = 0;
                i++;
            }
        }
    }
    return lps;
}
int KMP(const string &text, const string &pattern)
{
    int n = text.size();
    int m = pattern.size();
    vector<int> lps = computeLPS(pattern);
    int count = 0;
    int i = 0, j = 0;
    while (i < n)
    {
        if (pattern[j] == text[i])
            i++, j++;
        if (j == m)
        {
            count++, j = lps[j - 1];
        }
        else if (i < n && pattern[j] != text[i])
        {
            if (j != 0)
                j = lps[j - 1];
            else
                i++;
        }
    }
    return count;
}
`;

  const ETF_code = `int ETF(int n)
{
    vi phi(n + 1, 0);
    phi[0] = 0;
    phi[1] = 1;
    for (int i = 2; i <= n; i++)
        phi[i] = i;
    for (int i = 2; i <= n; i++)
        if (phi[i] == i)
            for (int j = i; j <= n; j += i)
                phi[j] = phi[j] - phi[j] / i;
    return phi[n];
}
`;
  const SOE_code = `const int N = 10005;
bool sieve[N];
bool segsieve[N];
vector<int> primes;
void initsieve(int n)
{
    memset(sieve, true, sizeof(sieve));
    sieve[0] = false;
    sieve[1] = false;
    primes.clear();
    for (int i = 2; i * i <= n; i++)
        if (sieve[i] == true)
        {
            primes.pb(i);
            for (int j = i * i; j <= n; j += i)
                sieve[j] = false;
        }
}
void initsegsieve(int l, int r)
{
    memset(segsieve, true, sizeof(segsieve));
    ll sqr = sqrt(r);
    initsieve(sqr);
    for (ll p : primes)
    {
        ll sm = (l / p) * p;

        if (sm < l)
            sm += p;
        for (ll i = sm; i <= r; i += p)
        {
            segsieve[i - l] = false;
        }
    }
}
`;

  // Clipboard hooks
  const { hasCopied: hasCopiedSegmentTree, onCopy: onCopySegmentTree } =
    useClipboard(segmentTreeCode);
  const { hasCopied: hasCopiedBinarySearch, onCopy: onCopyBinarySearch } =
    useClipboard(binarySearchCode);
  const { hasCopied: hasCopiedDijkstra, onCopy: onCopyDijkstra } =
    useClipboard(dijkstraCode);
  const { hasCopied: hasCopiedMatrix, onCopy: onCopyMatrix } =
    useClipboard(matrix_code);
  const { hasCopied: hasCopiedDSU, onCopy: onCopyDSU } =
    useClipboard(DSU_code);
  const { hasCopied: hasCopied_substring_hash_code, onCopy: onCopy_substring_hash_code } =
    useClipboard(substring_hash_code);
  const { hasCopied: hasCopied_hld_code, onCopy: onCopy_hld_code } =
    useClipboard(substring_hash_code);
  const { hasCopied: hasCopied_add_multiply_strings, onCopy: onCopy_add_multiply_strings } =
    useClipboard(add_multiply_strings_code);
  const { hasCopied: hasCopied_multiply_polynomials, onCopy: onCopy_multiply_polynomials } =
    useClipboard(multiply_polynomials_code);
  const { hasCopied: hasCopied_KMP, onCopy: onCopy_KMP } =
    useClipboard(KMP_code);
  const { hasCopied: hasCopied_ETF, onCopy: onCopy_ETF } =
    useClipboard(ETF_code);
  const { hasCopied: hasCopied_SOE, onCopy: onCopy_SOE } =
    useClipboard(SOE_code);
  const { hasCopied: hasCopied_simple_segment_tree, onCopy: onCopy_simple_segment_tree } =
    useClipboard(simple_segment_tree_code);

  return (
    <Box p={8}>
      <Heading as="h1" size="xl" mb={6}>
        Competitive Programming Snippets Modified and Assembled by <u><a href="https://www.linkedin.com/in/gourabmodak/">spexcher</a></u>
      </Heading>

      <VStack align="start" spacing={8}>
      {/* Simple Segment Tree */}
      <Box width="100%">
          <HStack justifyContent="space-between" width="100%">
            <Heading as="h2" size="lg" mb={2}>
              Simple Segment Tree
            </Heading>
            <Button size="sm" onClick={onCopy_simple_segment_tree}>
              {hasCopied_simple_segment_tree ? "Copied!" : "Copy"}
            </Button>
          </HStack>
          <Text mb={2}>
            A segment tree is a data structure that allows querying and updating
            ranges of an array efficiently.
          </Text>
          <Code
            p={4}
            rounded="md"
            bg="gray.800"
            color="green.300"
            display="block"
            whiteSpace="pre-wrap"
          >
            {simple_segment_tree_code}
          </Code>
        </Box>
        <Divider />
        {/* Segment Tree */}
        <Box width="100%">
          <HStack justifyContent="space-between" width="100%">
            <Heading as="h2" size="lg" mb={2}>
              Segment Tree
            </Heading>
            <Button size="sm" onClick={onCopySegmentTree}>
              {hasCopiedSegmentTree ? "Copied!" : "Copy"}
            </Button>
          </HStack>
          <Text mb={2}>
            A segment tree is a data structure that allows querying and updating
            ranges of an array efficiently.
          </Text>
          <Code
            p={4}
            rounded="md"
            bg="gray.800"
            color="green.300"
            display="block"
            whiteSpace="pre-wrap"
          >
            {segmentTreeCode}
          </Code>
        </Box>
        <Divider />
        {/* Segment Tree */}
        <Box width="100%">
          <HStack justifyContent="space-between" width="100%">
            <Heading as="h2" size="lg" mb={2}>
              DSU (Disjoint Set Union / Union Find Data Structure)
            </Heading>
            <Button size="sm" onClick={onCopyDSU}>
              {hasCopiedDSU ? "Copied!" : "Copy"}
            </Button>
          </HStack>
          <Text mb={2}>
          The Disjoint Set Union (DSU) data structure, which allows you to add edges to a graph and test whether two vertices of the graph are connected.
          </Text>
          <Code
            p={4}
            rounded="md"
            bg="gray.800"
            color="green.300"
            display="block"
            whiteSpace="pre-wrap"
          >
            {DSU_code}
          </Code>
        </Box>

        <Divider />

        {/* Binary Search */}
        <Box width="100%">
          <HStack justifyContent="space-between" width="100%">
            <Heading as="h2" size="lg" mb={2}>
              Binary Search
            </Heading>
            <Button size="sm" onClick={onCopyBinarySearch}>
              {hasCopiedBinarySearch ? "Copied!" : "Copy"}
            </Button>
          </HStack>
          <Text mb={2}>
            Binary search is a classic algorithm for finding an element in a
            sorted array.
          </Text>
          <Code
            p={4}
            rounded="md"
            bg="gray.800"
            color="green.300"
            display="block"
            whiteSpace="pre-wrap"
          >
            {binarySearchCode}
          </Code>
        </Box>

        <Divider />
        {/* HLD Code */}
        <Box width="100%">
          <HStack justifyContent="space-between" width="100%">
            <Heading as="h2" size="lg" mb={2}>
              Heavy Light Decomposition
            </Heading>
            <Button size="sm" onClick={onCopy_hld_code}>
              {hasCopied_hld_code ? "Copied!" : "Copy"}
            </Button>
          </HStack>
          <Text mb={2}>
            Operations on trees !
          </Text>
          <Code
            p={4}
            rounded="md"
            bg="gray.800"
            color="green.300"
            display="block"
            whiteSpace="pre-wrap"
          >
            {hld_code}
          </Code>
        </Box>
        <Divider />
        {/* Substring Hash Code */}
        <Box width="100%">
          <HStack justifyContent="space-between" width="100%">
            <Heading as="h2" size="lg" mb={2}>
            Substring Hash Code
            </Heading>
            <Button size="sm" onClick={onCopy_substring_hash_code}>
              {hasCopied_substring_hash_code ? "Copied!" : "Copy"}
            </Button>
          </HStack>
          <Text mb={2}>
            Substring Hash in O(1)
          </Text>
          <Code
            p={4}
            rounded="md"
            bg="gray.800"
            color="green.300"
            display="block"
            whiteSpace="pre-wrap"
          >
            {substring_hash_code}
          </Code>
        </Box>

        <Divider />

        {/* Dijkstra's Algorithm */}
        <Box width="100%">
          <HStack justifyContent="space-between" width="100%">
            <Heading as="h2" size="lg" mb={2}>
              Dijkstra's Algorithm
            </Heading>
            <Button size="sm" onClick={onCopyDijkstra}>
              {hasCopiedDijkstra ? "Copied!" : "Copy"}
            </Button>
          </HStack>
          <Text mb={2}>
            Dijkstra's algorithm finds the shortest path between nodes in a
            graph.
          </Text>
          <Code
            p={4}
            rounded="md"
            bg="gray.800"
            color="green.300"
            display="block"
            whiteSpace="pre-wrap"
          >
            {dijkstraCode}
          </Code>
        </Box>
        <Divider />

        {/* Matrix Operations */}
        <Box width="100%">
          <HStack justifyContent="space-between" width="100%">
            <Heading as="h2" size="lg" mb={2}>
              Matrix Operations
            </Heading>
            <Button size="sm" onClick={onCopyMatrix}>
              {hasCopiedMatrix ? "Copied!" : "Copy"}
            </Button>
          </HStack>
          <Text mb={2}>
            Common Matrix operations and exponentiation
          </Text>
          <Code
            p={4}
            rounded="md"
            bg="gray.800"
            color="green.300"
            display="block"
            whiteSpace="pre-wrap"
          >
            {matrix_code}
          </Code>
        </Box>
        <Divider />
        {/* Add, Multiply Strings */}
        <Box width="100%">
          <HStack justifyContent="space-between" width="100%">
            <Heading as="h2" size="lg" mb={2}>
            Add, Multiply Strings
            </Heading>
            <Button size="sm" onClick={onCopy_add_multiply_strings}>
              {hasCopied_add_multiply_strings ? "Copied!" : "Copy"}
            </Button>
          </HStack>
          <Text mb={2}>
           Functions to Add and Multiply Strings
          </Text>
          <Code
            p={4}
            rounded="md"
            bg="gray.800"
            color="green.300"
            display="block"
            whiteSpace="pre-wrap"
          >
            {add_multiply_strings_code}
          </Code>
        </Box>
        <Divider />
        {/* Multiply Polynomials */}
        <Box width="100%">
          <HStack justifyContent="space-between" width="100%">
            <Heading as="h2" size="lg" mb={2}>
            Multiply Polynomials
            </Heading>
            <Button size="sm" onClick={onCopy_multiply_polynomials}>
              {hasCopied_multiply_polynomials ? "Copied!" : "Copy"}
            </Button>
          </HStack>
          <Text mb={2}>
          Multiply Polynomials by Karastuba Method
          </Text>
          <Code
            p={4}
            rounded="md"
            bg="gray.800"
            color="green.300"
            display="block"
            whiteSpace="pre-wrap"
          >
            {multiply_polynomials_code}
          </Code>
        </Box>
        <Divider />
        {/* KMP */}
        <Box width="100%">
          <HStack justifyContent="space-between" width="100%">
            <Heading as="h2" size="lg" mb={2}>
            KMP
            </Heading>
            <Button size="sm" onClick={onCopy_KMP}>
              {hasCopied_KMP ? "Copied!" : "Copy"}
            </Button>
          </HStack>
          <Text mb={2}>
          the Knuth-Morris-Pratt algorithm (or KMP algorithm) is a string-searching algorithm that searches for occurrences of a "word" W within a main "text string" S by employing the observation that when a mismatch occurs, the word itself embodies sufficient information to determine where the next match could begin, thus bypassing re-examination of previously matched characters. 
          </Text>
          <Code
            p={4}
            rounded="md"
            bg="gray.800"
            color="green.300"
            display="block"
            whiteSpace="pre-wrap"
          >
            {KMP_code}
          </Code>
        </Box>
        <Divider />
        {/* ETF */}
        <Box width="100%">
          <HStack justifyContent="space-between" width="100%">
            <Heading as="h2" size="lg" mb={2}>
            Euler Totient Function
            </Heading>
            <Button size="sm" onClick={onCopy_ETF}>
              {hasCopied_ETF ? "Copied!" : "Copy"}
            </Button>
          </HStack>
          <Text mb={2}>
          In number theory, Euler's totient function counts the positive integers up to a given integer n that are relatively prime to n. It is written using the Greek letter phi as φ ( n ) , and may also be called Euler's phi function. In other words, it is the number of integers k in the range 1 ≤ k ≤ n for which the greatest common divisor gcd(n, k) is equal to 1. The integers k of this form are sometimes referred to as totatives of n.
          </Text>
          <Code
            p={4}
            rounded="md"
            bg="gray.800"
            color="green.300"
            display="block"
            whiteSpace="pre-wrap"
          >
            {ETF_code}
          </Code>
        </Box>
        <Divider />
        {/* Sieve of Eratostenes */}
        <Box width="100%">
          <HStack justifyContent="space-between" width="100%">
            <Heading as="h2" size="lg" mb={2}>
            Sieve of Eratostenes 
            </Heading>
            <Button size="sm" onClick={onCopy_SOE}>
              {hasCopied_SOE ? "Copied!" : "Copy"}
            </Button>
          </HStack>
          <Text mb={2}>
          In mathematics, the sieve of Eratosthenes is an ancient algorithm for finding all prime numbers up to any given limit. 
          </Text>
          <Code
            p={4}
            rounded="md"
            bg="gray.800"
            color="green.300"
            display="block"
            whiteSpace="pre-wrap"
          >
            {SOE_code}
          </Code>
        </Box>

        <Divider />

      </VStack>
    </Box>
  );
};

export default SnippetPage;
