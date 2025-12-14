export const simple_segment_tree_code = `vector<int> segtree(4 * size);
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
export const matrix_code = `struct matrix
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

// Binary Search snippet text
export const binarySearchCode = `auto poss = [&](int mid) {

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
export const dijkstraCode = `
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
export const substring_hash_code = `string s;
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

export const DSU_code = `struct DSU
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

export const multiply_polynomials_code = `vector<int> karatsuba_multiply_polynomials(const vector<int> &a, const vector<int> &b)
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

export const KMP_code = `// KMP Algorithm template --------------------------------
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

export const ETF_code = `int ETF(int n)
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
export const SOE_code = `const int N = 10005;
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
