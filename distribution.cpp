#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <algorithm>
#include <unordered_map>

using namespace std;

/* =========================================
 * 1. Graph Structure Definition & File I/O
 * ========================================= */
struct Graph {
    int n; 
    vector<vector<int>> adj;
    explicit Graph(int N = 0) : n(N), adj(N) {}
    void add_edge(int u, int v) { 
        adj[u].push_back(v); 
        adj[v].push_back(u); 
    }
};

// Read graph from an edgelist file and automatically map to contiguous IDs from 0 to n-1
Graph read_edge(const string& f){
    ifstream fin(f); 
    if(!fin) {
        cerr << "Error: Cannot open file " << f << endl;
        exit(1);
    }
    vector<pair<int,int>> E; 
    vector<int> ids; 
    string ln;
    while(getline(fin,ln)){
        if(ln.empty() || ln[0]=='#') continue;
        istringstream is(ln); 
        int u, v; 
        if(is >> u >> v){
            E.emplace_back(u, v);
            ids.push_back(u);
            ids.push_back(v);
        }
    }
    sort(ids.begin(), ids.end()); 
    ids.erase(unique(ids.begin(), ids.end()), ids.end());
    unordered_map<int,int> mp; 
    for(int i = 0; i < (int)ids.size(); ++i) {
        mp[ids[i]] = i;
    }
    Graph G(ids.size()); 
    for(auto [u, v] : E) {
        G.add_edge(mp[u], mp[v]);
    }
    return G;
}

// Compute the inner product of two vectors
double dot_product(const vector<double>& s, const vector<double>& w) {
    double res = 0.0;
    for (size_t i = 0; i < s.size(); ++i) res += s[i] * w[i];
    return res;
}

// Exact state vector evolution: s <- s * P_G
vector<double> mult_P(const Graph& G, const vector<double>& s) {
    vector<double> next_s(G.n, 0.0);
    for (int u = 0; u < G.n; ++u) {
        if (s[u] > 0 && !G.adj[u].empty()) {
            double share = s[u] / G.adj[u].size();
            for (int v : G.adj[u]) next_s[v] += share;
        }
    }
    return next_s;
}

/* =========================================
 * 2. Core Sampling Module: Lazy Random Walk
 * ========================================= */
pair<int, int> lazy_walk(const Graph& G, const vector<double>& s, double p, int max_steps, mt19937& rng) {
    double sum_s = 0;
    for(double val : s) sum_s += val;
    if (sum_s <= 1e-9) return {0, 0};

    discrete_distribution<> start_dist(s.begin(), s.end());
    int curr = start_dist(rng);
    
    uniform_real_distribution<> unif(0.0, 1.0);
    int k = 0;
    for (; k < max_steps; ++k) {
        if (unif(rng) < p || G.adj[curr].empty()) break;
        uniform_int_distribution<> neighbor_dist(0, G.adj[curr].size() - 1);
        curr = G.adj[curr][neighbor_dist(rng)];
    }
    return {curr, k};
}

/* =========================================
 * 3. Distributional GRW Algorithm
 * ========================================= */
double dist_grw(const Graph& G, const Graph& H,
                const vector<double>& v_G, const vector<double>& w_G,
                const vector<double>& v_H, const vector<double>& w_H,
                int L, double alpha, int T, 
                long W_G, long W_H, 
                int l_G, int l_H, 
                mt19937& rng) 
{
    double p = 1.0 - sqrt(1.0 - alpha);
    vector<double> h_G(L + 1, 0.0), h_H(L + 1, 0.0);
    vector<double> z_G(T + 1, 0.0), z_H(T + 1, 0.0);
    
    vector<vector<int>> g(T + 1, vector<int>(L + 1));
    uniform_int_distribution<> rad_dist(0, 1);
    for (int tau = 1; tau <= T; ++tau) {
        for (int l = 0; l <= L; ++l) g[tau][l] = rad_dist(rng) ? 1 : -1;
    }

    // Phase 1: Deterministic traversal prefix
    vector<double> s_G = v_G;
    for (int l = 0; l < l_G; ++l) {
        h_G[l] = dot_product(s_G, w_G);
        s_G = mult_P(G, s_G); 
    }
    vector<double> s_H = v_H;
    for (int l = 0; l < l_H; ++l) {
        h_H[l] = dot_product(s_H, w_H);
        s_H = mult_P(H, s_H); 
    }

    // Phase 2: Lazy-walk tails
    for (long w = 1; w <= W_G; ++w) {
        auto [x, k] = lazy_walk(G, s_G, p, L - l_G, rng);
        h_G[l_G + k] += (1.0 / W_G) * w_G[x];
    }
    for (long w = 1; w <= W_H; ++w) {
        auto [y, k] = lazy_walk(H, s_H, p, L - l_H, rng);
        h_H[l_H + k] += (1.0 / W_H) * w_H[y];
    }

    // Phase 3: Batched Rademacher Projection
    for (int l = 0; l <= L; ++l) {
        double w_l = sqrt(alpha) * pow(1.0 - alpha, l / 2.0);
        for (int tau = 1; tau <= T; ++tau) {
            z_G[tau] += w_l * h_G[l] * g[tau][l];
            z_H[tau] += w_l * h_H[l] * g[tau][l];
        }
    }

    // Phase 4: Final decoupled estimator
    double est = 0.0;
    for (int tau = 1; tau <= T; ++tau) est += z_G[tau] * z_H[tau];
    
    double norm_factor = alpha / pow(1.0 - sqrt(1.0 - alpha), 2);
    return norm_factor * (est / T);
}

/* =========================================
 * 4. Main Function
 * ========================================= */
int main() {
    mt19937 rng(123); 

    cout << "Loading graphs from files..." << endl;
    // Assuming two files for comparison. If only one is available, pass the same file to measure the self-kernel.
    Graph G = read_edge("./graphs/facebook.txt");
    Graph H = read_edge("./graphs/facebook.txt");  

    cout << "Graph G: " << G.n << " nodes." << endl;
    cout << "Graph H: " << H.n << " nodes." << endl;

    // === [Key Modification: Uniform Distribution Initialization] ===
    // Set both v and w to a uniform distribution 1/n across the entire graph
    vector<double> v_G(G.n, 1.0 / G.n);
    vector<double> w_G(G.n, 1.0 / G.n);
    
    vector<double> v_H(H.n, 1.0 / H.n);
    vector<double> w_H(H.n, 1.0 / H.n);

    // Algorithm parameters
    int L = 100;            
    double alpha = 0.15;   
    int T = 1000;           
    long W_G = 50000;      
    long W_H = 50000;      
    int l_G = 3;           
    int l_H = 3;           

    cout << "Running Distributional GRW Algorithm..." << endl;
    double estimated_kernel = dist_grw(G, H, 
                                       v_G, w_G, 
                                       v_H, w_H, 
                                       L, alpha, T, 
                                       W_G, W_H, 
                                       l_G, l_H, rng);

    cout << "---------------------------------------" << endl;
    cout << "Estimated Distributional Kernel Value: " 
         << setprecision(8) << estimated_kernel << endl;

    return 0;
}
