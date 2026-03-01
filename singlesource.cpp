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
 * 3. Single-Source Node Feature Extraction (Node-level Fingerprinting)
 * ========================================= */
// Returns: T-dimensional feature matrix Z[u][tau] for each node
vector<vector<double>> build_node_fingerprints(
    const Graph& G, int source, 
    int L, double alpha, int T, long W, int l_det, 
    mt19937& rng, const vector<vector<int>>& g) 
{
    double p = 1.0 - sqrt(1.0 - alpha);
    
    // H_G[u][l] records the probability mass of reaching node u at step l from the source
    vector<vector<double>> H_G(G.n, vector<double>(L + 1, 0.0));
    
    // Phase 1: Deterministic prefix
    vector<double> s_vec(G.n, 0.0);
    s_vec[source] = 1.0;
    for (int l = 0; l < l_det; ++l) {
        for (int u = 0; u < G.n; ++u) {
            H_G[u][l] = s_vec[u];
        }
        s_vec = mult_P(G, s_vec); 
    }
    
    // Phase 2: Lazy-walk tails
    for (long w = 0; w < W; ++w) {
        auto [x, k] = lazy_walk(G, s_vec, p, L - l_det, rng);
        // Increment probability mass by 1/W for each sample hit
        H_G[x][l_det + k] += 1.0 / W;
    }
    
    // Phase 3: Node-level Rademacher Projection
    vector<vector<double>> Z(G.n, vector<double>(T + 1, 0.0));
    for (int u = 0; u < G.n; ++u) {
        for (int l = 0; l <= L; ++l) {
            if (H_G[u][l] > 0) { // Sparsity optimization: only project non-zero histogram entries
                double w_l = sqrt(alpha) * pow(1.0 - alpha, l / 2.0);
                for (int tau = 1; tau <= T; ++tau) {
                    Z[u][tau] += w_l * H_G[u][l] * g[tau][l];
                }
            }
        }
    }
    return Z;
}

/* =========================================
 * 4. Arbitrary Pair Kernel Query (Decoupled Estimator)
 * ========================================= */
double estimate_single_pair(
    const vector<vector<double>>& Z_G, 
    const vector<vector<double>>& Z_H, 
    int u, int v, int T, double alpha) 
{
    double est = 0.0;
    // Directly compute the inner product of the two node features
    for (int tau = 1; tau <= T; ++tau) {
        est += Z_G[u][tau] * Z_H[v][tau];
    }
    double norm_factor = alpha / pow(1.0 - sqrt(1.0 - alpha), 2);
    return norm_factor * (est / T);
}

/* =========================================
 * 5. Main Function
 * ========================================= */
int main() {
    mt19937 rng(123); 

    cout << "Loading graphs from files..." << endl;
    Graph G = read_edge("./graphs/facebook.txt");
    Graph H = read_edge("./graphs/facebook.txt");  

    // Set the starting nodes for the Single-source query
    int s_G = 0; 
    int s_H = 0;

    // Algorithm parameters
    int L = 100;            
    double alpha = 0.15;   
    int T = 1000;           
    long W_G = 50000;      
    long W_H = 50000;      
    int l_G = 3;           
    int l_H = 3;           

    // Pre-generate the globally shared Rademacher matrix
    vector<vector<int>> g(T + 1, vector<int>(L + 1));
    uniform_int_distribution<> rad_dist(0, 1);
    for (int tau = 1; tau <= T; ++tau) {
        for (int l = 0; l <= L; ++l) {
            g[tau][l] = rad_dist(rng) ? 1 : -1;
        }
    }

    cout << "Building Node-level Fingerprints (Phase 1-3)..." << endl;
    // Build feature vectors for all nodes in graph G and H respectively (computed only once)
    vector<vector<double>> Z_G = build_node_fingerprints(G, s_G, L, alpha, T, W_G, l_G, rng, g);
    vector<vector<double>> Z_H = build_node_fingerprints(H, s_H, L, alpha, T, W_H, l_H, rng, g);

    cout << "Ready for O(T) pair queries!" << endl;
    cout << "---------------------------------------" << endl;

    // Phase 4: Lightning-fast kernel queries for any node pair in O(T) time
    // Test querying a few different target pairs (u, v)
    vector<pair<int, int>> query_targets = {
        {10, 20}, {1726, 2351}, {100, 100}, {5, 5}
    };

    for (auto [u, v] : query_targets) {
        // Ensure node IDs are within bounds
        if (u < G.n && v < H.n) {
            double kernel_score = estimate_single_pair(Z_G, Z_H, u, v, T, alpha);
            cout << "Query from (" << s_G << "," << s_H << ") to (" 
                 << u << "," << v << ") -> Kernel: " 
                 << setprecision(8) << kernel_score << endl;
        }
    }

    return 0;
}
