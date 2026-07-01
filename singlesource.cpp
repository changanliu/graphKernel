#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

using namespace std;

struct Graph {
    int n;
    vector<vector<int>> adj;

    explicit Graph(int n_nodes = 0) : n(n_nodes), adj(n_nodes) {}

    void add_edge(int u, int v) {
        if (u == v) return;
        adj[u].push_back(v);
        adj[v].push_back(u);
    }

    long long edges() const {
        long long twice_m = 0;
        for (const auto& nbrs : adj) twice_m += (long long)nbrs.size();
        return twice_m / 2;
    }
};

Graph read_edge(const string& path) {
    ifstream fin(path);
    if (!fin) {
        cerr << "Cannot open graph file: " << path << "\n";
        exit(1);
    }

    vector<pair<int, int>> edges;
    vector<int> ids;
    string line;
    while (getline(fin, line)) {
        if (line.empty() || line[0] == '#') continue;
        istringstream iss(line);
        int u, v;
        if (!(iss >> u >> v) || u == v) continue;
        edges.emplace_back(u, v);
        ids.push_back(u);
        ids.push_back(v);
    }

    sort(ids.begin(), ids.end());
    ids.erase(unique(ids.begin(), ids.end()), ids.end());

    unordered_map<int, int> id_map;
    id_map.reserve(ids.size() * 2 + 1);
    for (int i = 0; i < (int)ids.size(); ++i) id_map[ids[i]] = i;

    Graph graph((int)ids.size());
    for (auto [u0, v0] : edges) graph.add_edge(id_map[u0], id_map[v0]);
    for (auto& nbrs : graph.adj) {
        sort(nbrs.begin(), nbrs.end());
        nbrs.erase(unique(nbrs.begin(), nbrs.end()), nbrs.end());
    }
    return graph;
}

vector<double> multiply_transition(const Graph& graph, const vector<double>& state) {
    vector<double> next(graph.n, 0.0);
    for (int u = 0; u < graph.n; ++u) {
        double mass = state[u];
        if (mass == 0.0) continue;
        const auto& nbrs = graph.adj[u];
        if (nbrs.empty()) {
            next[u] += mass;
            continue;
        }
        double share = mass / (double)nbrs.size();
        for (int v : nbrs) next[v] += share;
    }
    return next;
}

double target_weight(double alpha, int ell) {
    return sqrt(alpha) * pow(1.0 - alpha, 0.5 * ell);
}

int sample_index(const vector<double>& weights, double total, mt19937& rng) {
    if (total <= 0.0) return -1;
    uniform_real_distribution<double> unif(0.0, total);
    double r = unif(rng);
    double prefix = 0.0;
    for (int i = 0; i < (int)weights.size(); ++i) {
        prefix += weights[i];
        if (prefix >= r) return i;
    }
    return (int)weights.size() - 1;
}

struct WalkSample {
    bool ok = false;
    int node = -1;
    int extra_steps = 0;
    double start_total = 0.0;
};

WalkSample lazy_walk_tail(
    const Graph& graph,
    const vector<double>& start_mass,
    double stop_prob,
    int max_extra_steps,
    mt19937& rng
) {
    double start_total = accumulate(start_mass.begin(), start_mass.end(), 0.0);
    int curr = sample_index(start_mass, start_total, rng);
    if (curr < 0) return {};

    uniform_real_distribution<double> unif(0.0, 1.0);
    int k = 0;
    while (true) {
        if (k > max_extra_steps) return {};
        if (unif(rng) < stop_prob) {
            return {true, curr, k, start_total};
        }
        const auto& nbrs = graph.adj[curr];
        uniform_int_distribution<int> choose(0, (int)nbrs.size() - 1);
        curr = nbrs[choose(rng)];
        ++k;
    }
}

vector<vector<double>> build_node_histograms(
    const Graph& graph,
    int source,
    int L,
    long long W,
    int l_det,
    double stop_prob,
    double alpha,
    mt19937& rng
) {
    vector<vector<double>> hist(graph.n, vector<double>(L + 1, 0.0));
    vector<double> state(graph.n, 0.0);
    state[source] = 1.0;

    // hist[u][ell] stores the weighted contribution w_ell * e_source^T P^ell e_u.
    int exact_steps = min(l_det, L + 1);
    for (int ell = 0; ell < exact_steps; ++ell) {
        double weight = target_weight(alpha, ell);
        for (int u = 0; u < graph.n; ++u) {
            if (state[u] != 0.0) hist[u][ell] = weight * state[u];
        }
        state = multiply_transition(graph, state);
    }

    if (l_det > L) return hist;

    int max_extra_steps = L - l_det;
    for (long long w = 0; w < W; ++w) {
        WalkSample sample = lazy_walk_tail(graph, state, stop_prob, max_extra_steps, rng);
        if (!sample.ok) continue;
        int ell = l_det + sample.extra_steps;
        double proposal_len_prob = stop_prob * pow(1.0 - stop_prob, sample.extra_steps);
        if (proposal_len_prob <= 0.0) continue;
        // Correct the proposal length probability; projection adds no extra length weight.
        hist[sample.node][ell] += sample.start_total * target_weight(alpha, ell)
                                / (proposal_len_prob * (double)W);
    }
    return hist;
}

vector<vector<int>> make_signs(long long T, int L, mt19937& rng) {
    uniform_int_distribution<int> rad(0, 1);
    vector<vector<int>> signs(T, vector<int>(L + 1));
    for (long long t = 0; t < T; ++t) {
        for (int ell = 0; ell <= L; ++ell) signs[t][ell] = rad(rng) ? 1 : -1;
    }
    return signs;
}

vector<vector<double>> project_fingerprints(
    const vector<vector<double>>& hist,
    const vector<vector<int>>& signs
) {
    int n = (int)hist.size();
    long long T = (long long)signs.size();
    int L = (int)signs[0].size() - 1;
    vector<vector<double>> z(n, vector<double>(T, 0.0));

    for (int u = 0; u < n; ++u) {
        for (int ell = 0; ell <= L; ++ell) {
            double value = hist[u][ell];
            if (value == 0.0) continue;
            for (long long t = 0; t < T; ++t) z[u][t] += signs[t][ell] * value;
        }
    }
    return z;
}

double estimate_pair(
    const vector<vector<double>>& z_g,
    const vector<vector<double>>& z_h,
    int u,
    int v
) {
    long long T = (long long)z_g[u].size();
    double acc = 0.0;
    for (long long t = 0; t < T; ++t) acc += z_g[u][t] * z_h[v][t];
    return acc / (double)T;
}

vector<pair<int, int>> sample_target_pairs(
    const Graph& G,
    const Graph& H,
    int count,
    mt19937& rng
) {
    vector<pair<int, int>> out;
    out.reserve(count);
    if (count < 0 || (long long)count > 1LL * G.n * H.n) {
        cerr << "query-count must be between 0 and |V_G|*|V_H|.\n";
        exit(1);
    }
    unordered_set<long long> seen;
    uniform_int_distribution<int> dist_g(0, G.n - 1);
    uniform_int_distribution<int> dist_h(0, H.n - 1);
    while ((int)out.size() < count) {
        int u = dist_g(rng);
        int v = dist_h(rng);
        long long key = 1LL * u * H.n + v;
        if (seen.insert(key).second) out.emplace_back(u, v);
    }
    return out;
}

vector<pair<int, int>> parse_targets(const string& text) {
    vector<pair<int, int>> targets;
    stringstream ss(text);
    string item;
    while (getline(ss, item, ',')) {
        size_t pos = item.find(':');
        if (pos == string::npos) {
            cerr << "Bad target pair '" << item << "'. Use u:v,u:v format.\n";
            exit(1);
        }
        int u = stoi(item.substr(0, pos));
        int v = stoi(item.substr(pos + 1));
        targets.emplace_back(u, v);
    }
    return targets;
}

struct Args {
    string graph_g = "graphs/facebook.txt";
    string graph_h = "graphs/facebook.txt";
    double alpha = 0.15;
    int L = 60;
    long long T = 1000;
    long long W_g = 10000;
    long long W_h = 10000;
    int l_det = 3;
    double p_multiplier = 1.0;
    int source_g = 0;
    int source_h = 0;
    int query_count = 4;
    string targets;
    unsigned seed = 123;
    bool use_theory_budget = false;
    double epsilon = 0.10;
    double delta = 0.01;
};

void print_help(const char* prog) {
    cout << "Usage: " << prog << " [options]\n"
         << "Single-source CANWAS demo with current proposal-corrected lazy walks.\n\n"
         << "Options:\n"
         << "  --graph-g PATH           first graph edge list (default graphs/facebook.txt)\n"
         << "  --graph-h PATH           second graph edge list (default graphs/facebook.txt)\n"
         << "  --source-g INT           source node in graph G after remapping (default 0)\n"
         << "  --source-h INT           source node in graph H after remapping (default 0)\n"
         << "  --targets u:v,u:v        explicit target pairs after remapping\n"
         << "  --query-count INT        random target pairs if --targets is omitted (default 4)\n"
         << "  --alpha FLOAT            RWK decay parameter (default 0.15)\n"
         << "  --L INT                  truncation length (default 60)\n"
         << "  --T INT                  Rademacher sketch trials (default 1000)\n"
         << "  --W INT                  lazy walks per graph (default 10000)\n"
         << "  --W-g INT                lazy walks for graph G\n"
         << "  --W-h INT                lazy walks for graph H\n"
         << "  --l-det INT              exact deterministic prefix length (default 3)\n"
         << "  --p-multiplier FLOAT     multiplier for p*=1-sqrt(1-alpha) (default 1)\n"
         << "  --use-theory-budget      set L,T,W from epsilon/delta split\n"
         << "  --epsilon FLOAT          target additive epsilon for theory budget\n"
         << "  --delta FLOAT            failure probability for theory budget\n"
         << "  --seed INT               RNG seed (default 123)\n";
}

Args parse_args(int argc, char** argv) {
    Args args;
    for (int i = 1; i < argc; ++i) {
        string key = argv[i];
        auto value = [&]() -> string {
            if (i + 1 >= argc) {
                cerr << "Missing value after " << key << "\n";
                exit(1);
            }
            return argv[++i];
        };
        if (key == "--help" || key == "-h") {
            print_help(argv[0]);
            exit(0);
        } else if (key == "--graph-g") args.graph_g = value();
        else if (key == "--graph-h") args.graph_h = value();
        else if (key == "--source-g") args.source_g = stoi(value());
        else if (key == "--source-h") args.source_h = stoi(value());
        else if (key == "--targets") args.targets = value();
        else if (key == "--query-count") args.query_count = stoi(value());
        else if (key == "--alpha") args.alpha = stod(value());
        else if (key == "--L") args.L = stoi(value());
        else if (key == "--T") args.T = stoll(value());
        else if (key == "--W") args.W_g = args.W_h = stoll(value());
        else if (key == "--W-g") args.W_g = stoll(value());
        else if (key == "--W-h") args.W_h = stoll(value());
        else if (key == "--l-det") args.l_det = stoi(value());
        else if (key == "--p-multiplier") args.p_multiplier = stod(value());
        else if (key == "--seed") args.seed = (unsigned)stoul(value());
        else if (key == "--use-theory-budget") args.use_theory_budget = true;
        else if (key == "--epsilon") args.epsilon = stod(value());
        else if (key == "--delta") args.delta = stod(value());
        else {
            cerr << "Unknown option: " << key << "\n";
            print_help(argv[0]);
            exit(1);
        }
    }
    return args;
}

void apply_theory_budget(Args& args) {
    double eps_t = 0.20 * args.epsilon;
    double eps_r = 0.60 * args.epsilon;
    double eps_w = 0.20 * args.epsilon;
    args.L = (int)ceil(log(eps_t) / log(1.0 - args.alpha));
    double log_term = log(4.0 * (args.L + 1.0) / args.delta);
    args.T = (long long)ceil(3.0 * (args.L + 1.0) * log_term / (eps_r * eps_r));
    long long W = (long long)ceil(pow(args.L + 1.0, 2.0) * log_term / (eps_w * eps_w));
    args.W_g = W;
    args.W_h = W;
}

int main(int argc, char** argv) {
    Args args = parse_args(argc, argv);
    if (!(args.alpha > 0.0 && args.alpha < 1.0)) {
        cerr << "--alpha must be in (0,1)\n";
        return 1;
    }
    if (args.use_theory_budget) {
        if (!(args.epsilon > 0.0) || !(args.delta > 0.0 && args.delta < 1.0)) {
            cerr << "Theory budget requires epsilon > 0 and delta in (0,1).\n";
            return 1;
        }
        apply_theory_budget(args);
    }
    if (args.L < 0 || args.T <= 0 || args.W_g <= 0 || args.W_h <= 0 || args.l_det < 0) {
        cerr << "Require L >= 0, T > 0, W_G > 0, W_H > 0, and l_det >= 0.\n";
        return 1;
    }
    if (args.query_count < 0) {
        cerr << "--query-count must be non-negative.\n";
        return 1;
    }

    double p_star = 1.0 - sqrt(1.0 - args.alpha);
    double stop_prob = args.p_multiplier * p_star;
    if (!(stop_prob > 0.0 && stop_prob < 1.0)) {
        cerr << "Invalid proposal stop probability: " << stop_prob << "\n";
        return 1;
    }

    mt19937 rng(args.seed);
    Graph G = read_edge(args.graph_g);
    Graph H = read_edge(args.graph_h);
    if (G.n == 0 || H.n == 0) {
        cerr << "Input graphs must be non-empty.\n";
        return 1;
    }
    if (args.source_g < 0 || args.source_g >= G.n || args.source_h < 0 || args.source_h >= H.n) {
        cerr << "Source node out of range after graph remapping.\n";
        return 1;
    }

    vector<pair<int, int>> targets;
    if (!args.targets.empty()) {
        targets = parse_targets(args.targets);
    } else {
        targets = sample_target_pairs(G, H, args.query_count, rng);
    }
    for (auto [u, v] : targets) {
        if (u < 0 || u >= G.n || v < 0 || v >= H.n) {
            cerr << "Target pair out of range after graph remapping: "
                 << u << ":" << v << "\n";
            return 1;
        }
    }

    vector<vector<int>> signs = make_signs(args.T, args.L, rng);
    vector<vector<double>> hist_g = build_node_histograms(
        G, args.source_g, args.L, args.W_g, args.l_det, stop_prob, args.alpha, rng);
    vector<vector<double>> hist_h = build_node_histograms(
        H, args.source_h, args.L, args.W_h, args.l_det, stop_prob, args.alpha, rng);
    vector<vector<double>> z_g = project_fingerprints(hist_g, signs);
    vector<vector<double>> z_h = project_fingerprints(hist_h, signs);

    cout << fixed << setprecision(8);
    cout << "Graph G: n=" << G.n << " m=" << G.edges() << "\n";
    cout << "Graph H: n=" << H.n << " m=" << H.edges() << "\n";
    cout << "alpha=" << args.alpha << " L=" << args.L << " T=" << args.T
         << " W_G=" << args.W_g << " W_H=" << args.W_h
         << " l_det=" << args.l_det << " p=" << stop_prob << "\n";
    cout << "sources=(" << args.source_g << "," << args.source_h << ")\n";

    for (auto [u, v] : targets) {
        double score = estimate_pair(z_g, z_h, u, v);
        cout << "target=(" << u << "," << v << ") kernel=" << score << "\n";
    }
    return 0;
}
