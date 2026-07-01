# graphKernel code

- `distribution.cpp`: distributional random-walk-kernel query.
- `singlesource.cpp`: single-source random-walk-kernel queries.
- `graphs/facebook.txt`: example graph.


## Build

```bash
g++ -O3 -std=c++17 distribution.cpp -o distribution
g++ -O3 -std=c++17 singlesource.cpp -o singlesource
```

## Distributional query

The code uses uniform start distributions on both graphs.  The default target
is an all-ones target-weight vector with entries in [0,1].

```bash
./distribution
./distribution --graph-g graphs/facebook.txt --graph-h graphs/facebook.txt \
  --alpha 0.15 --L 60 --T 1000 --W 10000 --l-det 3
```

## Single-source query

The code first builds node fingerprints for all targets from the selected
source nodes and then answers target-pair queries in `O(T)` time.

```bash
./singlesource
./singlesource --source-g 0 --source-h 0
```
