# Conversion measurements

Benchmarks compare the reviewed original `main` commit
`9dc62f74b9ce15d883e4324a89887b76dd8b4fe3` with the 2.0.0 source fingerprint
recorded in [the raw receipt](review_20261021.json). Each sample runs in a fresh
process; three samples per case were collected serially after correctness tests
finished. Chain/star inputs contain 20,000 segments and 19,999 directed links.
The third input is the bundled DRB1 graph. Input hashes, dependency versions,
platform, individual timings, peak process RSS, and adjacency fingerprints are
in the receipt. No stored sequences/tags or alignment subdivision were requested.

Times cover parsing/building. Peak RSS includes interpreter/library imports and
output fingerprint verification; it is not a graph-only memory estimate. These
are observations on one machine, not performance guarantees or capacity tests.

| Input | Backend / output | Original median ms | 2.0 median ms | Original peak MiB (median) | 2.0 peak MiB (median) | Equivalent weighted adjacency |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| chain | networkx / graph | 65.24 | 102.96 | 84.7 | 89.4 | yes |
| chain | networkx / matrix | 42.09 | 60.58 | 69.0 | 71.1 | yes |
| chain | igraph / graph | 1901.49 | 69.15 | 75.8 | 79.9 | yes |
| chain | igraph / matrix | error | 60.67 | — | 71.3 | original error |
| star | networkx / graph | 62.03 | 105.85 | 82.6 | 85.6 | yes |
| star | networkx / matrix | 41.82 | 60.74 | 69.1 | 71.3 | yes |
| star | igraph / graph | 2093.50 | 69.02 | 76.7 | 79.9 | yes |
| star | igraph / matrix | error | 60.06 | — | 71.3 | original error |
| bundled-drb1 | networkx / graph | 18.47 | 31.41 | 69.1 | 70.3 | intentional policy change |
| bundled-drb1 | networkx / matrix | 13.66 | 20.75 | 66.1 | 66.6 | yes |
| bundled-drb1 | igraph / graph | 246.68 | 22.90 | 69.3 | 68.9 | yes |
| bundled-drb1 | igraph / matrix | error | 20.68 | — | 66.4 | original error |

Batched igraph construction is faster on these inputs. NetworkX conversion has
additional validation and aggregation/provenance work and is slower in these
measurements. Peak RSS does not improve uniformly. Typed matrix buffers reduce
Python object bookkeeping, but these measurements do not establish lower total
RSS at large scale. No general speed or memory reduction is claimed.

The original igraph matrix path fails because it passes an unsupported argument
at adjacency export. DRB1 NetworkX graph fingerprints differ intentionally: the
original simple graph discarded repeated-link weights, while 2.0 sums them.
Its original matrix already summed those records, so matrix fingerprints match.
The original igraph graph retained parallel records; summing them for the
fingerprint gives the same weighted adjacency as 2.0. Raw edge counts therefore
need not match even where weighted adjacency does.

## Reproduce

```bash
python scripts/benchmark.py --nodes 20000 --repeats 3 --output benchmark.json
# Optional comparison with an existing checkout of the original revision:
python scripts/benchmark.py --baseline /path/to/original-checkout --output comparison.json
```

Correctness tests do not assert timing ratios. Biological validity and
million-node capacity require separate evidence.
