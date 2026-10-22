#!/usr/bin/env python3
"""Path-aware topology report for a GFA with genome paths and source FASTA.

Writes, into ``--out``:

``manifest.json``
    Input hashes, parameters, software versions and tool availability.
``fidelity.json``
    Path-fidelity receipt against the source FASTA, with mismatch classes.
``content_groups.tsv``
    Group-by-group summary of base-aware shared content (all selected paths).
``subset.tsv`` / ``subset_pairs.tsv``
    A seeded random subset per group, and for every subset pair: shared
    content, collinear chain, graph connectivity distance and an exact k-mer
    Jaccard computed from the FASTA alone.
``bridges.json`` / ``bridge_sizes.tsv``
    Blocks of contiguous sequence shared across groups, with path witnesses,
    contexts and exact-occurrence support in the FASTA.
``sensitivity.tsv``
    The subset measures after renaming and after chopping segments.
``summary.md``
    The numbers above in prose, with their limits.

Groups must come from outside the graph (``--labels`` TSV or
``--label-regex``). Nothing here infers recombination, ancestry or homology:
blocks and distances describe the graph construction and are checked against
the FASTA where possible.

Example::

    python examples/topology_report.py --gfa graph.gfa.gz --reference genomes.fa \\
        --label-regex 'GENUS=([^_]+)' --circular --ambiguity as-n --out report/
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import platform
import random
import re
import shutil
import sys
import tempfile

import numpy as np
import scipy
from scipy.stats import spearmanr

import gfa2network
from gfa2network.analysis import genome_distance_matrix
from gfa2network.genome_paths import _AS_N, load_genome_paths, read_fasta
from gfa2network.model import reverse_complement
from gfa2network.neighborhoods import bridge_blocks
from gfa2network.path_compare import compare_paths, shared_content_matrix
from gfa2network.transform import chop, rename


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def classify(spelled, reference, circular):
    """Name the relation between a spelled path and its source sequence."""
    s, r = spelled.upper(), reference.upper()
    if s == r:
        return "identical"
    sn, rn = s.translate(_AS_N), r.translate(_AS_N)
    if sn == rn:
        return "ambiguity_codes_only"
    if reverse_complement(sn) == rn:
        return "reverse_complement"
    if (
        circular
        and len(sn) == len(rn)
        and (sn in rn + rn or reverse_complement(sn) in rn + rn)
    ):
        return "rotation"
    if sn in rn or reverse_complement(sn) in rn:
        return "graph_is_substring_of_source"
    if rn in sn or reverse_complement(rn) in sn:
        return "source_is_substring_of_graph"
    if len(sn) == len(rn):
        if all(a == b or a == ord("N") or b == ord("N") for a, b in zip(sn, rn)):
            return "differs_only_at_N"
        return "substitutions"
    return "other_length_difference"


def kmers(sequence, k):
    sequence = sequence.upper()
    found = set()
    for i in range(len(sequence) - k + 1):
        word = sequence[i : i + k]
        if word.translate(None, b"ACGT"):
            continue
        found.add(min(word, reverse_complement(word)))
    return found


def jaccard(first, second):
    union = len(first | second)
    return len(first & second) / union if union else math.nan


def finite(value):
    return value if value is not None and math.isfinite(value) else None


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--gfa", required=True)
    parser.add_argument("--reference", help="Source FASTA")
    parser.add_argument(
        "--reference-key", help="Regex whose group 1 maps a path name to its FASTA name"
    )
    parser.add_argument("--labels", help="TSV: path name, group")
    parser.add_argument(
        "--label-regex",
        help="Regex whose group 1 is the group, read from the path name",
    )
    parser.add_argument("--paths-regex", help="Analyze only matching path names")
    parser.add_argument("--circular", action="store_true")
    parser.add_argument("--ambiguity", choices=["exact", "as-n"], default="exact")
    parser.add_argument(
        "--unknown-overlap", choices=["strict", "zero"], default="strict"
    )
    parser.add_argument("--subset-per-group", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20261022)
    parser.add_argument("--min-bases", type=int, default=21)
    parser.add_argument("--kmer", type=int, default=21)
    parser.add_argument(
        "--chop",
        type=int,
        default=1,
        help="Max segment length for the chop check; 0 skips",
    )
    parser.add_argument(
        "--max-rotations",
        type=int,
        default=8,
        help="Circular chain rotations tried per pair",
    )
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    if not args.labels and not args.label_regex:
        parser.error("group labels are required: --labels or --label-regex")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    paths = load_genome_paths(args.gfa)
    selected = [
        n for n in paths if not args.paths_regex or re.search(args.paths_regex, n)
    ]
    groups = {}
    if args.labels:
        for line in Path(args.labels).read_text().splitlines():
            if line.strip() and not line.startswith("#"):
                name, group = line.split("\t")[:2]
                groups[name] = group
    if args.label_regex:
        for name in selected:
            match = re.search(args.label_regex, name)
            if match:
                groups[name] = match.group(1)
    groups = {name: group for name, group in groups.items() if name in set(selected)}
    references = read_fasta(args.reference) if args.reference else {}

    def key(name):
        if not args.reference_key:
            return name
        match = re.search(args.reference_key, name)
        return match.group(1) if match else name

    # ---------------------------------------------------------- fidelity
    mapping = {name: key(name) for name in selected}
    report = paths.audit(
        references,
        names=selected,
        reference_names=mapping,
        circular=args.circular,
        ambiguity=args.ambiguity,
        unknown_overlap=args.unknown_overlap,
    )
    classes, examples = Counter(), {}
    audited = {item.name: item for item in report.paths}
    for name in selected:
        spelled = paths.spell(name, unknown_overlap=args.unknown_overlap).sequence
        reference = references.get(mapping[name])
        if spelled is None or reference is None:
            label = "not_spelled" if spelled is None else "no_reference"
        else:
            label = classify(spelled, reference, args.circular)
            item = audited.get(name)
            if (
                label not in ("identical", "ambiguity_codes_only")
                and item
                and item.passed
            ):
                # The audit applies policies classify() does not repeat, such
                # as trimming an evidenced circular closing overlap.
                label = item.comparison
        classes[label] += 1
        examples.setdefault(label, name)
    fidelity = {
        "summary": report.summary,
        "policy": report.policy,
        "ok": report.ok,
        "classes": dict(classes.most_common()),
        "class_examples": examples,
    }
    (out / "fidelity.json").write_text(json.dumps(fidelity, indent=2) + "\n")
    faithful = {
        name
        for name, item in zip(selected, report.paths)
        if item.passed and item.comparison != "no_reference"
    } or set(selected)

    # ----------------------------------------------------------- content
    labeled = [name for name in selected if name in groups]
    _, content = shared_content_matrix(
        paths,
        labeled,
        unknown_overlap=args.unknown_overlap,
    )
    index = defaultdict(list)
    for i, name in enumerate(labeled):
        index[groups[name]].append(i)
    order = sorted(index, key=lambda g: (-len(index[g]), g))
    jac = content["jaccard"]
    rows = [
        "group_a\tgroup_b\tpaths_a\tpaths_b\tpairs\tmean_jaccard\tmedian_jaccard\tfraction_pairs_sharing"
    ]
    for x, ga in enumerate(order):
        for gb in order[x:]:
            block = jac[np.ix_(index[ga], index[gb])]
            if ga == gb:
                block = block[~np.eye(len(index[ga]), dtype=bool)]
            values = block[np.isfinite(block)].ravel()
            rows.append(
                f"{ga}\t{gb}\t{len(index[ga])}\t{len(index[gb])}\t{values.size}\t"
                f"{values.mean() if values.size else float('nan'):.4g}\t"
                f"{np.median(values) if values.size else float('nan'):.4g}\t"
                f"{(values > 0).mean() if values.size else float('nan'):.4g}"
            )
    (out / "content_groups.tsv").write_text("\n".join(rows) + "\n")

    # ------------------------------------------------------------ subset
    rng = random.Random(args.seed)
    subset = []
    for group in order:
        members = sorted(n for n in labeled if groups[n] == group and n in faithful)
        subset.extend(rng.sample(members, min(args.subset_per_group, len(members))))
    (out / "subset.tsv").write_text(
        "path\tgroup\n" + "".join(f"{n}\t{groups[n]}\n" for n in subset)
    )

    def measures(collection, gfa_path):
        result = {}
        if len(subset) < 2:
            return result, None
        for i, a in enumerate(subset):
            for b in subset[i + 1 :]:
                pair = compare_paths(
                    collection, a, b,
                    orientation="both",
                    circular=args.circular,
                    unknown_overlap=args.unknown_overlap,
                    max_rotations=args.max_rotations,
                )  # fmt: skip
                chains = [pair.collinear] + (
                    [pair.reverse_collinear] if pair.reverse_collinear else []
                )
                best = max(chains, key=lambda c: c.bases)
                result[(a, b)] = {
                    "jaccard": pair.content.jaccard,
                    "collinear_bases": best.bases,
                    "collinear_fraction": min(
                        best.fraction_a or 0, best.fraction_b or 0
                    ),
                    "collinear_orientation": best.orientation,
                    "exhaustive": all(c.exhaustive for c in chains),
                }
        distances, details = genome_distance_matrix(
            str(gfa_path),
            method="mean-nearest",
            names=subset,
            projection="traversal",
            cost="hops",
            return_details=True,
        )
        matrix = np.asarray(distances, dtype=float)
        for i, a in enumerate(subset):
            for j, b in enumerate(subset):
                if j > i:
                    result[(a, b)]["connectivity_hops"] = finite(float(matrix[i, j]))
        return result, details["projection"]["vertices"]

    original, vertices = measures(paths, args.gfa)
    sequences = {
        n: kmers(references[mapping[n]], args.kmer)
        for n in subset
        if mapping[n] in references
    }
    lines = [
        "path_a\tgroup_a\tpath_b\tgroup_b\tcontent_jaccard\tcollinear_bases\tcollinear_fraction\torientation\tconnectivity_hops\tkmer_jaccard"
    ]
    table = []
    for (a, b), item in original.items():
        k = (
            jaccard(sequences[a], sequences[b])
            if a in sequences and b in sequences
            else math.nan
        )
        item["kmer_jaccard"] = k
        table.append(item)
        content_value = "NA" if item["jaccard"] is None else f"{item['jaccard']:.6g}"
        lines.append(
            f"{a}\t{groups[a]}\t{b}\t{groups[b]}\t{content_value}\t{item['collinear_bases']}\t"
            f"{item['collinear_fraction']:.6g}\t{item['collinear_orientation']}\t"
            f"{item['connectivity_hops'] if item['connectivity_hops'] is not None else 'inf'}\t{k:.6g}"
        )
    (out / "subset_pairs.tsv").write_text("\n".join(lines) + "\n")

    def rho(x, y):
        pairs = [
            (a, b)
            for a, b in zip(x, y)
            if a is not None and b is not None and math.isfinite(a) and math.isfinite(b)
        ]
        if len(pairs) < 3:
            return None
        x, y = zip(*pairs)
        if len(set(x)) < 2 or len(set(y)) < 2:
            return None  # undefined for constant input
        value = spearmanr(x, y).correlation
        return None if value is None or math.isnan(value) else round(float(value), 4)

    correlations = {
        "content_vs_kmer": rho(
            [t["jaccard"] for t in table], [t["kmer_jaccard"] for t in table]
        ),
        "collinear_vs_kmer": rho(
            [t["collinear_fraction"] for t in table], [t["kmer_jaccard"] for t in table]
        ),
        "connectivity_vs_kmer": rho(
            [t["connectivity_hops"] for t in table], [t["kmer_jaccard"] for t in table]
        ),
        "connectivity_vs_content": rho(
            [t["connectivity_hops"] for t in table], [t["jaccard"] for t in table]
        ),
    }
    within = [t for (a, b), t in original.items() if groups[a] == groups[b]]
    between = [t for (a, b), t in original.items() if groups[a] != groups[b]]

    # ----------------------------------------------------------- bridges
    blocks = bridge_blocks(
        paths, groups, min_bases=args.min_bases, unknown_overlap=args.unknown_overlap
    )
    every = bridge_blocks(paths, groups, unknown_overlap=args.unknown_overlap)

    def size(bases):
        if bases >= 100:
            return "100+"
        low = bases // 10 * 10
        return f"{low}-{low + 9}"

    bins = Counter(size(b.bases or 0) for b in every)
    (out / "bridge_sizes.tsv").write_text(
        "bases\tblocks\n"
        + "".join(
            f"{k}\t{v}\n"
            for k, v in sorted(bins.items(), key=lambda kv: (len(kv[0]), kv[0]))
        )
    )
    bridge_records = []
    reference_upper = {k: v.upper() for k, v in references.items()}
    reverse_keys = {value: name for name, value in mapping.items()}
    for block in blocks:
        record = block.to_dict()
        if block.sequence is not None and reference_upper:
            forward, backward = (
                block.sequence.upper(),
                reverse_complement(block.sequence.upper()),
            )
            hits = {
                reverse_keys.get(k, k)
                for k, seq in reference_upper.items()
                if forward in seq or backward in seq
            }
            members = set(block.paths)
            record["fasta_support"] = {
                "genomes_containing_sequence": len(hits),
                "members_containing_sequence": len(members & hits),
                "members": len(members),
                "nonmember_genomes_containing_sequence": dict(
                    Counter(groups.get(h, "unlabeled") for h in hits - members)
                ),
            }
            checks = []
            for item in block.traversals:
                w, ref = (
                    item.witness,
                    reference_upper.get(mapping.get(item.path, item.path)),
                )
                if (
                    w
                    and w.start is not None
                    and w.end is not None
                    and ref is not None
                    and not w.wraps
                ):
                    checks.append(
                        ref[w.start : w.end] == (backward if w.reverse else forward)
                    )
            record["fasta_support"]["witness_coordinates_match_source"] = (
                f"{sum(checks)}/{len(checks)}"
            )
        bridge_records.append(record)
    (out / "bridges.json").write_text(json.dumps(
        {"min_bases": args.min_bases, "min_groups": 2, "all_cross_group_blocks": len(every),
         "blocks": bridge_records}, indent=2) + "\n")  # fmt: skip

    # ------------------------------------------------------- sensitivity
    sensitivity = [
        "representation\tvertices\tmax_abs_delta_content_jaccard\tmax_abs_delta_collinear_bases\tmax_relative_delta_connectivity\tbridge_bases_equal"
    ]
    sensitivity.append(f"original\t{vertices}\t0\t0\t0\tTrue")
    bridge_bases = sorted(b.bases or 0 for b in blocks)
    variants = [("renamed", lambda src, dst: rename(src, dst, prefix="v"))]
    if args.chop:
        variants.append(
            (f"chopped_max{args.chop}", lambda src, dst: chop(src, dst, args.chop))
        )
    with tempfile.TemporaryDirectory() as scratch:
        for label, transform in variants:
            target = Path(scratch) / f"{label}.gfa"
            try:
                transform(args.gfa, target)
            except ValueError as exc:
                sensitivity.append(f"{label}\tnot run: {exc}\t\t\t\t")
                continue
            changed = load_genome_paths(target)
            result, count = measures(changed, target)
            diffs = [
                abs(result[p]["jaccard"] - original[p]["jaccard"])
                for p in original
                if result[p]["jaccard"] is not None
                and original[p]["jaccard"] is not None
            ]
            if any(
                (result[p]["jaccard"] is None) != (original[p]["jaccard"] is None)
                for p in original
            ):
                diffs.append(math.inf)
            dj = max(diffs) if diffs else 0
            dc = (
                max(
                    abs(result[p]["collinear_bases"] - original[p]["collinear_bases"])
                    for p in original
                )
                if original
                else 0
            )
            ratios = [
                abs(result[p]["connectivity_hops"] - original[p]["connectivity_hops"])
                / original[p]["connectivity_hops"]
                for p in original
                if original[p]["connectivity_hops"]
                and result[p]["connectivity_hops"] is not None
            ]
            moved = sorted(
                b.bases or 0
                for b in bridge_blocks(
                    changed,
                    groups,
                    min_bases=args.min_bases,
                    unknown_overlap=args.unknown_overlap,
                )
            )
            sensitivity.append(
                f"{label}\t{count}\t{dj:.3g}\t{dc}\t{max(ratios) if ratios else 0:.3g}\t{moved == bridge_bases}"
            )
            del changed
    (out / "sensitivity.tsv").write_text("\n".join(sensitivity) + "\n")

    # ---------------------------------------------------------- manifest
    manifest = {
        "gfa": {"path": args.gfa, "sha256": sha256(args.gfa)},
        "reference": {"path": args.reference, "sha256": sha256(args.reference)}
        if args.reference
        else None,
        "labels": {"path": args.labels, "sha256": sha256(args.labels)}
        if args.labels
        else None,
        "parameters": {
            k: v
            for k, v in vars(args).items()
            if k not in ("gfa", "reference", "labels", "out")
        },
        "paths": {
            "selected": len(selected),
            "labeled": len(labeled),
            "groups": dict(Counter(groups.values()).most_common()),
        },
        "software": {
            "gfa2network": gfa2network.__version__,
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "odgi": shutil.which("odgi") or "not available",
        },
        "connectivity": "mean-nearest, traversal projection, hop cost, undirected",
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    def mean(values):
        values = [v for v in values if v is not None and math.isfinite(v)]
        return f"{sum(values) / len(values):.3g}" if values else "n/a"

    summary = f"""# Topology report

Graph `{Path(args.gfa).name}` (sha256 `{manifest["gfa"]["sha256"][:12]}`), {len(selected)} paths
analyzed, {len(labeled)} with external group labels in {len(set(groups.values()))} groups.

## Path fidelity

{json.dumps(dict(classes.most_common()))}. Audit policy: {json.dumps(report.policy)}.
Transitions: {json.dumps(report.summary["transitions"])}.

## Shared content and order (subset of {len(subset)} paths, seed {args.seed})

Within-group pairs: mean content Jaccard {mean([t["jaccard"] for t in within])}, mean
collinear fraction {mean([t["collinear_fraction"] for t in within])}, mean k-mer Jaccard
{mean([t["kmer_jaccard"] for t in within])}. Between-group pairs: {mean([t["jaccard"] for t in between])},
{mean([t["collinear_fraction"] for t in between])} and {mean([t["kmer_jaccard"] for t in between])}.
Spearman correlations: {json.dumps(correlations)}. Circular rotation search was
exhaustive (all block starts tried, `--max-rotations {args.max_rotations}`) for
{sum(t["exhaustive"] for t in table)}/{len(table)} pairs; otherwise the chain is a lower bound.

## Cross-group blocks

{len(every)} blocks of contiguous sequence are traversed by paths of two or more
groups; {len(blocks)} are at least {args.min_bases} bp. Sizes: see `bridge_sizes.tsv`;
witnesses, contexts and FASTA support: `bridges.json`.

## Representation sensitivity

See `sensitivity.tsv`. Sequence-level measures should not change under renaming or
chopping; vertex-count connectivity distances may.

## Limits

Blocks and connectivity describe this graph construction. They are not evidence of
recombination, ancestry or homology. ODGI: {manifest["software"]["odgi"]}.
"""
    (out / "summary.md").write_text(summary)
    print(summary)


if __name__ == "__main__":
    sys.exit(main())
