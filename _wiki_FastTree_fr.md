# FastTree

FastTree infers maximum likelihood phylogenetic trees from nucleotide or protein sequence alignments. FastTree can handle alignments with up to a million sequences in reasonable time and memory.

## Environment Modules

We offer modules for single-precision and double-precision calculations. Single-precision calculations are faster, but double-precision calculations are more accurate. Double precision is recommended when using a highly biased transition matrix or if you wish to accurately resolve very short branches.

To see available modules:

```bash
module spider fasttree
```

To load a single-precision module:

```bash
module load fasttree/2.1.11
```

To load a double-precision module:

```bash
module load fasttree-double/2.1.11
```

## Troubleshooting

### Error Message

`WARNING! This alignment consists of closely-related and very long sequences`

This typically leads to very short branches, sometimes even negative branch lengths.


## References

* [FastTree webpage](link_to_fasttree_webpage_here)  *(Please replace `link_to_fasttree_webpage_here` with the actual link)*
