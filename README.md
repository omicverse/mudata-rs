# mudata-rs

Rust implementation of MuData — the scverse multimodal AnnData
container. Mirrors the role `anndata-rs` plays for AnnData.

## Layout

| Crate      | Purpose                                                            |
|------------|--------------------------------------------------------------------|
| `mudata`   | Core `MuData<B>` Rust type + `.h5mu` reader/writer                 |
| `pymudata` | PyO3 bindings — exposes `MuData` and `read_h5mu` to Python         |

The high-level Python wrapper that consumes `pymudata` is the
out-of-tree `mudataoom` package.

## Why?

`mudata.MuData` (Python) is fine for joint metadata, but each modality
is a full `AnnData` whose `X` lives in RAM. For 1M-cell multimodal
atlases this means >100 GB. Per-modality `X` here stays on disk via
`anndata-rs`'s backed reader; only joint obs/var/obsm and a few
strings sit in memory.

## Read path

```
.h5mu (on disk)
   └── /mod/<name>/…   (one AnnData group per modality)
       └── HDF5 ExternalLink — read by anndata-rs as a "thin .h5ad"
           └── AnnData<H5> with backed X (no copy)
```

Thin links live in a temp directory owned by `MuData`; they are
removed on drop.

## Build (Linux)

Needs HDF5 dev headers + libs (conda `hdf5` works) and a recent Rust
toolchain (≥1.88). Set `CARGO_HOME=/scratch/...` to keep the registry
out of your home quota.

```bash
cargo build --release
maturin build --release \
    --manifest-path pymudata/Cargo.toml \
    -i /path/to/python \
    --out wheels
```

## License

MIT.
