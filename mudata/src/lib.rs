//! mudata-rs — Rust core for MuData (multimodal AnnData).
//!
//! Mirrors the role `anndata-rs` plays for AnnData. The crate exposes:
//!
//! * [`MuData`] — a multimodal container whose modalities are
//!   [`anndata::AnnData<B>`] over an arbitrary [`Backend`], plus
//!   joint sample/feature metadata kept in memory.
//! * [`read_h5mu`] / [`write_h5mu`] — readers and writers for the
//!   scverse `.h5mu` on-disk format. Per-modality `X` stays on disk
//!   throughout; modalities are opened lazily via anndata-rs's
//!   existing backed-reader machinery by exposing each `/mod/<name>/`
//!   subtree to anndata-rs through an HDF5 ExternalLink-based "thin"
//!   `.h5ad` file.
//!
//! The crate intentionally does not implement its own HDF5 reader for
//! `X`/`obs`/`var`/`obsm`/... — that responsibility is delegated to
//! anndata-rs to keep a single source of truth for the encoding spec.

pub mod h5mu;
pub mod mudata;

pub use h5mu::{read_h5mu, write_h5mu, ThinH5muWorkspace};
pub use mudata::MuData;

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    /// Smoke test: workspace temp dir is created, the type compiles,
    /// and a constructed-empty `MuData` carries the expected zero state.
    #[test]
    fn empty_mudata_has_no_modalities() {
        let m: MuData = MuData::empty();
        assert_eq!(m.mod_names().len(), 0);
        assert_eq!(m.n_obs(), 0);
        assert_eq!(m.n_vars(), 0);
    }

    #[test]
    fn workspace_dir_is_writable() {
        let w = ThinH5muWorkspace::new().expect("workspace");
        let p: PathBuf = w.directory().join("scratch");
        std::fs::write(&p, b"ok").expect("write");
        let body = std::fs::read(&p).expect("read");
        assert_eq!(&body[..], b"ok");
    }
}
