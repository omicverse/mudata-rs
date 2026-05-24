//! Core [`MuData<B>`] type.
//!
//! Holds an ordered map of modalities (each an [`anndata::AnnData<B>`])
//! plus joint sample / feature metadata as in-memory state. The on-disk
//! ``.h5mu`` representation is owned by [`crate::h5mu`] which composes
//! `MuData<B>` from the file and writes it back.

use anndata::AnnData;
use anndata::backend::Backend;
use anndata_hdf5::H5;
use indexmap::IndexMap;
use std::path::PathBuf;

/// Multimodal AnnData container.
///
/// The matrix `X` of each modality lives in `mods[name].get_x()` and
/// stays on disk through anndata-rs's backed-reader machinery. The
/// joint metadata (`obs`, `var`, ...) is materialised into Rust types
/// because it is tiny compared to per-modality `X`.
///
/// `B` is generic to mirror anndata-rs; in practice only `H5` is wired
/// up today but a `Zarr` flavour can be added later without touching
/// the public API.
pub struct MuData<B: Backend = H5> {
    /// Source `.h5mu` path if this object was loaded via
    /// [`crate::read_h5mu`]; `None` for in-memory construction.
    pub(crate) source_path: Option<PathBuf>,
    /// Ordered map of modality name → backed AnnData.
    pub(crate) mods: IndexMap<String, AnnData<B>>,
    /// MuData "shared axis" flag — 0 if observations are aligned across
    /// modalities (CITE-seq-like) and 1 if features are aligned.
    pub(crate) axis: u8,
    /// Cached number of joint observations.
    pub(crate) n_obs_cached: usize,
    /// Cached number of joint variables.
    pub(crate) n_vars_cached: usize,
    /// Names of joint observations (the joint `obs.index`).
    pub(crate) obs_names: Vec<String>,
    /// Names of joint variables (the joint `var.index`).
    pub(crate) var_names: Vec<String>,
}

impl<B: Backend> std::fmt::Debug for MuData<B> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MuData")
            .field("source", &self.source_path)
            .field("n_obs", &self.n_obs_cached)
            .field("n_vars", &self.n_vars_cached)
            .field("axis", &self.axis)
            .field(
                "mod",
                &self.mods.keys().cloned().collect::<Vec<_>>(),
            )
            .finish()
    }
}

impl<B: Backend> std::fmt::Display for MuData<B> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(
            f,
            "MuData [{} × {}, axis={}]",
            self.n_obs_cached, self.n_vars_cached, self.axis
        )?;
        writeln!(f, "  mod ({})", self.mods.len())?;
        for (name, ad) in &self.mods {
            writeln!(f, "    {name}: {ad}")?;
        }
        Ok(())
    }
}

impl<B: Backend> MuData<B> {
    /// Build an empty `MuData` — no modalities, no joint axes. Useful
    /// for tests and as the seed for programmatic construction.
    pub fn empty() -> Self {
        Self {
            source_path: None,
            mods: IndexMap::new(),
            axis: 0,
            n_obs_cached: 0,
            n_vars_cached: 0,
            obs_names: Vec::new(),
            var_names: Vec::new(),
        }
    }

    /// Source `.h5mu` path if loaded from disk.
    pub fn source_path(&self) -> Option<&PathBuf> {
        self.source_path.as_ref()
    }

    /// Modality names in stable order.
    pub fn mod_names(&self) -> Vec<String> {
        self.mods.keys().cloned().collect()
    }

    /// Number of modalities.
    pub fn n_mod(&self) -> usize {
        self.mods.len()
    }

    /// Borrow a modality by name.
    pub fn get(&self, name: &str) -> Option<&AnnData<B>> {
        self.mods.get(name)
    }

    /// Mutably borrow a modality by name.
    pub fn get_mut(&mut self, name: &str) -> Option<&mut AnnData<B>> {
        self.mods.get_mut(name)
    }

    /// MuData shared axis — 0 for cell-aligned, 1 for feature-aligned.
    pub fn axis(&self) -> u8 {
        self.axis
    }

    /// Joint number of observations.
    pub fn n_obs(&self) -> usize {
        self.n_obs_cached
    }

    /// Joint number of variables.
    pub fn n_vars(&self) -> usize {
        self.n_vars_cached
    }

    /// Borrow joint obs_names.
    pub fn obs_names(&self) -> &[String] {
        &self.obs_names
    }

    /// Borrow joint var_names.
    pub fn var_names(&self) -> &[String] {
        &self.var_names
    }

    /// Iterate `(name, modality)` pairs in stable order.
    pub fn iter(&self) -> impl Iterator<Item = (&String, &AnnData<B>)> {
        self.mods.iter()
    }

    /// Insert a modality. Returns the previous value at `name`, if any.
    pub fn insert(&mut self, name: impl Into<String>, ad: AnnData<B>) -> Option<AnnData<B>> {
        self.mods.insert(name.into(), ad)
    }
}
