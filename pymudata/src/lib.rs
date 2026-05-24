//! PyO3 bindings for `mudata-rs`.
//!
//! Mirrors `pyanndata`'s role for `anndata-rs`. Exposes [`MuData`] as
//! a Python type so the Python `mudataoom` package can route through
//! the Rust backend with no `mudata` Python dependency.

use anndata_hdf5::H5;
use mudata as core_mudata;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

/// Python view of a [`mudata::MuData<H5>`].
///
/// Holds a reference-counted handle to both the core `MuData` (which
/// owns the per-modality [`anndata::AnnData<H5>`] backed handles) and
/// the [`ThinH5muWorkspace`] (which owns the temp ExternalLink files
/// that anndata-rs is reading through). When the last Python reference
/// drops, the temp files are cleaned up.
#[pyclass(name = "MuData", subclass)]
pub struct PyMuData {
    inner: Arc<Mutex<core_mudata::MuData<H5>>>,
    // Workspace must outlive `inner` — keep it alive on the same Arc lifetime.
    workspace: Arc<core_mudata::ThinH5muWorkspace>,
}

#[pymethods]
impl PyMuData {
    /// Path to the source `.h5mu` file if loaded via :func:`read_h5mu`.
    #[getter]
    fn source_path(&self) -> PyResult<Option<PathBuf>> {
        let g = self.inner.lock().unwrap();
        Ok(g.source_path().cloned())
    }

    /// List modality names in stable order.
    #[getter]
    fn mod_names(&self) -> PyResult<Vec<String>> {
        let g = self.inner.lock().unwrap();
        Ok(g.mod_names())
    }

    /// Number of modalities.
    #[getter]
    fn n_mod(&self) -> PyResult<usize> {
        let g = self.inner.lock().unwrap();
        Ok(g.n_mod())
    }

    /// MuData shared axis (0 = cell-aligned, 1 = feature-aligned).
    #[getter]
    fn axis(&self) -> PyResult<u8> {
        let g = self.inner.lock().unwrap();
        Ok(g.axis())
    }

    /// Joint number of observations.
    #[getter]
    fn n_obs(&self) -> PyResult<usize> {
        let g = self.inner.lock().unwrap();
        Ok(g.n_obs())
    }

    /// Joint number of variables.
    #[getter]
    fn n_vars(&self) -> PyResult<usize> {
        let g = self.inner.lock().unwrap();
        Ok(g.n_vars())
    }

    /// ``(n_obs, n_vars)``.
    #[getter]
    fn shape(&self) -> PyResult<(usize, usize)> {
        let g = self.inner.lock().unwrap();
        Ok((g.n_obs(), g.n_vars()))
    }

    /// Joint obs index labels.
    #[getter]
    fn obs_names(&self) -> PyResult<Vec<String>> {
        let g = self.inner.lock().unwrap();
        Ok(g.obs_names().to_vec())
    }

    /// Joint var index labels.
    #[getter]
    fn var_names(&self) -> PyResult<Vec<String>> {
        let g = self.inner.lock().unwrap();
        Ok(g.var_names().to_vec())
    }

    fn __len__(&self) -> PyResult<usize> {
        self.n_obs()
    }

    fn __repr__(&self) -> PyResult<String> {
        let g = self.inner.lock().unwrap();
        Ok(format!("{g}"))
    }

    fn __contains__(&self, name: &str) -> PyResult<bool> {
        let g = self.inner.lock().unwrap();
        Ok(g.get(name).is_some())
    }

    /// True if `name` is a modality on this object.
    fn has_modality(&self, name: &str) -> PyResult<bool> {
        self.__contains__(name)
    }

    /// `{modality_name: thin_h5ad_path}` mapping.
    ///
    /// Each thin file is an HDF5 file in a temp directory whose root
    /// keys are ExternalLinks into the source ``.h5mu``. Opening one
    /// with :func:`anndataoom.read` gives you an :class:`AnnDataOOM`
    /// whose `X` is read directly from the original ``.h5mu`` on
    /// demand (no copy). The files are removed when this MuData is
    /// garbage-collected — keep this object alive while you read them.
    #[getter]
    fn thin_paths<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let dict = PyDict::new(py);
        for (name, path) in self.workspace.files() {
            dict.set_item(name, path)?;
        }
        Ok(dict)
    }

    /// Absolute path of the temp directory holding the thin .h5ad
    /// files. Provided for diagnostics.
    #[getter]
    fn thin_directory(&self) -> PyResult<PathBuf> {
        Ok(self.workspace.directory().to_path_buf())
    }

    /// Write to a new `.h5mu` file.
    fn write_h5mu(&self, path: PathBuf) -> PyResult<()> {
        let g = self.inner.lock().unwrap();
        core_mudata::write_h5mu(&path, &g)
            .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))
    }
}

/// Open a `.h5mu` file lazily.
///
/// Per-modality `X` stays on disk; reading returns a :class:`MuData`
/// whose modalities are routed through anndata-rs's backed reader.
#[pyfunction]
pub fn read_h5mu(path: PathBuf) -> PyResult<PyMuData> {
    let (m, ws) = core_mudata::read_h5mu(&path)
        .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;
    Ok(PyMuData {
        inner: Arc::new(Mutex::new(m)),
        workspace: Arc::new(ws),
    })
}

/// Register every Python class and function this crate exposes into
/// the given `_backend` PyModule. Called by downstream extension
/// crates (e.g. `mudataoom`) from inside their own `#[pymodule]`
/// block.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyMuData>()?;
    m.add_function(wrap_pyfunction!(read_h5mu, m)?)?;
    Ok(())
}
