//! ``.h5mu`` reader / writer for [`crate::MuData`].
//!
//! Reading strategy: rather than re-implement the entire AnnData
//! encoding (X / obs / var / obsm / varm / varp / layers / raw / uns)
//! inside mudata-rs, we **delegate** modality loading to anndata-rs.
//! The catch is that anndata-rs' [`AnnData::open`] takes a whole
//! "store" (HDF5 file root); it does not accept a sub-group like
//! ``/mod/<name>/``. To bridge the gap we synthesise a temporary
//! "thin" `.h5ad` file per modality whose every root-level entry is an
//! HDF5 ExternalLink pointing back at the corresponding location in
//! the source `.h5mu`. libhdf5 follows external links transparently,
//! so anndata-rs sees a perfectly normal `.h5ad` while the on-disk
//! bytes for `X` and friends are never copied.
//!
//! The same workspace owns the lifetime of every temp link: when the
//! [`MuData`] is dropped, the temp dir is removed.

use crate::mudata::MuData;

use anndata::AnnData;
use anndata::backend::{
    Backend, Compression, WriteConfig, get_default_write_config, set_default_write_config,
};
use anndata_hdf5::H5;
use anyhow::{anyhow, Context, Result};
use hdf5::Location;
use indexmap::IndexMap;
use std::path::{Path, PathBuf};
use tempfile::TempDir;

/// AnnData keys that may appear at the root of a `/mod/<name>/` group.
/// Reflects anndata >= 0.8 conventions.
const ANNDATA_ROOT_KEYS: &[&str] = &[
    "X", "obs", "var", "obsm", "varm", "obsp", "varp", "layers", "raw", "uns",
];

/// Owns the temp directory of thin `.h5ad` ExternalLink files that
/// lets anndata-rs lazily read each `/mod/<name>/` subtree of a `.h5mu`
/// without copying the actual `X` bytes.
///
/// Holding a `ThinH5muWorkspace` keeps the temp dir alive; dropping it
/// removes every thin file.
pub struct ThinH5muWorkspace {
    dir: TempDir,
    files: Vec<(String, PathBuf)>,
}

impl ThinH5muWorkspace {
    /// Create a fresh workspace with a unique temp directory.
    pub fn new() -> Result<Self> {
        let dir = tempfile::Builder::new()
            .prefix("mudata-rs-thin-")
            .tempdir()
            .context("could not create temp dir for thin .h5ad files")?;
        Ok(Self {
            dir,
            files: Vec::new(),
        })
    }

    /// Absolute path to the temp directory.
    pub fn directory(&self) -> &Path {
        self.dir.path()
    }

    /// `(modality_name, thin_path)` pairs in insertion order.
    pub fn files(&self) -> &[(String, PathBuf)] {
        &self.files
    }

    /// Create a thin `.h5ad` for `mod_name` that links back into
    /// `h5mu_path:/mod/<mod_name>/`. Returns the thin file's path.
    pub fn add_modality(&mut self, h5mu_path: &Path, mod_name: &str) -> Result<PathBuf> {
        let thin_path = self.dir.path().join(format!("{mod_name}.h5ad"));
        materialise_thin_h5ad(h5mu_path, mod_name, &thin_path)?;
        self.files.push((mod_name.to_string(), thin_path.clone()));
        Ok(thin_path)
    }
}

/// Create a thin `.h5ad` file at `thin_path` that mirrors
/// `/mod/<mod_name>/` of `h5mu_path` via HDF5 ExternalLinks.
///
/// The link targets are stored as relative paths from the thin file's
/// directory so the pair can be moved as a unit. libhdf5 will follow
/// them transparently as long as both files are reachable.
pub fn materialise_thin_h5ad(
    h5mu_path: &Path,
    mod_name: &str,
    thin_path: &Path,
) -> Result<()> {
    let h5mu_path = h5mu_path.canonicalize().with_context(|| {
        format!("could not canonicalise source h5mu path {}", h5mu_path.display())
    })?;

    // What entries actually exist under /mod/<name>/?  Open r/o briefly.
    let (root_attrs, keys_present) = {
        let src = hdf5::File::open(&h5mu_path)
            .with_context(|| format!("open {} for reading", h5mu_path.display()))?;
        let group_path = format!("mod/{mod_name}");
        let g = src
            .group(&group_path)
            .with_context(|| format!("modality {mod_name:?} not found in {}", h5mu_path.display()))?;

        let mut found = Vec::new();
        for k in ANNDATA_ROOT_KEYS {
            if g.link_exists(k) {
                found.push((*k).to_string());
            }
        }

        // We can't carry hdf5 attrs across the close, so just record
        // the attr names; the second pass below re-opens and clones them.
        let attr_names: Vec<String> = g.attr_names()?.into_iter().collect();
        (attr_names, found)
    };

    // Stable relative target path so moving (h5mu, thin/) as a unit
    // doesn't break.  Fall back to absolute on any failure.
    let link_target = thin_path
        .parent()
        .and_then(|d| pathdiff(d, &h5mu_path))
        .unwrap_or_else(|| h5mu_path.clone());

    // Now build the thin file.
    {
        let dst = hdf5::File::create(thin_path)
            .with_context(|| format!("create thin .h5ad at {}", thin_path.display()))?;

        // Mirror attributes from /mod/<name> onto the new root group
        // and ensure the AnnData encoding markers are present even if
        // the source omitted them.
        {
            let src = hdf5::File::open(&h5mu_path)?;
            let src_g = src.group(&format!("mod/{mod_name}"))?;
            for name in &root_attrs {
                if let Ok(attr) = src_g.attr(name) {
                    let _ = clone_attr(&attr, &dst);
                }
            }
        }
        ensure_anndata_root_attrs(&dst)?;

        // Write one external link per AnnData root-level entry.
        for key in &keys_present {
            let target = format!("/mod/{mod_name}/{key}");
            create_external_link(&dst, key, &link_target, &target)?;
        }
    }

    Ok(())
}

/// Open a `.h5mu` file and return a fully-constructed [`MuData<H5>`]
/// plus the [`ThinH5muWorkspace`] that owns its temp files (must
/// outlive the MuData).
pub fn read_h5mu<P: AsRef<Path>>(path: P) -> Result<(MuData<H5>, ThinH5muWorkspace)> {
    let h5mu_path = path.as_ref().to_path_buf();
    if !h5mu_path.exists() {
        return Err(anyhow!("not found: {}", h5mu_path.display()));
    }

    // ---- Scan structure ----
    let mod_order: Vec<String>;
    let axis: u8;
    {
        let src = hdf5::File::open(&h5mu_path)
            .with_context(|| format!("open {} for reading", h5mu_path.display()))?;
        let mod_group = src
            .group("mod")
            .with_context(|| format!("{} has no /mod group — not a .h5mu file", h5mu_path.display()))?;

        mod_order = read_mod_order(&mod_group)?;
        axis = src
            .attr("axis")
            .ok()
            .and_then(|a| a.read_scalar::<i64>().ok())
            .map(|v| v as u8)
            .unwrap_or(0);
    }

    // ---- Build thin .h5ad workspace ----
    let mut workspace = ThinH5muWorkspace::new()?;
    let mut mods: IndexMap<String, AnnData<H5>> = IndexMap::new();
    for name in &mod_order {
        let thin = workspace.add_modality(&h5mu_path, name)?;
        let store = H5::open(&thin)
            .with_context(|| format!("anndata-rs could not open thin h5ad {}", thin.display()))?;
        let adata = AnnData::<H5>::open(store)
            .with_context(|| format!("anndata-rs failed to parse modality {name}"))?;
        mods.insert(name.clone(), adata);
    }

    // ---- Compute joint shape from the first modality (axis==0 case) ----
    // Full joint obs/var DataFrame loading is a follow-up; demo focuses
    // on shape + obs_names / var_names which are tiny string lists.
    let (n_obs_cached, n_vars_cached, obs_names, var_names) =
        joint_dims_from_h5mu(&h5mu_path).unwrap_or_else(|_| (0, 0, Vec::new(), Vec::new()));

    let mdata = MuData::<H5> {
        source_path: Some(h5mu_path),
        mods,
        axis,
        n_obs_cached,
        n_vars_cached,
        obs_names,
        var_names,
    };
    Ok((mdata, workspace))
}

/// Joint root-level groups that may appear in a `.h5mu`. Used by the
/// writer's "copy joint metadata from source" pass.
const JOINT_ROOT_KEYS: &[&str] = &[
    "obs", "var", "obsm", "varm", "obsp", "varp", "obsmap", "varmap", "uns",
];

/// Persist `mdata` to a new ``.h5mu`` file.
///
/// Layout written:
///
/// * 512-byte HDF5 userblock stamped with the MuData magic header
///   (`MuData (format-version=...; creator=...)`).
/// * `/mod/<name>/` per modality — each one is the full AnnData group
///   written by anndata-rs and copied in at the libhdf5 level (no
///   Python decoding of `X`).
/// * `/obs`, `/var`, `/obsm`, `/varm`, `/obsp`, `/varp`, `/obsmap`,
///   `/varmap`, `/uns` at the root — copied verbatim from the source
///   `.h5mu` when one is available. (For an in-memory `MuData` that
///   was never loaded from disk these are skipped; programmatic
///   construction of joint metadata is a follow-up.)
/// * `axis`, `encoding-type`, `encoding-version` attrs on the root.
///
/// Compression: anndata-rs's default is `Compression::Zst(5)`, which
/// stock h5py wheels cannot read because the blosc/zstd HDF5 filter
/// isn't shipped with h5py. The writer temporarily switches the
/// thread-local default to `Gzip(6)` so downstream tools (h5py,
/// upstream `mudata.read_h5mu`, scanpy etc.) can open the result
/// without installing extra filter plugins. The previous setting is
/// restored when the function returns.
pub fn write_h5mu<P: AsRef<Path>>(path: P, mdata: &MuData<H5>) -> Result<()> {
    let dest = path.as_ref();

    // ---- 1) switch anndata-rs to h5py-compatible compression ----
    let prev_write_config = get_default_write_config();
    set_default_write_config(WriteConfig {
        compression: Some(Compression::Gzip(6)),
        block_size: None,
    });
    let result = write_h5mu_inner(dest, mdata);
    set_default_write_config(prev_write_config);
    result
}

fn write_h5mu_inner(dest: &Path, mdata: &MuData<H5>) -> Result<()> {
    let tmp = tempfile::Builder::new()
        .prefix("mudata-rs-write-")
        .tempdir()?;

    // ---- 2) dump each AnnDataOOM modality to a temp .h5ad ----
    let mut per_mod = Vec::new();
    for (name, ad) in mdata.iter() {
        let p = tmp.path().join(format!("{name}.h5ad"));
        ad.write::<H5, _>(&p, None, None)
            .with_context(|| format!("dump modality {name}"))?;
        per_mod.push((name.clone(), p));
    }

    // ---- 3) build the destination .h5mu ----
    let dst = hdf5::FileBuilder::new()
        .with_fcpl(|b| b.userblock(512))
        .create(dest)
        .with_context(|| format!("create {}", dest.display()))?;

    let mod_group = dst.create_group("mod")?;
    for (name, p) in &per_mod {
        let src = hdf5::File::open(p)?;
        hdf5_copy(&src, "/", &mod_group, name)?;
    }
    let names: Vec<String> = mdata.mod_names();
    write_str_array_attr(loc_of_group(&mod_group), "mod-order", &names)?;

    // ---- 4) copy joint metadata from the source .h5mu (if any) ----
    if let Some(src_path) = mdata.source_path() {
        if src_path.exists() && src_path != dest {
            let src = hdf5::File::open(src_path).with_context(|| {
                format!("re-open source {} for joint metadata copy", src_path.display())
            })?;
            for key in JOINT_ROOT_KEYS {
                if src.link_exists(key) {
                    hdf5_copy(&src, &format!("/{key}"), &dst.group("/")?, key)
                        .with_context(|| format!("copy joint /{key}"))?;
                }
            }
        }
    }

    write_str_attr(loc_of_file(&dst), "encoding-type", "MuData")?;
    write_str_attr(loc_of_file(&dst), "encoding-version", "0.1.0")?;
    write_i64_attr(loc_of_file(&dst), "axis", mdata.axis() as i64)?;

    drop(dst);

    // ---- 5) stamp the MuData magic into the userblock ----
    let magic = b"MuData (format-version=0.1.0;creator=mudata-rs;creator-version=0.1.0)";
    let mut f = std::fs::OpenOptions::new().write(true).open(dest)?;
    use std::io::{Seek, SeekFrom, Write};
    f.seek(SeekFrom::Start(0))?;
    f.write_all(magic)?;
    let pad = vec![0u8; 512 - magic.len()];
    f.write_all(&pad)?;
    Ok(())
}

// =====================================================================
// Internal helpers
// =====================================================================

fn read_mod_order(mod_group: &hdf5::Group) -> Result<Vec<String>> {
    // Prefer the `mod-order` attribute if present; fall back to
    // alphabetical iteration of children.
    if let Ok(attr) = mod_group.attr("mod-order") {
        if let Ok(arr) = attr.read_1d::<hdf5::types::VarLenUnicode>() {
            return Ok(arr.into_iter().map(|s| s.to_string()).collect());
        }
        if let Ok(arr) = attr.read_1d::<hdf5::types::VarLenAscii>() {
            return Ok(arr.into_iter().map(|s| s.to_string()).collect());
        }
    }
    let names = mod_group.member_names()?;
    Ok(names)
}

fn joint_dims_from_h5mu(h5mu_path: &Path) -> Result<(usize, usize, Vec<String>, Vec<String>)> {
    let src = hdf5::File::open(h5mu_path)?;

    // The joint /obs and /var are DataFrames serialised by anndata's
    // pandas encoder. The index column's name is recorded as the
    // `_index` attr on the group; the dataset under that name holds
    // the row labels.
    let obs_names = read_dataframe_index(&src, "obs").unwrap_or_default();
    let var_names = read_dataframe_index(&src, "var").unwrap_or_default();
    Ok((obs_names.len(), var_names.len(), obs_names, var_names))
}

fn read_dataframe_index(file: &hdf5::File, name: &str) -> Result<Vec<String>> {
    let g = file.group(name)?;
    let index_col = g
        .attr("_index")
        .ok()
        .and_then(|a| a.read_scalar::<hdf5::types::VarLenUnicode>().ok())
        .map(|s| s.to_string())
        .unwrap_or_else(|| "_index".to_string());
    let ds = g.dataset(&index_col)?;
    if let Ok(arr) = ds.read_1d::<hdf5::types::VarLenUnicode>() {
        return Ok(arr.into_iter().map(|s| s.to_string()).collect());
    }
    if let Ok(arr) = ds.read_1d::<hdf5::types::VarLenAscii>() {
        return Ok(arr.into_iter().map(|s| s.to_string()).collect());
    }
    Err(anyhow!("could not decode {name}/{index_col} as a string index"))
}

fn ensure_anndata_root_attrs(loc: &hdf5::File) -> Result<()> {
    let l = loc_of_file(loc);
    if l.attr("encoding-type").is_err() {
        write_str_attr(l, "encoding-type", "anndata")?;
    }
    if l.attr("encoding-version").is_err() {
        write_str_attr(l, "encoding-version", "0.1.0")?;
    }
    Ok(())
}

fn clone_attr(src: &hdf5::Attribute, dst: &hdf5::File) -> Result<()> {
    // Best-effort: only copy string attrs, which is what mudata uses
    // for AnnData root group markers.
    let name = src.name();
    if let Ok(v) = src.read_scalar::<hdf5::types::VarLenUnicode>() {
        write_str_attr(loc_of_file(dst), &name, &v.to_string())?;
        return Ok(());
    }
    Ok(())
}

/// Reach the `&Location` view of a `&File` via the documented deref
/// chain (`File -> Group -> Location` in hdf5-metno 0.12).
fn loc_of_file(f: &hdf5::File) -> &Location {
    use std::ops::Deref;
    (**f).deref()
}

/// Reach the `&Location` view of a `&Group`.
fn loc_of_group(g: &hdf5::Group) -> &Location {
    use std::ops::Deref;
    g.deref()
}

fn write_str_attr(loc: &Location, name: &str, value: &str) -> Result<()> {
    let s: hdf5::types::VarLenUnicode = value.parse().unwrap();
    let attr = loc
        .new_attr::<hdf5::types::VarLenUnicode>()
        .shape(())
        .create(name)?;
    attr.write_scalar(&s)?;
    Ok(())
}

fn write_i64_attr(loc: &Location, name: &str, value: i64) -> Result<()> {
    let attr = loc.new_attr::<i64>().shape(()).create(name)?;
    attr.write_scalar(&value)?;
    Ok(())
}

fn write_str_array_attr(loc: &Location, name: &str, values: &[String]) -> Result<()> {
    let typed: Vec<hdf5::types::VarLenUnicode> = values
        .iter()
        .map(|v| v.parse::<hdf5::types::VarLenUnicode>().unwrap())
        .collect();
    let attr = loc
        .new_attr::<hdf5::types::VarLenUnicode>()
        .shape(typed.len())
        .create(name)?;
    attr.write(&typed)?;
    Ok(())
}

fn create_external_link(
    dst: &hdf5::File,
    name: &str,
    target_file: &Path,
    target_path: &str,
) -> Result<()> {
    use hdf5_sys::h5l::H5Lcreate_external;
    use std::ffi::CString;

    let c_target_file = CString::new(target_file.to_string_lossy().into_owned()).unwrap();
    let c_target_path = CString::new(target_path).unwrap();
    let c_name = CString::new(name).unwrap();
    let r = unsafe {
        H5Lcreate_external(
            c_target_file.as_ptr(),
            c_target_path.as_ptr(),
            dst.id(),
            c_name.as_ptr(),
            hdf5_sys::h5p::H5P_DEFAULT,
            hdf5_sys::h5p::H5P_DEFAULT,
        )
    };
    if r < 0 {
        return Err(anyhow!(
            "H5Lcreate_external failed for {} -> {}:{}",
            name,
            target_file.display(),
            target_path
        ));
    }
    Ok(())
}

fn hdf5_copy(
    src: &hdf5::File,
    src_path: &str,
    dst_parent: &hdf5::Group,
    dst_name: &str,
) -> Result<()> {
    use hdf5_sys::h5o::H5Ocopy;
    use std::ffi::CString;

    let c_src_path = CString::new(src_path).unwrap();
    let c_dst_name = CString::new(dst_name).unwrap();
    let r = unsafe {
        H5Ocopy(
            src.id(),
            c_src_path.as_ptr(),
            dst_parent.id(),
            c_dst_name.as_ptr(),
            hdf5_sys::h5p::H5P_DEFAULT,
            hdf5_sys::h5p::H5P_DEFAULT,
        )
    };
    if r < 0 {
        return Err(anyhow!(
            "H5Ocopy failed: {} -> {}",
            src_path,
            dst_name
        ));
    }
    Ok(())
}

/// Stripped-down `pathdiff::diff_paths` (avoids pulling in another crate).
fn pathdiff(base: &Path, target: &Path) -> Option<PathBuf> {
    use std::path::Component;

    if target.is_absolute() != base.is_absolute() {
        return None;
    }
    let mut tb = target.components().peekable();
    let mut bb = base.components().peekable();
    while tb.peek() == bb.peek() && tb.peek().is_some() {
        tb.next();
        bb.next();
    }
    let mut out = PathBuf::new();
    for c in bb {
        if let Component::Normal(_) | Component::CurDir = c {
            out.push("..");
        }
    }
    for c in tb {
        out.push(c.as_os_str());
    }
    Some(out)
}

