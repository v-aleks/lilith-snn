use lmdb::{Environment, Database, Transaction, Cursor, WriteFlags};
use lmdb::DatabaseFlags;
use std::path::Path;
use std::sync::Arc;
use crate::config::NordConfig;
use burn::tensor::{Tensor, backend::Backend};
use anyhow::{Result, anyhow};

/// Simple LMDB dataset loader.
///
/// The Python version stores tokenized sequences in an LMDB database where each
/// entry is a serialized byte vector of `u32` token ids. This stub opens the LMDB
/// environment, reads entries lazily, and provides a method to retrieve a batch of
/// token tensors.
pub struct LMDBDataset {
    env: Arc<Environment>,
    db: Database,
    cfg: NordConfig,
}

impl LMDBDataset {
    /// Open an LMDB dataset located at `path`.
    ///
    /// Returns an error if the environment cannot be opened.
    pub fn open<P: AsRef<Path>>(path: P, cfg: NordConfig) -> Result<Self> {
        let env = Environment::new()
            .set_max_dbs(1)
            .set_map_size(10 * 1024 * 1024 * 1024) // 10 GiB map size – adjust as needed
            .open(path.as_ref())
            .map_err(|e| anyhow!("Failed to open LMDB env: {}", e))?;
        let db = env
            .open_db(Some("data"))
            .or_else(|_| env.open_db(None))
            .map_err(|e| anyhow!("Failed to open LMDB db: {}", e))?;
        Ok(Self { env: Arc::new(env), db, cfg })
    }

    /// Retrieve a single entry by key and return a tensor of shape `[seq_len, D]`.
    ///
    /// The stored value is expected to be a little‑endian `u32` array of token
    /// ids. This function deserializes the bytes and creates a 1‑D tensor.
    pub fn get_entry<B: Backend>(&self, key: &[u8]) -> Result<Tensor<B, 1>> {
        let txn = self.env.begin_ro_txn()?;
        let bytes = txn.get(self.db, &key)?;
        // Assume each token id is a u32 (4 bytes).
        if bytes.len() % 4 != 0 {
            return Err(anyhow!("Corrupt LMDB entry: length not multiple of 4"));
        }
        let token_count = bytes.len() / 4;
        let mut ids = Vec::with_capacity(token_count);
        for i in 0..token_count {
            let start = i * 4;
            let id = u32::from_le_bytes([bytes[start], bytes[start + 1], bytes[start + 2], bytes[start + 3]]);
            ids.push(id as f32);
        }
        // Create a 1‑D tensor of token ids (as f32 for compatibility with Burn).
        Ok(Tensor::from_data(ids.into()))
    }

    /// Iterate over all entries and collect them into a vector of tensors.
    /// This is a simple utility for small datasets; for large corpora you would
    /// stream batches instead.
    pub fn all_entries<B: Backend>(&self) -> Result<Vec<Tensor<B, 1>>> {
        let txn = self.env.begin_ro_txn()?;
        let mut cursor = txn.open_ro_cursor(self.db)?;
        let mut result = Vec::new();
        for (key, _val) in cursor.iter_start() {
            let entry = self.get_entry::<B>(key)?;
            result.push(entry);
        }
        Ok(result)
    }
}
