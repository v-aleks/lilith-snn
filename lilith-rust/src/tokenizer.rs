use tokenizers::Tokenizer;
use tokenizers::models::bpe::BPE;
use tokenizers::normalizers::unicode::{NFKC, NFKD};
use tokenizers::pre_tokenizers::whitespace::Whitespace;
use tokenizers::processors::template::TemplateProcessing;
use tokenizers::decoders::byte_level::ByteLevel;
use tokenizers::Encoding;
use anyhow::{Result, anyhow};
use crate::config::NordConfig;

/// Wrapper around the `tokenizers` crate for the Llama‑3.2 tokenizer.
///
/// The wrapper loads a pretrained tokenizer identified by `config.tokenizer_id`
/// (e.g., "meta-llama/Llama-3.2-1B") and exposes simple `encode`/`decode`
/// methods used by the SNN model.
pub struct TokenizerWrapper {
    tokenizer: Tokenizer,
    pad_id: u32,
}

impl TokenizerWrapper {
    /// Create a new tokenizer wrapper from a configuration.
    ///
    /// Returns an `anyhow::Result` because loading a pretrained tokenizer can
    /// fail (missing files, network errors, etc.).
    pub fn new(cfg: &NordConfig) -> Result<Self> {
        // The tokenizers crate provides `from_pretrained` which downloads the model
        // files if they are not present locally. This call may error, so we propagate
        // the error using `anyhow`.
        let tokenizer = Tokenizer::from_pretrained(&cfg.tokenizer_id)
            .map_err(|e| anyhow!("Failed to load tokenizer '{}': {}", cfg.tokenizer_id, e))?;

        // Determine the padding token id (if the tokenizer defines one).
        let pad_id = tokenizer.get_vocab().get("<pad>")
            .cloned()
            .unwrap_or(0) as u32;

        Ok(Self { tokenizer, pad_id })
    }

    /// Encode a string into a vector of token ids.
    ///
    /// Returns a `Result` to surface any encoding errors.
    pub fn encode(&self, text: &str) -> Result<Vec<u32>> {
        let encoding = self.tokenizer.encode(text, true)
            .map_err(|e| anyhow!("Encoding error: {}", e))?;
        Ok(encoding.get_ids().to_vec())
    }

    /// Decode a slice of token ids back into a string.
    ///
    /// Returns a `Result` because the underlying tokenizer may fail.
    pub fn decode(&self, ids: &[u32]) -> Result<String> {
        let encoding = Encoding::new(ids.to_vec(), None, None, None, None, None);
        self.tokenizer.decode(encoding.get_ids(), true)
            .map_err(|e| anyhow!("Decoding error: {}", e))
    }

    /// Return the padding token id.
    pub fn pad_id(&self) -> u32 {
        self.pad_id
    }
}
