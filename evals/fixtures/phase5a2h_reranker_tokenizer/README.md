# Frozen cross-encoder tokenizer fixture

These files were copied from the existing local snapshot of
`cross-encoder/ms-marco-MiniLM-L-6-v2` revision
`c5ee24cb16019beea0893ab7796b1df96625c6b8`, without a download.
The adjacent Phase 5A2H contract pins their exact SHA-256 identities.

The fixture supports offline pair-length and tokenizer tests without model
weights or a populated Hugging Face cache. Inference still requires the complete
pinned local model snapshot and refuses to download or substitute it.

The upstream config's historical `_name_or_path` mentions L-12. This is retained
unchanged and disclosed in the contract; the cached revision has six encoder
layers and a one-output `[1, 384]` classifier, independently verified in its
safetensors weights. The scoring activation is Identity (raw relevance logit).
