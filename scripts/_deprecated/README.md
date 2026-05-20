# Deprecated scripts

These scripts are no longer part of the active pipeline. They are kept on disk for reproducibility (the thesis's methodological journey is partly documented by what was tried and abandoned). **Do not run them as part of the current pipeline.**

| Script | Why deprecated |
|---|---|
| `08_download_adi17_msa.py` | Attempted to find MSA in ADI17 train shards. Confirmed empirically that ADI17 contains 0 MSA across all 40 shards / 990,821 rows. Superseded by FLEURS-based MSA collection. See FINDINGS §8.2.3. |
| `09_find_msa_files.py` | Diagnostic footer-only Parquet scan to identify MSA-bearing shards. Found column-level stats unavailable on the ADI17 dataset. Of historical interest only. |
| `colab_extract_msa.py` | Google Colab script that exhaustively iterated ADI17 train split via `hf_transfer`. Confirmed 0 MSA. Retired. |
| `colab_extract_msa_fleurs.py` | First-pass FLEURS MSA extraction attempt in Colab that failed silently 2026-04-27. Superseded by `colab_extract_msa_cv.py` (kept in main scripts/). |
| `colab_extract_embeddings.py` | Earlier Colab embedding-extraction script. Pivoted to local CPU extraction (`12_extract_embeddings_local.py`) due to Colab + file-host friction. |
| `colab_finetune_xlsr.py` | Colab T4-GPU fine-tuning script. Pivoted to local CPU fine-tuning (`16_finetune_xlsr_local.py`) for full control and resumability. |

If any of these are ever needed again, move back to `scripts/` and update any internal path references.
