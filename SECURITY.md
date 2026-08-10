# Security

## Reporting

Report vulnerabilities privately through GitHub's "Report a vulnerability" flow rather
than a public issue.

## Scope

**The trust boundary is the model files.** This engine parses binary and JSON input that
users routinely download from third-party mirrors: safetensors headers, a packed trunk
file, config.json, and tokenizer files. It treats all of them as untrusted.

The parsers bound what they read, nesting depth, tensor element counts, header lengths,
index sizes, and refuse implausible values rather than trusting the file. A crafted
checkpoint that causes an out-of-bounds read or write, an unbounded allocation, or
execution of attacker-controlled data is a vulnerability, and is in scope.

The model WEIGHTS themselves are not a trust boundary this engine can defend: a
checkpoint with valid structure and hostile parameter values will produce hostile output,
and no parser check can prevent that.

`scripts/download-model.sh` verifies the checkpoint against its published **sizes**
shard count, summed byte total, and every per-shard size. Note the limit: a size check
detects truncation and short reads, but not a substituted shard of identical length,
which is exactly the crafted-checkpoint case this section is about. Verify against
published hashes if your threat model includes a hostile mirror.

Out of scope: the quality or safety of generated text, and denial of service through
legitimately large models.

## Credentials

The download script reads a HuggingFace token from the environment and never echoes it.
`.gitignore` excludes `hf_token*` and key material. If you add tooling that touches
credentials, keep them out of logs, this project produces logs intended to be published.
