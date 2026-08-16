// Tokenise a prepared JSONL dataset against a model's OWN vocab + chat template, and report
// the real token budget -- the figure the "train in under a day" arithmetic actually needs.
//
// The prep script (tools/prep_redteam.py) dedupes text; THIS validates that every example
// tokenises, that the response mask is non-empty, and prints the true per-epoch token count
// (not the ~4-chars/token planning guess). It loads only GGUF metadata, not the weights, so
// it runs in a second on a 17 GB checkpoint.
//
//     dataprep MODEL_DIR prepared.jsonl [--hist]
fn main() -> Result<(), String> {
    let a: Vec<String> = std::env::args().collect();
    let model = a.get(1).ok_or("usage: dataprep MODEL_DIR prepared.jsonl [--hist]")?;
    let data = a.get(2).ok_or("usage: dataprep MODEL_DIR prepared.jsonl [--hist]")?;
    let hist = a.iter().any(|x| x == "--hist");

    let st = k3::st::St::open(std::path::Path::new(model)).map_err(|e| e.to_string())?;
    let meta = st.meta.as_ref().ok_or("no gguf metadata")?;
    let tok = k3::tok::Tok::from_gguf(meta)?;
    let tmpl = k3::chat::Template::from_gguf(meta);
    if !tmpl.is_jinja() {
        return Err("model has no chat template; cannot build SFT examples".into());
    }

    let rows = k3::train::load_jsonl(data)?;
    println!("loaded {} deduped rows from {data}", rows.len());

    let mut total_ids = 0usize;
    let mut total_scored = 0usize;
    let mut max_len = 0usize;
    let mut lens: Vec<usize> = Vec::with_capacity(rows.len());
    let mut empty = 0usize;
    let mut failed = 0usize;
    for r in &rows {
        match k3::train::build_example(&tmpl, &tok, &r.system, &r.user, &r.response) {
            Ok(ex) => {
                if ex.scored_tokens() == 0 {
                    empty += 1;
                }
                total_ids += ex.ids.len();
                total_scored += ex.scored_tokens();
                max_len = max_len.max(ex.ids.len());
                lens.push(ex.ids.len());
            }
            Err(_) => failed += 1,
        }
    }
    lens.sort_unstable();
    let median = lens.get(lens.len() / 2).copied().unwrap_or(0);
    let p95 = lens.get(lens.len() * 95 / 100).copied().unwrap_or(0);

    println!("---- real token budget (model vocab) ----");
    println!("examples tokenised   : {}", lens.len());
    if failed > 0 {
        println!("FAILED to render     : {failed}  (template rejected the triple)");
    }
    if empty > 0 {
        println!("empty response mask  : {empty}  (would contribute no loss)");
    }
    println!("sequence length      : median {median}  p95 {p95}  max {max_len}");
    println!("tokens / epoch (all) : {total_ids}  (every token sees the full stack -- forward cost)");
    println!("tokens / epoch scored: {total_scored}  (response-only, the loss denominator)");

    if hist {
        // Sequence-length histogram in power-of-two buckets, to size the max context and
        // spot outliers that would blow up a fixed-width batch.
        println!("---- sequence-length histogram ----");
        let mut buckets = std::collections::BTreeMap::new();
        for &l in &lens {
            let b = (usize::BITS - l.next_power_of_two().leading_zeros()) as usize;
            *buckets.entry(1usize << b).or_insert(0usize) += 1;
        }
        for (bound, count) in buckets {
            println!("  <= {bound:>6} : {count}");
        }
    }
    Ok(())
}
