/* test_cfg.c - prove the config reader handles both shapes and refuses bad ones.
 *
 * Portable C: no weights, no Linux, no checkpoint. The point is that the reader that
 * decides which of 93 layers are MLA can be verified on any machine in a second.
 *
 *   test_cfg real     <config.json>    released nested shape; asserts the K3 constants
 *   test_cfg fixture  <ref_k3.json>    flat shape, config nested under "config"
 *   test_cfg reject   <file.json>      expects the load to FAIL (negative test)
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "k3_cfg.h"

static int fails = 0;

static void ck(int cond, const char *what, long got, long want)
{
    if (cond) { printf("  ok    %-28s %ld\n", what, got); }
    else      { printf("  FAIL  %-28s got %ld, want %ld\n", what, got, want); fails++; }
}

int main(int argc, char **argv)
{
    if (argc < 3) { fprintf(stderr, "usage: test_cfg <real|fixture|reject> <path>\n"); return 2; }
    const char *mode = argv[1], *path = argv[2];

    K3Cfg c; static int fa[128];

    if (!strcmp(mode, "reject")) {
        int ok = k3_cfg_load_file(&c, fa, 128, path);
        if (ok) { printf("  FAIL  expected rejection, but the config loaded\n"); return 1; }
        printf("  ok    correctly rejected %s\n", path);
        return 0;
    }

    if (!strcmp(mode, "fixture")) {
        /* ref_k3.json wraps the config in a "config" member. */
        FILE *f = fopen(path, "rb");
        if (!f) { perror(path); return 2; }
        fseek(f, 0, SEEK_END); long n = ftell(f); fseek(f, 0, SEEK_SET);
        char *txt = (char *)malloc((size_t)n + 1);
        size_t got = fread(txt, 1, (size_t)n, f); fclose(f); txt[got] = 0;
        char *arena = NULL;
        jval *root = json_parse(txt, &arena);
        jval *cfg = json_get(root, "config");
        if (!cfg) { fprintf(stderr, "%s has no \"config\" member\n", path); return 2; }
        if (!k3_cfg_load(&c, fa, 128, cfg, path)) return 1;

        /* The tiny oracle's own numbers, from make_k3_oracle.py. */
        ck(c.hidden == 128,        "hidden",          c.hidden, 128);
        ck(c.n_layers == 13,       "layers",          c.n_layers, 13);
        ck(c.vocab == 256,         "vocab",           c.vocab, 256);
        ck(c.kda_heads == 4,       "kda heads",       c.kda_heads, 4);
        ck(c.kda_head_dim == 16,   "kda head_dim",    c.kda_head_dim, 16);
        ck(c.n_experts == 8,       "experts",         c.n_experts, 8);
        ck(c.topk == 2,            "topk",            c.topk, 2);
        ck(c.latent == 64,         "latent",          c.latent, 64);
        ck(c.attn_res_block == 3,  "attn_res_block",  c.attn_res_block, 3);
        ck(c.situ_b1 == 4.0f,      "situ b1",         (long)c.situ_b1, 4);
        ck(c.situ_b2 == 25.0f,     "situ b2",         (long)c.situ_b2, 25);
        ck(c.n_full_attn == 4,     "full_attn count", c.n_full_attn, 4);
        ck(c.gate_lb == -5.0f,     "gate_lb (x-1)",   (long)(-c.gate_lb), 5);
        printf("\n%s\n", fails ? "FIXTURE CONFIG: FAIL" : "FIXTURE CONFIG: PASS");
        return fails ? 1 : 0;
    }

    if (!strcmp(mode, "real")) {
        if (!k3_cfg_load_file(&c, fa, 128, path)) return 1;

        /* Every one of these is independently attested in docs/the tech report, so a
         * silent config change shows up here rather than 93 layers later. */
        ck(c.hidden == 7168,        "hidden",         c.hidden, 7168);
        ck(c.n_layers == 93,        "layers",         c.n_layers, 93);
        ck(c.vocab == 163840,       "vocab",          c.vocab, 163840);
        ck(c.kda_heads == 96,       "kda heads",      c.kda_heads, 96);
        ck(c.kda_head_dim == 128,   "kda head_dim",   c.kda_head_dim, 128);
        ck(c.conv_k == 4,           "short conv k",   c.conv_k, 4);
        ck(c.n_heads == 96,         "attn heads",     c.n_heads, 96);
        ck(c.q_lora == 1536,        "q_lora",         c.q_lora, 1536);
        ck(c.kv_lora == 512,        "kv_lora",        c.kv_lora, 512);
        ck(c.qk_nope == 128,        "qk_nope",        c.qk_nope, 128);
        ck(c.qk_rope == 64,         "qk_rope",        c.qk_rope, 64);
        ck(c.v_head == 128,         "v_head",         c.v_head, 128);
        ck(c.n_experts == 896,      "experts",        c.n_experts, 896);
        ck(c.topk == 16,            "topk",           c.topk, 16);
        ck(c.n_shared == 2,         "shared experts", c.n_shared, 2);
        ck(c.latent == 3584,        "latent",         c.latent, 3584);
        ck(c.moe_inter == 3072,     "moe_inter",      c.moe_inter, 3072);
        ck(c.first_dense == 1,      "dense layers",   c.first_dense, 1);
        ck(c.dense_inter == 33792,  "dense_inter",    c.dense_inter, 33792);
        ck(c.attn_res_block == 12,  "attn_res_block", c.attn_res_block, 12);
        ck(c.situ_b1 == 4.0f,       "situ b1",        (long)c.situ_b1, 4);
        ck(c.situ_b2 == 25.0f,      "situ b2",        (long)c.situ_b2, 25);
        ck(c.gate_lb == -5.0f,      "gate_lb (x-1)",  (long)(-c.gate_lb), 5);

        /* THE check this file exists for. 24 MLA layers, one-based 4,8,...,92 plus a
         * final 93 so the last layer is global attention (tech report 2.1). If the
         * nesting were mishandled this count would be 0 and every layer would run KDA. */
        ck(c.n_full_attn == 24,     "MLA layer count", c.n_full_attn, 24);
        ck(fa[0] == 4,              "first MLA layer", fa[0], 4);
        ck(fa[22] == 92,            "MLA layer 23",    fa[22], 92);
        ck(fa[23] == 93,            "last MLA layer",  fa[23], 93);

        /* And the derived split the engine actually branches on. */
        int nmla = 0, nkda = 0;
        for (int i = 0; i < c.n_layers; i++) k3_is_mla(&c, i) ? nmla++ : nkda++;
        ck(nmla == 24,              "k3_is_mla says",  nmla, 24);
        ck(nkda == 69,              "k3_is_kda says",  nkda, 69);
        /* Layers 91 and 92 zero-based are BOTH MLA (one-based 92 and 93). */
        ck(k3_is_mla(&c, 91) && k3_is_mla(&c, 92), "layers 91,92 both MLA", 1, 1);
        ck(!k3_is_mla(&c, 0),       "layer 0 is KDA",  k3_is_mla(&c, 0) ? 1 : 0, 0);

        printf("\n%s\n", fails ? "REAL CONFIG: FAIL" : "REAL CONFIG: PASS");
        return fails ? 1 : 0;
    }

    fprintf(stderr, "unknown mode %s\n", mode);
    return 2;
}
