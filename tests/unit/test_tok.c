/* test_tok.c - exercise the C tokenizer against the Python oracle (tools/tok.py).
 *
 * The engine proper is Linux-only (O_DIRECT, posix_memalign, sys/resource). The
 * tokenizer is not: tok.h, k3_tok.h and json.h are portable C99 over stdio/stdlib/
 * string only. So this harness builds and runs anywhere, which is the point --
 * tokenizer parity can be established without a Linux box or a single model weight.
 *
 *   test_tok <files_dir> encode     "text"     -> comma-separated ids, one line
 *   test_tok <files_dir> encodefile <path>    -> same, but text read as raw bytes
 *   test_tok <files_dir> decode     1,2,3     -> the text
 *   test_tok <files_dir> roundtrip  <path>    -> encode then decode; PASS/FAIL on
 *                                                byte-exact recovery of the input
 *
 * `encode` output is deliberately byte-identical in format to `tok.py encode` so the
 * two can be diffed directly with no post-processing.
 *
 * USE encodefile FOR ANYTHING NON-ASCII. argv is re-encoded by the host shell -- on
 * Windows it arrives in the active code page, so "ÿ" reaches main() as one byte
 * rather than the two UTF-8 bytes the tokenizer must see. That is a property of the
 * process boundary, not of the tokenizer, and it makes argv useless as a parity
 * channel. encodefile reads the bytes verbatim and is the one that can fail.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "k3_tok.h"

#define MAXIDS  (1 << 20)
#define MAXTEXT (1 << 22)

static int parse_ids(const char *s, int *out, int max)
{
    int n = 0;
    while (*s && n < max) {
        while (*s == ',' || *s == ' ') s++;
        if (!*s) break;
        out[n++] = atoi(s);
        while (*s && *s != ',') s++;
    }
    return n;
}

int main(int argc, char **argv)
{
    if (argc < 3) {
        fprintf(stderr,
            "usage: test_tok <files_dir> encode \"text\"\n"
            "       test_tok <files_dir> decode 1,2,3\n"
            "       test_tok <files_dir> roundtrip <textfile>\n");
        return 2;
    }
    const char *dir = argv[1], *mode = argv[2];

    Tok T;
    k3_tok_load(&T, dir);

    static int ids[MAXIDS];
    static char text[MAXTEXT];

    if (!strcmp(mode, "encode") && argc > 3) {
        int n = tok_encode(&T, argv[3], (int)strlen(argv[3]), ids, MAXIDS);
        for (int i = 0; i < n; i++) printf("%s%d", i ? "," : "", ids[i]);
        printf("\n");
        return 0;
    }

    if (!strcmp(mode, "encodefile") && argc > 3) {
        FILE *f = fopen(argv[3], "rb");
        if (!f) { perror(argv[3]); return 1; }
        size_t n = fread(text, 1, MAXTEXT - 1, f);
        fclose(f);
        text[n] = 0;
        int m = tok_encode(&T, text, (int)n, ids, MAXIDS);
        for (int i = 0; i < m; i++) printf("%s%d", i ? "," : "", ids[i]);
        printf("\n");
        return 0;
    }

    if (!strcmp(mode, "decode") && argc > 3) {
        int n = parse_ids(argv[3], ids, MAXIDS);
        int m = tok_decode(&T, ids, n, text, MAXTEXT - 1);
        text[m] = 0;
        fwrite(text, 1, (size_t)m, stdout);
        printf("\n");
        return 0;
    }

    if (!strcmp(mode, "roundtrip") && argc > 3) {
        FILE *f = fopen(argv[3], "rb");
        if (!f) { perror(argv[3]); return 1; }
        size_t n = fread(text, 1, MAXTEXT - 1, f);
        fclose(f);
        text[n] = 0;

        int ni = tok_encode(&T, text, (int)n, ids, MAXIDS);
        static char back[MAXTEXT];
        int nb = tok_decode(&T, ids, ni, back, MAXTEXT - 1);

        /* Byte-exact recovery is the bar. A tokenizer that loses a byte somewhere is
         * a tokenizer that will silently corrupt a prompt. */
        int ok = (nb == (int)n) && !memcmp(back, text, n);
        printf("roundtrip: %d bytes -> %d ids -> %d bytes : %s\n",
               (int)n, ni, nb, ok ? "PASS" : "FAIL");
        if (!ok) {
            int lim = nb < (int)n ? nb : (int)n;
            int d = 0; while (d < lim && back[d] == text[d]) d++;
            fprintf(stderr, "first divergence at byte %d\n", d);
            fprintf(stderr, "  in : %.40s\n", text + d);
            fprintf(stderr, "  out: %.40s\n", back + d);
            return 1;
        }
        return 0;
    }

    fprintf(stderr, "unknown mode '%s'\n", mode);
    return 2;
}
