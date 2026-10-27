/* all-MiniLM-L6-v2 inference in plain C: WordPiece tokenizer, six BERT layers,
 * mean pooling, L2 normalization. Single-threaded, float32, no dependencies.
 *
 *   embed <model-dir>/c
 *
 * Reads one JSON string per line from stdin and writes one JSON object per line:
 *   {"ids":[101,...,102],"vector":[0.0123,...]}
 * The first input is answered only after the weights are loaded, so a caller
 * can exclude load time by waiting for the first reply.
 *
 * Tokenizer coverage: ASCII is handled exactly as the BERT tokenizer does
 * (lowercase, punctuation split, whitespace). For non-ASCII text, Latin-1 and
 * Latin Extended-A letters are lowercased and stripped of accents, CJK
 * ideographs are isolated, and General Punctuation is split. Other scripts
 * pass through to WordPiece unchanged, which may differ from the reference
 * tokenizer's Unicode handling. Special token names such as "[SEP]" written
 * literally in the input are ordinary text here, not special tokens.
 */

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define HIDDEN 384
#define HEADS 12
#define HEAD_DIM 32
#define INTERMEDIATE 1536
#define LAYERS 6
#define MAX_TOKENS 256
#define MAX_WORD_CHARS 100
#define CLS 101
#define SEP 102
#define UNK 100
#define LAYER_NORM_EPSILON 1e-12f

/* ------------------------------------------------------------ weights */

typedef struct {
    const float *wq, *bq, *wk, *bk, *wv, *bv, *wo, *bo;
    const float *ln1_w, *ln1_b, *wi, *bi, *wo2, *bo2, *ln2_w, *ln2_b;
} Layer;

static struct {
    const float *word, *position, *type, *ln_w, *ln_b;
    Layer layers[LAYERS];
} model;

static void fail(const char *message) {
    fprintf(stderr, "embed: %s\n", message);
    exit(1);
}

static uint32_t read_u32(FILE *file) {
    uint32_t value;
    if (fread(&value, 4, 1, file) != 1) fail("truncated weights file");
    return value;
}

/* Reads the tensor stream written by export_weights.py; names fix the layout. */
static void load_weights(const char *path) {
    FILE *file = fopen(path, "rb");
    if (!file) fail("cannot open weights.bin");
    char magic[4];
    if (fread(magic, 1, 4, file) != 4 || memcmp(magic, "MLM6", 4) != 0) fail("bad magic");
    uint32_t count = read_u32(file);
    for (uint32_t index = 0; index < count; index++) {
        char name[128];
        uint32_t length = read_u32(file);
        if (length >= sizeof name || fread(name, 1, length, file) != length) fail("bad tensor name");
        name[length] = 0;
        uint32_t rank = read_u32(file), elements = 1;
        for (uint32_t d = 0; d < rank; d++) elements *= read_u32(file);
        float *data = malloc(elements * sizeof(float));
        if (!data || fread(data, sizeof(float), elements, file) != elements) fail("truncated tensor");

        int layer;
        char role[96];
        if (sscanf(name, "encoder.layer.%d.%95s", &layer, role) == 2) {
            Layer *l = &model.layers[layer];
            const float **slot =
                !strcmp(role, "attention.self.query.weight") ? &l->wq :
                !strcmp(role, "attention.self.query.bias") ? &l->bq :
                !strcmp(role, "attention.self.key.weight") ? &l->wk :
                !strcmp(role, "attention.self.key.bias") ? &l->bk :
                !strcmp(role, "attention.self.value.weight") ? &l->wv :
                !strcmp(role, "attention.self.value.bias") ? &l->bv :
                !strcmp(role, "attention.output.dense.weight") ? &l->wo :
                !strcmp(role, "attention.output.dense.bias") ? &l->bo :
                !strcmp(role, "attention.output.LayerNorm.weight") ? &l->ln1_w :
                !strcmp(role, "attention.output.LayerNorm.bias") ? &l->ln1_b :
                !strcmp(role, "intermediate.dense.weight") ? &l->wi :
                !strcmp(role, "intermediate.dense.bias") ? &l->bi :
                !strcmp(role, "output.dense.weight") ? &l->wo2 :
                !strcmp(role, "output.dense.bias") ? &l->bo2 :
                !strcmp(role, "output.LayerNorm.weight") ? &l->ln2_w :
                !strcmp(role, "output.LayerNorm.bias") ? &l->ln2_b : NULL;
            if (!slot) fail(name);
            *slot = data;
        } else if (!strcmp(name, "embeddings.word_embeddings.weight")) model.word = data;
        else if (!strcmp(name, "embeddings.position_embeddings.weight")) model.position = data;
        else if (!strcmp(name, "embeddings.token_type_embeddings.weight")) model.type = data;
        else if (!strcmp(name, "embeddings.LayerNorm.weight")) model.ln_w = data;
        else if (!strcmp(name, "embeddings.LayerNorm.bias")) model.ln_b = data;
        else fail(name);
    }
    fclose(file);
}

/* ------------------------------------------------------------ vocabulary */

#define VOCAB_BUCKETS 65536  /* power of two above 30522 entries */

static struct { char **tokens; int count; int32_t buckets[VOCAB_BUCKETS]; } vocab;

static uint32_t hash(const char *text, size_t length) {
    uint32_t value = 2166136261u;
    for (size_t i = 0; i < length; i++) value = (value ^ (unsigned char)text[i]) * 16777619u;
    return value;
}

static void load_vocab(const char *path) {
    FILE *file = fopen(path, "r");
    if (!file) fail("cannot open vocab.txt");
    memset(vocab.buckets, -1, sizeof vocab.buckets);
    vocab.tokens = malloc(40000 * sizeof(char *));
    char line[512];
    while (fgets(line, sizeof line, file)) {
        size_t length = strcspn(line, "\n");
        line[length] = 0;
        char *token = strdup(line);
        int id = vocab.count++;
        vocab.tokens[id] = token;
        for (uint32_t slot = hash(token, length) & (VOCAB_BUCKETS - 1);; slot = (slot + 1) & (VOCAB_BUCKETS - 1)) {
            if (vocab.buckets[slot] < 0) { vocab.buckets[slot] = id; break; }
        }
    }
    fclose(file);
}

static int lookup(const char *text, size_t length) {
    for (uint32_t slot = hash(text, length) & (VOCAB_BUCKETS - 1);; slot = (slot + 1) & (VOCAB_BUCKETS - 1)) {
        int id = vocab.buckets[slot];
        if (id < 0) return -1;
        if (strlen(vocab.tokens[id]) == length && memcmp(vocab.tokens[id], text, length) == 0) return id;
    }
}

/* ------------------------------------------------------------ tokenizer */

static int decode_utf8(const unsigned char *s, size_t length, size_t *i, uint32_t *out) {
    unsigned char c = s[*i];
    int extra = c < 0x80 ? 0 : (c & 0xE0) == 0xC0 ? 1 : (c & 0xF0) == 0xE0 ? 2 : (c & 0xF8) == 0xF0 ? 3 : -1;
    if (extra < 0 || *i + extra >= length + (extra == 0)) { *out = 0xFFFD; (*i)++; return 1; }
    uint32_t code = extra == 0 ? c : extra == 1 ? c & 0x1F : extra == 2 ? c & 0x0F : c & 0x07;
    for (int k = 1; k <= extra; k++) code = (code << 6) | (s[*i + k] & 0x3F);
    *i += extra + 1;
    *out = code;
    return 1;
}

static size_t encode_utf8(uint32_t code, char *out) {
    if (code < 0x80) { out[0] = (char)code; return 1; }
    if (code < 0x800) { out[0] = (char)(0xC0 | code >> 6); out[1] = (char)(0x80 | (code & 0x3F)); return 2; }
    if (code < 0x10000) {
        out[0] = (char)(0xE0 | code >> 12); out[1] = (char)(0x80 | ((code >> 6) & 0x3F));
        out[2] = (char)(0x80 | (code & 0x3F)); return 3;
    }
    out[0] = (char)(0xF0 | code >> 18); out[1] = (char)(0x80 | ((code >> 12) & 0x3F));
    out[2] = (char)(0x80 | ((code >> 6) & 0x3F)); out[3] = (char)(0x80 | (code & 0x3F)); return 4;
}

/* Lowercases U+00C0..U+017F and strips accents the way NFD plus mark removal
 * does. '.' marks letters without a base letter, which are only lowercased. */
static uint32_t normalize_latin(uint32_t c) {
    static const char latin1[] =
        "aaaaaa.ceeeeiiii" ".nooooo..uuuuy.." "aaaaaa.ceeeeiiii" ".nooooo..uuuuy.y";
    static const char extended[] =
        "aaaaaaccccccccdd" "..eeeeeeeeeegggg" "gggghh..iiiiiiii" "i...jjkk.llllll."
        "...nnnnnn...oooo" "oo..rrrrrrssssss" "sstttt..uuuuuuuu" "uuuuwwyyyzzzzzz.";
    if (c >= 0xC0 && c <= 0xFF) {
        char base = latin1[c - 0xC0];
        if (base != '.') return (unsigned char)base;
        return c < 0xE0 && c != 0xD7 ? c + 0x20 : c;
    }
    if (c >= 0x100 && c <= 0x17F) {
        char base = extended[c - 0x100];
        if (base != '.') return (unsigned char)base;
        switch (c) {
            case 0x110: case 0x126: case 0x132: case 0x13F: case 0x141:
            case 0x14A: case 0x152: case 0x166: return c + 1;
            default: return c;
        }
    }
    return c;
}

static int is_space(uint32_t c) { return c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == 0xA0 || c == 0x3000 || (c >= 0x2000 && c <= 0x200A); }
static int is_control(uint32_t c) { return (c < 0x20 && !is_space(c)) || c == 0x7F || (c >= 0x80 && c < 0xA0) || c == 0 || c == 0xFFFD; }
static int is_cjk(uint32_t c) {
    return (c >= 0x4E00 && c <= 0x9FFF) || (c >= 0x3400 && c <= 0x4DBF) || (c >= 0x20000 && c <= 0x2A6DF) ||
           (c >= 0x2A700 && c <= 0x2CEAF) || (c >= 0xF900 && c <= 0xFAFF) || (c >= 0x2F800 && c <= 0x2FA1F);
}
static int is_punctuation(uint32_t c) {
    return (c >= 33 && c <= 47) || (c >= 58 && c <= 64) || (c >= 91 && c <= 96) || (c >= 123 && c <= 126) ||
           (c >= 0x2010 && c <= 0x2027) || (c >= 0x2030 && c <= 0x205E) || (c >= 0x3001 && c <= 0x3003) ||
           (c >= 0x3008 && c <= 0x3011) || c == 0xA1 || c == 0xA7 || c == 0xAB || c == 0xB6 || c == 0xB7 || c == 0xBB || c == 0xBF;
}

/* Appends the WordPiece IDs for one word to ids; words are pre-tokenized by caller. */
static int wordpiece(const char *word, size_t length, size_t chars, int *ids, int count) {
    if (count >= MAX_TOKENS) return count;
    if (chars > MAX_WORD_CHARS) { ids[count++] = UNK; return count; }
    int start_count = count;
    size_t start = 0;
    char buffer[2 * MAX_WORD_CHARS * 4 + 3];
    while (start < length) {
        size_t end = length;
        int id = -1;
        while (start < end) {
            size_t piece = end - start;
            if (start > 0) { memcpy(buffer, "##", 2); memcpy(buffer + 2, word + start, piece); piece += 2; }
            else memcpy(buffer, word + start, piece);
            id = lookup(buffer, piece);
            if (id >= 0) break;
            /* Step back to the previous code point boundary. */
            do end--; while (end > start && ((unsigned char)word[end] & 0xC0) == 0x80);
        }
        if (id < 0) { ids[start_count] = UNK; return start_count + 1; }
        if (count >= MAX_TOKENS) return count;
        ids[count++] = id;
        start = end;
    }
    return count;
}

/* Normalizes, pre-tokenizes, and encodes text into [CLS] ids [SEP], at most MAX_TOKENS. */
static int tokenize(const char *text, size_t length, int *ids) {
    int count = 0;
    ids[count++] = CLS;
    char word[MAX_WORD_CHARS * 4 + 8];
    size_t word_length = 0, word_chars = 0;
    size_t i = 0;
    /* Content tokens are capped so [SEP] always fits. */
    int limit = MAX_TOKENS - 1;
    while (i < length) {
        uint32_t c;
        decode_utf8((const unsigned char *)text, length, &i, &c);
        if (is_control(c)) continue;
        c = normalize_latin(c);
        if (c >= 'A' && c <= 'Z') c += 32;
        int boundary = is_space(c) || is_punctuation(c) || is_cjk(c);
        if (boundary && word_length) {
            if (count < limit) count = wordpiece(word, word_length, word_chars, ids, count);
            if (count > limit) count = limit;
            word_length = word_chars = 0;
        }
        if (is_space(c)) continue;
        if (boundary) {
            char single[4];
            size_t n = encode_utf8(c, single);
            if (count < limit) count = wordpiece(single, n, 1, ids, count);
            continue;
        }
        if (word_chars < MAX_WORD_CHARS + 1 && word_length + 4 < sizeof word) {
            word_length += encode_utf8(c, word + word_length);
        }
        word_chars++;
    }
    if (word_length && count < limit) count = wordpiece(word, word_length, word_chars, ids, count);
    if (count > limit) count = limit;
    ids[count++] = SEP;
    return count;
}

/* ------------------------------------------------------------ model */

static void layer_norm(float *x, int tokens, const float *weight, const float *bias) {
    for (int t = 0; t < tokens; t++) {
        float *row = x + t * HIDDEN, mean = 0, variance = 0;
        for (int i = 0; i < HIDDEN; i++) mean += row[i];
        mean /= HIDDEN;
        for (int i = 0; i < HIDDEN; i++) { float d = row[i] - mean; variance += d * d; }
        float scale = 1.0f / sqrtf(variance / HIDDEN + LAYER_NORM_EPSILON);
        for (int i = 0; i < HIDDEN; i++) row[i] = (row[i] - mean) * scale * weight[i] + bias[i];
    }
}

/* y[t] = x[t] @ W + b with W stored [in][out]. Eight weight rows are applied
 * per pass so each output element is loaded and stored once per eight
 * multiply-adds; the sums are still accumulated in row order, and the inner
 * loop runs along out, so it vectorizes without reassociating anything. */
static void linear(const float *restrict x, int tokens, int in, const float *restrict w,
                   const float *restrict b, int out, float *restrict y) {
    for (int t = 0; t < tokens; t++) memcpy(y + t * out, b, out * sizeof(float));
    for (int i = 0; i < in; i += 8) {
        const float *w0 = w + i * out, *w1 = w0 + out, *w2 = w1 + out, *w3 = w2 + out;
        const float *w4 = w3 + out, *w5 = w4 + out, *w6 = w5 + out, *w7 = w6 + out;
        for (int t = 0; t < tokens; t++) {
            const float *xi = x + t * in + i;
            float x0 = xi[0], x1 = xi[1], x2 = xi[2], x3 = xi[3];
            float x4 = xi[4], x5 = xi[5], x6 = xi[6], x7 = xi[7];
            float *result = y + t * out;
            for (int j = 0; j < out; j++) {
                float r = result[j];
                r += x0 * w0[j]; r += x1 * w1[j]; r += x2 * w2[j]; r += x3 * w3[j];
                r += x4 * w4[j]; r += x5 * w5[j]; r += x6 * w6[j]; r += x7 * w7[j];
                result[j] = r;
            }
        }
    }
}

static float gelu(float x) { return 0.5f * x * (1.0f + erff(x / 1.4142135f)); }

static float q[MAX_TOKENS * HIDDEN], k[MAX_TOKENS * HIDDEN], v[MAX_TOKENS * HIDDEN];
static float context[MAX_TOKENS * HIDDEN], scratch[MAX_TOKENS * HIDDEN], inter[MAX_TOKENS * INTERMEDIATE];
static float scores[MAX_TOKENS];

static void attention(float *x, int tokens, const Layer *l) {
    linear(x, tokens, HIDDEN, l->wq, l->bq, HIDDEN, q);
    linear(x, tokens, HIDDEN, l->wk, l->bk, HIDDEN, k);
    linear(x, tokens, HIDDEN, l->wv, l->bv, HIDDEN, v);
    for (int h = 0; h < HEADS; h++) {
        for (int i = 0; i < tokens; i++) {
            const float *qi = q + i * HIDDEN + h * HEAD_DIM;
            float max = -INFINITY;
            for (int j = 0; j < tokens; j++) {
                const float *kj = k + j * HIDDEN + h * HEAD_DIM;
                float dot = 0;
                for (int d = 0; d < HEAD_DIM; d++) dot += qi[d] * kj[d];
                scores[j] = dot * 0.17677669529663687f;  /* 1 / sqrt(HEAD_DIM) */
                if (scores[j] > max) max = scores[j];
            }
            float sum = 0;
            for (int j = 0; j < tokens; j++) { scores[j] = expf(scores[j] - max); sum += scores[j]; }
            float *out = context + i * HIDDEN + h * HEAD_DIM;
            memset(out, 0, HEAD_DIM * sizeof(float));
            for (int j = 0; j < tokens; j++) {
                float p = scores[j] / sum;
                const float *vj = v + j * HIDDEN + h * HEAD_DIM;
                for (int d = 0; d < HEAD_DIM; d++) out[d] += p * vj[d];
            }
        }
    }
    linear(context, tokens, HIDDEN, l->wo, l->bo, HIDDEN, scratch);
    for (int i = 0; i < tokens * HIDDEN; i++) x[i] += scratch[i];
    layer_norm(x, tokens, l->ln1_w, l->ln1_b);
}

static void feed_forward(float *x, int tokens, const Layer *l) {
    linear(x, tokens, HIDDEN, l->wi, l->bi, INTERMEDIATE, inter);
    for (int i = 0; i < tokens * INTERMEDIATE; i++) inter[i] = gelu(inter[i]);
    linear(inter, tokens, INTERMEDIATE, l->wo2, l->bo2, HIDDEN, scratch);
    for (int i = 0; i < tokens * HIDDEN; i++) x[i] += scratch[i];
    layer_norm(x, tokens, l->ln2_w, l->ln2_b);
}

static float hidden[MAX_TOKENS * HIDDEN];

/* Writes the normalized 384-dimensional sentence embedding for ids into vector. */
static void embed(const int *ids, int tokens, float *vector) {
    for (int t = 0; t < tokens; t++)
        for (int i = 0; i < HIDDEN; i++)
            hidden[t * HIDDEN + i] = model.word[ids[t] * HIDDEN + i] + model.type[i] + model.position[t * HIDDEN + i];
    layer_norm(hidden, tokens, model.ln_w, model.ln_b);
    for (int l = 0; l < LAYERS; l++) {
        attention(hidden, tokens, &model.layers[l]);
        feed_forward(hidden, tokens, &model.layers[l]);
    }
    float norm = 0;
    for (int i = 0; i < HIDDEN; i++) {
        float sum = 0;
        for (int t = 0; t < tokens; t++) sum += hidden[t * HIDDEN + i];
        vector[i] = sum / tokens;
        norm += vector[i] * vector[i];
    }
    norm = sqrtf(norm);
    if (norm < 1e-12f) norm = 1e-12f;
    for (int i = 0; i < HIDDEN; i++) vector[i] /= norm;
}

/* ------------------------------------------------------------ line protocol */

/* Decodes a JSON string literal in place; returns the decoded length or -1. */
static long decode_json_string(char *line) {
    char *p = line;
    while (*p == ' ' || *p == '\t') p++;
    if (*p != '"') return -1;
    p++;
    char *out = line;
    while (*p && *p != '"') {
        if (*p != '\\') { *out++ = *p++; continue; }
        p++;
        switch (*p++) {
            case '"': *out++ = '"'; break;
            case '\\': *out++ = '\\'; break;
            case '/': *out++ = '/'; break;
            case 'b': *out++ = '\b'; break;
            case 'f': *out++ = '\f'; break;
            case 'n': *out++ = '\n'; break;
            case 'r': *out++ = '\r'; break;
            case 't': *out++ = '\t'; break;
            case 'u': {
                char digits[5] = {0};
                memcpy(digits, p, 4);
                uint32_t code = (uint32_t)strtoul(digits, NULL, 16);
                p += 4;
                if (code >= 0xD800 && code <= 0xDBFF && p[0] == '\\' && p[1] == 'u') {
                    memcpy(digits, p + 2, 4);
                    uint32_t low = (uint32_t)strtoul(digits, NULL, 16);
                    code = 0x10000 + ((code - 0xD800) << 10) + (low - 0xDC00);
                    p += 6;
                }
                out += encode_utf8(code, out);
                break;
            }
            default: return -1;
        }
    }
    if (*p != '"') return -1;
    return out - line;
}

int main(int argc, char **argv) {
    if (argc != 2) fail("usage: embed <model-dir>/c");
    char path[4096];
    snprintf(path, sizeof path, "%s/weights.bin", argv[1]);
    load_weights(path);
    snprintf(path, sizeof path, "%s/vocab.txt", argv[1]);
    load_vocab(path);

    static char line[1 << 20];
    static int ids[MAX_TOKENS];
    static float vector[HIDDEN];
    while (fgets(line, sizeof line, stdin)) {
        long length = decode_json_string(line);
        if (length < 0) { puts("{\"error\":\"expected a JSON string\"}"); fflush(stdout); continue; }
        int tokens = tokenize(line, (size_t)length, ids);
        embed(ids, tokens, vector);
        fputs("{\"ids\":[", stdout);
        for (int t = 0; t < tokens; t++) printf(t ? ",%d" : "%d", ids[t]);
        fputs("],\"vector\":[", stdout);
        for (int i = 0; i < HIDDEN; i++) printf(i ? ",%.9g" : "%.9g", vector[i]);
        puts("]}");
        fflush(stdout);
    }
    return 0;
}
