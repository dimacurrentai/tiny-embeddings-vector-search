/* all-MiniLM-L6-v2 inference in plain C: WordPiece tokenizer, six BERT layers,
 * mean pooling, L2 normalization. Float32, no dependencies beyond libc and
 * pthreads.
 *
 *   embed <model-dir>/c [--threads N] [--time]
 *
 * --threads splits every matrix product and the attention heads across N
 * threads; one thread is the default. With --time, the microseconds spent
 * tokenizing and embedding each request are written to stderr, excluding
 * JSON parsing and output.
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
#include <pthread.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#ifdef __APPLE__
#include <pthread/qos.h>
#endif

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

#define TILE_OUT 16

/* Reorders a [in][out] weight into [out / TILE_OUT][in][TILE_OUT] so each tile
 * streams its sixteen columns contiguously instead of striding across rows. */
static float *pack(float *w, uint32_t in, uint32_t out) {
    float *packed = malloc((size_t)in * out * sizeof(float));
    if (!packed) exit(1);
    for (uint32_t j0 = 0; j0 < out; j0 += TILE_OUT)
        for (uint32_t i = 0; i < in; i++)
            memcpy(packed + ((size_t)j0 / TILE_OUT * in + i) * TILE_OUT, w + (size_t)i * out + j0, TILE_OUT * sizeof(float));
    free(w);
    return packed;
}

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
        uint32_t dims[2] = {0, 0};
        fseek(file, -(long)(rank * 4), SEEK_CUR);
        for (uint32_t d = 0; d < rank; d++) dims[d < 2 ? d : 1] = read_u32(file);
        float *data = malloc(elements * sizeof(float));
        if (!data || fread(data, sizeof(float), elements, file) != elements) fail("truncated tensor");
        if (rank == 2 && strstr(name, "encoder.layer.") && strstr(name, ".weight")) data = pack(data, dims[0], dims[1]);

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

/* ------------------------------------------------------------ thread pool */

/* Workers spin on the job generation while a request is in flight and sleep
 * on a condition variable between requests. The main thread takes share 0. */
typedef void (*Job)(void *context, int begin, int end);

static struct {
    int threads;
    pthread_mutex_t mutex;
    pthread_cond_t wake;
    _Atomic int generation, remaining, busy;
    Job job;
    void *context;
    int count;
} pool = {1, PTHREAD_MUTEX_INITIALIZER, PTHREAD_COND_INITIALIZER, 0, 0, 0, NULL, NULL, 0};

static void run_share(int index) {
    int begin = (int)((long)pool.count * index / pool.threads);
    int end = (int)((long)pool.count * (index + 1) / pool.threads);
    if (begin < end) pool.job(pool.context, begin, end);
}

static void *worker(void *argument) {
#ifdef __APPLE__
    pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0);
#endif
    int index = (int)(intptr_t)argument, seen = 0;
    for (;;) {
        while (atomic_load_explicit(&pool.generation, memory_order_acquire) == seen) {
            if (!atomic_load(&pool.busy)) {
                pthread_mutex_lock(&pool.mutex);
                while (!atomic_load(&pool.busy) && atomic_load(&pool.generation) == seen)
                    pthread_cond_wait(&pool.wake, &pool.mutex);
                pthread_mutex_unlock(&pool.mutex);
            }
        }
        seen++;
        run_share(index);
        atomic_fetch_sub_explicit(&pool.remaining, 1, memory_order_release);
    }
    return NULL;
}

static void start_pool(int threads) {
    pool.threads = threads;
    for (int index = 1; index < threads; index++) {
        pthread_t thread;
        if (pthread_create(&thread, NULL, worker, (void *)(intptr_t)index) != 0) fail("cannot start thread");
        pthread_detach(thread);
    }
}

/* Runs job over [0, count) split evenly across the pool and waits for all shares. */
static void parallel(int count, Job job, void *context) {
    pool.job = job, pool.context = context, pool.count = count;
    if (pool.threads == 1) { run_share(0); return; }
    atomic_store(&pool.remaining, pool.threads - 1);
    atomic_fetch_add_explicit(&pool.generation, 1, memory_order_release);
    run_share(0);
    while (atomic_load_explicit(&pool.remaining, memory_order_acquire) != 0) ;
}

static void request_begin(void) {
    if (pool.threads == 1) return;
    pthread_mutex_lock(&pool.mutex);
    atomic_store(&pool.busy, 1);
    pthread_cond_broadcast(&pool.wake);
    pthread_mutex_unlock(&pool.mutex);
}

static void request_end(void) { atomic_store(&pool.busy, 0); }

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

/* y[t] = x[t] @ W + b with W packed [out / TILE_OUT][in][TILE_OUT], computed in
 * tiles of up to four tokens by sixteen outputs whose accumulators stay in
 * vector registers for the whole in loop. Each element still sums its products
 * in row order, so results match a plain loop bit for bit. A column block is
 * read from memory once and reused from cache for every token tile; tiles of
 * 4, 2, and 1 tokens cover any token count without padding rows. */

/* One ROWS x TILE_OUT tile: x points at the first token row, w at the packed
 * column block, b and y at column j0. */
#define DEFINE_TILE(NAME, ROWS) \
static void NAME(const float *restrict x, int in, const float *restrict w, \
                 const float *restrict b, int out, float *restrict y) { \
    float acc[ROWS][TILE_OUT]; \
    for (int r = 0; r < ROWS; r++) \
        for (int j = 0; j < TILE_OUT; j++) acc[r][j] = b[j]; \
    for (int i = 0; i < in; i++, w += TILE_OUT) { \
        float s[ROWS]; \
        for (int r = 0; r < ROWS; r++) s[r] = x[r * in + i]; \
        for (int r = 0; r < ROWS; r++) \
            for (int j = 0; j < TILE_OUT; j++) acc[r][j] += s[r] * w[j]; \
    } \
    for (int r = 0; r < ROWS; r++) memcpy(y + r * out, acc[r], sizeof acc[r]); \
}

DEFINE_TILE(tile4, 4)
DEFINE_TILE(tile2, 2)
DEFINE_TILE(tile1, 1)

typedef struct { const float *x, *w, *b; float *y; int tokens, in, out; } Linear;

static void linear_blocks(void *context, int begin, int end) {
    const Linear *l = context;
    for (int block = begin; block < end; block++) {
        int j0 = block * TILE_OUT;
        const float *w = l->w + (size_t)j0 * l->in, *b = l->b + j0;
        float *y = l->y + j0;
        int t = 0;
        for (; t + 4 <= l->tokens; t += 4) tile4(l->x + t * l->in, l->in, w, b, l->out, y + t * l->out);
        if (t + 2 <= l->tokens) { tile2(l->x + t * l->in, l->in, w, b, l->out, y + t * l->out); t += 2; }
        if (t < l->tokens) tile1(l->x + t * l->in, l->in, w, b, l->out, y + t * l->out);
    }
}

static void linear(const float *x, int tokens, int in, const float *w, const float *b, int out, float *y) {
    Linear l = {x, w, b, y, tokens, in, out};
    parallel(out / TILE_OUT, linear_blocks, &l);
}

/* exp(x) for x <= 0, accurate to about one float ulp, written so the GELU loop
 * vectorizes: 2^n is assembled from the exponent bits instead of calling ldexp. */
static inline float exp_negative(float x) {
    if (x < -87.0f) x = -87.0f;
    float n = (float)(int)(x * 1.44269504f - 0.5f);
    float r = x - n * 0.693145751953125f - n * 1.428606765330187e-06f;
    float p = 1.9875691500e-4f;
    p = p * r + 1.3981999507e-3f;
    p = p * r + 8.3334519073e-3f;
    p = p * r + 4.1665795894e-2f;
    p = p * r + 1.6666665459e-1f;
    p = p * r + 5.0000001201e-1f;
    p = p * r * r + r + 1.0f;
    int32_t bits = ((int32_t)n + 127) << 23;
    float scale;
    memcpy(&scale, &bits, sizeof scale);
    return p * scale;
}

/* erf by the Abramowitz and Stegun 7.1.26 rational form, absolute error below
 * 1.5e-7, branch-free so the GELU loop vectorizes. */
static inline float erf_vector(float x) {
    float sign = x < 0 ? -1.0f : 1.0f, a = fabsf(x);
    float t = 1.0f / (1.0f + 0.3275911f * a);
    float poly = ((((1.061405429f * t - 1.453152027f) * t + 1.421413741f) * t - 0.284496736f) * t + 0.254829592f) * t;
    return sign * (1.0f - poly * exp_negative(-a * a));
}

static inline float gelu(float x) { return 0.5f * x * (1.0f + erf_vector(x * 0.70710678f)); }

static void gelu_rows(void *context, int begin, int end) {
    float *values = context;
    for (int i = begin * INTERMEDIATE; i < end * INTERMEDIATE; i++) values[i] = gelu(values[i]);
}

static float q[MAX_TOKENS * HIDDEN], k[MAX_TOKENS * HIDDEN], v[MAX_TOKENS * HIDDEN];
static float context[MAX_TOKENS * HIDDEN], scratch[MAX_TOKENS * HIDDEN], inter[MAX_TOKENS * INTERMEDIATE];

typedef struct { int tokens; } Heads;

static void attention_heads(void *argument, int begin, int end) {
    int tokens = ((Heads *)argument)->tokens;
    float scores[MAX_TOKENS];
    for (int h = begin; h < end; h++) {
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
}

static void attention(float *x, int tokens, const Layer *l) {
    linear(x, tokens, HIDDEN, l->wq, l->bq, HIDDEN, q);
    linear(x, tokens, HIDDEN, l->wk, l->bk, HIDDEN, k);
    linear(x, tokens, HIDDEN, l->wv, l->bv, HIDDEN, v);
    Heads heads = {tokens};
    parallel(HEADS, attention_heads, &heads);
    linear(context, tokens, HIDDEN, l->wo, l->bo, HIDDEN, scratch);
    for (int i = 0; i < tokens * HIDDEN; i++) x[i] += scratch[i];
    layer_norm(x, tokens, l->ln1_w, l->ln1_b);
}

static void feed_forward(float *x, int tokens, const Layer *l) {
    linear(x, tokens, HIDDEN, l->wi, l->bi, INTERMEDIATE, inter);
    parallel(tokens, gelu_rows, inter);
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
    const char *directory = NULL;
    int threads = 1, timing = 0;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--time")) timing = 1;
        else if (!strcmp(argv[i], "--threads") && i + 1 < argc) threads = atoi(argv[++i]);
        else if (argv[i][0] != '-' && !directory) directory = argv[i];
        else fail("usage: embed <model-dir>/c [--threads N] [--time]");
    }
    if (!directory || threads < 1 || threads > 64) fail("usage: embed <model-dir>/c [--threads N] [--time]");
#ifdef __APPLE__
    pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0);
#endif
    char path[4096];
    snprintf(path, sizeof path, "%s/weights.bin", directory);
    load_weights(path);
    snprintf(path, sizeof path, "%s/vocab.txt", directory);
    load_vocab(path);
    start_pool(threads);

    static char line[1 << 20];
    static int ids[MAX_TOKENS];
    static float vector[HIDDEN];
    while (fgets(line, sizeof line, stdin)) {
        long length = decode_json_string(line);
        if (length < 0) { puts("{\"error\":\"expected a JSON string\"}"); fflush(stdout); continue; }
        struct timespec started, finished;
        clock_gettime(CLOCK_MONOTONIC, &started);
        request_begin();
        int tokens = tokenize(line, (size_t)length, ids);
        embed(ids, tokens, vector);
        request_end();
        clock_gettime(CLOCK_MONOTONIC, &finished);
        if (timing) fprintf(stderr, "%ld\n", (finished.tv_sec - started.tv_sec) * 1000000L + (finished.tv_nsec - started.tv_nsec) / 1000);
        fputs("{\"ids\":[", stdout);
        for (int t = 0; t < tokens; t++) printf(t ? ",%d" : "%d", ids[t]);
        fputs("],\"vector\":[", stdout);
        for (int i = 0; i < HIDDEN; i++) printf(i ? ",%.9g" : "%.9g", vector[i]);
        puts("]}");
        fflush(stdout);
    }
    return 0;
}
