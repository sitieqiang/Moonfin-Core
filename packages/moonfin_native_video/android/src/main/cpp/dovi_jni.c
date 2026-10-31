// JNI shim between DoviRpu.kt and libdovi. libdovi is dlopened so this
// library always loads, and a build that ships without libdovi.so degrades to
// the caller's strip path instead of failing to link.
#include <dlfcn.h>
#include <jni.h>
#include <stddef.h>
#include <stdint.h>

typedef struct DoviData {
    const uint8_t *data;
    size_t len;
} DoviData;

typedef void DoviRpuOpaque;

// Mirrored by DoviRpu.kt so a report can say why conversion is unavailable.
#define DOVI_STATUS_READY 0
#define DOVI_STATUS_UNRESOLVED 1
#define DOVI_STATUS_NOT_FOUND 2
#define DOVI_STATUS_MISSING_SYMBOLS 3

typedef DoviRpuOpaque *(*dovi_parse_unspec62_nalu_fn)(const uint8_t *, size_t);
typedef int (*dovi_convert_rpu_with_mode_fn)(DoviRpuOpaque *, uint8_t);
typedef const DoviData *(*dovi_write_unspec62_nalu_fn)(DoviRpuOpaque *);
typedef void (*dovi_data_free_fn)(const DoviData *);
typedef void (*dovi_rpu_free_fn)(DoviRpuOpaque *);

static dovi_parse_unspec62_nalu_fn p_parse;
static dovi_convert_rpu_with_mode_fn p_convert;
static dovi_write_unspec62_nalu_fn p_write;
static dovi_data_free_fn p_data_free;
static dovi_rpu_free_fn p_rpu_free;

// Profile 5 reshaping. These mirror libdovi 3.4.0's repr(C) structs up to the
// last field read here, and are resolved separately so a libdovi without them
// only loses reshaping.
typedef struct { const uint16_t *data; size_t len; } DoviU16Data;
typedef struct { const uint64_t *data; size_t len; } DoviU64Data;
typedef struct { const int64_t *data; size_t len; } DoviI64Data;
typedef struct { const DoviU64Data *const *list; size_t len; } DoviU64Data2D;
typedef struct { const DoviI64Data *const *list; size_t len; } DoviI64Data2D;
typedef struct { const DoviU64Data2D *const *list; size_t len; } DoviU64Data3D;
typedef struct { const DoviI64Data2D *const *list; size_t len; } DoviI64Data3D;

typedef struct {
    DoviU64Data poly_order_minus1;
    DoviData linear_interp_flag;
    DoviI64Data2D poly_coef_int;
    DoviU64Data2D poly_coef;
} DoviPolynomialCurve;

typedef struct {
    DoviData mmr_order_minus1;
    DoviI64Data mmr_constant_int;
    DoviU64Data mmr_constant;
    DoviI64Data3D mmr_coef_int;
    DoviU64Data3D mmr_coef;
} DoviMMRCurve;

typedef struct {
    uint64_t num_pivots_minus2;
    DoviU16Data pivots;
    uint8_t mapping_idc;
    const DoviPolynomialCurve *polynomial;
    const DoviMMRCurve *mmr;
} DoviReshapingCurve;

typedef struct {
    uint64_t vdr_rpu_id;
    uint64_t mapping_color_space;
    uint64_t mapping_chroma_format_idc;
    uint64_t num_x_partitions_minus1;
    uint64_t num_y_partitions_minus1;
    DoviReshapingCurve curves[3];
} DoviRpuDataMapping;

typedef struct {
    uint8_t guessed_profile;
    const char *el_type;
    uint8_t rpu_nal_prefix;
    uint8_t rpu_type;
    uint16_t rpu_format;
    uint8_t vdr_rpu_profile;
    uint8_t vdr_rpu_level;
    uint8_t vdr_seq_info_present_flag;
    uint8_t chroma_resampling_explicit_filter_flag;
    uint8_t coefficient_data_type;
    uint64_t coefficient_log2_denom;
    uint8_t vdr_rpu_normalized_idc;
    uint8_t bl_video_full_range_flag;
    uint64_t bl_bit_depth_minus8;
} DoviRpuDataHeader;

typedef struct {
    uint8_t compressed;
    uint64_t affected_dm_metadata_id;
    uint64_t current_dm_metadata_id;
    uint64_t scene_refresh_flag;
    int16_t ycc_to_rgb_coef[9];
    uint32_t ycc_to_rgb_offset[3];
    int16_t rgb_to_lms_coef[9];
} DoviVdrDmData;

typedef const DoviRpuDataHeader *(*dovi_rpu_get_header_fn)(const DoviRpuOpaque *);
typedef void (*dovi_rpu_free_header_fn)(const DoviRpuDataHeader *);
typedef const DoviRpuDataMapping *(*dovi_rpu_get_data_mapping_fn)(const DoviRpuOpaque *);
typedef void (*dovi_rpu_free_data_mapping_fn)(const DoviRpuDataMapping *);
typedef const DoviVdrDmData *(*dovi_rpu_get_vdr_dm_data_fn)(const DoviRpuOpaque *);
typedef void (*dovi_rpu_free_vdr_dm_data_fn)(const DoviVdrDmData *);

static dovi_rpu_get_header_fn p_get_header;
static dovi_rpu_free_header_fn p_free_header;
static dovi_rpu_get_data_mapping_fn p_get_mapping;
static dovi_rpu_free_data_mapping_fn p_free_mapping;
static dovi_rpu_get_vdr_dm_data_fn p_get_dm;
static dovi_rpu_free_vdr_dm_data_fn p_free_dm;

static int resolved;
static int status = DOVI_STATUS_UNRESOLVED;

// Profile 8.1 compatible output where luma and chroma mapping become no-ops,
// which is what a profile 7 stream needs once its enhancement layer is gone.
// libdovi numbers these modes independently of its own enum, so 2 is the
// value to pass here and the neighbouring numbers mean other conversions.
#define DOVI_CONVERT_MODE_P81 2

static void resolve_libdovi(void) {
    if (resolved) {
        return;
    }
    resolved = 1;
    void *handle = dlopen("libdovi.so", RTLD_NOW | RTLD_LOCAL);
    if (handle == NULL) {
        status = DOVI_STATUS_NOT_FOUND;
        return;
    }
    p_parse = (dovi_parse_unspec62_nalu_fn)dlsym(handle, "dovi_parse_unspec62_nalu");
    p_convert = (dovi_convert_rpu_with_mode_fn)dlsym(handle, "dovi_convert_rpu_with_mode");
    p_write = (dovi_write_unspec62_nalu_fn)dlsym(handle, "dovi_write_unspec62_nalu");
    p_data_free = (dovi_data_free_fn)dlsym(handle, "dovi_data_free");
    p_rpu_free = (dovi_rpu_free_fn)dlsym(handle, "dovi_rpu_free");
    if (p_parse == NULL || p_convert == NULL || p_write == NULL ||
        p_data_free == NULL || p_rpu_free == NULL) {
        p_parse = NULL;
        p_convert = NULL;
        p_write = NULL;
        p_data_free = NULL;
        p_rpu_free = NULL;
        status = DOVI_STATUS_MISSING_SYMBOLS;
        return;
    }
    status = DOVI_STATUS_READY;

    p_get_header = (dovi_rpu_get_header_fn)dlsym(handle, "dovi_rpu_get_header");
    p_free_header = (dovi_rpu_free_header_fn)dlsym(handle, "dovi_rpu_free_header");
    p_get_mapping = (dovi_rpu_get_data_mapping_fn)dlsym(handle, "dovi_rpu_get_data_mapping");
    p_free_mapping = (dovi_rpu_free_data_mapping_fn)dlsym(handle, "dovi_rpu_free_data_mapping");
    p_get_dm = (dovi_rpu_get_vdr_dm_data_fn)dlsym(handle, "dovi_rpu_get_vdr_dm_data");
    p_free_dm = (dovi_rpu_free_vdr_dm_data_fn)dlsym(handle, "dovi_rpu_free_vdr_dm_data");
}

JNIEXPORT jint JNICALL
Java_org_moonfin_nativevideo_DoviRpu_nativeStatus(JNIEnv *env, jobject thiz) {
    (void)env;
    (void)thiz;
    resolve_libdovi();
    return (jint)status;
}

JNIEXPORT jbyteArray JNICALL
Java_org_moonfin_nativevideo_DoviRpu_nativeConvert(
    JNIEnv *env,
    jobject thiz,
    jbyteArray data,
    jint offset,
    jint length) {
    (void)thiz;
    resolve_libdovi();
    if (p_parse == NULL || length <= 0) {
        return NULL;
    }

    jbyte *bytes = (*env)->GetByteArrayElements(env, data, NULL);
    if (bytes == NULL) {
        return NULL;
    }

    jbyteArray result = NULL;
    DoviRpuOpaque *rpu = p_parse((const uint8_t *)bytes + offset, (size_t)length);
    (*env)->ReleaseByteArrayElements(env, data, bytes, JNI_ABORT);
    if (rpu == NULL) {
        return NULL;
    }

    if (p_convert(rpu, DOVI_CONVERT_MODE_P81) == 0) {
        const DoviData *written = p_write(rpu);
        if (written != NULL) {
            result = (*env)->NewByteArray(env, (jsize)written->len);
            if (result != NULL) {
                (*env)->SetByteArrayRegion(
                    env, result, 0, (jsize)written->len, (const jbyte *)written->data);
            }
            p_data_free(written);
        }
    }
    p_rpu_free(rpu);
    return result;
}

// Fills out with what the profile 5 shader needs, laid out as DoviP5Renderer.kt
// reads it, and returns a bit set of what the RPU carried:
// 1 for a reshaping mapping, 2 for color matrices, or -1 when it did not parse.
//   [0..8]  ycc_to_rgb, [9..11] ycc_to_rgb offset, [12..20] rgb_to_lms
//   then for each of 3 components, starting at 21 + c * 210:
//     [0] piece count, [1..9] pivots,
//     [10 + p * 4]  piece head: is_mmr, then poly c0 c1 c2, or constant, order
//     [42 + p * 21] mmr coefficients, 7 per order
#define P5_MAX_PIECES 8
#define P5_MAX_MMR_ORDER 3
#define P5_COMPONENT_BLOCK 210
#define P5_OUT_LENGTH (21 + 3 * P5_COMPONENT_BLOCK)

static void write_p5_curve(const DoviReshapingCurve *curve, float scale, float pivot_scale, float *out) {
    size_t pieces = curve->pivots.len > 0 ? curve->pivots.len - 1 : 0;
    if (pieces > P5_MAX_PIECES) {
        pieces = P5_MAX_PIECES;
    }
    out[0] = (float)pieces;
    // The RPU codes each pivot as the step from the one before it.
    uint32_t pivot = 0;
    for (size_t i = 0; i <= pieces; i++) {
        pivot += curve->pivots.data[i];
        out[1 + i] = (float)pivot * pivot_scale;
    }
    for (size_t p = 0; p < pieces; p++) {
        float *head = out + 10 + p * 4;
        float *mmr = out + 42 + p * 21;
        if (curve->mapping_idc == 0 && curve->polynomial != NULL) {
            const DoviPolynomialCurve *poly = curve->polynomial;
            const DoviI64Data *ints = poly->poly_coef_int.list[p];
            const DoviU64Data *fracs = poly->poly_coef.list[p];
            head[0] = 0.0f;
            for (size_t k = 0; k < ints->len && k < 3; k++) {
                head[1 + k] = (float)ints->data[k] + (float)fracs->data[k] * scale;
            }
        } else if (curve->mapping_idc == 1 && curve->mmr != NULL) {
            const DoviMMRCurve *m = curve->mmr;
            size_t order = (size_t)m->mmr_order_minus1.data[p] + 1;
            if (order > P5_MAX_MMR_ORDER) {
                order = P5_MAX_MMR_ORDER;
            }
            head[0] = 1.0f;
            head[1] = (float)m->mmr_constant_int.data[p] + (float)m->mmr_constant.data[p] * scale;
            head[2] = (float)order;
            const DoviI64Data2D *ints = m->mmr_coef_int.list[p];
            const DoviU64Data2D *fracs = m->mmr_coef.list[p];
            for (size_t j = 0; j < order; j++) {
                const DoviI64Data *row_int = ints->list[j];
                const DoviU64Data *row_frac = fracs->list[j];
                for (size_t k = 0; k < row_int->len && k < 7; k++) {
                    mmr[j * 7 + k] = (float)row_int->data[k] + (float)row_frac->data[k] * scale;
                }
            }
        }
    }
}

JNIEXPORT jint JNICALL
Java_org_moonfin_nativevideo_DoviRpu_nativeP5Mapping(
    JNIEnv *env,
    jobject thiz,
    jbyteArray data,
    jfloatArray out) {
    (void)thiz;
    resolve_libdovi();
    if (p_parse == NULL || p_get_header == NULL || p_get_mapping == NULL || p_get_dm == NULL ||
        p_free_header == NULL || p_free_mapping == NULL || p_free_dm == NULL) {
        return -1;
    }
    if ((*env)->GetArrayLength(env, out) < P5_OUT_LENGTH) {
        return -1;
    }

    jsize length = (*env)->GetArrayLength(env, data);
    jbyte *bytes = (*env)->GetByteArrayElements(env, data, NULL);
    if (bytes == NULL) {
        return -1;
    }
    DoviRpuOpaque *rpu = p_parse((const uint8_t *)bytes, (size_t)length);
    (*env)->ReleaseByteArrayElements(env, data, bytes, JNI_ABORT);
    if (rpu == NULL) {
        return -1;
    }

    jint flags = -1;
    const DoviRpuDataHeader *header = p_get_header(rpu);
    if (header != NULL) {
        flags = 0;
        jfloat *values = (*env)->GetFloatArrayElements(env, out, NULL);
        if (values != NULL) {
            const DoviRpuDataMapping *mapping = p_get_mapping(rpu);
            if (mapping != NULL) {
                float scale = 1.0f / (float)(1ULL << header->coefficient_log2_denom);
                float pivot_scale = 1.0f / (float)((1U << (header->bl_bit_depth_minus8 + 8)) - 1U);
                for (int c = 0; c < 3; c++) {
                    float *block = values + 21 + c * P5_COMPONENT_BLOCK;
                    for (int i = 0; i < P5_COMPONENT_BLOCK; i++) {
                        block[i] = 0.0f;
                    }
                    write_p5_curve(&mapping->curves[c], scale, pivot_scale, block);
                }
                p_free_mapping(mapping);
                flags |= 1;
            }
            const DoviVdrDmData *dm = p_get_dm(rpu);
            if (dm != NULL) {
                for (int i = 0; i < 9; i++) {
                    values[i] = (float)dm->ycc_to_rgb_coef[i] / 8192.0f;
                    values[12 + i] = (float)dm->rgb_to_lms_coef[i] / 16384.0f;
                }
                for (int i = 0; i < 3; i++) {
                    values[9 + i] = (float)dm->ycc_to_rgb_offset[i] / 268435456.0f;
                }
                p_free_dm(dm);
                flags |= 2;
            }
            (*env)->ReleaseFloatArrayElements(env, out, values, 0);
        }
        p_free_header(header);
    }
    p_rpu_free(rpu);
    return flags;
}
