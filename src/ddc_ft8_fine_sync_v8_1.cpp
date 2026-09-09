// ============================================================
// test_refine_ft8_delay_v8_1_1.cpp
//
// FAST FT8 DELAY REFINEMENT V8 - DUAL PLATFORM
// Linux / Ubuntu + ESP32-S3 Arduino / LittleFS
//
// Test WAV:
//   /test_real_ft8_D1500.wav
//   /test_real_ft8_D2000.wav
//   /test_real_ft8_D2500.wav
//
// Fs = 12000 Hz, FT8 symbol = 1920 samples = 160 ms
//
// V8:
//   coarse search +/- 100 ms, step 16 samples
//   local refinement +/- 16 samples, step 1 sample
//   complex sliding correlation
//
// IMPORTANT FIX:
//   incomplete final FT8 symbols are ignored.  This prevents
//   reading beyond the WAV buffer, especially for D2500 where
//   79*1920 + 30000 = 181680 > 180000 samples.
// ============================================================

#if defined(ARDUINO)
#include <Arduino.h>
#include <LittleFS.h>
#else
#include <chrono>
#include <fstream>
#endif

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#ifndef PI
#define PI 3.14159265358979323846f
#endif
#ifndef TWO_PI
#define TWO_PI 6.28318530717958647692f
#endif

static constexpr int SAMPLE_RATE = 12000;
static constexpr int NTONES = 79;
static constexpr int SPS = 1920;
static constexpr float TONE_SPACING = 6.25f;
static constexpr float TEST_FREQ = 1500.0f;

// ============================================================
// refine_ft8_delay_v8_1
// ============================================================

// ============================================================
// Detailed timing instrumentation
// ============================================================
struct RefineTiming
{
    uint64_t coarse_search_us = 0;
    uint64_t coarse_max_us = 0;
    uint64_t fine_search_us = 0;
    uint64_t fine_max_us = 0;
};

static inline uint64_t bench_now_us()
{
#if defined(ARDUINO)
    return (uint64_t)micros();
#else
    return (uint64_t)std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
#endif
}

static float refine_ft8_delay_v8_1(
    const float* samples,
    int num_samples,
    const uint8_t* tones,
    float delay0,
    float freq,
    int cand_to_print,
    RefineTiming* timing = nullptr)
{
    constexpr int COARSE_RADIUS = 1200;   // +/- 100 ms
    constexpr int COARSE_STEP = 16;
    constexpr int MAX_COARSE = 2 * COARSE_RADIUS / COARSE_STEP + 1;
    constexpr int FINE_RADIUS = 16;

    static_assert(MAX_COARSE <= 151, "MAX_COARSE too small");
    (void)cand_to_print;
    if (timing) *timing = RefineTiming{};

    if (!samples || !tones || num_samples <= 0)
        return delay0;

    const int center_delay =
        (int)lroundf(delay0 * (float)SAMPLE_RATE);

    const int first_delay =
        std::max(0, center_delay - COARSE_RADIUS);

    const int last_delay =
        std::min(num_samples - 1,
                 center_delay + COARSE_RADIUS);

    if (last_delay < first_delay)
        return delay0;

    const int ncoarse =
        (last_delay - first_delay) / COARSE_STEP + 1;

    float score_total[MAX_COARSE] = {};
    int used_count[MAX_COARSE] = {};

    // Correlate exactly one complete 1920-sample FT8 symbol.
    // The caller guarantees start + SPS <= num_samples.
    auto initial_correlation =
        [&](int start, float cd, float sd,
            float& ci, float& cq)
    {
        ci = 0.0f;
        cq = 0.0f;

        float c = 1.0f;
        float s = 0.0f;
        int n = 0;

        for (; n + 3 < SPS; n += 4)
        {
            const float x0 = samples[start + n];
            const float x1 = samples[start + n + 1];
            const float x2 = samples[start + n + 2];
            const float x3 = samples[start + n + 3];

            ci += x0 * c;
            cq -= x0 * s;

            const float c1 = c * cd - s * sd;
            const float s1 = s * cd + c * sd;
            ci += x1 * c1;
            cq -= x1 * s1;

            const float c2 = c1 * cd - s1 * sd;
            const float s2 = s1 * cd + c1 * sd;
            ci += x2 * c2;
            cq -= x2 * s2;

            const float c3 = c2 * cd - s2 * sd;
            const float s3 = s2 * cd + c2 * sd;
            ci += x3 * c3;
            cq -= x3 * s3;

            c = c3 * cd - s3 * sd;
            s = s3 * cd + c3 * sd;
        }

        for (; n < SPS; ++n)
        {
            const float x = samples[start + n];
            ci += x * c;
            cq -= x * s;

            const float nc = c * cd - s * sd;
            const float ns = s * cd + c * sd;
            c = nc;
            s = ns;
        }
    };

    // --------------------------------------------------------
    // COARSE SEARCH
    // --------------------------------------------------------
    const uint64_t t_coarse0 = bench_now_us();
    for (int k = 0; k < NTONES; ++k)
    {
        const float f = freq + 6.25f * (float)tones[k];
        const float w = TWO_PI * f / (float)SAMPLE_RATE;

        const float cd = cosf(w);
        const float sd = sinf(w);
        const float pc = cd;
        const float ps = sd;

        const float end_angle = -w * (float)SPS;
        const float ec = cosf(end_angle);
        const float es = sinf(end_angle);

        const int first_start =
            first_delay + k * SPS;

        // FIX: incomplete symbols are not correlated.
        if (first_start < 0 ||
            first_start + SPS > num_samples)
            continue;

        float ci, cq;
        initial_correlation(first_start, cd, sd, ci, cq);

        int pos = first_start;

        // Only candidates whose entire 1920-sample window fits
        // in the WAV are considered.
        const int max_d_for_tone =
            std::min(
                ncoarse - 1,
                (num_samples - first_start - SPS) / COARSE_STEP);

        for (int d = 0; d <= max_d_for_tone; ++d)
        {
            score_total[d] += ci * ci + cq * cq;
            used_count[d]++;

            if (d == max_d_for_tone)
                break;

            for (int step = 0; step < COARSE_STEP; ++step)
            {
                const int old_pos = pos + step;
                const int new_sample = old_pos + SPS;

                // Defensive bounds; normally guaranteed by max_d_for_tone.
                if (old_pos < 0 || new_sample >= num_samples)
                    break;

                const float x_old = samples[old_pos];
                const float x_new = samples[new_sample];

                const float ti = ci - x_old + x_new * ec;
                const float tq = cq + x_new * es;

                ci = ti * pc - tq * ps;
                cq = ti * ps + tq * pc;
            }

            pos += COARSE_STEP;
        }
    }

    if (timing) timing->coarse_search_us = bench_now_us() - t_coarse0;

    // --------------------------------------------------------
    // Find coarse maximum
    // --------------------------------------------------------
    const uint64_t t_coarse_max0 = bench_now_us();
    int best_coarse_index = 0;
    float best_coarse_score = -1.0f;

    for (int d = 0; d < ncoarse; ++d)
    {
        if (used_count[d] == 0)
            continue;

        const float score =
            score_total[d] / (float)used_count[d];

        if (score > best_coarse_score)
        {
            best_coarse_score = score;
            best_coarse_index = d;
        }
    }

    const int coarse_delay =
        first_delay + best_coarse_index * COARSE_STEP;

    if (timing) timing->coarse_max_us = bench_now_us() - t_coarse_max0;

    // --------------------------------------------------------
    // LOCAL REFINEMENT SEARCH
    // --------------------------------------------------------
    const uint64_t t_fine0 = bench_now_us();
    const int fine_first =
        std::max(first_delay, coarse_delay - FINE_RADIUS);

    const int fine_last =
        std::min(last_delay, coarse_delay + FINE_RADIUS);

    const int nfine = fine_last - fine_first + 1;

    float accumulated_fine[33] = {};

    for (int k = 0; k < NTONES; ++k)
    {
        const float f = freq + 6.25f * (float)tones[k];
        const float w = TWO_PI * f / (float)SAMPLE_RATE;

        const float cd = cosf(w);
        const float sd = sinf(w);
        const float pc = cd;
        const float ps = sd;

        const float end_angle = -w * (float)SPS;
        const float ec = cosf(end_angle);
        const float es = sinf(end_angle);

        const int pos = fine_first + k * SPS;

        // FIX: ignore a partial final symbol.
        if (pos < 0 || pos + SPS > num_samples)
            continue;

        float ci, cq;
        initial_correlation(pos, cd, sd, ci, cq);

        const int max_d_for_tone =
            std::min(
                nfine - 1,
                num_samples - pos - SPS);

        for (int d = 0; d <= max_d_for_tone; ++d)
        {
            accumulated_fine[d] += ci * ci + cq * cq;

            if (d == max_d_for_tone)
                break;

            const int old_pos = pos + d;
            const int new_sample = old_pos + SPS;

            if (old_pos < 0 || new_sample >= num_samples)
                break;

            const float x_old = samples[old_pos];
            const float x_new = samples[new_sample];

            const float ti = ci - x_old + x_new * ec;
            const float tq = cq + x_new * es;

            ci = ti * pc - tq * ps;
            cq = ti * ps + tq * pc;
        }
    }

    if (timing) timing->fine_search_us = bench_now_us() - t_fine0;

    // --------------------------------------------------------
    // Find fine maximum
    // --------------------------------------------------------
    const uint64_t t_fine_max0 = bench_now_us();
    float fine_best_score = -1.0f;
    int fine_best_delay = coarse_delay;

    for (int d = 0; d < nfine; ++d)
    {
        if (accumulated_fine[d] > fine_best_score)
        {
            fine_best_score = accumulated_fine[d];
            fine_best_delay = fine_first + d;
        }
    }

    if (timing) timing->fine_max_us = bench_now_us() - t_fine_max0;

    return fine_best_delay / (float)SAMPLE_RATE;
}

// ============================================================
// WAV LOADER - ESP32 / LittleFS
// ============================================================
#if defined(ARDUINO)

static float* load_wav(const char* path,
                       int* out_num_samples,
                       int* out_num_channels,
                       int* out_sample_rate)
{
    File f = LittleFS.open(path, "r");
    if (!f) {
        Serial.printf("ERROR: cannot open %s\n", path);
        return nullptr;
    }

    uint8_t riff[12];
    if (f.read(riff, 12) != 12) {
        Serial.println("ERROR: WAV header too short");
        f.close(); return nullptr;
    }
    if (memcmp(riff, "RIFF", 4) != 0 || memcmp(riff + 8, "WAVE", 4) != 0) {
        Serial.println("ERROR: not a RIFF/WAVE file");
        f.close(); return nullptr;
    }

    uint16_t audio_format = 0, num_channels = 0, bits_per_sample = 0;
    uint32_t sample_rate = 0, data_pos = 0, data_size = 0;
    bool have_fmt = false, have_data = false;

    while (f.available())
    {
        uint8_t h[8];
        if (f.read(h, 8) != 8) break;

        const uint32_t chunk_size =
            h[4] | ((uint32_t)h[5] << 8) |
            ((uint32_t)h[6] << 16) | ((uint32_t)h[7] << 24);
        const uint32_t chunk_start = f.position();

        if (memcmp(h, "fmt ", 4) == 0)
        {
            if (chunk_size < 16) { f.close(); return nullptr; }
            uint8_t fmt[16];
            if (f.read(fmt, 16) != 16) { f.close(); return nullptr; }
            audio_format = fmt[0] | ((uint16_t)fmt[1] << 8);
            num_channels = fmt[2] | ((uint16_t)fmt[3] << 8);
            sample_rate = fmt[4] | ((uint32_t)fmt[5] << 8) |
                          ((uint32_t)fmt[6] << 16) | ((uint32_t)fmt[7] << 24);
            bits_per_sample = fmt[14] | ((uint16_t)fmt[15] << 8);
            have_fmt = true;
        }
        else if (memcmp(h, "data", 4) == 0)
        {
            data_pos = f.position();
            data_size = chunk_size;
            have_data = true;
        }

        f.seek(chunk_start + chunk_size);
        if (chunk_size & 1) f.seek(f.position() + 1);
        if (have_fmt && have_data) break;
    }

    if (!have_fmt || !have_data) { f.close(); return nullptr; }
    if (audio_format != 1 || bits_per_sample != 16 || num_channels < 1) {
        f.close(); return nullptr;
    }

    const int bytes_per_sample = 2;
    const int frame_bytes = bytes_per_sample * num_channels;
    const int num_samples = data_size / frame_bytes;

    float* samples = (float*)malloc((size_t)num_samples * sizeof(float));
    if (!samples) { f.close(); return nullptr; }

    f.seek(data_pos);
    for (int n = 0; n < num_samples; ++n)
    {
        uint8_t b[2];
        if (f.read(b, 2) != 2) {
            free(samples); f.close(); return nullptr;
        }
        const int16_t v = (int16_t)(b[0] | ((uint16_t)b[1] << 8));
        samples[n] = (float)v / 32768.0f;
        for (int ch = 1; ch < num_channels; ++ch)
            f.seek(f.position() + bytes_per_sample);
    }

    f.close();
    *out_num_samples = num_samples;
    *out_num_channels = num_channels;
    *out_sample_rate = (int)sample_rate;
    return samples;
}

// ============================================================
// WAV LOADER - Linux / Ubuntu
// ============================================================
#else

static uint16_t rd_u16(const uint8_t* p)
{
    return (uint16_t)p[0] | ((uint16_t)p[1] << 8);
}

static uint32_t rd_u32(const uint8_t* p)
{
    return (uint32_t)p[0] | ((uint32_t)p[1] << 8) |
           ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
}

static float* load_wav(const char* path,
                       int* out_num_samples,
                       int* out_num_channels,
                       int* out_sample_rate)
{
    std::ifstream f(path, std::ios::binary);
    if (!f && path[0] == '/') {
        f.clear();
        f.open(path + 1, std::ios::binary);
    }
    if (!f) {
        printf("ERROR: cannot open %s\n", path);
        return nullptr;
    }

    uint8_t riff[12];
    if (!f.read((char*)riff, 12)) return nullptr;
    if (memcmp(riff, "RIFF", 4) != 0 || memcmp(riff + 8, "WAVE", 4) != 0)
        return nullptr;

    uint16_t audio_format = 0, num_channels = 0;
    uint16_t bits_per_sample = 0, block_align = 0;
    uint32_t sample_rate = 0, data_size = 0;
    std::streamoff data_pos = 0;
    bool have_fmt = false, have_data = false;

    while (f)
    {
        uint8_t h[8];
        if (!f.read((char*)h, 8)) break;

        const uint32_t chunk_size = rd_u32(h + 4);
        const std::streamoff chunk_start = f.tellg();

        if (memcmp(h, "fmt ", 4) == 0)
        {
            if (chunk_size < 16) return nullptr;
            std::vector<uint8_t> fmt(chunk_size);
            if (!f.read((char*)fmt.data(), chunk_size)) return nullptr;
            audio_format = rd_u16(fmt.data());
            num_channels = rd_u16(fmt.data() + 2);
            sample_rate = rd_u32(fmt.data() + 4);
            block_align = rd_u16(fmt.data() + 12);
            bits_per_sample = rd_u16(fmt.data() + 14);
            have_fmt = true;
        }
        else if (memcmp(h, "data", 4) == 0)
        {
            data_pos = chunk_start;
            data_size = chunk_size;
            have_data = true;
        }
        else
        {
            f.seekg((std::streamoff)chunk_size, std::ios::cur);
        }

        if (chunk_size & 1) f.seekg(1, std::ios::cur);
        if (have_fmt && have_data) break;
    }

    if (!have_fmt || !have_data) return nullptr;
    if (audio_format != 1 || bits_per_sample != 16 || num_channels < 1)
        return nullptr;
    if (block_align == 0) block_align = (uint16_t)(num_channels * 2);

    const int num_samples = (int)(data_size / block_align);
    if (num_samples <= 0) return nullptr;

    float* samples = (float*)malloc((size_t)num_samples * sizeof(float));
    if (!samples) return nullptr;

    f.clear();
    f.seekg(data_pos);
    for (int n = 0; n < num_samples; ++n)
    {
        uint8_t b[2];
        if (!f.read((char*)b, 2)) {
            free(samples); return nullptr;
        }
        const int16_t v = (int16_t)(b[0] | ((uint16_t)b[1] << 8));
        samples[n] = (float)v / 32768.0f;
        const std::streamoff skip = (std::streamoff)block_align - 2;
        if (skip > 0) f.seekg(skip, std::ios::cur);
    }

    *out_num_samples = num_samples;
    *out_num_channels = (int)num_channels;
    *out_sample_rate = (int)sample_rate;
    return samples;
}
#endif

// ============================================================
// Tone estimator
// ============================================================

static uint8_t estimate_tone(const float* samples,
                             int num_samples,
                             int start,
                             float freq)
{
    float best_power = -1.0f;
    int best_tone = 0;

    if (start < 0 || start >= num_samples)
        return 0;

    const int count = std::min(SPS, num_samples - start);
    if (count <= 0) return 0;

    for (int tone = 0; tone < 8; ++tone)
    {
        const float f = freq + tone * TONE_SPACING;
        const float w = TWO_PI * f / SAMPLE_RATE;
        const float c = cosf(w);
        const float s = sinf(w);
        float ci = 0.0f, cq = 0.0f;
        float phase_c = 1.0f, phase_s = 0.0f;

        for (int n = 0; n < count; ++n)
        {
            const float x = samples[start + n];
            ci += x * phase_c;
            cq -= x * phase_s;

            const float nc = phase_c * c - phase_s * s;
            const float ns = phase_s * c + phase_c * s;
            phase_c = nc;
            phase_s = ns;
        }

        const float power = ci * ci + cq * cq;
        if (power > best_power) {
            best_power = power;
            best_tone = tone;
        }
    }
    return (uint8_t)best_tone;
}

static bool extract_ft8_tones(const float* samples,
                              int num_samples,
                              float delay0,
                              float freq,
                              uint8_t tones[NTONES])
{
    const int start = (int)lroundf(delay0 * SAMPLE_RATE);
    if (start < 0 || start >= num_samples) {
        printf("ERROR: invalid FT8 start position\n");
        return false;
    }

    for (int k = 0; k < NTONES; ++k)
        tones[k] = estimate_tone(samples, num_samples,
                                  start + k * SPS, freq);
    return true;
}

static void print_tones(const uint8_t tones[NTONES])
{
    printf("Detected FT8 tones:");
    for (int k = 0; k < NTONES; ++k) {
        printf("%2d:%d ", k, tones[k]);
        if ((k % 13) == 12) printf("\n");
    }
    printf("\n");
}

// ============================================================
// One test
// ============================================================

static void run_one_test(const char* filename, float true_delay)
{
    printf("\n============================================================\n");
    printf("FILE: %s\n", filename);
    printf("TRUE DELAY: %.6f s\n", true_delay);
    printf("============================================================\n");

    int num_samples = 0, num_channels = 0, sample_rate = 0;

#if defined(ARDUINO)
    const uint32_t heap_before = ESP.getFreeHeap();
#else
    const uint32_t heap_before = 0;
#endif

    float* samples = load_wav(filename, &num_samples,
                              &num_channels, &sample_rate);

#if defined(ARDUINO)
    const uint32_t heap_after_load = ESP.getFreeHeap();
#else
    const uint32_t heap_after_load = 0;
#endif

    if (!samples) {
        printf("LOAD FAILED\n");
        return;
    }

    printf("Samples          : %d\n", num_samples);
    printf("Duration         : %.3f s\n", (float)num_samples / sample_rate);
    printf("Channels         : %d\n", num_channels);
    printf("Sample rate      : %d Hz\n", sample_rate);
    printf("Heap before load : %u\n", heap_before);
    printf("Heap after load  : %u\n", heap_after_load);

    if (sample_rate != SAMPLE_RATE || num_channels != 1) {
        printf("ERROR: WAV format mismatch\n");
        free(samples);
        return;
    }

    uint8_t tones[NTONES];
    if (!extract_ft8_tones(samples, num_samples, true_delay,
                            TEST_FREQ, tones)) {
        free(samples);
        return;
    }

    print_tones(tones);
    printf("\nRunning refine_ft8_delay_v8_1()...\n");

#if defined(ARDUINO)
    const uint32_t t0 = micros();
#else
    const auto t0 = std::chrono::steady_clock::now();
#endif

    RefineTiming timing;
    const float estimated_delay =
        refine_ft8_delay_v8_1(samples, num_samples, tones,
                            true_delay, TEST_FREQ, 0, &timing);

#if defined(ARDUINO)
    const uint32_t elapsed_us = micros() - t0;
#else
    const auto elapsed_us =
        std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::steady_clock::now() - t0).count();
#endif

    printf("\nDetailed timing:\n");
    printf("  Coarse search : %llu us\n", (unsigned long long)timing.coarse_search_us);
    printf("  Coarse max    : %llu us\n", (unsigned long long)timing.coarse_max_us);
    printf("  Local refine  : %llu us\n", (unsigned long long)timing.fine_search_us);
    printf("  Fine max      : %llu us\n", (unsigned long long)timing.fine_max_us);

    const float error_s = estimated_delay - true_delay;
    const float error_ms = error_s * 1000.0f;
    const float error_samples = error_s * SAMPLE_RATE;

    printf("\n------------------------------------------------------------\n");
    printf("True delay       : %10.6f s\n", true_delay);
    printf("Estimated delay  : %10.6f s\n", estimated_delay);
    printf("Error            : %+10.6f s\n", error_s);
    printf("Error            : %+10.3f ms\n", error_ms);
    printf("Error            : %+10.3f samples\n", error_samples);
    printf("Execution time   : %llu us (%.3f ms)\n",
           (unsigned long long)elapsed_us,
           (double)elapsed_us / 1000.0);

#if defined(ESP32)
    printf("Free heap        : %u bytes\n", ESP.getFreeHeap());
#endif

    printf("------------------------------------------------------------\n");

    free(samples);

#if defined(ESP32)
    printf("After free() heap: %u bytes\n", ESP.getFreeHeap());
#endif

    printf("WAV released.\n");
}

// ============================================================
// Benchmark driver
// ============================================================

static void run_all_tests()
{
    printf("\r\n\r\n");
    printf("############################################################\n");
    printf(" FT8 refine_ft8_delay_v8_1 - REAL WAV BENCHMARK\n");
    printf("############################################################\n");

#if defined(ESP32)
    printf("Platform         : ESP32-S3 / Arduino\n");
    printf("CPU frequency    : %u MHz\n", ESP.getCpuFreqMHz());
#else
    printf("Platform         : Linux / Ubuntu\n");
#endif

    printf("Sample rate      : %d Hz\n", SAMPLE_RATE);
    printf("Symbol duration  : %d samples = %.3f ms\n",
           SPS, 1000.0f * SPS / SAMPLE_RATE);
    printf("Test frequency   : %.3f Hz\n", TEST_FREQ);

#if defined(ESP32)
    printf("Free heap        : %u bytes\n", ESP.getFreeHeap());
    printf("Free PSRAM       : %u bytes\n", ESP.getFreePsram());
    printf("\r\nMounting LittleFS...\n");
    if (!LittleFS.begin(true)) {
        printf("ERROR: LittleFS.begin() failed\n");
        return;
    }
    printf("LittleFS mounted.\n");
#endif

    run_one_test("/test_real_ft8_D1500.wav", 1.500f);
    run_one_test("/test_real_ft8_D2000.wav", 2.000f);
    run_one_test("/test_real_ft8_D2500.wav", 2.500f);

    printf("\r\n\r\n");
    printf("############################################################\n");
    printf("BENCHMARK COMPLETE\n");
#if defined(ESP32)
    printf("Final free heap : %u bytes\n", ESP.getFreeHeap());
    printf("Final free PSRAM: %u bytes\n", ESP.getFreePsram());
#endif
    printf("############################################################\n");
}

#if defined(ARDUINO)
void setup()
{
    Serial.begin(115200);
    delay(2000);
    run_all_tests();
}

void loop()
{
    delay(1000);
}
#else
int main()
{
    run_all_tests();
    return 0;
}
#endif
