/*
 * ============================================================
 * FT8 DDC V2
 * ============================================================
 *
 * 12 kHz real
 *      |
 *      | complex mixer @ 1450 Hz
 *      v
 *  FIR low-pass 80 Hz / 121 taps
 *      |
 *      | decimate by 60
 *      v
 *  200 Hz complex IQ
 *      |
 *      v
 *  FFT32 -> 160 ms / symbol
 *
 * V2 TEST:
 *
 *   180000 input samples
 *        ->
 *   3000 complex IQ samples
 *
 *   FFT32 every 32 IQ samples
 *
 *   Costas verification:
 *
 *       symbols 0..6
 *       symbols 36..42
 *       symbols 72..78
 *
 *   Expected residual frequency:
 *
 *       1500 - 1450 = 50 Hz
 *
 *   FFT bin:
 *
 *       50 / (200/32) = 8
 *
 *   Costas bins:
 *
 *       8 + {3,1,4,0,6,5,2}
 *       =
 *       {11,9,12,8,14,13,10}
 *
 * IMPORTANT:
 *
 *   The FIR delay is reported separately.
 *
 *   The FT8 verification windows remain referenced to the
 *   original signal start and the decimation grid:
 *
 *       2.000 ... 2.160 s
 *       7.760 ... 7.920 s
 *      13.520 ... 13.680 s
 *
 * ============================================================
 */

#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cmath>
#include <cstring>
#include <vector>
#include <algorithm>
#include <time.h>

#ifdef ARDUINO
#include <Arduino.h>
#endif


// ============================================================
// PARAMETERS
// ============================================================

static constexpr int INPUT_FS  = 12000;
static constexpr int OUTPUT_FS = 200;
static constexpr int DECIM     = 60;

static constexpr float LO_FREQ    = 1450.0f;
static constexpr float FIR_CUTOFF = 80.0f;
static constexpr int FIR_TAPS     = 121;

static constexpr int FFT_N = 32;

static constexpr double SIGNAL_START_SEC = 2.000;

static constexpr int SYMBOL_SAMPLES =
    OUTPUT_FS * 160 / 1000;       // 32

static constexpr int SIGNAL_START_INPUT =
    static_cast<int>(
        SIGNAL_START_SEC * INPUT_FS);

static constexpr int FIR_DELAY =
    (FIR_TAPS - 1) / 2;            // 60 input samples

/*
 * IMPORTANT:
 *
 * Timing of the FFT test is tied to the original signal
 * start and the decimation grid.
 *
 * 24000 / 60 = 400
 *
 * therefore:
 *
 * FFT #0 starts at 400 * 5 ms = 2.000 s
 */
static constexpr int SIGNAL_START_OUTPUT =
    SIGNAL_START_INPUT / DECIM;


// ============================================================
// COSTAS
// ============================================================

static constexpr int COSTAS_LEN = 7;

static constexpr int COSTAS[COSTAS_LEN] =
{
    3, 1, 4, 0, 6, 5, 2
};

static constexpr int COSTAS_START[3] =
{
    0, 36, 72
};


// ============================================================
// COMPLEX SAMPLE
// ============================================================

struct IQ
{
    float i;
    float q;
};


// ============================================================
// TIME
// ============================================================

static double elapsed_ms(
    const struct timespec *t0,
    const struct timespec *t1)
{
    long sec =
        t1->tv_sec - t0->tv_sec;

    long nsec =
        t1->tv_nsec - t0->tv_nsec;

    return
        static_cast<double>(sec) * 1000.0 +
        static_cast<double>(nsec) / 1000000.0;
}


// ============================================================
// FIR DESIGN
// ============================================================

static void design_fir(float *h)
{
    const double fc =
        static_cast<double>(FIR_CUTOFF) /
        static_cast<double>(INPUT_FS);

    const int M = FIR_TAPS - 1;
    const int mid = M / 2;

    double sum = 0.0;

    for (int n = 0; n < FIR_TAPS; ++n)
    {
        const int k = n - mid;

        double sinc;

        if (k == 0)
        {
            sinc = 2.0 * fc;
        }
        else
        {
            const double x =
                2.0 * M_PI *
                fc *
                static_cast<double>(k);

            sinc =
                std::sin(x) /
                (M_PI * static_cast<double>(k));
        }

        const double w =
            0.54 -
            0.46 *
            std::cos(
                2.0 * M_PI *
                static_cast<double>(n) /
                static_cast<double>(M));

        h[n] =
            static_cast<float>(
                sinc * w);

        sum += h[n];
    }

    for (int n = 0; n < FIR_TAPS; ++n)
    {
        h[n] =
            static_cast<float>(
                static_cast<double>(h[n]) /
                sum);
    }
}


// ============================================================
// FAST DDC V2
// ============================================================
//
// Only decimated output samples are calculated.
//
// 3000 output samples
// 121 FIR taps
//
// = 363000 FIR coefficients
//
// No complete mixed input buffer is generated.
//
// ============================================================

struct ComplexCoeff
{
    float re;
    float im;
};

void ddc_process_fast(
    const float *__restrict input,
    int num_samples,
    std::vector<IQ> &output)
{
    static ComplexCoeff hc[FIR_TAPS];
    static bool initialized = false;

    // ============================================================
    // PRECOMPUTE COMPLEX FIR COEFFICIENTS
    // ============================================================

    if (!initialized)
    {
        float h[FIR_TAPS];

        design_fir(h);

        const float omega =
            2.0f * static_cast<float>(M_PI) *
            LO_FREQ /
            static_cast<float>(INPUT_FS);

        const float cs = std::cos(omega);
        const float sn = std::sin(omega);

        float ph_re = 1.0f;
        float ph_im = 0.0f;

        for (int k = 0; k < FIR_TAPS; ++k)
        {
            hc[k].re = h[k] * ph_re;
            hc[k].im = h[k] * ph_im;

            const float nr =
                ph_re * cs - ph_im * sn;

            const float ni =
                ph_re * sn + ph_im * cs;

            ph_re = nr;
            ph_im = ni;
        }

        initialized = true;
    }

    // ============================================================
    // OUTPUT SIZE
    // ============================================================

    const int count =
        (num_samples + DECIM - 1) / DECIM;

    output.resize(count);

    int out_index = 0;

    // ============================================================
    // DECIMATED FIR
    // ============================================================

    for (int n = 0; n < num_samples; n += DECIM)
    {
        float acc_re = 0.0f;
        float acc_im = 0.0f;

        // --------------------------------------------------------
        // Startup section
        //
        // Only the first 60 output positions need boundary
        // checking. After that all 121 taps are valid.
        // --------------------------------------------------------

        if (n < FIR_TAPS)
        {
            int k = 0;

            for (; k + 3 < FIR_TAPS && k + 3 <= n; k += 4)
            {
                const float s0 = input[n - k];
                const float s1 = input[n - k - 1];
                const float s2 = input[n - k - 2];
                const float s3 = input[n - k - 3];

                acc_re +=
                    s0 * hc[k].re +
                    s1 * hc[k + 1].re +
                    s2 * hc[k + 2].re +
                    s3 * hc[k + 3].re;

                acc_im +=
                    s0 * hc[k].im +
                    s1 * hc[k + 1].im +
                    s2 * hc[k + 2].im +
                    s3 * hc[k + 3].im;
            }

            for (; k <= n && k < FIR_TAPS; ++k)
            {
                const float s = input[n - k];

                acc_re += s * hc[k].re;
                acc_im += s * hc[k].im;
            }
        }
        else
        {
            // ----------------------------------------------------
            // FULL FIR
            //
            // No boundary test.
            // 121 taps = 30 groups of 4 + 1 tap.
            // ----------------------------------------------------

            int k = 0;

            for (; k < 120; k += 4)
            {
                const float s0 = input[n - k];
                const float s1 = input[n - k - 1];
                const float s2 = input[n - k - 2];
                const float s3 = input[n - k - 3];

                acc_re +=
                    s0 * hc[k].re +
                    s1 * hc[k + 1].re +
                    s2 * hc[k + 2].re +
                    s3 * hc[k + 3].re;

                acc_im +=
                    s0 * hc[k].im +
                    s1 * hc[k + 1].im +
                    s2 * hc[k + 2].im +
                    s3 * hc[k + 3].im;
            }

            // Tap 120
            const float s = input[n - 120];

            acc_re += s * hc[120].re;
            acc_im += s * hc[120].im;
        }

        // ========================================================
        // FINAL ROTATION
        //
        // LO = 1450 Hz
        // Fs = 12000 Hz
        // DECIM = 60
        //
        // Phase increment between output samples:
        //
        // -2*pi*1450*60/12000 = -pi/2 (mod 2*pi)
        //
        // Therefore:
        //
        // 0 ->  1
        // 1 -> -j
        // 2 -> -1
        // 3 -> +j
        // ========================================================

        switch (out_index & 3)
        {
            case 0:
                output[out_index].i = acc_re;
                output[out_index].q = acc_im;
                break;

            case 1:
                output[out_index].i = acc_im;
                output[out_index].q = -acc_re;
                break;

            case 2:
                output[out_index].i = -acc_re;
                output[out_index].q = -acc_im;
                break;

            default:
                output[out_index].i = -acc_im;
                output[out_index].q = acc_re;
                break;
        }

        ++out_index;
    }
}


// ============================================================
// FINE SYNC ESTIMATION (frequency + timing)
// ============================================================
//
// Once an FT8 message has already been decoded, its coarse
// residual frequency (base_bin) and coarse symbol timing
// (signal_start_output) are known from the decoder. This
// estimator refines both, using only the 21 Costas sync
// symbols, so the decoded signal can later be regenerated and
// subtracted (in the time domain) from the raw IQ stream.
//
// Frequency:
//
//   A direct DFT (single-bin Goertzel-style) value is evaluated
//   at the *expected* Costas bin for each of the 21 sync
//   symbols. Consecutive Costas symbols are exactly FFT_N output
//   samples apart, so any residual (sub-bin) frequency offset
//   appears as a constant phase rotation between consecutive
//   symbols:
//
//       phase_step = 2*pi * df * (FFT_N / OUTPUT_FS)
//
//   df is recovered with a standard phase-difference ("Kay")
//   estimator: sum X[s] * conj(X[s-1]) over every consecutive
//   in-block symbol pair, then take the angle of the resulting
//   complex sum.
//
// Timing:
//
//   Using the corrected frequency, the total sync energy
//   (sum of |X|^2 at the expected bins, de-rotated by the fine
//   frequency estimate) is evaluated at the coarse start and at
//   +-1 output sample. A parabolic interpolation across these
//   three energy values gives the sub-sample timing offset that
//   maximizes coherence, i.e. the true message start time.
//
// ============================================================

struct SyncEstimate
{
    float freq_offset_hz;    // fine correction, added to base_bin frequency
    float time_offset_samp;  // fine correction, added to signal_start_output (output samples)
};

// Direct DFT value at one exact (non-searched) bin.
static IQ dft_bin(
    const std::vector<IQ> &x,
    int start,
    int bin)
{
    float re = 0.0f;
    float im = 0.0f;

    for (int n = 0; n < FFT_N; ++n)
    {
        const float angle =
            -2.0f * static_cast<float>(M_PI) *
            static_cast<float>(bin * n) /
            static_cast<float>(FFT_N);

        const float c = std::cos(angle);
        const float s = std::sin(angle);

        re += x[start + n].i * c - x[start + n].q * s;
        im += x[start + n].i * s + x[start + n].q * c;
    }

    return IQ{re, im};
}

// Sum of |X|^2 at the expected Costas bins, for a given
// output-sample start shift, after de-rotating each symbol's
// DFT value by the candidate fine frequency correction.
static float costas_sync_energy(
    const std::vector<IQ> &x,
    int signal_start_output,
    int base_bin,
    const int *costas,
    int costas_len,
    const int *costas_start,
    int costas_arrays,
    float freq_offset_hz,
    int start_shift)
{
    float energy = 0.0f;

    for (int array = 0; array < costas_arrays; ++array)
    {
        const int first = costas_start[array];

        for (int s = 0; s < costas_len; ++s)
        {
            const int symbol = first + s;

            const int fft_start =
                signal_start_output + start_shift +
                symbol * FFT_N;

            if (fft_start < 0 ||
                fft_start + FFT_N > static_cast<int>(x.size()))
                continue;

            const int bin = base_bin + costas[s];

            const IQ X = dft_bin(x, fft_start, bin);

            const float t =
                static_cast<float>(fft_start) /
                static_cast<float>(OUTPUT_FS);

            const float angle =
                -2.0f * static_cast<float>(M_PI) *
                freq_offset_hz * t;

            const float c = std::cos(angle);
            const float sn = std::sin(angle);

            const float re = X.i * c - X.q * sn;
            const float im = X.i * sn + X.q * c;

            energy += re * re + im * im;
        }
    }

    return energy;
}

// ------------------------------------------------------------
// Public entry point.
//
// costas[costas_len]        : Costas bin offsets (e.g. {3,1,4,0,6,5,2})
// costas_start[costas_arrays]: first symbol index of each Costas array
// ------------------------------------------------------------

void ddc_estimate_fine_sync(
    const std::vector<IQ> &output,
    int signal_start_output,
    int base_bin,
    const int *costas,
    int costas_len,
    const int *costas_start,
    int costas_arrays,
    float *out_freq_offset_hz,
    float *out_time_offset_samples)
{
    // --------------------------------------------------------
    // 0) Coarse timing search.
    //
    // The caller's signal_start_output may be off by more than
    // one output sample (e.g. tens of ms), so first do an
    // integer-sample grid search for the shift that maximizes
    // Costas sync energy (frequency assumed 0 for this step;
    // a few Hz of residual error only mildly attenuates the
    // energy metric and does not bias the peak location).
    // --------------------------------------------------------

    static constexpr int COARSE_SEARCH_RANGE = 16;  // +-80 ms @ 200 Hz

    int best_shift = 0;
    float best_energy = -1.0f;

    for (int shift = -COARSE_SEARCH_RANGE;
         shift <= COARSE_SEARCH_RANGE;
         ++shift)
    {
        const float e =
            costas_sync_energy(
                output, signal_start_output, base_bin,
                costas, costas_len, costas_start, costas_arrays,
                0.0f, shift);

        if (e > best_energy)
        {
            best_energy = e;
            best_shift = shift;
        }
    }

    const int aligned_start =
        signal_start_output + best_shift;

    // --------------------------------------------------------
    // 1) Fine frequency: averaged phase difference between
    //    consecutive Costas symbols at their expected bins,
    //    evaluated at the coarse-aligned start.
    // --------------------------------------------------------

    float sum_re = 0.0f;
    float sum_im = 0.0f;

    for (int array = 0; array < costas_arrays; ++array)
    {
        const int first = costas_start[array];

        IQ prev{0.0f, 0.0f};
        bool have_prev = false;

        for (int s = 0; s < costas_len; ++s)
        {
            const int symbol = first + s;

            const int fft_start =
                aligned_start + symbol * FFT_N;

            if (fft_start < 0 ||
                fft_start + FFT_N > static_cast<int>(output.size()))
            {
                have_prev = false;
                continue;
            }

            const int bin = base_bin + costas[s];

            const IQ X = dft_bin(output, fft_start, bin);

            if (have_prev)
            {
                // X * conj(prev)
                sum_re += X.i * prev.i + X.q * prev.q;
                sum_im += X.q * prev.i - X.i * prev.q;
            }

            prev = X;
            have_prev = true;
        }
    }

    const float phase_step =
        std::atan2(sum_im, sum_re);

    const float symbol_period =
        static_cast<float>(FFT_N) /
        static_cast<float>(OUTPUT_FS);

    const float freq_offset_hz =
        phase_step /
        (2.0f * static_cast<float>(M_PI) * symbol_period);

    // --------------------------------------------------------
    // 2) Fine timing: parabolic interpolation of sync energy
    //    around the coarse-aligned (integer output-sample)
    //    start, now using the recovered fine frequency.
    // --------------------------------------------------------

    const float e_minus =
        costas_sync_energy(
            output, aligned_start, base_bin,
            costas, costas_len, costas_start, costas_arrays,
            freq_offset_hz, -1);

    const float e_zero =
        costas_sync_energy(
            output, aligned_start, base_bin,
            costas, costas_len, costas_start, costas_arrays,
            freq_offset_hz, 0);

    const float e_plus =
        costas_sync_energy(
            output, aligned_start, base_bin,
            costas, costas_len, costas_start, costas_arrays,
            freq_offset_hz, +1);

    const float denom =
        e_minus - 2.0f * e_zero + e_plus;

    float subsample_offset = 0.0f;

    if (std::fabs(denom) > 1e-9f)
    {
        subsample_offset =
            0.5f * (e_minus - e_plus) / denom;

        // Parabolic fit is only valid within +-1 sample.
        if (subsample_offset > 1.0f)  subsample_offset = 1.0f;
        if (subsample_offset < -1.0f) subsample_offset = -1.0f;
    }

    *out_freq_offset_hz = freq_offset_hz;
    *out_time_offset_samples =
        static_cast<float>(best_shift) + subsample_offset;
}


// ============================================================
// WAV READER
// ============================================================

#ifndef ARDUINO

static uint16_t read_u16(FILE *f)
{
    uint8_t b[2];

    if (fread(b, 1, 2, f) != 2)
        return 0;

    return
        static_cast<uint16_t>(b[0]) |
        (static_cast<uint16_t>(b[1]) << 8);
}


static uint32_t read_u32(FILE *f)
{
    uint8_t b[4];

    if (fread(b, 1, 4, f) != 4)
        return 0;

    return
        static_cast<uint32_t>(b[0]) |
        (static_cast<uint32_t>(b[1]) << 8) |
        (static_cast<uint32_t>(b[2]) << 16) |
        (static_cast<uint32_t>(b[3]) << 24);
}


static bool read_wav(
    const char *filename,
    std::vector<float> &samples,
    int &sample_rate)
{
    FILE *f =
        std::fopen(filename, "rb");

    if (!f)
    {
        std::printf(
            "ERROR: cannot open %s\n",
            filename);

        return false;
    }


    char riff[4];
    char wave[4];


    if (fread(riff, 1, 4, f) != 4)
    {
        fclose(f);
        return false;
    }

    (void)read_u32(f);


    if (fread(wave, 1, 4, f) != 4)
    {
        fclose(f);
        return false;
    }


    if (std::memcmp(riff, "RIFF", 4) != 0 ||
        std::memcmp(wave, "WAVE", 4) != 0)
    {
        fclose(f);
        return false;
    }


    uint16_t audio_format = 0;
    uint16_t channels = 0;
    uint16_t bits = 0;

    std::vector<uint8_t> data;


    while (!feof(f))
    {
        char id[4];

        if (fread(id, 1, 4, f) != 4)
            break;


        const uint32_t size =
            read_u32(f);


        if (std::memcmp(id, "fmt ", 4) == 0)
        {
            audio_format =
                read_u16(f);

            channels =
                read_u16(f);

            sample_rate =
                static_cast<int>(
                    read_u32(f));

            (void)read_u32(f);
            (void)read_u16(f);

            bits =
                read_u16(f);


            if (size > 16)
            {
                std::fseek(
                    f,
                    static_cast<long>(size - 16),
                    SEEK_CUR);
            }
        }
        else if (std::memcmp(id, "data", 4) == 0)
        {
            data.resize(size);

            if (size != 0)
            {
                fread(
                    data.data(),
                    1,
                    size,
                    f);
            }

            break;
        }
        else
        {
            std::fseek(
                f,
                static_cast<long>(size),
                SEEK_CUR);
        }
    }


    fclose(f);


    if (audio_format != 1 ||
        channels != 1 ||
        bits != 16 ||
        sample_rate != INPUT_FS)
    {
        std::printf(
            "ERROR: WAV must be mono 16-bit %d Hz\n",
            INPUT_FS);

        return false;
    }


    const size_t nsamples =
        data.size() / 2;

    samples.resize(nsamples);


    for (size_t i = 0;
         i < nsamples;
         ++i)
    {
        int16_t v =
            static_cast<int16_t>(
                static_cast<uint16_t>(
                    data[2*i]) |
                (static_cast<uint16_t>(
                    data[2*i+1]) << 8));


        samples[i] =
            static_cast<float>(v) /
            32768.0f;
    }


    return true;
}


// ============================================================
// FFT32
// ============================================================

static int fft_peak_bin(
    const std::vector<IQ> &x,
    int start)
{
    float mag[FFT_N];


    for (int k = 0;
         k < FFT_N;
         ++k)
    {
        float re = 0.0f;
        float im = 0.0f;


        for (int n = 0;
             n < FFT_N;
             ++n)
        {
            const float angle =
                -2.0f *
                static_cast<float>(M_PI) *
                static_cast<float>(k*n) /
                static_cast<float>(FFT_N);


            const float c =
                std::cos(angle);

            const float s =
                std::sin(angle);


            re +=
                x[start+n].i * c -
                x[start+n].q * s;

            im +=
                x[start+n].i * s +
                x[start+n].q * c;
        }


        mag[k] =
            re*re + im*im;
    }


    int peak = 0;


    for (int k = 1;
         k < FFT_N;
         ++k)
    {
        if (mag[k] > mag[peak])
            peak = k;
    }


    return peak;
}


// ============================================================
// MAIN
// ============================================================

int main(int argc, char **argv)
{
    const char *filename =
        (argc > 1)
        ? argv[1]
        : "test_real_ft8_D2000.wav";


    std::printf(
        "\n############################################################\n");

    std::printf(
        " FT8 DDC V2 - COSTAS 21/21 TEST\n");

    std::printf(
        "############################################################\n");


    std::printf(
        "Input Fs            : %d Hz\n",
        INPUT_FS);

    std::printf(
        "LO                  : %.1f Hz\n",
        LO_FREQ);

    std::printf(
        "FIR                 : %d taps\n",
        FIR_TAPS);

    std::printf(
        "FIR cutoff          : %.1f Hz\n",
        FIR_CUTOFF);

    std::printf(
        "Decimation          : %d\n",
        DECIM);

    std::printf(
        "Output Fs           : %d Hz\n",
        OUTPUT_FS);

    std::printf(
        "IQ samples expected : 3000\n");

    std::printf(
        "FFT                  : %d complex samples\n",
        FFT_N);

    std::printf(
        "FFT duration        : %.3f ms\n",
        1000.0 *
        static_cast<double>(FFT_N) /
        OUTPUT_FS);


    // --------------------------------------------------------
    // WAV
    // --------------------------------------------------------

    std::vector<float> input;

    int fs = 0;


    if (!read_wav(
            filename,
            input,
            fs))
    {
        return 1;
    }


    std::printf(
        "\nInput samples       : %zu\n",
        input.size());

    std::printf(
        "Duration            : %.3f s\n",
        static_cast<double>(input.size()) /
        INPUT_FS);


    // --------------------------------------------------------
    // DDC benchmark
    // --------------------------------------------------------

    std::vector<IQ> output;


    struct timespec t0;
    struct timespec t1;


    clock_gettime(
        CLOCK_MONOTONIC,
        &t0);


    ddc_process_fast(
        input.data(),
        static_cast<int>(input.size()),
        output);


    clock_gettime(
        CLOCK_MONOTONIC,
        &t1);


    const double ddc_time_ms =
        elapsed_ms(
            &t0,
            &t1);


    std::printf(
        "\n------------------------------------------------------------\n");

    std::printf(
        " DDC BENCHMARK\n");

    std::printf(
        "------------------------------------------------------------\n");

    std::printf(
        "DDC + FIR + decimation : %.3f ms\n",
        ddc_time_ms);

    std::printf(
        "Complex output samples : %zu\n",
        output.size());


    // --------------------------------------------------------
    // Output count verification
    // --------------------------------------------------------

    const bool output_count_ok =
        output.size() == 3000;


    std::printf(
        "Expected output        : 3000\n");

    std::printf(
        "Output count           : %s\n",
        output_count_ok
            ? "OK"
            : "WRONG");


    // --------------------------------------------------------
    // Timing alignment
    // --------------------------------------------------------

    std::printf(
        "\n============================================================\n");

    std::printf(
        " TIMING ALIGNMENT\n");

    std::printf(
        "============================================================\n");


    std::printf(
        "Signal start           : %.3f s\n",
        SIGNAL_START_SEC);

    std::printf(
        "Input sample           : %d\n",
        SIGNAL_START_INPUT);

    std::printf(
        "FIR group delay        : %d samples = %.3f ms\n",
        FIR_DELAY,
        1000.0 *
        static_cast<double>(FIR_DELAY) /
        INPUT_FS);

    std::printf(
        "Signal output index    : %d\n",
        SIGNAL_START_OUTPUT);


    for (int array = 0;
         array < 3;
         ++array)
    {
        const int symbol =
            COSTAS_START[array];

        const int fft_start =
            SIGNAL_START_OUTPUT +
            symbol * FFT_N;


        const double t_start =
            static_cast<double>(fft_start) /
            OUTPUT_FS;

        const double t_end =
            static_cast<double>(
                fft_start + FFT_N) /
            OUTPUT_FS;


        std::printf(
            "FFT #%d   symbols %d..%d : %.3f ... %.3f s\n",
            symbol,
            symbol,
            symbol + 6,
            t_start,
            t_end);
    }


    // --------------------------------------------------------
    // Costas frequency
    // --------------------------------------------------------

    const float base_freq =
        1500.0f - LO_FREQ;


    const float bin_spacing =
        static_cast<float>(OUTPUT_FS) /
        FFT_N;


    const int base_bin =
        static_cast<int>(
            std::lround(
                base_freq /
                bin_spacing));


    std::printf(
        "\n============================================================\n");

    std::printf(
        " FT8 COSTAS VERIFICATION\n");

    std::printf(
        "============================================================\n");


    std::printf(
        "RF tone             : 1500.00 Hz\n");

    std::printf(
        "LO                  : %.2f Hz\n",
        LO_FREQ);

    std::printf(
        "Residual frequency  : %.2f Hz\n",
        base_freq);

    std::printf(
        "FFT bin spacing     : %.2f Hz\n",
        bin_spacing);

    std::printf(
        "Base FFT bin        : %d\n",
        base_bin);


    std::printf(
        "Costas pattern      : ");

    for (int i = 0;
         i < COSTAS_LEN;
         ++i)
    {
        std::printf(
            "%d%s",
            COSTAS[i],
            (i == COSTAS_LEN-1)
                ? "\n"
                : " ");
    }


    std::printf(
        "Expected bins       : ");

    for (int i = 0;
         i < COSTAS_LEN;
         ++i)
    {
        std::printf(
            "%d%s",
            base_bin + COSTAS[i],
            (i == COSTAS_LEN-1)
                ? "\n"
                : " ");
    }


    // --------------------------------------------------------
    // Test 21 Costas symbols
    // --------------------------------------------------------

    int correct = 0;
    int tested = 0;


    for (int array = 0;
         array < 3;
         ++array)
    {
        const int first =
            COSTAS_START[array];


        std::printf(
            "\n------------------------------------------------------------\n");

        std::printf(
            " COSTAS ARRAY %d   symbols %d..%d\n",
            array + 1,
            first,
            first + 6);

        std::printf(
            "------------------------------------------------------------\n");

        std::printf(
            " SYMBOL   TIME        EXPECTED       MEASURED      RESULT\n");


        for (int s = 0;
             s < COSTAS_LEN;
             ++s)
        {
            const int symbol =
                first + s;


            const int fft_start =
                SIGNAL_START_OUTPUT +
                symbol * FFT_N;


            const int expected =
                base_bin +
                COSTAS[s];


            if (fft_start + FFT_N >
                static_cast<int>(
                    output.size()))
            {
                std::printf(
                    "%5d   OUT OF RANGE\n",
                    symbol);

                continue;
            }


            const int peak =
                fft_peak_bin(
                    output,
                    fft_start);


            const bool ok =
                peak == expected;


            if (ok)
                ++correct;

            ++tested;


            const double t_start =
                static_cast<double>(
                    fft_start) /
                OUTPUT_FS;

            const double t_end =
                static_cast<double>(
                    fft_start + FFT_N) /
                OUTPUT_FS;


            std::printf(
                "%5d   %5.3f-%5.3f    "
                "%2d (%6.2f)    "
                "%2d (%6.2f)    %s\n",

                symbol,

                t_start,
                t_end,

                expected,
                expected *
                    static_cast<float>(
                        OUTPUT_FS) /
                    FFT_N,

                peak,
                peak *
                    static_cast<float>(
                        OUTPUT_FS) /
                    FFT_N,

                ok
                    ? "OK"
                    : "WRONG");
        }
    }


    // --------------------------------------------------------
    // Fine sync estimation
    //
    // Refines residual frequency and symbol timing using the
    // already-decoded Costas sync symbols. These values are
    // needed later to regenerate and subtract this signal from
    // the raw stream (time-domain cancellation).
    // --------------------------------------------------------

    float fine_freq_offset_hz = 0.0f;
    float fine_time_offset_samp = 0.0f;

    ddc_estimate_fine_sync(
        output,
        SIGNAL_START_OUTPUT,
        base_bin,
        COSTAS,
        COSTAS_LEN,
        COSTAS_START,
        3,
        &fine_freq_offset_hz,
        &fine_time_offset_samp);

    const float exact_freq_hz =
        LO_FREQ +
        base_bin * bin_spacing +
        fine_freq_offset_hz;

    const double exact_time_sec =
        (static_cast<double>(SIGNAL_START_OUTPUT) +
         fine_time_offset_samp) /
        OUTPUT_FS;

    std::printf(
        "\n============================================================\n");

    std::printf(
        " FINE SYNC ESTIMATE\n");

    std::printf(
        "============================================================\n");

    std::printf(
        "Coarse freq bin      : %d (%.2f Hz residual)\n",
        base_bin,
        base_bin * bin_spacing);

    std::printf(
        "Fine freq correction : %+.3f Hz\n",
        fine_freq_offset_hz);

    std::printf(
        "Exact carrier freq   : %.3f Hz\n",
        exact_freq_hz);

    std::printf(
        "Fine time correction : %+.3f output samples "
        "(%+.3f ms)\n",
        fine_time_offset_samp,
        1000.0 * fine_time_offset_samp / OUTPUT_FS);

    std::printf(
        "Exact message start  : %.6f s\n",
        exact_time_sec);


    // --------------------------------------------------------
    // Stress test: wrong coarse guess
    //
    // Feed the estimator a deliberately wrong coarse guess:
    //   coarse freq  = 1502 Hz   (true carrier is 1500 Hz)
    //   coarse delay = 2020 ms   (true start is   2000 ms)
    //
    // ddc_estimate_fine_sync() must correct these back to the
    // true values using only the 21 Costas sync symbols.
    // --------------------------------------------------------

    {
        const float coarse_freq_guess_hz = 1502.0f;
        const double coarse_time_guess_sec = 2.020;

        const int guess_base_bin =
            static_cast<int>(
                std::lround(
                    (coarse_freq_guess_hz - LO_FREQ) /
                    bin_spacing));

        const int guess_start_output =
            static_cast<int>(
                std::lround(
                    coarse_time_guess_sec * OUTPUT_FS));

        float stress_freq_offset_hz = 0.0f;
        float stress_time_offset_samp = 0.0f;

        ddc_estimate_fine_sync(
            output,
            guess_start_output,
            guess_base_bin,
            COSTAS,
            COSTAS_LEN,
            COSTAS_START,
            3,
            &stress_freq_offset_hz,
            &stress_time_offset_samp);

        const float stress_exact_freq_hz =
            LO_FREQ +
            guess_base_bin * bin_spacing +
            stress_freq_offset_hz;

        const double stress_exact_time_sec =
            (static_cast<double>(guess_start_output) +
             stress_time_offset_samp) /
            OUTPUT_FS;

        std::printf(
            "\n============================================================\n");

        std::printf(
            " STRESS TEST: WRONG COARSE GUESS\n");

        std::printf(
            "============================================================\n");

        std::printf(
            "Coarse freq guess    : %.1f Hz (bin %d)\n",
            coarse_freq_guess_hz,
            guess_base_bin);

        std::printf(
            "Coarse delay guess   : %.1f ms (output sample %d)\n",
            coarse_time_guess_sec * 1000.0,
            guess_start_output);

        std::printf(
            "Fine freq correction : %+.3f Hz\n",
            stress_freq_offset_hz);

        std::printf(
            "Fine time correction : %+.3f samples (%+.3f ms)\n",
            stress_time_offset_samp,
            1000.0 * stress_time_offset_samp / OUTPUT_FS);

        std::printf(
            "Recovered exact freq : %.3f Hz   expected ~1500.000 Hz\n",
            stress_exact_freq_hz);

        std::printf(
            "Recovered exact delay: %.3f ms   expected ~2000.000 ms\n",
            stress_exact_time_sec * 1000.0);

        const bool freq_ok =
            std::fabs(stress_exact_freq_hz - 1500.0f) < 0.5f;

        const bool time_ok =
            std::fabs(stress_exact_time_sec * 1000.0 - 2000.0) < 5.0;

        std::printf(
            "Stress test result   : %s\n",
            (time_ok && freq_ok) ? "PASS" : "FAIL");
    }


    // --------------------------------------------------------
    // Final result
    // --------------------------------------------------------

    std::printf(
        "\n============================================================\n");

    std::printf(
        " V2 RESULT\n");

    std::printf(
        "============================================================\n");


    std::printf(
        "IQ samples           : %zu / 3000\n",
        output.size());

    std::printf(
        "DDC time             : %.3f ms\n",
        ddc_time_ms);

    std::printf(
        "Costas tested        : %d\n",
        tested);

    std::printf(
        "Costas correct       : %d\n",
        correct);

    std::printf(
        "Costas wrong         : %d\n",
        tested - correct);


    if (output_count_ok &&
        tested == 21 &&
        correct == 21)
    {
        std::printf(
            "\n*** V2 PASS: 3000 IQ + COSTAS 21/21 ***\n");
    }
    else
    {
        std::printf(
            "\n*** V2 FAIL ***\n");
    }


    std::printf(
        "============================================================\n");


    return 0;
}

#endif  // ARDUINO