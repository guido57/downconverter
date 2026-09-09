
#include <Arduino.h>
#include <LittleFS.h>
#include <vector>
#include <stdint.h>
#include <time.h>
#include <cmath>
#include <cstring>
#include <algorithm>


// ============================================================
// PARAMETERS
// ============================================================

static constexpr int INPUT_FS  = 12000;
static constexpr int OUTPUT_FS = 200;
static constexpr int DECIM     = 60;

static constexpr float LO_FREQ = 1450.0f;

static constexpr int FFT_N = 32;

static constexpr float RF_TEST_FREQ = 1500.0f;

static constexpr float FFT_BIN_SPACING =
    static_cast<float>(OUTPUT_FS) / FFT_N;


// ============================================================
// FT8 TIMING
// ============================================================

static constexpr double SIGNAL_START_SEC = 2.000;

static constexpr int SIGNAL_START_INPUT =
    static_cast<int>(
        SIGNAL_START_SEC * INPUT_FS);

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
// DDC
// ============================================================

extern void ddc_process_fast(
    const float *input,
    int num_samples,
    std::vector<IQ> &output);

// ============================================================
// FINE SYNC ESTIMATION
// ============================================================
//
// Refines the coarse decoder frequency/timing using the 21
// Costas sync symbols, at 200 Hz. Used to prepare exact
// frequency and time-delay values for later coherent
// subtraction of an already-decoded FT8 signal.

extern void ddc_estimate_fine_sync(
    const std::vector<IQ> &output,
    int signal_start_output,
    int base_bin,
    const int *costas,
    int costas_len,
    const int *costas_start,
    int costas_arrays,
    float *out_freq_offset_hz,
    float *out_time_offset_samples);


// ============================================================
// WAV HEADER
// ============================================================

#pragma pack(push, 1)

struct WAVHeader
{
    char     riff[4];
    uint32_t file_size;
    char     wave[4];

    char     fmt[4];
    uint32_t fmt_size;

    uint16_t audio_format;
    uint16_t channels;
    uint32_t sample_rate;
    uint32_t byte_rate;
    uint16_t block_align;
    uint16_t bits_per_sample;

    char     data[4];
    uint32_t data_size;
};

#pragma pack(pop)


// ============================================================
// READ WAV FROM LITTLEFS
// ============================================================

bool load_wav(
    const char *filename,
    std::vector<float> &samples)
{
    File file = LittleFS.open(
        filename,
        "r");

    if (!file)
    {
        Serial.printf(
            "ERROR: cannot open %s\n",
            filename);

        return false;
    }


    WAVHeader hdr;

    if (file.read(
            (uint8_t *)&hdr,
            sizeof(WAVHeader)) !=
        sizeof(WAVHeader))
    {
        Serial.println(
            "ERROR: cannot read WAV header");

        file.close();

        return false;
    }


    // --------------------------------------------------------
    // Basic validation
    // --------------------------------------------------------

    if (memcmp(hdr.riff, "RIFF", 4) != 0 ||
        memcmp(hdr.wave, "WAVE", 4) != 0)
    {
        Serial.println(
            "ERROR: not a WAV file");

        file.close();

        return false;
    }


    if (hdr.audio_format != 1)
    {
        Serial.println(
            "ERROR: WAV is not PCM");

        file.close();

        return false;
    }


    if (hdr.channels != 1)
    {
        Serial.println(
            "ERROR: WAV is not mono");

        file.close();

        return false;
    }


    if (hdr.sample_rate != INPUT_FS)
    {
        Serial.printf(
            "ERROR: sample rate = %lu Hz, "
            "expected %d Hz\n",
            (unsigned long)hdr.sample_rate,
            INPUT_FS);

        file.close();

        return false;
    }


    if (hdr.bits_per_sample != 16)
    {
        Serial.println(
            "ERROR: WAV is not 16 bit");

        file.close();

        return false;
    }


    // --------------------------------------------------------
    // Number of samples
    // --------------------------------------------------------

    const int num_samples =
        hdr.data_size / 2;


    Serial.printf(
        "WAV samples        : %d\n",
        num_samples);

    Serial.printf(
        "WAV duration       : %.3f s\n",
        (float)num_samples / INPUT_FS);


    // --------------------------------------------------------
    // Allocate input
    // --------------------------------------------------------

    samples.resize(num_samples);


    if (samples.empty())
    {
        Serial.println(
            "ERROR: empty WAV");

        file.close();

        return false;
    }


    // --------------------------------------------------------
    // Read PCM16
    // --------------------------------------------------------

    for (int n = 0;
         n < num_samples;
         ++n)
    {
        uint8_t b[2];

        if (file.read(b, 2) != 2)
        {
            Serial.printf(
                "ERROR: WAV read failed "
                "at sample %d\n",
                n);

            file.close();

            return false;
        }


        const int16_t s =
            (int16_t)(
                (uint16_t)b[0] |
                ((uint16_t)b[1] << 8));


        samples[n] =
            (float)s / 32768.0f;
    }


    file.close();

    return true;
}


// ============================================================
// TIME
// ============================================================

static double elapsed_ms(
    const struct timespec *t0,
    const struct timespec *t1)
{
    long sec =
        t1->tv_sec -
        t0->tv_sec;

    long nsec =
        t1->tv_nsec -
        t0->tv_nsec;


    return
        (double)sec * 1000.0 +
        (double)nsec / 1000000.0;
}


// ============================================================
// FFT32 PEAK
// ============================================================
//
// Direct DFT.
//
// Input:
//   32 complex IQ samples
//
// Output:
//   strongest positive-frequency bin
//
// Bin spacing:
//   200 / 32 = 6.25 Hz
//
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
                static_cast<float>(k * n) /
                static_cast<float>(FFT_N);


            const float c =
                std::cos(angle);

            const float s =
                std::sin(angle);


            // ------------------------------------------------
            // Complex FFT:
            //
            // X[k] = sum x[n] * exp(-j*2*pi*k*n/N)
            //
            // x[n] = I + jQ
            // ------------------------------------------------

            re +=
                x[start + n].i * c -
                x[start + n].q * s;

            im +=
                x[start + n].i * s +
                x[start + n].q * c;
        }


        mag[k] =
            re * re +
            im * im;
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
// SETUP
// ============================================================

void setup()
{
    Serial.begin(115200);

    delay(1000);


    Serial.println();

    Serial.println(
        "############################################################");

    Serial.println(
        " FT8 DDC V2 - COSTAS 21/21 TEST");

    Serial.println(
        "############################################################");


    // ========================================================
    // PARAMETERS
    // ========================================================

    Serial.printf(
        "Input Fs            : %d Hz\n",
        INPUT_FS);

    Serial.printf(
        "LO                  : %.1f Hz\n",
        LO_FREQ);

    Serial.printf(
        "Decimation          : %d\n",
        DECIM);

    Serial.printf(
        "Output Fs           : %d Hz\n",
        OUTPUT_FS);

    Serial.printf(
        "IQ samples expected : 3000\n");

    Serial.printf(
        "FFT                  : %d complex samples\n",
        FFT_N);

    Serial.printf(
        "FFT duration        : %.3f ms\n",
        1000.0 *
        static_cast<double>(FFT_N) /
        OUTPUT_FS);

    Serial.printf(
        "FFT bin spacing     : %.3f Hz\n",
        FFT_BIN_SPACING);


    // ========================================================
    // LITTLEFS
    // ========================================================

    if (!LittleFS.begin(false))
    {
        Serial.println(
            "ERROR: LittleFS mount failed");

        return;
    }


    Serial.println(
        "LittleFS mounted");


    // ========================================================
    // LOAD WAV
    // ========================================================

    std::vector<float> input;


    if (!load_wav(
            "/test_real_ft8_D2000.wav",
            input))
    {
        return;
    }


    Serial.printf(
        "Input vector       : %d samples\n",
        (int)input.size());


    // ========================================================
    // DDC
    // ========================================================

    std::vector<IQ> output;


    struct timespec t0;
    struct timespec t1;


    clock_gettime(
        CLOCK_MONOTONIC,
        &t0);


    // IMPORTANT:
    // Keep ddc_process_fast() unchanged.

    ddc_process_fast(
        input.data(),
        (int)input.size(),
        output);


    clock_gettime(
        CLOCK_MONOTONIC,
        &t1);


    const double ddc_time_ms =
        elapsed_ms(
            &t0,
            &t1);


    // ========================================================
    // DDC RESULT
    // ========================================================

    Serial.println();

    Serial.println(
        "------------------------------------------------------------");

    Serial.println(
        " DDC RESULT");

    Serial.println(
        "------------------------------------------------------------");


    Serial.printf(
        "DDC time           : %.3f ms\n",
        ddc_time_ms);

    Serial.printf(
        "Input samples      : %d\n",
        (int)input.size());

    Serial.printf(
        "IQ output samples  : %d\n",
        (int)output.size());

    Serial.printf(
        "Expected IQ        : %d\n",
        (int)input.size() / DECIM);


    const bool output_count_ok =
        output.size() ==
        (input.size() + DECIM - 1) / DECIM;


    Serial.printf(
        "Output count       : %s\n",
        output_count_ok
            ? "OK"
            : "WRONG");


    // ========================================================
    // TIMING ALIGNMENT
    // ========================================================

    Serial.println();

    Serial.println(
        "============================================================");

    Serial.println(
        " TIMING ALIGNMENT");

    Serial.println(
        "============================================================");


    Serial.printf(
        "Signal start       : %.3f s\n",
        SIGNAL_START_SEC);

    Serial.printf(
        "Input sample       : %d\n",
        SIGNAL_START_INPUT);

    Serial.printf(
        "Signal output index: %d\n",
        SIGNAL_START_OUTPUT);

    Serial.printf(
        "Output sample time : %.3f s\n",
        (double)SIGNAL_START_OUTPUT /
        OUTPUT_FS);


    for (int array = 0;
         array < 3;
         ++array)
    {
        const int first =
            COSTAS_START[array];


        const int fft_start =
            SIGNAL_START_OUTPUT +
            first * FFT_N;


        const double t_start =
            static_cast<double>(fft_start) /
            OUTPUT_FS;


        const double t_end =
            static_cast<double>(
                fft_start + FFT_N) /
            OUTPUT_FS;


        Serial.printf(
            "Costas %d first symbol: "
            "symbol %d, %.3f ... %.3f s\n",
            array + 1,
            first,
            t_start,
            t_end);
    }


    // ========================================================
    // COSTAS PARAMETERS
    // ========================================================

    const float residual_freq =
        RF_TEST_FREQ -
        LO_FREQ;


    const int base_bin =
        static_cast<int>(
            std::lround(
                residual_freq /
                FFT_BIN_SPACING));


    Serial.println();

    Serial.println(
        "============================================================");

    Serial.println(
        " FT8 COSTAS VERIFICATION");

    Serial.println(
        "============================================================");


    Serial.printf(
        "RF tone             : %.2f Hz\n",
        RF_TEST_FREQ);

    Serial.printf(
        "LO                  : %.2f Hz\n",
        LO_FREQ);

    Serial.printf(
        "Residual frequency  : %.2f Hz\n",
        residual_freq);

    Serial.printf(
        "FFT bin spacing     : %.3f Hz\n",
        FFT_BIN_SPACING);

    Serial.printf(
        "Base FFT bin        : %d\n",
        base_bin);


    Serial.print(
        "Costas pattern      : ");


    for (int i = 0;
         i < COSTAS_LEN;
         ++i)
    {
        Serial.printf(
            "%d%s",
            COSTAS[i],
            (i == COSTAS_LEN - 1)
                ? "\n"
                : " ");
    }


    Serial.print(
        "Expected bins       : ");


    for (int i = 0;
         i < COSTAS_LEN;
         ++i)
    {
        Serial.printf(
            "%d%s",
            base_bin + COSTAS[i],
            (i == COSTAS_LEN - 1)
                ? "\n"
                : " ");
    }


    // ========================================================
    // TEST 21 COSTAS SYMBOLS
    // ========================================================

    int tested = 0;
    int correct = 0;


    for (int array = 0;
         array < 3;
         ++array)
    {
        const int first =
            COSTAS_START[array];


        Serial.println();

        Serial.println(
            "------------------------------------------------------------");

        Serial.printf(
            " COSTAS ARRAY %d   symbols %d..%d\n",
            array + 1,
            first,
            first + COSTAS_LEN - 1);

        Serial.println(
            "------------------------------------------------------------");

        Serial.println(
            " SYMBOL   TIME          "
            "EXPECTED          MEASURED        RESULT");


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


            // ------------------------------------------------
            // Range check
            // ------------------------------------------------

            if (fft_start + FFT_N >
                (int)output.size())
            {
                Serial.printf(
                    "%5d   OUT OF RANGE\n",
                    symbol);

                continue;
            }


            // ------------------------------------------------
            // FFT32
            // ------------------------------------------------

            const int peak =
                fft_peak_bin(
                    output,
                    fft_start);


            const bool ok =
                peak == expected;


            ++tested;


            if (ok)
                ++correct;


            const double t_start =
                static_cast<double>(fft_start) /
                OUTPUT_FS;


            const double t_end =
                static_cast<double>(
                    fft_start + FFT_N) /
                OUTPUT_FS;


            const float expected_freq =
                expected *
                FFT_BIN_SPACING;


            const float measured_freq =
                peak *
                FFT_BIN_SPACING;


            Serial.printf(
                "%5d   %5.3f-%5.3f    "
                "%2d (%6.2f Hz)    "
                "%2d (%6.2f Hz)    %s\n",

                symbol,

                t_start,
                t_end,

                expected,
                expected_freq,

                peak,
                measured_freq,

                ok
                    ? "OK"
                    : "WRONG");
        }
    }


    // ========================================================
    // FINE SYNC ESTIMATION
    //
    // Refines residual frequency and symbol timing using the
    // already-decoded Costas sync symbols. These values are
    // needed later to regenerate and subtract this signal from
    // the raw stream (time-domain cancellation).
    // ========================================================

    float fine_freq_offset_hz = 0.0f;
    float fine_time_offset_samp = 0.0f;

    struct timespec t2;
    struct timespec t3;

    clock_gettime(
        CLOCK_MONOTONIC,
        &t2);


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

    clock_gettime(
        CLOCK_MONOTONIC,
        &t3);


    const float exact_freq_hz =
        LO_FREQ +
        base_bin * FFT_BIN_SPACING +
        fine_freq_offset_hz;

    const double exact_time_sec =
        (static_cast<double>(SIGNAL_START_OUTPUT) +
         fine_time_offset_samp) /
        OUTPUT_FS;

    Serial.println();

    Serial.println(
        "============================================================");

    Serial.println(
        " FINE SYNC ESTIMATE");

    Serial.println(
        "============================================================");

    Serial.printf(
        "Coarse freq bin      : %d (%.2f Hz residual)\n",
        base_bin,
        base_bin * FFT_BIN_SPACING);

    Serial.printf(
        "Fine freq correction : %+.3f Hz\n",
        fine_freq_offset_hz);

    Serial.printf(
        "Exact carrier freq   : %.3f Hz\n",
        exact_freq_hz);

    Serial.printf(
        "Fine time correction : %+.3f output samples "
        "(%+.3f ms)\n",
        fine_time_offset_samp,
        1000.0 * fine_time_offset_samp / OUTPUT_FS);

    Serial.printf(
        "Exact message start  : %.6f s\n",
        exact_time_sec);


    // ========================================================
    // FINAL RESULT
    // ========================================================

    Serial.println();

    Serial.println(
        "============================================================");

    Serial.println(
        " V2 RESULT");

    Serial.println(
        "============================================================");


    Serial.printf(
        "IQ samples           : %d / 3000\n",
        (int)output.size());

    Serial.printf(
        "DDC time             : %.3f ms\n",
        ddc_time_ms);

    Serial.printf(
        "Costas tested        : %d\n",
        tested);

    Serial.printf(
        "Costas correct       : %d\n",
        correct);

    Serial.printf(
        "Costas wrong         : %d\n",
        tested - correct);

    Serial.printf(
        "FINE SYNC time       : %.3f ms\n",
        elapsed_ms(&t2, &t3));


    if (output_count_ok &&
        tested == 21 &&
        correct == 21)
    {
        Serial.println();

        Serial.println(
            "*** V2 PASS: 3000 IQ + COSTAS 21/21 ***");
    }
    else
    {
        Serial.println();

        Serial.println(
            "*** V2 FAIL ***");
    }


    Serial.println(
        "============================================================");
}


// ============================================================
// LOOP
// ============================================================

void loop()
{
    delay(1000);
}

