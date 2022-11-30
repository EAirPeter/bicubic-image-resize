#ifndef RESIZE_H_
#define RESIZE_H_

#include "utils.hpp"
#include <cmath>

#include <immintrin.h>

#ifdef HAVE_PROFILER
//#undef HAVE_PROFILER
#endif

#ifdef HAVE_PROFILER

struct ScopedProfiler
{
    ScopedProfiler(bool cond)
        : _cond(cond) { if (cond) PROF_START(); }
    ~ScopedProfiler() { if (_cond) PROF_STOP(); }
    bool _cond;
};

struct ScopedProfilerMarker
{
    ScopedProfilerMarker(const char* name) { PROF_PUSH_MARKER(name); }
    ~ScopedProfilerMarker() { PROF_POP_MARKER(); }
};

struct ScopedProfilerArrayMarker
{
    ScopedProfilerArrayMarker(const char* name, const void* ptr, size_t size)
        : _ptr(ptr) { PROF_MARK_MEMORY(name, ptr, size); }
    ~ScopedProfilerArrayMarker() { PROF_UNMARK_MEMORY(_ptr); }
    const void* _ptr;
};

#define PROF_CONCAT_(a, b) a ## b
#define PROF_CONCAT(a, b) PROF_CONCAT_(a, b)

#define PROF_SCOPED_CAPTURE() ScopedProfiler PROF_CONCAT(_zw_sp_, __LINE__)(true)
#define PROF_SCOPED_COND_CAPTURE(cond) ScopedProfiler PROF_CONCAT(_zw_sp_, __LINE__)((cond))
#define PROF_SCOPED_MARKER(name) ScopedProfilerMarker PROF_CONCAT(_zw_spm_, __LINE__)(name)
#define PROF_SCOPED_MEMORY(name, ptr, size) ScopedProfilerArrayMarker PROF_CONCAT(_zw_spam, __LINE__)(name, ptr, size)
#else
#define PROF_START()
#define PROF_STOP()
#define PROF_PUSH_MARKER(name)
#define PROF_POP_MARKER()
#define PROF_MARK_MEMORY(name, ptr, size)
#define PROF_UNMARK_MEMORY(ptr)

#define PROF_SCOPED_CAPTURE()
#define PROF_SCOPED_COND_CAPTURE(cond)
#define PROF_SCOPED_MARKER(name)
#define PROF_SCOPED_MEMORY(name, ptr, size)
#endif

constexpr float WeightCoeff(float x, float a)
{
    if (x <= 1)
        return 1 - (a + 3) * x * x + (a + 2) * x * x * x;
    else if (x < 2)
        return -4 * a + 8 * a * x - 5 * a * x * x + a * x * x * x;
    return 0.0;
}

constexpr void CalcCoeff4x4(float u, float v, float outCoeff[4][4])
{
    constexpr float a = -0.5f;

    u += 1.0f;
    v += 1.0f;

    for (int i = 0; i < 4; i++)
        for (int j = 0; j < 4; j++)
            outCoeff[i][j] = WeightCoeff(__builtin_fabs(u - i), a) * WeightCoeff(__builtin_fabs(v - j), a);
}

constexpr auto kNChannel = 3;
constexpr auto kRatio = 5;
constexpr auto kRatioFloat = static_cast<float>(kRatio);
static_assert(kRatio * kNChannel <= 16, "YMM");

struct CoeffTable
{
    constexpr CoeffTable()
    {
        for (auto ir = 0; ir < kRatio; ++ir)
        {
            const auto u = static_cast<float>(ir) / kRatioFloat;
            for (auto ic = 0; ic < kRatio; ++ic)
            {
                const auto v = static_cast<float>(ic) / kRatioFloat;
                CalcCoeff4x4(u, v, m_Data[ir][ic]);
            }
        }
    }

    constexpr decltype(auto) operator[](int ir) const { return (m_Data[ir]); }

    alignas(64) float m_Data[kRatio][kRatio][4][4]{};
};

struct CoeffTableSwizzled
{
    constexpr CoeffTableSwizzled()
    {
        constexpr CoeffTable kCoeffs;
        for (auto ir = 0; ir < kRatio; ++ir)
            for (auto i = 0; i < 4; ++i)
                for (auto j = 0; j < 4; ++j)
                    for (auto ic = 0; ic < kRatio; ++ic)
                        for (auto ch = 0; ch < kNChannel; ++ch)
                            m_Data[ir][i][j][ic * kNChannel + ch] = kCoeffs[ir][ic][i][j];
    }

    constexpr decltype(auto) operator[](int ir) const { return (m_Data[ir]); }

    alignas(64) float m_Data[kRatio][4][4][16]{};
};

static constexpr CoeffTableSwizzled kCoeffsSwizzled;

RGBImage ResizeImage(RGBImage src, float ratio) {
    if (kNChannel != src.channels || kRatio != ratio)
        return {};

    Timer timer("resize image by 5x");
    //PROF_SCOPED_CAPTURE();

    const auto nRow = src.rows;
    const auto nCol = src.cols;
    const auto nResRow = nRow * kRatio;
    const auto nResCol = nCol * kRatio;

    PROF_SCOPED_MEMORY("Source", src.data, kNChannel * nRow * nCol);

    printf("resize to: %d x %d\n", nResRow, nResCol);

    const auto pRes = new unsigned char[kNChannel * nResRow * nResCol]{};
    PROF_SCOPED_MEMORY("Result", pRes, kNChannel * nResRow * nResCol);

    // Analysis of check_perimeter() in vanilla code:
    // srcRow = r + ir / kRatio
    // resRow = r * kRatio + ir
    // * srcRow <= 1
    //   * r + ir / kRatio <= 1
    //   * (r == 0) || (r == 1 && ir == 0)
    //   * resRow <= kRatio
    // * srcRow >= nRow - 2
    //   * r + ir / kRatio >= nRow - 2
    //   * (r >= nRow - 2)
    //   * resRow >= (nRow - 2) * kRatio
    // * 1 < srcRow < nRow - 2
    //   * (r == 1 && ir > 0) || (2 <= r < nRow - 2)
    //   * kRatio < resRow < nResRow - 2 * kRatio
    // For the sake of simplicity, we change the limit to
    // * 1 <= srcRow < nRow - 2
    //  * 1 <= r < nRow - 2
    //  * kRatio <= resRow < nResRow - 2 * kRatio
    // The above also holds for columns

#define SIMPLIFY_START 0

#define USE_ASM_LOAD 1

    PROF_SCOPED_MARKER("WorkLoop");

    #pragma omp parallel for
    for (auto r = 1; r < nRow - 2; ++r)
    {
        PROF_SCOPED_COND_CAPTURE(r == 123);
        //PROF_SCOPED_MARKER("SourceRow");
        alignas(64) float in012[4][4][16];
        {
            PROF_SCOPED_MARKER("LoadInput");
        #if USE_ASM_LOAD
            #define LoadYmm(y, p) asm("vpmovzxbd %1, %0" : "=x"(y) : "m"(*(unsigned char(*)[8])(p)))
            #define LoadXmm(x, p) asm("vpmovzxbd %1, %0" : "=x"(x) : "m"(*(unsigned char(*)[4])(p)))
        #else
            #define LoadYmm(y, p) (y = _mm256_cvtepu8_epi32(*(const __m128i*)(p)));
            #define LoadXmm(x, p) (x = _mm_cvtepu8_epi32(   *(const __m128i*)(p)))
        #endif

            //const auto ydw00 = _mm256_cvtepu8_epi32(*(const __m128i*)&src.data[((r - 1) * nCol + (1 - 1)) * kNChannel + 0]);
            //const auto xdw01 = _mm_cvtepu8_epi32(   *(const __m128i*)&src.data[((r - 1) * nCol + (1 - 1)) * kNChannel + 8]);
            //const auto ydw10 = _mm256_cvtepu8_epi32(*(const __m128i*)&src.data[((r - 0) * nCol + (1 - 1)) * kNChannel + 0]);
            //const auto xdw11 = _mm_cvtepu8_epi32(   *(const __m128i*)&src.data[((r - 0) * nCol + (1 - 1)) * kNChannel + 8]);
            //const auto ydw20 = _mm256_cvtepu8_epi32(*(const __m128i*)&src.data[((r + 1) * nCol + (1 - 1)) * kNChannel + 0]);
            //const auto xdw21 = _mm_cvtepu8_epi32(   *(const __m128i*)&src.data[((r + 1) * nCol + (1 - 1)) * kNChannel + 8]);
            //const auto ydw30 = _mm256_cvtepu8_epi32(*(const __m128i*)&src.data[((r + 2) * nCol + (1 - 1)) * kNChannel + 0]);
            //const auto xdw31 = _mm_cvtepu8_epi32(   *(const __m128i*)&src.data[((r + 2) * nCol + (1 - 1)) * kNChannel + 8]);

            #define GET_SW0(y, _2, _1, _0) _mm256_permutevar8x32_ps(y, _mm256_set_epi32(_1, _0, _2, _1, _0, _2, _1, _0))
            #define GET_SW1(y, _2, _1, _0) _mm256_permutevar8x32_ps(y, _mm256_set_epi32(_0, _2, _1, _0, _2, _1, _0, _2))

            #define MAKE_SW(i) \
                __m256i ydw##i##0; __m128i xdw##i##1; \
                LoadYmm(ydw##i##0, &src.data[((r + i - 1) * nCol + (1 - 1)) * kNChannel + 0]); \
                LoadXmm(xdw##i##1, &src.data[((r + i - 1) * nCol + (1 - 1)) * kNChannel + 8]); \
                const auto yf##i##0 = _mm256_cvtepi32_ps(ydw##i##0);                      /* 00 01 02 10 11 12 20 21 */ \
                const auto yf##i##1 = _mm256_castps128_ps256(_mm_cvtepi32_ps(xdw##i##1)); /* 22 30 31 32 ?? ?? ?? ?? */ \
                const auto yf##i##2 = _mm256_blend_ps(yf##i##0, yf##i##1, 0b00001111);    /* 22 ?? ?? ?? ?? ?? 20 21 */ \
                const auto ysw##i##00 = GET_SW0(yf##i##0, 2, 1, 0); \
                const auto ysw##i##01 = GET_SW1(yf##i##0, 2, 1, 0); \
                const auto ysw##i##10 = GET_SW0(yf##i##0, 5, 4, 3); \
                const auto ysw##i##11 = GET_SW1(yf##i##0, 5, 4, 3); \
                const auto ysw##i##20 = GET_SW0(yf##i##2, 0, 7, 6); \
                const auto ysw##i##21 = GET_SW1(yf##i##2, 0, 7, 6); \
                const auto ysw##i##30 = GET_SW0(yf##i##2, 3, 2, 1); \
                const auto ysw##i##31 = GET_SW1(yf##i##2, 3, 2, 1); \
                _mm256_store_ps(&in012[i][0][0], ysw##i##00); \
                _mm256_store_ps(&in012[i][0][8], ysw##i##01); \
                _mm256_store_ps(&in012[i][1][0], ysw##i##10); \
                _mm256_store_ps(&in012[i][1][8], ysw##i##11); \
                _mm256_store_ps(&in012[i][2][0], ysw##i##20); \
                _mm256_store_ps(&in012[i][2][8], ysw##i##21); \
                _mm256_store_ps(&in012[i][3][0], ysw##i##30); \
                _mm256_store_ps(&in012[i][3][8], ysw##i##31);

            MAKE_SW(0);
            MAKE_SW(1);
            MAKE_SW(2);
            MAKE_SW(3);

            #undef GET_SW0
            #undef GET_SW1
            #undef MAKE_SW
        }

        for (auto c = 1; c < nCol - 2; ++c)
        {
            //PROF_SCOPED_MARKER("SourceColumn");
            if (c > 1)
            {
                constexpr auto j = 3;

            #if USE_ASM_LOAD
                const auto* in = &src.data[(r - 1) * nCol + (c + j - 1) * kNChannel];
                const auto  inStep = ptrdiff_t(nCol * kNChannel);

                ptrdiff_t s0;
                __m128 x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, xA, xB;
                __m256 yC, yD, yE, yF;

                asm(
                    R"???(
                        lea (%[step],%[step],2), %[tmp]

                        vpmovzxbd (%[src]),           %[xs0]
                        vpmovzxbd (%[src],%[step]),   %[xs1]
                        vpmovzxbd (%[src],%[step],2), %[xs2]
                        vpmovzxbd (%[src],%[tmp]),    %[xs3]

                        vcvtdq2ps %[xs0], %[xs0]
                        vcvtdq2ps %[xs1], %[xs1]
                        vcvtdq2ps %[xs2], %[xs2]
                        vcvtdq2ps %[xs3], %[xs3]

                        vpermilps %[m1201], %[xs0], %[xt0]
                        vpermilps %[m2012], %[xs0], %[xu0]
                        vpermilps %[m0120], %[xs0], %[xs0]

                        vpermilps %[m1201], %[xs1], %[xt1]
                        vpermilps %[m2012], %[xs1], %[xu1]
                        vpermilps %[m0120], %[xs1], %[xs1]

                        vpermilps %[m1201], %[xs2], %[xt2]
                        vpermilps %[m2012], %[xs2], %[xu2]
                        vpermilps %[m0120], %[xs2], %[xs2]

                        vpermilps %[m1201], %[xs3], %[xt3]
                        vpermilps %[m2012], %[xs3], %[xu3]
                        vpermilps %[m0120], %[xs3], %[xs3]

                        vmovaps %c[i010](%[dst]), %[y0]
                        vmovaps %c[i018](%[dst]), %[y1]
                        vmovaps %c[i020](%[dst]), %[y2]
                        vmovaps %c[i028](%[dst]), %[y3]

                        vmovaps %[y0], %c[i000](%[dst])
                        vmovaps %[y1], %c[i008](%[dst])
                        vmovaps %[y2], %c[i010](%[dst])
                        vmovaps %[y3], %c[i018](%[dst])

                        vmovaps %c[i030](%[dst]), %[y0]
                        vmovaps %c[i038](%[dst]), %[y1]

                        vmovaps %[y0], %c[i020](%[dst])
                        vmovaps %[y1], %c[i028](%[dst])

                        vmovaps %[xs0], %c[i030](%[dst])
                        vmovaps %[xt0], %c[i034](%[dst])
                        vmovaps %[xu0], %c[i038](%[dst])
                        vmovaps %[xs0], %c[i03C](%[dst])

                        vmovaps %c[i110](%[dst]), %[y0]
                        vmovaps %c[i118](%[dst]), %[y1]
                        vmovaps %c[i120](%[dst]), %[y2]
                        vmovaps %c[i128](%[dst]), %[y3]
                        vmovaps %c[i130](%[dst]), %t[xs0]
                        vmovaps %c[i138](%[dst]), %t[xt0]

                        vmovaps %[y0],   %c[i100](%[dst])
                        vmovaps %[y1],   %c[i108](%[dst])
                        vmovaps %[y2],   %c[i110](%[dst])
                        vmovaps %[y3],   %c[i118](%[dst])
                        vmovaps %t[xs0], %c[i120](%[dst])
                        vmovaps %t[xt0], %c[i128](%[dst])

                        vmovaps %[xs1], %c[i130](%[dst])
                        vmovaps %[xt1], %c[i134](%[dst])
                        vmovaps %[xu1], %c[i138](%[dst])
                        vmovaps %[xs1], %c[i13C](%[dst])

                        vmovaps %c[i210](%[dst]), %[y0]
                        vmovaps %c[i218](%[dst]), %[y1]
                        vmovaps %c[i220](%[dst]), %[y2]
                        vmovaps %c[i228](%[dst]), %[y3]
                        vmovaps %c[i230](%[dst]), %t[xs0]
                        vmovaps %c[i238](%[dst]), %t[xt0]

                        vmovaps %[y0],   %c[i200](%[dst])
                        vmovaps %[y1],   %c[i208](%[dst])
                        vmovaps %[y2],   %c[i210](%[dst])
                        vmovaps %[y3],   %c[i218](%[dst])
                        vmovaps %t[xs0], %c[i220](%[dst])
                        vmovaps %t[xt0], %c[i228](%[dst])

                        vmovaps %[xs2], %c[i230](%[dst])
                        vmovaps %[xt2], %c[i234](%[dst])
                        vmovaps %[xu2], %c[i238](%[dst])
                        vmovaps %[xs2], %c[i23C](%[dst])

                        vmovaps %c[i310](%[dst]), %[y0]
                        vmovaps %c[i318](%[dst]), %[y1]
                        vmovaps %c[i320](%[dst]), %[y2]
                        vmovaps %c[i328](%[dst]), %[y3]
                        vmovaps %c[i330](%[dst]), %t[xs0]
                        vmovaps %c[i338](%[dst]), %t[xt0]

                        vmovaps %[y0],   %c[i300](%[dst])
                        vmovaps %[y1],   %c[i308](%[dst])
                        vmovaps %[y2],   %c[i310](%[dst])
                        vmovaps %[y3],   %c[i318](%[dst])
                        vmovaps %t[xs0], %c[i320](%[dst])
                        vmovaps %t[xt0], %c[i328](%[dst])

                        vmovaps %[xs3], %c[i330](%[dst])
                        vmovaps %[xt3], %c[i334](%[dst])
                        vmovaps %[xu3], %c[i338](%[dst])
                        vmovaps %[xs3], %c[i33C](%[dst])
                    )???"
                    : [xs0]"=x"(x0), [xs1]"=x"(x1), [xs2]"=x"(x2), [xs3]"=x"(x3)
                    , [xt0]"=x"(x4), [xu0]"=x"(x5), [xt1]"=x"(x6), [xu1]"=x"(x7)
                    , [xt2]"=x"(x8), [xu2]"=x"(x9), [xt3]"=x"(xA), [xu3]"=x"(xB)
                    , [y0]"=x"(yC), [y1]"=x"(yD), [y2]"=x"(yE), [y3]"=x"(yF)
                    , [tmp]"=&r"(s0)
                    : [step]"r"(inStep), [dst]"r"(&in012), [src]"r"(in)
                    , [m0120]"i"(_MM_SHUFFLE(0, 2, 1, 0))
                    , [m1201]"i"(_MM_SHUFFLE(1, 0, 2, 1))
                    , [m2012]"i"(_MM_SHUFFLE(2, 1, 0, 2))
                    , [i000]"i"(((0 * 4 + 0) * 16 +  0) * 4)
                    , [i008]"i"(((0 * 4 + 0) * 16 +  8) * 4)
                    , [i010]"i"(((0 * 4 + 1) * 16 +  0) * 4)
                    , [i018]"i"(((0 * 4 + 1) * 16 +  8) * 4)
                    , [i020]"i"(((0 * 4 + 2) * 16 +  0) * 4)
                    , [i028]"i"(((0 * 4 + 2) * 16 +  8) * 4)
                    , [i030]"i"(((0 * 4 + 3) * 16 +  0) * 4)
                    , [i034]"i"(((0 * 4 + 3) * 16 +  4) * 4)
                    , [i038]"i"(((0 * 4 + 3) * 16 +  8) * 4)
                    , [i03C]"i"(((0 * 4 + 3) * 16 + 12) * 4)
                    , [i100]"i"(((1 * 4 + 0) * 16 +  0) * 4)
                    , [i108]"i"(((1 * 4 + 0) * 16 +  8) * 4)
                    , [i110]"i"(((1 * 4 + 1) * 16 +  0) * 4)
                    , [i118]"i"(((1 * 4 + 1) * 16 +  8) * 4)
                    , [i120]"i"(((1 * 4 + 2) * 16 +  0) * 4)
                    , [i128]"i"(((1 * 4 + 2) * 16 +  8) * 4)
                    , [i130]"i"(((1 * 4 + 3) * 16 +  0) * 4)
                    , [i134]"i"(((1 * 4 + 3) * 16 +  4) * 4)
                    , [i138]"i"(((1 * 4 + 3) * 16 +  8) * 4)
                    , [i13C]"i"(((1 * 4 + 3) * 16 + 12) * 4)
                    , [i200]"i"(((2 * 4 + 0) * 16 +  0) * 4)
                    , [i208]"i"(((2 * 4 + 0) * 16 +  8) * 4)
                    , [i210]"i"(((2 * 4 + 1) * 16 +  0) * 4)
                    , [i218]"i"(((2 * 4 + 1) * 16 +  8) * 4)
                    , [i220]"i"(((2 * 4 + 2) * 16 +  0) * 4)
                    , [i228]"i"(((2 * 4 + 2) * 16 +  8) * 4)
                    , [i230]"i"(((2 * 4 + 3) * 16 +  0) * 4)
                    , [i234]"i"(((2 * 4 + 3) * 16 +  4) * 4)
                    , [i238]"i"(((2 * 4 + 3) * 16 +  8) * 4)
                    , [i23C]"i"(((2 * 4 + 3) * 16 + 12) * 4)
                    , [i300]"i"(((3 * 4 + 0) * 16 +  0) * 4)
                    , [i308]"i"(((3 * 4 + 0) * 16 +  8) * 4)
                    , [i310]"i"(((3 * 4 + 1) * 16 +  0) * 4)
                    , [i318]"i"(((3 * 4 + 1) * 16 +  8) * 4)
                    , [i320]"i"(((3 * 4 + 2) * 16 +  0) * 4)
                    , [i328]"i"(((3 * 4 + 2) * 16 +  8) * 4)
                    , [i330]"i"(((3 * 4 + 3) * 16 +  0) * 4)
                    , [i334]"i"(((3 * 4 + 3) * 16 +  4) * 4)
                    , [i338]"i"(((3 * 4 + 3) * 16 +  8) * 4)
                    , [i33C]"i"(((3 * 4 + 3) * 16 + 12) * 4)
                    : "memory"
                );
            #else

                const auto xdw0 = _mm_cvtepu8_epi32(*(const __m128i*)&src.data[((r - 1) * nCol + (c + j - 1)) * kNChannel]);
                const auto xdw1 = _mm_cvtepu8_epi32(*(const __m128i*)&src.data[((r - 0) * nCol + (c + j - 1)) * kNChannel]);
                const auto xdw2 = _mm_cvtepu8_epi32(*(const __m128i*)&src.data[((r + 1) * nCol + (c + j - 1)) * kNChannel]);
                const auto xdw3 = _mm_cvtepu8_epi32(*(const __m128i*)&src.data[((r + 2) * nCol + (c + j - 1)) * kNChannel]);
                const auto xf0 = _mm_cvtepi32_ps(xdw0);
                const auto xf1 = _mm_cvtepi32_ps(xdw1);
                const auto xf2 = _mm_cvtepi32_ps(xdw2);
                const auto xf3 = _mm_cvtepi32_ps(xdw3);
                #define MAKE_SW(i) \
                    const auto xsw##i##0 = _mm_permute_ps(xf##i, _MM_SHUFFLE(0, 2, 1, 0)); /* 00 01 02 00 */ \
                    const auto xsw##i##1 = _mm_permute_ps(xf##i, _MM_SHUFFLE(1, 0, 2, 1)); /* 01 02 00 01 */ \
                    const auto xsw##i##2 = _mm_permute_ps(xf##i, _MM_SHUFFLE(2, 1, 0, 2)); /* 02 00 01 02 */ \
                    const auto ycp##i##00 = _mm256_load_ps(&in012[i][1][0]); \
                    const auto ycp##i##01 = _mm256_load_ps(&in012[i][1][8]); \
                    const auto ycp##i##10 = _mm256_load_ps(&in012[i][2][0]); \
                    const auto ycp##i##11 = _mm256_load_ps(&in012[i][2][8]); \
                    const auto ycp##i##20 = _mm256_load_ps(&in012[i][3][0]); \
                    const auto ycp##i##21 = _mm256_load_ps(&in012[i][3][8]); \
                    _mm256_store_ps(&in012[i][0][0], ycp##i##00); \
                    _mm256_store_ps(&in012[i][0][8], ycp##i##01); \
                    _mm256_store_ps(&in012[i][1][0], ycp##i##10); \
                    _mm256_store_ps(&in012[i][1][8], ycp##i##11); \
                    _mm256_store_ps(&in012[i][2][0], ycp##i##20); \
                    _mm256_store_ps(&in012[i][2][8], ycp##i##21); \
                    _mm_store_ps(&in012[i][3][ 0], xsw##i##0); \
                    _mm_store_ps(&in012[i][3][ 4], xsw##i##1); \
                    _mm_store_ps(&in012[i][3][ 8], xsw##i##2); \
                    _mm_store_ps(&in012[i][3][12], xsw##i##0);

                MAKE_SW(0);
                MAKE_SW(1);
                MAKE_SW(2);
                MAKE_SW(3);

                #undef MAKE_SW
            #endif
            }

            //PROF_SCOPED_MARKER("YieldTile");

        //#if SIMPLIFY_START
        //    for (auto ir = 0; ir < kRatio; ++ir)
        //#else
        //    for (auto ir = r == 1 ? 1 : 0; ir < kRatio; ++ir)
        //#endif
        //        _mm_prefetch(&pRes[((r * kRatio + ir) * nResCol + c * kRatio) * kNChannel], _MM_HINT_NTA);
        #if SIMPLIFY_START
            for (auto ir = 0; ir < kRatio; ++ir)
        #else
            for (auto ir = r == 1 ? 1 : 0; ir < kRatio; ++ir)
        #endif
            {
                //_mm_prefetch(&pRes[((r * kRatio + ir) * nResCol + c * kRatio) * kNChannel], _MM_HINT_NTA);
                const auto& coeffs = kCoeffsSwizzled.m_Data[ir];
                auto yf0 = _mm256_setzero_ps();
                auto yf1 = _mm256_setzero_ps();

                for (auto i = 0; i < 4; ++i)
                    for (auto j = 0; j < 4; ++j)
                    {
                        yf0 = _mm256_fmadd_ps(_mm256_load_ps(&coeffs[i][j][0]), _mm256_load_ps(&in012[i][j][0]), yf0);
                        yf1 = _mm256_fmadd_ps(_mm256_load_ps(&coeffs[i][j][8]), _mm256_load_ps(&in012[i][j][8]), yf1);
                    }

                const auto ydw0 = _mm256_cvttps_epi32(yf0);
                const auto ydw1 = _mm256_cvttps_epi32(yf1);
                const auto xdw00 = _mm256_castsi256_si128(ydw0);
                const auto xdw10 = _mm256_castsi256_si128(ydw1);
                const auto xdw01 = _mm256_extracti128_si256(ydw0, 1);
                const auto xdw11 = _mm256_extracti128_si256(ydw1, 1);
                const auto xw0 = _mm_packus_epi32(xdw00, xdw01);
                const auto xw1 = _mm_packus_epi32(xdw10, xdw11);
                const auto xw = _mm_packus_epi16(xw0, xw1);
                _mm_storeu_si128((__m128i*)&pRes[((r * kRatio + ir) * nResCol + c * kRatio) * kNChannel], xw);
            }
        }
    }

    return RGBImage{nResCol, nResRow, kNChannel, pRes};
}

#endif
