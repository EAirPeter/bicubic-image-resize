#ifndef RESIZE_H_
#define RESIZE_H_

#include "utils.hpp"
#include <cmath>

#include <immintrin.h>

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

    alignas(32) float m_Data[kRatio][kRatio][4][4]{};
};

static constexpr CoeffTable kCoeffs;

struct CoeffTableSwizzled
{
    constexpr CoeffTableSwizzled()
    {
        for (auto ir = 0; ir < kRatio; ++ir)
            for (auto i = 0; i < 4; ++i)
                for (auto j = 0; j < 4; ++j)
                    for (auto ic = 0; ic < kRatio; ++ic)
                        for (auto ch = 0; ch < kNChannel; ++ch)
                            m_Data[ir][i][j][ic * kNChannel + ch] = kCoeffs[ir][ic][i][j];
    }

    constexpr decltype(auto) operator[](int ir) const { return (m_Data[ir]); }

    alignas(32) float m_Data[kRatio][4][4][16]{};
};

static constexpr CoeffTableSwizzled kCoeffsSwizzled;

RGBImage ResizeImage(RGBImage src, float ratio) {
    if (kNChannel != src.channels || kRatio != ratio)
        return {};

    Timer timer("resize image by 5x");

    const auto nRow = src.rows;
    const auto nCol = src.cols;
    const auto nResRow = nRow * kRatio;
    const auto nResCol = nCol * kRatio;

    printf("resize to: %d x %d\n", nResRow, nResCol);

    const auto pRes = new unsigned char[kNChannel * nResRow * nResCol]{};

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

#define PRECOMPUTE_COEFFS 1
#define SIMPLIFY_START 0

#define LOAD_IN_YMM 1
#define LOAD_IN_XMM 0
#define LOAD_IN_INTRIN 1

    #pragma omp parallel for
    for (auto r = 1; r < nRow - 2; ++r)
    {
        for (auto c = 1; c < nCol - 2; ++c)
        {
            alignas(32) float in012[4][4][16];
            {
            #if LOAD_IN_INTRIN
                #if LOAD_IN_YMM
                    const auto ydw00 = _mm256_cvtepu8_epi32(*(const __m128i*)&src.data[((r - 1) * nCol + (c - 1)) * kNChannel + 0]);
                    const auto xdw01 = _mm_cvtepu8_epi32(   *(const __m128i*)&src.data[((r - 1) * nCol + (c - 1)) * kNChannel + 8]);
                    const auto ydw10 = _mm256_cvtepu8_epi32(*(const __m128i*)&src.data[((r - 0) * nCol + (c - 1)) * kNChannel + 0]);
                    const auto xdw11 = _mm_cvtepu8_epi32(   *(const __m128i*)&src.data[((r - 0) * nCol + (c - 1)) * kNChannel + 8]);
                    const auto ydw20 = _mm256_cvtepu8_epi32(*(const __m128i*)&src.data[((r + 1) * nCol + (c - 1)) * kNChannel + 0]);
                    const auto xdw21 = _mm_cvtepu8_epi32(   *(const __m128i*)&src.data[((r + 1) * nCol + (c - 1)) * kNChannel + 8]);
                    const auto ydw30 = _mm256_cvtepu8_epi32(*(const __m128i*)&src.data[((r + 2) * nCol + (c - 1)) * kNChannel + 0]);
                    const auto xdw31 = _mm_cvtepu8_epi32(   *(const __m128i*)&src.data[((r + 2) * nCol + (c - 1)) * kNChannel + 8]);

                    #define GET_SW0(y, _2, _1, _0) _mm256_permutevar8x32_ps(y, _mm256_set_epi32(_1, _0, _2, _1, _0, _2, _1, _0))
                    #define GET_SW1(y, _2, _1, _0) _mm256_permutevar8x32_ps(y, _mm256_set_epi32(_0, _2, _1, _0, _2, _1, _0, _2))

                    #define MAKE_SW(i) \
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
                #elif LOAD_IN_XMM
                    const auto xdw00 = _mm_cvtepu8_epi32(*(const __m128i*)&src.data[((r - 1) * nCol + (c - 1)) * kNChannel + 0]);
                    const auto xdw01 = _mm_cvtepu8_epi32(*(const __m128i*)&src.data[((r - 1) * nCol + (c - 1)) * kNChannel + 4]);
                    const auto xdw02 = _mm_cvtepu8_epi32(*(const __m128i*)&src.data[((r - 1) * nCol + (c - 1)) * kNChannel + 8]);
                    const auto xdw10 = _mm_cvtepu8_epi32(*(const __m128i*)&src.data[((r - 0) * nCol + (c - 1)) * kNChannel + 0]);
                    const auto xdw11 = _mm_cvtepu8_epi32(*(const __m128i*)&src.data[((r - 0) * nCol + (c - 1)) * kNChannel + 4]);
                    const auto xdw12 = _mm_cvtepu8_epi32(*(const __m128i*)&src.data[((r - 0) * nCol + (c - 1)) * kNChannel + 8]);
                    const auto xdw20 = _mm_cvtepu8_epi32(*(const __m128i*)&src.data[((r + 1) * nCol + (c - 1)) * kNChannel + 0]);
                    const auto xdw21 = _mm_cvtepu8_epi32(*(const __m128i*)&src.data[((r + 1) * nCol + (c - 1)) * kNChannel + 4]);
                    const auto xdw22 = _mm_cvtepu8_epi32(*(const __m128i*)&src.data[((r + 1) * nCol + (c - 1)) * kNChannel + 8]);
                    const auto xdw30 = _mm_cvtepu8_epi32(*(const __m128i*)&src.data[((r + 2) * nCol + (c - 1)) * kNChannel + 0]);
                    const auto xdw31 = _mm_cvtepu8_epi32(*(const __m128i*)&src.data[((r + 2) * nCol + (c - 1)) * kNChannel + 4]);
                    const auto xdw32 = _mm_cvtepu8_epi32(*(const __m128i*)&src.data[((r + 2) * nCol + (c - 1)) * kNChannel + 8]);
                    const auto xf00 = _mm_cvtepi32_ps(xdw00); // 000 001 002 010
                    const auto xf01 = _mm_cvtepi32_ps(xdw01); // 011 012 020 021
                    const auto xf02 = _mm_cvtepi32_ps(xdw02); // 022 030 031 032
                    const auto xf10 = _mm_cvtepi32_ps(xdw10); // 100 101 102 110
                    const auto xf11 = _mm_cvtepi32_ps(xdw11); // 111 112 120 121
                    const auto xf12 = _mm_cvtepi32_ps(xdw12); // 122 130 131 132
                    const auto xf20 = _mm_cvtepi32_ps(xdw20); // 200 201 202 210
                    const auto xf21 = _mm_cvtepi32_ps(xdw21); // 211 212 220 221
                    const auto xf22 = _mm_cvtepi32_ps(xdw22); // 222 230 231 232
                    const auto xf30 = _mm_cvtepi32_ps(xdw30); // 300 301 302 310
                    const auto xf31 = _mm_cvtepi32_ps(xdw31); // 311 312 320 321
                    const auto xf32 = _mm_cvtepi32_ps(xdw32); // 322 330 331 332

                    #define MAKE_SW_(i, j, _2, _1, _0) \
                        const auto xsw##i##j##0 = _mm_permute_ps(xraw##i##j, _MM_SHUFFLE(_0, _2, _1, _0)); /* 00 01 02 00 */ \
                        const auto xsw##i##j##1 = _mm_permute_ps(xraw##i##j, _MM_SHUFFLE(_1, _0, _2, _1)); /* 01 02 00 01 */ \
                        const auto xsw##i##j##2 = _mm_permute_ps(xraw##i##j, _MM_SHUFFLE(_2, _1, _0, _2)); /* 02 00 01 02 */ \
                        _mm_store_ps(&in012[i][j][ 0], xsw##i##j##0); \
                        _mm_store_ps(&in012[i][j][ 4], xsw##i##j##1); \
                        _mm_store_ps(&in012[i][j][ 8], xsw##i##j##2); \
                        _mm_store_ps(&in012[i][j][12], xsw##i##j##0);
                    #define MAKE_SW(i) \
                        const auto xraw##i##0 = xf##i##0;                                 /* 00 01 02 ?? */ \
                        const auto xraw##i##1 = _mm_blend_ps(xf##i##0, xf##i##1, 0b0011); /* 11 12 ?? 10 */ \
                        const auto xraw##i##2 = _mm_blend_ps(xf##i##1, xf##i##2, 0b0011); /* 22 ?? 20 21 */ \
                        const auto xraw##i##3 = xf##i##2;                                 /* ?? 30 31 32 */ \
                        MAKE_SW_(i, 0, 2, 1, 0) \
                        MAKE_SW_(i, 1, 1, 0, 3) \
                        MAKE_SW_(i, 2, 0, 3, 2) \
                        MAKE_SW_(i, 3, 3, 2, 1)

                    MAKE_SW(0);
                    MAKE_SW(1);
                    MAKE_SW(2);
                    MAKE_SW(3);

                    #undef MAKE_SW_
                    #undef MAKE_SW
                #else
                    alignas(32) float in[4][4][kNChannel];
                    for (auto i = 0; i < 4; ++i)
                        for (auto j = 0; j < 4; ++j)
                            for (auto ch = 0; ch < kNChannel; ++ch)
                                in[i][j][ch] = src.data[((r + i - 1) * nCol + (c + j - 1)) * kNChannel + ch];
                    for (auto i = 0; i < 4; ++i)
                        for (auto j = 0; j < 4; ++j)
                            for (auto ic = 0; ic < kRatio; ++ic)
                                for (auto ch = 0; ch < kNChannel; ++ch)
                                    in012[i][j][ic * kNChannel + ch] = in[i][j][ch];
                #endif
            #else
                for (int i = 0; i < 4; ++i)
                    for (auto j = 0; j < 4; ++j)
                        for (auto ic = 0; ic < kRatio; ++ic)
                            for (auto ch = 0; ch < kNChannel; ++ch)
                                in012[i][j][ic * kNChannel + ch] = src.data[((r + i - 1) * nCol + (c + j - 1)) * kNChannel + ch];
            #endif
            }


        #if SIMPLIFY_START
            for (auto ir = 0; ir < kRatio; ++ir)
        #else
            for (auto ir = r == 1 ? 1 : 0; ir < kRatio; ++ir)
        #endif
            {
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
