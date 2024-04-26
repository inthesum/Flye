//(c) 2016 by Authors
//This file is a part of ABruijn program.
//Released under the BSD license (see LICENSE file)

#include "alignment_avx.h"
#include <immintrin.h> // Include SIMD intrinsics header

constexpr size_t batchSize = 4; // Use constexpr for batch size

AlignmentAVX::AlignmentAVX(size_t size, const SubstitutionMatrix& sm, const std::vector <std::string> &reads):
    _forwardScores(size / batchSize),
    _reverseScores(size / batchSize),
    _subsMatrix(sm),
    _subsScoresA(size / batchSize),
    _subsScoresC(size / batchSize),
    _subsScoresG(size / batchSize),
    _subsScoresT(size / batchSize)
{
    const size_t extendedK = reads.size();
    _readsSize = new AlnScoreType [extendedK];
    for (size_t k = 0; k < extendedK; k += batchSize)
    {
        size_t x = reads[k+batchSize-1].size();
        size_t y = batchSize;
        ScoreMatrix subsScoresA(x, y, 0);
        ScoreMatrix subsScoresC(x, y, 0);
        ScoreMatrix subsScoresG(x, y, 0);
        ScoreMatrix subsScoresT(x, y, 0);

        for(size_t b = 0; b < batchSize; b++) {
            const std::string w = reads[k + b];
            for (size_t i = 0; i < w.size(); i++) {
                subsScoresA.at(i, b) = _subsMatrix.getScore('A', w[i]);
                subsScoresC.at(i, b) = _subsMatrix.getScore('C', w[i]);
                subsScoresG.at(i, b) = _subsMatrix.getScore('G', w[i]);
                subsScoresT.at(i, b) = _subsMatrix.getScore('T', w[i]);
            }
        }

        _subsScoresA[k / batchSize] = std::move(subsScoresA);
        _subsScoresC[k / batchSize] = std::move(subsScoresC);
        _subsScoresG[k / batchSize] = std::move(subsScoresG);
        _subsScoresT[k / batchSize] = std::move(subsScoresT);
    }
}

AlignmentAVX::~AlignmentAVX() {
    delete[] _readsSize;
}


__m256i mm256_max_epi64(__m256i a, __m256i b) {
    // Compare a and b to get a mask of elements where a > b
    __m256i cmp_mask = _mm256_cmpgt_epi64(a, b);

    // Use the mask to select the maximum value from a and b
    __m256i result = _mm256_blendv_epi8(b, a, cmp_mask);

    return result;
}


AlnScoreType AlignmentAVX::globalAlignmentAVX(const std::string& consensus,
                                              const std::vector<std::string>& reads,
                                              const size_t readsNum)
{
    AlnScoreType finalScore = 0;
    const size_t K = readsNum;
    const size_t extendedK = reads.size();
    for (size_t k = 0; k < extendedK; k += batchSize)
    {
        size_t x = consensus.size() + 1;
        size_t y = reads[k+batchSize-1].size() + 1;
        size_t z = batchSize;
        ScoreMatrix3d scoreMatrix(x, y, z, 0);
        ScoreMatrix leftSubsMatrix(y - 1, z, 0);
        ScoreMatrix3d crossSubsMatrix(x - 1, y - 1, z, 0);

        const std::string v = consensus;
        alignas(32) AlnScoreType cols[batchSize];
        for(size_t b = 0; b < batchSize; b++) {
            const std::string w = reads[k + b];
            cols[b] = w.size();
            _readsSize[k + b] = w.size();

            for (size_t i = 0; i < v.size(); i++)
            {
                AlnScoreType s = _subsMatrix.getScore(v[i], '-');
                scoreMatrix.at(i+1, 0, b) = scoreMatrix.at(i, 0, b) + s;
            }

            for (size_t i = 0; i < w.size(); i++) {
                AlnScoreType s = _subsMatrix.getScore('-', w[i]);
                leftSubsMatrix.at(i, b) = s;
                scoreMatrix.at(0, i+1, b) = scoreMatrix.at(0, i, b) + s;
            }

            for (size_t i = 0; i < v.size(); i++)
                for (size_t j = 0; j < w.size(); j++)
                    crossSubsMatrix.at(i, j, b) = _subsMatrix.getScore(v[i], w[j]);
        }

        size_t leftScoreIndex = 1 * y * z;
        size_t crossScoreIndex = 0;
        size_t crossSubScoreIndex = 0;

        __m256i score = _mm256_set1_epi64x(0);
        __m256i _cols = _mm256_load_si256((__m256i*) cols);
        for (size_t i = 1; i < x; i++, leftScoreIndex += z, crossScoreIndex += z)
        {
            size_t leftSubScoreIndex = 0;
            __m256i upSubScore = _mm256_set1_epi64x(_subsMatrix.getScore(v[i - 1], '-'));

            for (size_t j = 1; j < y; j++, leftScoreIndex += z, crossScoreIndex += z,
                                           leftSubScoreIndex += z, crossSubScoreIndex += z)
            {
                __m256i leftScore = _mm256_loadu_si256((__m256i*)(scoreMatrix.data() + leftScoreIndex));
                __m256i leftSubScore = _mm256_loadu_si256((__m256i*)(leftSubsMatrix.data() + leftSubScoreIndex));
                __m256i left = _mm256_add_epi64(leftScore, leftSubScore);

                size_t upScoreIndex = crossScoreIndex + z;
                __m256i upScore = _mm256_loadu_si256((__m256i*)(scoreMatrix.data() + upScoreIndex));
                __m256i up = _mm256_add_epi64(upScore, upSubScore);

                __m256i crossScore = _mm256_loadu_si256((__m256i*)(scoreMatrix.data() + crossScoreIndex));
                __m256i crossSubScore = _mm256_loadu_si256((__m256i*)(crossSubsMatrix.data() + crossSubScoreIndex));
                __m256i cross = _mm256_add_epi64(crossScore, crossSubScore);

                __m256i prevScore = score;
                score = mm256_max_epi64(left, up);
                score = mm256_max_epi64(score, cross);
                __m256i _j = _mm256_set1_epi64x(j);
                __m256i cmp_mask = _mm256_cmpgt_epi64(_j, _cols);
                score = _mm256_blendv_epi8(score, prevScore, cmp_mask);

                _mm256_storeu_si256((__m256i*)(scoreMatrix.data() + leftScoreIndex + z), score);
            }
        }

        alignas(32) AlnScoreType scores[batchSize];
        _mm256_store_si256((__m256i*)scores, score);

        _forwardScores[k / batchSize] = std::move(scoreMatrix);

        for(size_t b = 0; b < batchSize; b++)
            if((k + b) < K)
                finalScore += scores[b];
    }


    for (size_t k = 0; k < extendedK; k += batchSize)
    {
        size_t x = consensus.size() + 1;
        size_t y = reads[k+batchSize-1].size() + 1;
        size_t z = batchSize;
        ScoreMatrix3d scoreMatrix(x, y, z, 0);
        ScoreMatrix leftSubsMatrix(y - 1, z, 0);
        ScoreMatrix3d crossSubsMatrix(x - 1, y - 1, z, 0);

        const std::string v = consensus;
        alignas(32) AlnScoreType cols[batchSize];
        for(size_t b = 0; b < batchSize; b++) {
            const std::string w = reads[k + b];
            cols[b] = w.size();

            for (int i = v.size() - 1; i >= 0; i--)
            {
                AlnScoreType s = _subsMatrix.getScore(v[i], '-');
                scoreMatrix.at(i, w.size(), b) = scoreMatrix.at(i + 1, w.size(), b) + s;
            }

            for (int i = w.size() - 1; i >= 0 ; i--) {
                AlnScoreType s = _subsMatrix.getScore('-', w[i]);
                leftSubsMatrix.at(i, b) = s;
                scoreMatrix.at(v.size(), i, b) = scoreMatrix.at(v.size(), i + 1, b) + s;
            }

            for (size_t i = 0; i < v.size(); i++)
                for (size_t j = 0; j < w.size(); j++)
                    crossSubsMatrix.at(i, j, b) = _subsMatrix.getScore(v[i], w[j]);
        }

        size_t rightScoreIndex = (x - 2) * y * z + (y - 1) * z; // (v.size() - 1, w.size())
        size_t crossScoreIndex = (x - 1) * y * z + (y - 1) * z; // (v.size(), w.size())
        size_t crossSubScoreIndex = (x - 2) * (y - 1) * z + (y - 2) * z; // (v.size() - 1, w.size() - 1)

        __m256i score = _mm256_set1_epi64x(0);
        __m256i _cols = _mm256_load_si256((__m256i*) cols);
        for (size_t i = x - 1; i >= 1; i--, rightScoreIndex -= z, crossScoreIndex -= z)
        {
            size_t rightSubScoreIndex = (y - 2) * z;
            __m256i downSubScore = _mm256_set1_epi64x(_subsMatrix.getScore(v[i - 1], '-'));

            for (size_t j = y - 1; j >= 1; j--, rightScoreIndex -= z, crossScoreIndex -= z,
                                                rightSubScoreIndex -= z, crossSubScoreIndex -= z)
            {
                __m256i rightScore = _mm256_loadu_si256((__m256i*)(scoreMatrix.data() + rightScoreIndex));
                __m256i rightSubScore = _mm256_loadu_si256((__m256i*)(leftSubsMatrix.data() + rightSubScoreIndex));
                __m256i right = _mm256_add_epi64(rightScore, rightSubScore);

                size_t downScoreIndex = crossScoreIndex - z;
                __m256i downScore = _mm256_loadu_si256((__m256i*)(scoreMatrix.data() + downScoreIndex));
                __m256i down = _mm256_add_epi64(downScore, downSubScore);

                __m256i crossScore = _mm256_loadu_si256((__m256i*)(scoreMatrix.data() + crossScoreIndex));
                __m256i crossSubScore = _mm256_loadu_si256((__m256i*)(crossSubsMatrix.data() + crossSubScoreIndex));
                __m256i cross = _mm256_add_epi64(crossScore, crossSubScore);

                __m256i prevScore = _mm256_loadu_si256((__m256i*)(scoreMatrix.data() + rightScoreIndex - z));
                score = mm256_max_epi64(right, down);
                score = mm256_max_epi64(score, cross);
                __m256i _j = _mm256_set1_epi64x(j);
                __m256i cmp_mask = _mm256_cmpgt_epi64(_j, _cols);
                score = _mm256_blendv_epi8(score, prevScore, cmp_mask);

                _mm256_storeu_si256((__m256i*)(scoreMatrix.data() + rightScoreIndex - z), score);
            }
        }

        _reverseScores[k / batchSize] = std::move(scoreMatrix);
    }

    return finalScore;
}


AlnScoreType AlignmentAVX::addDeletionAVX(unsigned int letterIndex, const size_t readsNum) const
{
    const size_t K = readsNum;
    const size_t extendedK = _forwardScores.size() * batchSize;

    size_t frontRow = letterIndex - 1;
    size_t revRow = letterIndex;

    AlnScoreType finalScore = 0;
    for (size_t k = 0; k < extendedK; k += batchSize) {
        const ScoreMatrix3d& forwardScores = _forwardScores[k / batchSize];
        const ScoreMatrix3d& reverseScores = _reverseScores[k / batchSize];

        size_t y = forwardScores.ncols();
        size_t z = batchSize;

        size_t frontIndex = frontRow * y * z; // (frontRow, 0)
        size_t reverseIndex = revRow * y * z; // (revRow, 0)

        AlnScoreType maxVal = std::numeric_limits<AlnScoreType>::lowest();
        __m256i _maxVal = _mm256_set1_epi64x(maxVal);
        __m256i _cols = _mm256_loadu_si256((__m256i*)(_readsSize + k));
        for (size_t col = 0; col < y; ++col, frontIndex += z, reverseIndex += z)
        {
            __m256i forwardScore = _mm256_loadu_si256((__m256i*)(forwardScores.data() + frontIndex));
            __m256i reverseScore = _mm256_loadu_si256((__m256i*)(reverseScores.data() + reverseIndex));
            __m256i sum = _mm256_add_epi64(forwardScore, reverseScore);

            __m256i _preMaxVal = _maxVal;
            __m256i _col = _mm256_set1_epi64x(col);
            __m256i cmp_mask = _mm256_cmpgt_epi64(_col, _cols);
            _maxVal = mm256_max_epi64(_maxVal, sum);
            _maxVal = _mm256_blendv_epi8(_maxVal, _preMaxVal, cmp_mask);
        }

        alignas(32) AlnScoreType scores[batchSize];
        _mm256_store_si256((__m256i*)scores, _maxVal);

        for(size_t b = 0; b < batchSize; b++)
            if((k + b) < K)
                finalScore += scores[b];
    }

    return finalScore;
}


AlnScoreType AlignmentAVX::addSubstitutionAVX(unsigned int letterIndex, char base,
                                           const std::vector<std::string>& reads,
                                           const size_t readsNum) const
{
    std::vector<ScoreMatrix> _subsScores;
    switch (base) {
        case 'A':
            _subsScores = _subsScoresA;
            break;
        case 'C':
            _subsScores = _subsScoresC;
            break;
        case 'G':
            _subsScores = _subsScoresG;
            break;
        case 'T':
            _subsScores = _subsScoresT;
            break;
        default:
            std::cout << "Wrong base!" << std::endl;
    }

    const size_t K = readsNum;
    const size_t extendedK = reads.size();

    size_t frontRow = letterIndex - 1;
    size_t revRow = letterIndex;

    AlnScoreType finalScore = 0;
    AlnScoreType baseScoreWithGap = _subsMatrix.getScore(base, '-');
    __m256i _baseScoreWithGap = _mm256_set1_epi64x(baseScoreWithGap);
    for (size_t k = 0; k < extendedK; k += batchSize) {
        const ScoreMatrix3d& forwardScores = _forwardScores[k / batchSize];
        const ScoreMatrix3d& reverseScores = _reverseScores[k / batchSize];
        const ScoreMatrix& subsScores = _subsScores[k / batchSize];

        size_t y = forwardScores.ncols();
        size_t z = batchSize;

        size_t frontIndex = frontRow * y * z; // (frontRow, 0)
        size_t reverseIndex = revRow * y * z; // (revRow, 0)
        size_t subIndex = 0;

        __m256i _cols = _mm256_loadu_si256((__m256i*)(_readsSize + k));
        __m256i _one = _mm256_set1_epi64x(-1);
        _cols = _mm256_add_epi64(_cols, _one);

        __m256i forwardScore = _mm256_loadu_si256((__m256i*)(forwardScores.data() + frontIndex));
        __m256i reverseScore = _mm256_loadu_si256((__m256i*)(reverseScores.data() + reverseIndex));
        __m256i _maxVal = _mm256_add_epi64(forwardScore, reverseScore);
        _maxVal = _mm256_add_epi64(_maxVal, _baseScoreWithGap);
        for (size_t col = 0; col < y - 1; ++col, subIndex += z, frontIndex += z, reverseIndex += z)
        {
            __m256i subScore = _mm256_loadu_si256((__m256i*)(subsScores.data() + subIndex));
            __m256i forwardScoreCurrent = _mm256_loadu_si256((__m256i*)(forwardScores.data() + frontIndex));
            __m256i forwardScoreNext = _mm256_loadu_si256((__m256i*)(forwardScores.data() + frontIndex + z));
            __m256i reverseScoreNext = _mm256_loadu_si256((__m256i*)(reverseScores.data() + reverseIndex + z));

            __m256i matchScore = _mm256_add_epi64(forwardScoreCurrent, subScore);
            __m256i insertScore = _mm256_add_epi64(forwardScoreNext, _baseScoreWithGap);
            __m256i tempMaxScore = mm256_max_epi64(matchScore, insertScore);
            __m256i sum = _mm256_add_epi64(reverseScoreNext, tempMaxScore);

            __m256i _preMaxVal = _maxVal;
            __m256i _col = _mm256_set1_epi64x(col);
            __m256i cmp_mask = _mm256_cmpgt_epi64(_col, _cols);
            _maxVal = mm256_max_epi64(_maxVal, sum);
            _maxVal = _mm256_blendv_epi8(_maxVal, _preMaxVal, cmp_mask);
        }

        alignas(32) AlnScoreType scores[batchSize];
        _mm256_store_si256((__m256i*)scores, _maxVal);

        for(size_t b = 0; b < batchSize; b++)
            if((k + b) < K)
                finalScore += scores[b];
    }

    return finalScore;
}


AlnScoreType AlignmentAVX::addInsertionAVX(unsigned int pos, char base,
                                        const std::vector<std::string>& reads,
                                        const size_t readsNum) const
{
    std::vector<ScoreMatrix> _subsScores;
    switch (base) {
        case 'A':
            _subsScores = _subsScoresA;
            break;
        case 'C':
            _subsScores = _subsScoresC;
            break;
        case 'G':
            _subsScores = _subsScoresG;
            break;
        case 'T':
            _subsScores = _subsScoresT;
            break;
        default:
            std::cout << "Wrong base!" << std::endl;
    }

    const size_t K = readsNum;
    const size_t extendedK = reads.size();

    size_t frontRow = pos - 1;
    size_t revRow = pos - 1;

    AlnScoreType finalScore = 0;
    AlnScoreType baseScoreWithGap = _subsMatrix.getScore(base, '-');

    __m256i _baseScoreWithGap = _mm256_set1_epi64x(baseScoreWithGap);
    for (size_t k = 0; k < extendedK; k += batchSize) {
        const ScoreMatrix3d& forwardScores = _forwardScores[k / 4];
        const ScoreMatrix3d& reverseScores = _reverseScores[k / 4];
        const ScoreMatrix& subsScores = _subsScores[k / 4];

        size_t y = forwardScores.ncols();
        size_t z = batchSize;

        size_t frontIndex = frontRow * y * z; // (frontRow, 0)
        size_t reverseIndex = revRow * y * z; // (revRow, 0)
        size_t subIndex = 0;

        __m256i _cols = _mm256_loadu_si256((__m256i*)(_readsSize + k));
        __m256i _one = _mm256_set1_epi64x(-1);
        _cols = _mm256_add_epi64(_cols, _one);

        __m256i forwardScore = _mm256_loadu_si256((__m256i*)(forwardScores.data() + frontIndex));
        __m256i reverseScore = _mm256_loadu_si256((__m256i*)(reverseScores.data() + reverseIndex));
        __m256i _maxVal = _mm256_add_epi64(forwardScore, reverseScore);
        _maxVal = _mm256_add_epi64(_maxVal, _baseScoreWithGap);
        for (size_t col = 0; col < y - 1; ++col, subIndex += z, frontIndex += z, reverseIndex += z)
        {
            __m256i subScore = _mm256_loadu_si256((__m256i*)(subsScores.data() + subIndex));
            __m256i forwardScoreCurrent = _mm256_loadu_si256((__m256i*)(forwardScores.data() + frontIndex));
            __m256i forwardScoreNext = _mm256_loadu_si256((__m256i*)(forwardScores.data() + frontIndex + z));
            __m256i reverseScoreNext = _mm256_loadu_si256((__m256i*)(reverseScores.data() + reverseIndex + z));

            __m256i matchScore = _mm256_add_epi64(forwardScoreCurrent, subScore);
            __m256i insertScore = _mm256_add_epi64(forwardScoreNext, _baseScoreWithGap);
            __m256i maxScore = mm256_max_epi64(matchScore, insertScore);
            __m256i sum = _mm256_add_epi64(reverseScoreNext, maxScore);

            __m256i _preMaxVal = _maxVal;
            __m256i _col = _mm256_set1_epi64x(col);
            __m256i cmp_mask = _mm256_cmpgt_epi64(_col, _cols);
            _maxVal = mm256_max_epi64(_maxVal, sum);
            _maxVal = _mm256_blendv_epi8(_maxVal, _preMaxVal, cmp_mask);
        }

        alignas(32) AlnScoreType scores[batchSize];
        _mm256_store_si256((__m256i*)scores, _maxVal);

        for(size_t b = 0; b < batchSize; b++)
            if((k + b) < K)
                finalScore += scores[b];
    }

    return finalScore;
}