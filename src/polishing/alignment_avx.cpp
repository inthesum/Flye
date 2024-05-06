//(c) 2016 by Authors
//This file is a part of ABruijn program.
//Released under the BSD license (see LICENSE file)

#include "alignment_avx.h"

constexpr size_t batchSize = 8; // Use constexpr for batch size

AlignmentAVX::AlignmentAVX(size_t size, const SubstitutionMatrix& sm, const std::vector <std::string> &reads):
    _subsMatrix(sm),
    _forwardScores(size/batchSize),
    _reverseScores(size/batchSize),
    _subsScoresA(size/batchSize),
    _subsScoresC(size/batchSize),
    _subsScoresG(size/batchSize),
    _subsScoresT(size/batchSize),
    _subsScores_(size/batchSize)
{
    const size_t extendedReads = reads.size();
    _readsSize = (AlnScoreType*)_mm_malloc(extendedReads * sizeof(AlnScoreType), 64);

    for (size_t readId = 0; readId < extendedReads; readId += batchSize)
    {
        size_t x = reads[readId + batchSize - 1].size();
        size_t y = batchSize;
        ScoreMatrix subsScoresA(x, y);
        ScoreMatrix subsScoresC(x, y);
        ScoreMatrix subsScoresG(x, y);
        ScoreMatrix subsScoresT(x, y);
        ScoreMatrix subsScores_(x, y);

        for(size_t b = 0; b < batchSize; b++) {
            const std::string w = reads[readId + b];
            for (size_t i = 0; i < w.size(); i++) {
                subsScoresA.at(i, b) = _subsMatrix.getScore('A', w[i]);
                subsScoresC.at(i, b) = _subsMatrix.getScore('C', w[i]);
                subsScoresG.at(i, b) = _subsMatrix.getScore('G', w[i]);
                subsScoresT.at(i, b) = _subsMatrix.getScore('T', w[i]);
                subsScores_.at(i, b) = _subsMatrix.getScore('-', w[i]);
            }
        }

        _subsScoresA[readId/batchSize] = std::move(subsScoresA);
        _subsScoresC[readId/batchSize] = std::move(subsScoresC);
        _subsScoresG[readId/batchSize] = std::move(subsScoresG);
        _subsScoresT[readId/batchSize] = std::move(subsScoresT);
        _subsScores_[readId/batchSize] = std::move(subsScores_);
    }
}

AlignmentAVX::~AlignmentAVX() {
    _mm_free(_readsSize);
}


AlnScoreType AlignmentAVX::globalAlignmentAVX(const std::string& consensus,
                                              const std::vector<std::string>& reads,
                                              const size_t readsNum)
//AlnScoreType AlignmentAVX::globalAlignmentAVX(const std::string& consensus,
//                                              const std::vector<std::string>& reads,
//                                              const size_t readsNum,
//                                              ScoreMemoryPool& memoryPool)
//AlnScoreType AlignmentAVX::globalAlignmentAVX(const std::string& consensus,
//                                              const std::vector<std::string>& reads,
//                                              const size_t readsNum,
////                                              ScoreMemoryPool& memoryPool,
//                                              std::chrono::duration<double>& alignmentDuration)
{
    AlnScoreType finalScore = 0;
    const size_t extendedReads = reads.size();

    // getScoringMatrix
    for (size_t readId = 0; readId < extendedReads; readId += batchSize)
    {
        size_t x = consensus.size() + 1;
        size_t y = reads[readId + batchSize - 1].size() + 1;
        size_t z = batchSize;

        ScoreMatrix3d scoreMatrix(x, y, z);
//        AlnScoreType* ptr = memoryPool.allocate(x * y * z);
//        ScoreMatrix3d scoreMatrix(ptr, x, y, z);

        const ScoreMatrix& leftSubsMatrix = _subsScores_[readId/batchSize];
        const ScoreMatrix& crossSubsMatrixA = _subsScoresA[readId/batchSize];
        const ScoreMatrix& crossSubsMatrixC = _subsScoresC[readId/batchSize];
        const ScoreMatrix& crossSubsMatrixG = _subsScoresG[readId/batchSize];
        const ScoreMatrix& crossSubsMatrixT = _subsScoresT[readId/batchSize];

        const std::string v = consensus;

        for(size_t b = 0; b < batchSize; b++) {
            const std::string w = reads[readId + b];
            _readsSize[readId + b] = w.size();

            scoreMatrix.at(0, 0, b) = 0;

            for (size_t i = 0; i < v.size(); i++)
                scoreMatrix.at(i+1, 0, b) = scoreMatrix.at(i, 0, b) + _subsMatrix.getScore(v[i], '-');

            for (size_t i = 0; i < w.size(); i++)
                scoreMatrix.at(0, i+1, b) = scoreMatrix.at(0, i, b) + _subsMatrix.getScore('-', w[i]);
        }

        size_t leftScoreIndex = 1 * y * z;
        size_t crossScoreIndex = 0;

        __m512i score = _mm512_set1_epi64(0);
        __m512i _cols = _mm512_load_si512((__m512i*)(_readsSize + readId));

        for (size_t i = 1; i < x; i++, leftScoreIndex += z, crossScoreIndex += z)
        {
            size_t leftSubScoreIndex = 0;
            __m512i upSubScore = _mm512_set1_epi64(_subsMatrix.getScore(v[i - 1], '-'));

            size_t crossSubScoreIndex = 0;
            const ScoreMatrix* crossSubsMatrixPtr;
            switch (v[i - 1]) {
                case 'A':
                    crossSubsMatrixPtr = &crossSubsMatrixA;
                    break;
                case 'C':
                    crossSubsMatrixPtr = &crossSubsMatrixC;
                    break;
                case 'G':
                    crossSubsMatrixPtr = &crossSubsMatrixG;
                    break;
                case 'T':
                    crossSubsMatrixPtr = &crossSubsMatrixT;
                    break;
                default:
                    std::cout << "Wrong base!" << std::endl;
            }
            const ScoreMatrix& crossSubsMatrix = *crossSubsMatrixPtr;

//            auto alignmentStart = std::chrono::high_resolution_clock::now();

            for (size_t j = 1; j < _readsSize[readId]; j++, leftScoreIndex += z, crossScoreIndex += z,
                                           leftSubScoreIndex += z, crossSubScoreIndex += z)
            {
                __m512i leftScore = _mm512_load_si512((__m512i*)(scoreMatrix.data() + leftScoreIndex));
                __m512i leftSubScore = _mm512_load_si512((__m512i*)(leftSubsMatrix.data() + leftSubScoreIndex));
                __m512i left = _mm512_add_epi64(leftScore, leftSubScore);

                __m512i upScore = _mm512_load_si512((__m512i*)(scoreMatrix.data() + crossScoreIndex + z));
                __m512i up = _mm512_add_epi64(upScore, upSubScore);

                __m512i crossScore = _mm512_load_si512((__m512i*)(scoreMatrix.data() + crossScoreIndex));
                __m512i crossSubScore = _mm512_load_si512((__m512i*)(crossSubsMatrix.data() + crossSubScoreIndex));
                __m512i cross = _mm512_add_epi64(crossScore, crossSubScore);

                score = _mm512_max_epi64(left, up);
                score = _mm512_max_epi64(score, cross);

                _mm512_store_si512((__m512i*)(scoreMatrix.data() + leftScoreIndex + z), score);
            }

            // Deal with various reads' length
            // _readsSize[readId] -> _readsSize[readId + batchSize - 1]
            for (size_t j = _readsSize[readId]; j < y; j++, leftScoreIndex += z, crossScoreIndex += z,
                                           leftSubScoreIndex += z, crossSubScoreIndex += z)
            {
                __m512i leftScore = _mm512_load_si512((__m512i*)(scoreMatrix.data() + leftScoreIndex));
                __m512i leftSubScore = _mm512_load_si512((__m512i*)(leftSubsMatrix.data() + leftSubScoreIndex));
                __m512i left = _mm512_add_epi64(leftScore, leftSubScore);

                __m512i upScore = _mm512_load_si512((__m512i*)(scoreMatrix.data() + crossScoreIndex + z));
                __m512i up = _mm512_add_epi64(upScore, upSubScore);

                __m512i crossScore = _mm512_load_si512((__m512i*)(scoreMatrix.data() + crossScoreIndex));
                __m512i crossSubScore = _mm512_load_si512((__m512i*)(crossSubsMatrix.data() + crossSubScoreIndex));
                __m512i cross = _mm512_add_epi64(crossScore, crossSubScore);

                __m512i prevScore = score;
                score = _mm512_max_epi64(left, up);
                score = _mm512_max_epi64(score, cross);
                __m512i _j = _mm512_set1_epi64(j);
                __mmask8 cmp_mask = _mm512_cmpgt_epi64_mask(_j, _cols);
                score = _mm512_mask_blend_epi64(cmp_mask, score, prevScore);

                _mm512_store_si512((__m512i*)(scoreMatrix.data() + leftScoreIndex + z), score);
            }

//            auto alignmentEnd = std::chrono::high_resolution_clock::now();
//            alignmentDuration += alignmentEnd - alignmentStart;
        }

        alignas(64) AlnScoreType scores[batchSize];
//        AlnScoreType scores[batchSize] __attribute((aligned(64)));
        _mm512_store_si512((__m512i*)scores, score);

        _forwardScores[readId / batchSize] = std::move(scoreMatrix);

        for(size_t b = 0; b < batchSize; b++)
            if((readId + b) < readsNum)
                finalScore += scores[b];
    }


    // getRevScoringMatrix
    for (size_t readId = 0; readId < extendedReads; readId += batchSize)
    {
        size_t x = consensus.size() + 1;
        size_t y = reads[readId+batchSize-1].size() + 1;
        size_t z = batchSize;

        ScoreMatrix3d scoreMatrix(x, y, z);
//        AlnScoreType* ptr = memoryPool.allocate(x * y * z);
//        ScoreMatrix3d scoreMatrix(ptr, x, y, z);

        const ScoreMatrix& leftSubsMatrix = _subsScores_[readId/batchSize];
        const ScoreMatrix& crossSubsMatrixA = _subsScoresA[readId/batchSize];
        const ScoreMatrix& crossSubsMatrixC = _subsScoresC[readId/batchSize];
        const ScoreMatrix& crossSubsMatrixG = _subsScoresG[readId/batchSize];
        const ScoreMatrix& crossSubsMatrixT = _subsScoresT[readId/batchSize];

        const std::string v = consensus;

        for(size_t b = 0; b < batchSize; b++) {
            const std::string w = reads[readId + b];

            scoreMatrix.at(v.size(), w.size(), b) = 0;

            for (int i = v.size() - 1; i >= 0; i--)
                scoreMatrix.at(i, w.size(), b) = scoreMatrix.at(i + 1, w.size(), b) + _subsMatrix.getScore(v[i], '-');

            for (int i = w.size() - 1; i >= 0 ; i--)
                scoreMatrix.at(v.size(), i, b) = scoreMatrix.at(v.size(), i + 1, b) + _subsMatrix.getScore('-', w[i]);
        }

        size_t rightScoreIndex = (x - 2) * y * z + (y - 1) * z; // (v.size() - 1, w.size())
        size_t crossScoreIndex = (x - 1) * y * z + (y - 1) * z; // (v.size(), w.size())

        __m512i score;
        __m512i _cols = _mm512_load_si512((__m512i*)(_readsSize + readId));

        for (size_t i = x - 1; i >= 1; i--, rightScoreIndex -= z, crossScoreIndex -= z)
        {

            size_t rightSubScoreIndex = (y - 2) * z;
            __m512i downSubScore = _mm512_set1_epi64(_subsMatrix.getScore(v[i - 1], '-'));

            size_t crossSubScoreIndex = (y - 2) * z; // (v.size() - 1, w.size() - 1)
            const ScoreMatrix* crossSubsMatrixPtr;
            switch (v[i - 1]) {
                case 'A':
                    crossSubsMatrixPtr = &crossSubsMatrixA;
                    break;
                case 'C':
                    crossSubsMatrixPtr = &crossSubsMatrixC;
                    break;
                case 'G':
                    crossSubsMatrixPtr = &crossSubsMatrixG;
                    break;
                case 'T':
                    crossSubsMatrixPtr = &crossSubsMatrixT;
                    break;
                default:
                    std::cout << "Wrong base!" << std::endl;
            }
            const ScoreMatrix& crossSubsMatrix = *crossSubsMatrixPtr;

//            auto alignmentStart = std::chrono::high_resolution_clock::now();

            // Deal with various reads' length
            // _readsSize[readId] -> _readsSize[readId + batchSize - 1]
            for (size_t j = y - 1; j >= _readsSize[readId]; j--, rightScoreIndex -= z, crossScoreIndex -= z,
                                                rightSubScoreIndex -= z, crossSubScoreIndex -= z)
            {
                __m512i rightScore = _mm512_load_si512((__m512i*)(scoreMatrix.data() + rightScoreIndex));
                __m512i rightSubScore = _mm512_load_si512((__m512i*)(leftSubsMatrix.data() + rightSubScoreIndex));
                __m512i right = _mm512_add_epi64(rightScore, rightSubScore);

                __m512i downScore = _mm512_load_si512((__m512i*)(scoreMatrix.data() + crossScoreIndex - z));
                __m512i down = _mm512_add_epi64(downScore, downSubScore);

                __m512i crossScore = _mm512_load_si512((__m512i*)(scoreMatrix.data() + crossScoreIndex));
                __m512i crossSubScore = _mm512_load_si512((__m512i*)(crossSubsMatrix.data() + crossSubScoreIndex));
                __m512i cross = _mm512_add_epi64(crossScore, crossSubScore);

                __m512i prevScore = _mm512_load_si512((__m512i*)(scoreMatrix.data() + rightScoreIndex - z));
                score = _mm512_max_epi64(right, down);
                score = _mm512_max_epi64(score, cross);
                __m512i _j = _mm512_set1_epi64(j);
                __mmask8 cmp_mask = _mm512_cmpgt_epi64_mask(_j, _cols);
                score = _mm512_mask_blend_epi64(cmp_mask, score, prevScore);

                _mm512_store_si512((__m512i*)(scoreMatrix.data() + rightScoreIndex - z), score);
            }

            for (size_t j = _readsSize[readId] - 1; j >= 1; j--, rightScoreIndex -= z, crossScoreIndex -= z,
                                                rightSubScoreIndex -= z, crossSubScoreIndex -= z)
            {
                __m512i rightScore = _mm512_load_si512((__m512i*)(scoreMatrix.data() + rightScoreIndex));
                __m512i rightSubScore = _mm512_load_si512((__m512i*)(leftSubsMatrix.data() + rightSubScoreIndex));
                __m512i right = _mm512_add_epi64(rightScore, rightSubScore);

                __m512i downScore = _mm512_load_si512((__m512i*)(scoreMatrix.data() + crossScoreIndex - z));
                __m512i down = _mm512_add_epi64(downScore, downSubScore);

                __m512i crossScore = _mm512_load_si512((__m512i*)(scoreMatrix.data() + crossScoreIndex));
                __m512i crossSubScore = _mm512_load_si512((__m512i*)(crossSubsMatrix.data() + crossSubScoreIndex));
                __m512i cross = _mm512_add_epi64(crossScore, crossSubScore);

                score = _mm512_max_epi64(right, down);
                score = _mm512_max_epi64(score, cross);

                _mm512_store_si512((__m512i*)(scoreMatrix.data() + rightScoreIndex - z), score);
            }

//            auto alignmentEnd = std::chrono::high_resolution_clock::now();
//            alignmentDuration += alignmentEnd - alignmentStart;
        }

        _reverseScores[readId / batchSize] = std::move(scoreMatrix);
    }

    return finalScore;
}


AlnScoreType AlignmentAVX::addDeletionAVX(unsigned int letterIndex, const size_t readsNum) const
//AlnScoreType AlignmentAVX::addDeletionAVX(unsigned int letterIndex,
//                                          const size_t readsNum,
//                                          std::chrono::duration<double>& deletionDuration) const
{
    const size_t extendedReadsNum = _forwardScores.size() * batchSize;

    size_t frontRow = letterIndex - 1;
    size_t revRow = letterIndex;

    AlnScoreType finalScore = 0;

    for (size_t readId = 0; readId < extendedReadsNum; readId += batchSize) {
        const ScoreMatrix3d& forwardScores = _forwardScores[readId / batchSize];
        const ScoreMatrix3d& reverseScores = _reverseScores[readId / batchSize];

        size_t y = forwardScores.ncols();
        size_t z = batchSize;

        size_t frontIndex = frontRow * y * z; // (frontRow, 0)
        size_t reverseIndex = revRow * y * z; // (revRow, 0)
        const AlnScoreType* frontPtr = forwardScores.data() + frontIndex;
        const AlnScoreType* reversePtr = reverseScores.data() + reverseIndex;

        AlnScoreType maxVal = std::numeric_limits<AlnScoreType>::lowest();
        __m512i _maxVal = _mm512_set1_epi64(maxVal);

//        auto deletionStart = std::chrono::high_resolution_clock::now();

        size_t shortestCol = _readsSize[readId];
        for (size_t col = 0; col < shortestCol; ++col)
        {
            __m512i forwardScore = _mm512_load_si512((__m512i*)(frontPtr + col * z));
            __m512i reverseScore = _mm512_load_si512((__m512i*)(reversePtr + col * z));
            __m512i sum = _mm512_add_epi64(forwardScore, reverseScore);
            _maxVal = _mm512_max_epi64(_maxVal, sum);
        }

//        auto deletionEnd = std::chrono::high_resolution_clock::now();
//        deletionDuration += deletionEnd - deletionStart;

        // Deal with various reads' length
        // _readsSize[readId] -> _readsSize[readId + batchSize - 1]
        __m512i _cols = _mm512_load_si512((__m512i*)(_readsSize + readId));
        for (size_t col = shortestCol; col < y; ++col)
        {
            __m512i forwardScore = _mm512_load_si512((__m512i*)(frontPtr + col * z));
            __m512i reverseScore = _mm512_load_si512((__m512i*)(reversePtr + col * z));
            __m512i sum = _mm512_add_epi64(forwardScore, reverseScore);

            __m512i _col = _mm512_set1_epi64(col);
            __mmask8 cmp_mask = _mm512_cmpgt_epi64_mask(_col, _cols);
            sum = _mm512_mask_blend_epi64(cmp_mask, sum, _maxVal);
            _maxVal = _mm512_max_epi64(_maxVal, sum);
        }

        alignas(64) AlnScoreType scores[batchSize];
//        AlnScoreType scores[batchSize] __attribute((aligned(64)));
        _mm512_store_si512((__m512i*)scores, _maxVal);

        for(size_t b = 0; b < batchSize; b++)
            if((readId + b) < readsNum)
                finalScore += scores[b];
    }

    return finalScore;
}


AlnScoreType AlignmentAVX::addSubstitutionAVX(unsigned int letterIndex, char base,
                                              const std::vector<std::string>& reads,
                                              const size_t readsNum) const
{
    size_t frontRow = letterIndex - 1;
    size_t revRow = letterIndex;

    return addSubsAndInsertAVX(frontRow, revRow, base, reads, readsNum);
}


AlnScoreType AlignmentAVX::addInsertionAVX(unsigned int pos, char base,
                                           const std::vector<std::string>& reads,
                                           const size_t readsNum) const
{
    size_t frontRow = pos - 1;
    size_t revRow = pos - 1;

    return addSubsAndInsertAVX(frontRow, revRow, base, reads, readsNum);
}


AlnScoreType AlignmentAVX::addSubsAndInsertAVX(size_t frontRow, size_t revRow,
                                 char base, const std::vector <std::string> &reads,
                                 const size_t readsNum) const
{
    const std::vector<ScoreMatrix>* _subsScoresPtr;
    switch (base) {
        case 'A':
            _subsScoresPtr = &_subsScoresA;
            break;
        case 'C':
            _subsScoresPtr = &_subsScoresC;
            break;
        case 'G':
            _subsScoresPtr = &_subsScoresG;
            break;
        case 'T':
            _subsScoresPtr = &_subsScoresT;
            break;
        default:
            std::cout << "Wrong base!" << std::endl;
            return -1;
    }
    const std::vector<ScoreMatrix>& _subsScores = *_subsScoresPtr;

    const size_t extendedReadsNum = reads.size();

    AlnScoreType finalScore = 0;

    AlnScoreType baseScoreWithGap = _subsMatrix.getScore(base, '-');
    __m512i _baseScoreWithGap = _mm512_set1_epi64(baseScoreWithGap);

    for (size_t readId = 0; readId < extendedReadsNum; readId += batchSize) {
        const ScoreMatrix3d& forwardScores = _forwardScores[readId / batchSize];
        const ScoreMatrix3d& reverseScores = _reverseScores[readId / batchSize];
        const ScoreMatrix& subsScores = _subsScores[readId / batchSize];

        size_t y = forwardScores.ncols();
        size_t z = batchSize;

        size_t frontIndex = frontRow * y * z; // (frontRow, 0)
        size_t reverseIndex = revRow * y * z; // (revRow, 0)
        size_t subIndex = 0;
        const AlnScoreType* frontPtr = forwardScores.data() + frontIndex;
        const AlnScoreType* reversePtr = reverseScores.data() + reverseIndex;
        const AlnScoreType* subPtr = subsScores.data() + subIndex;

        __m512i _cols = _mm512_load_si512((__m512i*)(_readsSize + readId));
        __m512i _one = _mm512_set1_epi64(-1);
        _cols = _mm512_add_epi64(_cols, _one);

        __m512i forwardScore = _mm512_load_si512((__m512i*) frontPtr);
        __m512i reverseScore = _mm512_load_si512((__m512i*) reversePtr);
        __m512i _maxVal = _mm512_add_epi64(forwardScore, reverseScore);
        _maxVal = _mm512_add_epi64(_maxVal, _baseScoreWithGap);

        for (size_t col = 0; col < _readsSize[readId]; ++col)
        {
            __m512i subScore = _mm512_load_si512((__m512i*)(subPtr + col * z));
            __m512i forwardScoreCurrent = _mm512_load_si512((__m512i*)(frontPtr + col * z));
            __m512i forwardScoreNext = _mm512_load_si512((__m512i*)(frontPtr + (col + 1) * z));
            __m512i reverseScoreNext = _mm512_load_si512((__m512i*)(reversePtr + (col + 1) * z));

            __m512i matchScore = _mm512_add_epi64(forwardScoreCurrent, subScore);
            __m512i insertScore = _mm512_add_epi64(forwardScoreNext, _baseScoreWithGap);
            __m512i tempMaxScore = _mm512_max_epi64(matchScore, insertScore);
            __m512i sum = _mm512_add_epi64(reverseScoreNext, tempMaxScore);
            _maxVal = _mm512_max_epi64(_maxVal, sum);
        }

        // Deal with various reads' length
        // _readsSize[readId] -> _readsSize[readId + batchSize - 1]
        for (size_t col = _readsSize[readId]; col < y - 1; ++col)
        {
            __m512i subScore = _mm512_load_si512((__m512i*)(subPtr + col * z));
            __m512i forwardScoreCurrent = _mm512_load_si512((__m512i*)(frontPtr + col * z));
            __m512i forwardScoreNext = _mm512_load_si512((__m512i*)(frontPtr + (col + 1) * z));
            __m512i reverseScoreNext = _mm512_load_si512((__m512i*)(reversePtr + (col + 1) * z));

            __m512i matchScore = _mm512_add_epi64(forwardScoreCurrent, subScore);
            __m512i insertScore = _mm512_add_epi64(forwardScoreNext, _baseScoreWithGap);
            __m512i tempMaxScore = _mm512_max_epi64(matchScore, insertScore);
            __m512i sum = _mm512_add_epi64(reverseScoreNext, tempMaxScore);

            __m512i _col = _mm512_set1_epi64(col);
            __mmask8 cmp_mask = _mm512_cmpgt_epi64_mask(_col, _cols);
            sum = _mm512_mask_blend_epi64(cmp_mask, sum, _maxVal);
            _maxVal = _mm512_max_epi64(_maxVal, sum);
        }

        alignas(64) AlnScoreType scores[batchSize];
//        AlnScoreType scores[batchSize] __attribute((aligned(64)));
        _mm512_store_si512((__m512i*)scores, _maxVal);

        for(size_t b = 0; b < batchSize; b++)
            if((readId + b) < readsNum)
                finalScore += scores[b];
    }

    return finalScore;
}