//(c) 2016 by Authors
//This file is a part of ABruijn program.
//Released under the BSD license (see LICENSE file)

#include "alignment_avx.h"

constexpr size_t batchSize = 4; // Use constexpr for batch size

AlignmentAVX::AlignmentAVX(size_t size, const SubstitutionMatrix& sm, const std::vector <std::string> &reads, ScoreMemoryPool& memoryPool):
    _subsMatrix(sm),
    batchNum(size/batchSize),
    memoryPool(memoryPool)
{
    _forwardScores.resize(batchNum);
    _reverseScores.resize(batchNum);
    _subsScoresA.resize(batchNum);
    _subsScoresC.resize(batchNum);
    _subsScoresG.resize(batchNum);
    _subsScoresT.resize(batchNum);
    _subsScores_.resize(batchNum);

    _readsSize = (AlnScoreType*)_mm_malloc(size * sizeof(AlnScoreType), 32);
//    _readsSize = memoryPool.allocate(size);

    for (size_t batchId = 0; batchId < batchNum; batchId++)
    {
        const size_t readId = batchId * batchSize;
        size_t x = reads[readId + batchSize - 1].size();
        size_t y = batchSize;

        ScoreMatrix subsScoresA(x, y);
        ScoreMatrix subsScoresC(x, y);
        ScoreMatrix subsScoresG(x, y);
        ScoreMatrix subsScoresT(x, y);
        ScoreMatrix subsScores_(x, y);

//        AlnScoreType* ptr = memoryPool.allocate(x * y);
//        ScoreMatrix subsScoresA(ptr, x, y);
//        ptr = memoryPool.allocate(x * y);
//        ScoreMatrix subsScoresC(ptr, x, y);
//        ptr = memoryPool.allocate(x * y);
//        ScoreMatrix subsScoresG(ptr, x, y);
//        ptr = memoryPool.allocate(x * y);
//        ScoreMatrix subsScoresT(ptr, x, y);
//        ptr = memoryPool.allocate(x * y);
//        ScoreMatrix subsScores_(ptr, x, y);

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

        _subsScoresA[batchId] = std::move(subsScoresA);
        _subsScoresC[batchId] = std::move(subsScoresC);
        _subsScoresG[batchId] = std::move(subsScoresG);
        _subsScoresT[batchId] = std::move(subsScoresT);
        _subsScores_[batchId] = std::move(subsScores_);
    }
}

AlignmentAVX::~AlignmentAVX() {
    _mm_free(_readsSize);
}





//AlnScoreType AlignmentAVX::globalAlignmentAVX(const std::string& consensus,
//                                              const std::vector<std::string>& reads,
//                                              const size_t readsNum)
AlnScoreType AlignmentAVX::globalAlignmentAVX(const std::string& consensus,
                                              const std::vector<std::string>& reads,
                                              const size_t readsNum)
{
    AlnScoreType finalScore = 0;

    // getScoringMatrix
    for (size_t batchId = 0; batchId < batchNum; batchId++)
    {
        const size_t readId = batchId * batchSize;
        size_t x = consensus.size() + 1;
        size_t y = reads[readId + batchSize - 1].size() + 1;
        size_t z = batchSize;

//        ScoreMatrix3d scoreMatrix(x, y, z);
        AlnScoreType* ptr = memoryPool.allocate(x * y * z);
        ScoreMatrix3d scoreMatrix(ptr, x, y, z);

        const ScoreMatrix& leftSubsMatrix = _subsScores_[batchId];
        const ScoreMatrix& crossSubsMatrixA = _subsScoresA[batchId];
        const ScoreMatrix& crossSubsMatrixC = _subsScoresC[batchId];
        const ScoreMatrix& crossSubsMatrixG = _subsScoresG[batchId];
        const ScoreMatrix& crossSubsMatrixT = _subsScoresT[batchId];

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

        __m256i score = _mm256_set1_epi64x(0);
        __m256i _cols = _mm256_load_si256((__m256i*)(_readsSize + readId));

        for (size_t i = 1; i < x; i++, leftScoreIndex += z, crossScoreIndex += z)
        {
            size_t leftSubScoreIndex = 0;
            __m256i upSubScore = _mm256_set1_epi64x(_subsMatrix.getScore(v[i - 1], '-'));

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
            const size_t shortestCol = _readsSize[readId];
            for (size_t j = 1; j < shortestCol; j++, leftScoreIndex += z, crossScoreIndex += z,
                                           leftSubScoreIndex += z, crossSubScoreIndex += z)
            {
                __m256i leftScore = _mm256_load_si256((__m256i*)(scoreMatrix.data() + leftScoreIndex));
                __m256i leftSubScore = _mm256_load_si256((__m256i*)(leftSubsMatrix.data() + leftSubScoreIndex));
                __m256i left = _mm256_add_epi64(leftScore, leftSubScore);

                __m256i upScore = _mm256_load_si256((__m256i*)(scoreMatrix.data() + crossScoreIndex + z));
                __m256i up = _mm256_add_epi64(upScore, upSubScore);

                __m256i crossScore = _mm256_load_si256((__m256i*)(scoreMatrix.data() + crossScoreIndex));
                __m256i crossSubScore = _mm256_load_si256((__m256i*)(crossSubsMatrix.data() + crossSubScoreIndex));
                __m256i cross = _mm256_add_epi64(crossScore, crossSubScore);

                score = mm256_max_epi64(left, up);
                score = mm256_max_epi64(score, cross);

                _mm256_store_si256((__m256i*)(scoreMatrix.data() + leftScoreIndex + z), score);
            }

            // Deal with various reads' length
            // _readsSize[readId] -> _readsSize[readId + batchSize - 1]
            for (size_t j = shortestCol; j < y; j++, leftScoreIndex += z, crossScoreIndex += z,
                                           leftSubScoreIndex += z, crossSubScoreIndex += z)
            {
                __m256i leftScore = _mm256_load_si256((__m256i*)(scoreMatrix.data() + leftScoreIndex));
                __m256i leftSubScore = _mm256_load_si256((__m256i*)(leftSubsMatrix.data() + leftSubScoreIndex));
                __m256i left = _mm256_add_epi64(leftScore, leftSubScore);

                __m256i upScore = _mm256_load_si256((__m256i*)(scoreMatrix.data() + crossScoreIndex + z));
                __m256i up = _mm256_add_epi64(upScore, upSubScore);

                __m256i crossScore = _mm256_load_si256((__m256i*)(scoreMatrix.data() + crossScoreIndex));
                __m256i crossSubScore = _mm256_load_si256((__m256i*)(crossSubsMatrix.data() + crossSubScoreIndex));
                __m256i cross = _mm256_add_epi64(crossScore, crossSubScore);

                __m256i prevScore = score;
                score = mm256_max_epi64(left, up);
                score = mm256_max_epi64(score, cross);
                __m256i _j = _mm256_set1_epi64x(j);
                __m256i cmp_mask = _mm256_cmpgt_epi64(_j, _cols);
                score = _mm256_blendv_epi8(score, prevScore, cmp_mask);

                _mm256_store_si256((__m256i*)(scoreMatrix.data() + leftScoreIndex + z), score);
            }

//            auto alignmentEnd = std::chrono::high_resolution_clock::now();
//            alignmentDuration += alignmentEnd - alignmentStart;
        }

        alignas(32) AlnScoreType scores[batchSize];
//        AlnScoreType scores[batchSize] __attribute((aligned(64)));
        _mm256_store_si256((__m256i*)scores, score);

        _forwardScores[batchId] = std::move(scoreMatrix);

        for(size_t b = 0; b < batchSize; b++)
            if((readId + b) < readsNum)
                finalScore += scores[b];
    }


    // getRevScoringMatrix
    for (size_t batchId = 0; batchId < batchNum; batchId++)
    {
        const size_t readId = batchId * batchSize;
        size_t x = consensus.size() + 1;
        size_t y = reads[readId+batchSize-1].size() + 1;
        size_t z = batchSize;

//        ScoreMatrix3d scoreMatrix(x, y, z);
        AlnScoreType* ptr = memoryPool.allocate(x * y * z);
        ScoreMatrix3d scoreMatrix(ptr, x, y, z);

        const ScoreMatrix& leftSubsMatrix = _subsScores_[batchId];
        const ScoreMatrix& crossSubsMatrixA = _subsScoresA[batchId];
        const ScoreMatrix& crossSubsMatrixC = _subsScoresC[batchId];
        const ScoreMatrix& crossSubsMatrixG = _subsScoresG[batchId];
        const ScoreMatrix& crossSubsMatrixT = _subsScoresT[batchId];

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

        __m256i score;
        __m256i _cols = _mm256_load_si256((__m256i*)(_readsSize + readId));

        for (size_t i = x - 1; i >= 1; i--, rightScoreIndex -= z, crossScoreIndex -= z)
        {

            size_t rightSubScoreIndex = (y - 2) * z;
            __m256i downSubScore = _mm256_set1_epi64x(_subsMatrix.getScore(v[i - 1], '-'));

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
            const size_t shortestCol = _readsSize[readId];
            for (size_t j = y - 1; j >= shortestCol; j--, rightScoreIndex -= z, crossScoreIndex -= z,
                                                rightSubScoreIndex -= z, crossSubScoreIndex -= z)
            {
                __m256i rightScore = _mm256_load_si256((__m256i*)(scoreMatrix.data() + rightScoreIndex));
                __m256i rightSubScore = _mm256_load_si256((__m256i*)(leftSubsMatrix.data() + rightSubScoreIndex));
                __m256i right = _mm256_add_epi64(rightScore, rightSubScore);

                __m256i downScore = _mm256_load_si256((__m256i*)(scoreMatrix.data() + crossScoreIndex - z));
                __m256i down = _mm256_add_epi64(downScore, downSubScore);

                __m256i crossScore = _mm256_load_si256((__m256i*)(scoreMatrix.data() + crossScoreIndex));
                __m256i crossSubScore = _mm256_load_si256((__m256i*)(crossSubsMatrix.data() + crossSubScoreIndex));
                __m256i cross = _mm256_add_epi64(crossScore, crossSubScore);

                __m256i prevScore = _mm256_load_si256((__m256i*)(scoreMatrix.data() + rightScoreIndex - z));
                score = mm256_max_epi64(right, down);
                score = mm256_max_epi64(score, cross);
                __m256i _j = _mm256_set1_epi64x(j);
                __m256i cmp_mask = _mm256_cmpgt_epi64(_j, _cols);
                score = _mm256_blendv_epi8(score, prevScore, cmp_mask);

                _mm256_store_si256((__m256i*)(scoreMatrix.data() + rightScoreIndex - z), score);
            }

            for (size_t j = shortestCol - 1; j >= 1; j--, rightScoreIndex -= z, crossScoreIndex -= z,
                                                rightSubScoreIndex -= z, crossSubScoreIndex -= z)
            {
                __m256i rightScore = _mm256_load_si256((__m256i*)(scoreMatrix.data() + rightScoreIndex));
                __m256i rightSubScore = _mm256_load_si256((__m256i*)(leftSubsMatrix.data() + rightSubScoreIndex));
                __m256i right = _mm256_add_epi64(rightScore, rightSubScore);

                __m256i downScore = _mm256_load_si256((__m256i*)(scoreMatrix.data() + crossScoreIndex - z));
                __m256i down = _mm256_add_epi64(downScore, downSubScore);

                __m256i crossScore = _mm256_load_si256((__m256i*)(scoreMatrix.data() + crossScoreIndex));
                __m256i crossSubScore = _mm256_load_si256((__m256i*)(crossSubsMatrix.data() + crossSubScoreIndex));
                __m256i cross = _mm256_add_epi64(crossScore, crossSubScore);

                score = mm256_max_epi64(right, down);
                score = mm256_max_epi64(score, cross);

                _mm256_store_si256((__m256i*)(scoreMatrix.data() + rightScoreIndex - z), score);
            }

//            auto alignmentEnd = std::chrono::high_resolution_clock::now();
//            alignmentDuration += alignmentEnd - alignmentStart;
        }

        _reverseScores[batchId] = std::move(scoreMatrix);
    }

    return finalScore;
}


AlnScoreType AlignmentAVX::addDeletionAVX(unsigned int letterIndex, const size_t readsNum)
{
    size_t frontRow = letterIndex - 1;
    size_t revRow = letterIndex;

    AlnScoreType finalScore = 0;

    for (size_t batchId = 0; batchId < batchNum; batchId++) {
        const size_t readId = batchId * batchSize;
        const ScoreMatrix3d& forwardScores = _forwardScores[batchId];
        const ScoreMatrix3d& reverseScores = _reverseScores[batchId];

        size_t y = forwardScores.ncols();
        size_t z = batchSize;

        size_t frontIndex = frontRow * y * z; // (frontRow, 0)
        size_t reverseIndex = revRow * y * z; // (revRow, 0)
        const AlnScoreType* frontPtr = forwardScores.data() + frontIndex;
        const AlnScoreType* reversePtr = reverseScores.data() + reverseIndex;

        AlnScoreType maxVal = std::numeric_limits<AlnScoreType>::lowest();
        __m256i _maxVal = _mm256_set1_epi64x(maxVal);

//        auto deletionStart = std::chrono::high_resolution_clock::now();

        const size_t shortestCol = _readsSize[readId];
        for (size_t col = 0; col < shortestCol; ++col)
        {
            __m256i forwardScore = _mm256_load_si256((__m256i*)(frontPtr + col * z));
            __m256i reverseScore = _mm256_load_si256((__m256i*)(reversePtr + col * z));
            __m256i sum = _mm256_add_epi64(forwardScore, reverseScore);
            _maxVal = mm256_max_epi64(_maxVal, sum);
        }

//        auto deletionEnd = std::chrono::high_resolution_clock::now();
//        deletionDuration += deletionEnd - deletionStart;

        // Deal with various reads' length
        // _readsSize[readId] -> _readsSize[readId + batchSize - 1]
        __m256i _cols = _mm256_load_si256((__m256i*)(_readsSize + readId));
        for (size_t col = shortestCol; col < y; ++col)
        {
            __m256i forwardScore = _mm256_load_si256((__m256i*)(frontPtr + col * z));
            __m256i reverseScore = _mm256_load_si256((__m256i*)(reversePtr + col * z));
            __m256i sum = _mm256_add_epi64(forwardScore, reverseScore);

            __m256i _col = _mm256_set1_epi64x(col);
            __m256i cmp_mask = _mm256_cmpgt_epi64(_col, _cols);
            sum = _mm256_blendv_epi8(sum, _maxVal, cmp_mask);
            _maxVal = mm256_max_epi64(_maxVal, sum);
        }

        alignas(32) AlnScoreType scores[batchSize];
//        AlnScoreType scores[batchSize] __attribute((aligned(64)));
        _mm256_store_si256((__m256i*)scores, _maxVal);

        for(size_t b = 0; b < batchSize; b++)
            if((readId + b) < readsNum)
                finalScore += scores[b];
    }

    return finalScore;
}


AlnScoreType AlignmentAVX::addSubstitutionAVX(unsigned int letterIndex, char base,
                                              const std::vector<std::string>& reads,
                                              const size_t readsNum)
{
    size_t frontRow = letterIndex - 1;
    size_t revRow = letterIndex;

    return addSubsAndInsertAVX(frontRow, revRow, base, reads, readsNum);
}


AlnScoreType AlignmentAVX::addInsertionAVX(unsigned int pos, char base,
                                           const std::vector<std::string>& reads,
                                           const size_t readsNum)
{
    size_t frontRow = pos - 1;
    size_t revRow = pos - 1;

    return addSubsAndInsertAVX(frontRow, revRow, base, reads, readsNum);
}


AlnScoreType AlignmentAVX::addSubsAndInsertAVX(size_t frontRow, size_t revRow,
                                 char base, const std::vector <std::string> &reads,
                                 const size_t readsNum)
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

    AlnScoreType finalScore = 0;

    AlnScoreType baseScoreWithGap = _subsMatrix.getScore(base, '-');
    __m256i _baseScoreWithGap = _mm256_set1_epi64x(baseScoreWithGap);

    const size_t batchNum = reads.size() / batchSize;
    for (size_t batchId = 0; batchId < batchNum; batchId++) {
        const size_t readId = batchId * batchSize;
        const ScoreMatrix3d& forwardScores = _forwardScores[batchId];
        const ScoreMatrix3d& reverseScores = _reverseScores[batchId];
        const ScoreMatrix& subsScores = _subsScores[batchId];

        size_t y = forwardScores.ncols();
        size_t z = batchSize;

        size_t frontIndex = frontRow * y * z; // (frontRow, 0)
        size_t reverseIndex = revRow * y * z; // (revRow, 0)
        size_t subIndex = 0;
        const AlnScoreType* frontPtr = forwardScores.data() + frontIndex;
        const AlnScoreType* reversePtr = reverseScores.data() + reverseIndex;
        const AlnScoreType* subPtr = subsScores.data() + subIndex;

        __m256i _cols = _mm256_load_si256((__m256i*)(_readsSize + readId));
        __m256i _one = _mm256_set1_epi64x(-1);
        _cols = _mm256_add_epi64(_cols, _one);

        __m256i forwardScore = _mm256_load_si256((__m256i*) frontPtr);
        __m256i reverseScore = _mm256_load_si256((__m256i*) reversePtr);
        __m256i _maxVal = _mm256_add_epi64(forwardScore, reverseScore);
        _maxVal = _mm256_add_epi64(_maxVal, _baseScoreWithGap);

        const size_t shortestCol = _readsSize[readId];
        for (size_t col = 0; col < shortestCol; ++col)
        {
            __m256i subScore = _mm256_load_si256((__m256i*)(subPtr + col * z));
            __m256i forwardScoreCurrent = _mm256_load_si256((__m256i*)(frontPtr + col * z));
            __m256i forwardScoreNext = _mm256_load_si256((__m256i*)(frontPtr + (col + 1) * z));
            __m256i reverseScoreNext = _mm256_load_si256((__m256i*)(reversePtr + (col + 1) * z));

            __m256i matchScore = _mm256_add_epi64(forwardScoreCurrent, subScore);
            __m256i insertScore = _mm256_add_epi64(forwardScoreNext, _baseScoreWithGap);
            __m256i tempMaxScore = mm256_max_epi64(matchScore, insertScore);
            __m256i sum = _mm256_add_epi64(reverseScoreNext, tempMaxScore);
            _maxVal = mm256_max_epi64(_maxVal, sum);
        }

        // Deal with various reads' length
        // _readsSize[readId] -> _readsSize[readId + batchSize - 1]
        for (size_t col = shortestCol; col < y - 1; ++col)
        {
            __m256i subScore = _mm256_load_si256((__m256i*)(subPtr + col * z));
            __m256i forwardScoreCurrent = _mm256_load_si256((__m256i*)(frontPtr + col * z));
            __m256i forwardScoreNext = _mm256_load_si256((__m256i*)(frontPtr + (col + 1) * z));
            __m256i reverseScoreNext = _mm256_load_si256((__m256i*)(reversePtr + (col + 1) * z));

            __m256i matchScore = _mm256_add_epi64(forwardScoreCurrent, subScore);
            __m256i insertScore = _mm256_add_epi64(forwardScoreNext, _baseScoreWithGap);
            __m256i tempMaxScore = mm256_max_epi64(matchScore, insertScore);
            __m256i sum = _mm256_add_epi64(reverseScoreNext, tempMaxScore);

            __m256i _col = _mm256_set1_epi64x(col);
            __m256i cmp_mask = _mm256_cmpgt_epi64(_col, _cols);
            sum = _mm256_blendv_epi8(sum, _maxVal, cmp_mask);
            _maxVal = mm256_max_epi64(_maxVal, sum);
        }

        alignas(32) AlnScoreType scores[batchSize];
//        AlnScoreType scores[batchSize] __attribute((aligned(64)));
        _mm256_store_si256((__m256i*)scores, _maxVal);

        for(size_t b = 0; b < batchSize; b++)
            if((readId + b) < readsNum)
                finalScore += scores[b];
    }

    return finalScore;
}