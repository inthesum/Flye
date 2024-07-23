//(c) 2016 by Authors
//This file is a part of ABruijn program.
//Released under the BSD license (see LICENSE file)

#include "alignment_score_only_avx.h"

constexpr size_t batchSize = 8; // Use constexpr for batch size

AlignmentScoreOnlyAVX::AlignmentScoreOnlyAVX(size_t size, const SubstitutionMatrix& sm, const std::vector <std::string> &reads):
_subsMatrix(sm),
batchNum(size/batchSize)
{
    _subsScoresA.resize(batchNum);
    _subsScoresC.resize(batchNum);
    _subsScoresG.resize(batchNum);
    _subsScoresT.resize(batchNum);
    _subsScores_.resize(batchNum);

    _readsSize = (AlnScoreType*)_mm_malloc(size * sizeof(AlnScoreType), 64);

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

AlignmentScoreOnlyAVX::~AlignmentScoreOnlyAVX() {
    _mm_free(_readsSize);
}


AlnScoreType AlignmentScoreOnlyAVX::globalAlignmentAVX(const std::string& consensus,
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

        ScoreMatrix3d scoreMatrix(x, y, z);

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

            const size_t shortestCol = _readsSize[readId];
            for (size_t j = 1; j < shortestCol; j++, leftScoreIndex += z, crossScoreIndex += z,
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
            for (size_t j = shortestCol; j < y; j++, leftScoreIndex += z, crossScoreIndex += z,
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
        }

        alignas(64) AlnScoreType scores[batchSize];
        _mm512_store_si512((__m512i*)scores, score);

        for(size_t b = 0; b < batchSize; b++)
            if((readId + b) < readsNum)
                finalScore += scores[b];
    }

    return finalScore;
}