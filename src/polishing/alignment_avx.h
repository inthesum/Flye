//(c) 2016 by Authors
//This file is a part of ABruijn program.
//Released under the BSD license (see LICENSE file)

#pragma once

#include <string>
#include <vector>
#include <algorithm>
#include <iomanip>
#include <stdexcept>
#include <memory>
#include <chrono>

#include "score_matrix.h"
#include "score_matrix3d.h"
#include "subs_matrix.h"
#include "score_memory_pool.h"


class AlignmentAVX {

public:
    AlignmentAVX(size_t size, const SubstitutionMatrix &sm, const std::vector<std::string> &reads, ScoreMemoryPool& memoryPool);
    ~AlignmentAVX();

    AlnScoreType globalAlignmentAVX(const std::string &consensus,
                                    const std::vector <std::string> &reads,
                                    const size_t readsNum);

    AlnScoreType addDeletionAVX(unsigned int letterIndex,
                                const size_t readsNum);

    AlnScoreType addSubsAndInsertAVX(size_t frontRow, size_t revRow,
                                     char base, const std::vector<std::string> &reads,
                                     const size_t readsNum);

    AlnScoreType addSubstitutionAVX(unsigned int letterIndex,
                                    char base, const std::vector<std::string> &reads,
                                    const size_t readsNum);

    AlnScoreType addInsertionAVX(unsigned int positionIndex,
                                 char base, const std::vector<std::string> &reads,
                                 const size_t readsNum);

private:
    ScoreMemoryPool& memoryPool;
    std::vector<ScoreMatrix3d> _forwardScores;
    std::vector<ScoreMatrix3d> _reverseScores;
    const SubstitutionMatrix &_subsMatrix;
    AlnScoreType* _readsSize;

    std::vector<ScoreMatrix> _subsScoresA;
    std::vector<ScoreMatrix> _subsScoresC;
    std::vector<ScoreMatrix> _subsScoresG;
    std::vector<ScoreMatrix> _subsScoresT;
    std::vector<ScoreMatrix> _subsScores_;

    const size_t batchNum;

    __m256i mm256_max_epi64(__m256i a, __m256i b) {
        __m256i cmp_mask = _mm256_cmpgt_epi64(a, b);
        __m256i result = _mm256_blendv_epi8(b, a, cmp_mask);

        return result;
    }
};

