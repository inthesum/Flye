//(c) 2016 by Authors
//This file is a part of ABruijn program.
//Released under the BSD license (see LICENSE file)

#pragma once

#include <string>
#include <vector>
#include <algorithm>
#include <chrono>

#include "score_matrix.h"
#include "score_matrix3d.h"
//#include "score_memory_pool.h"


class AlignmentAVX {

public:
    AlignmentAVX(size_t size, const SubstitutionMatrix &sm, const std::vector <std::string> &reads);
    ~AlignmentAVX();

    AlnScoreType globalAlignmentAVX(const std::string &consensus,
                                    const std::vector <std::string> &reads,
                                    const size_t readsNum);
//    AlnScoreType globalAlignmentAVX(const std::string &consensus,
//                                    const std::vector <std::string> &reads,
//                                    const size_t readsNum,
//                                    ScoreMemoryPool& memoryPool);
//    AlnScoreType globalAlignmentAVX(const std::string &consensus,
//                                    const std::vector <std::string> &reads,
//                                    const size_t readsNum,
////                                    ScoreMemoryPool& memoryPool,
//                                    std::chrono::duration<double>& alignmentDuration);

    AlnScoreType addDeletionAVX(unsigned int letterIndex,
                                const size_t readsNum) const;
//    AlnScoreType addDeletionAVX(unsigned int letterIndex,
//                                const size_t readsNum,
//                                std::chrono::duration<double>& deletionDuration) const;

    AlnScoreType addSubsAndInsertAVX(size_t frontRow, size_t revRow,
                                     char base, const std::vector <std::string> &reads,
                                     const size_t readsNum) const;

    AlnScoreType addSubstitutionAVX(unsigned int letterIndex,
                                    char base, const std::vector <std::string> &reads,
                                    const size_t readsNum) const;

    AlnScoreType addInsertionAVX(unsigned int positionIndex,
                                 char base, const std::vector <std::string> &reads,
                                 const size_t readsNum) const;

private:
    std::vector<ScoreMatrix3d> _forwardScores;
    std::vector<ScoreMatrix3d> _reverseScores;
    const SubstitutionMatrix &_subsMatrix;
    AlnScoreType* _readsSize;

    std::vector<ScoreMatrix> _subsScoresA;
    std::vector<ScoreMatrix> _subsScoresC;
    std::vector<ScoreMatrix> _subsScoresG;
    std::vector<ScoreMatrix> _subsScoresT;
    std::vector<ScoreMatrix> _subsScores_;
};

