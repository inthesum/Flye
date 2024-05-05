//(c) 2016 by Authors
//This file is a part of ABruijn program.
//Released under the BSD license (see LICENSE file)

#pragma once

#include <string>
#include <vector>
#include <algorithm>
#include <iomanip>
#include <stdexcept>
#include <chrono>

#include "score_matrix.h"


class Alignment {

public:
    Alignment(size_t size, const SubstitutionMatrix &sm);

    AlnScoreType globalAlignment(const std::string &consensus,
                                 const std::vector <std::string> &reads);
    AlnScoreType globalAlignment(const std::string &consensus,
                                 const std::vector <std::string> &reads,
                                 std::chrono::duration<double>& alignmentDuration);

    AlnScoreType addDeletion(unsigned int letterIndex) const;

    AlnScoreType addSubstitution(unsigned int letterIndex,
                                 char base, const std::vector <std::string> &reads) const;

    AlnScoreType addInsertion(unsigned int positionIndex,
                              char base, const std::vector <std::string> &reads) const;

private:
    std::vector <ScoreMatrix> _forwardScores;
    std::vector <ScoreMatrix> _reverseScores;
    const SubstitutionMatrix &_subsMatrix;

    AlnScoreType getScoringMatrix(const std::string &v, const std::string &w,
                                  ScoreMatrix &scoreMat);
    AlnScoreType getRevScoringMatrix(const std::string &v, const std::string &w,
                                  ScoreMatrix &scoreMat);
};
