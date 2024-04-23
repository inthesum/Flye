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

#include "../common/matrix.h"
#include "subs_matrix.h"


class Alignment {

public:
    Alignment(size_t size, const SubstitutionMatrix &sm);
    Alignment(size_t size, const SubstitutionMatrix &sm, const std::vector <std::string> &reads);
    ~Alignment();

    typedef Matrix<AlnScoreType> ScoreMatrix;

    AlnScoreType globalAlignment(const std::string &consensus,
                                 const std::vector <std::string> &reads);

    AlnScoreType addDeletion(unsigned int letterIndex) const;

    AlnScoreType addSubstitution(unsigned int letterIndex,
                                 char base, const std::vector <std::string> &reads) const;

    AlnScoreType addInsertion(unsigned int positionIndex,
                              char base, const std::vector <std::string> &reads) const;

private:
    std::vector <ScoreMatrix> _forwardScores;
    std::vector <ScoreMatrix> _reverseScores;
    const SubstitutionMatrix &_subsMatrix;
    AlnScoreType* _subsScoresA;
    AlnScoreType* _subsScoresC;
    AlnScoreType* _subsScoresG;
    AlnScoreType* _subsScoresT;

    AlnScoreType getScoringMatrix(const std::string &v,
                                  const std::string &w,
                                  ScoreMatrix &scoreMat);
    AlnScoreType getRevScoringMatrix(const std::string &v,
                                     const std::string &w,
                                     ScoreMatrix &scoreMat);
    AlnScoreType getScoringMatrixAVX2(const std::string& v,
                                      const std::string& w,
                                      ScoreMatrix &scoreMat);
};
