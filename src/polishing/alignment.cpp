//(c) 2016 by Authors
//This file is a part of ABruijn program.
//Released under the BSD license (see LICENSE file)

#include "alignment.h"
#include <immintrin.h> // Include SIMD intrinsics header


Alignment::Alignment(size_t size, const SubstitutionMatrix& sm):
	_forwardScores(size),
	_reverseScores(size),
	_subsMatrix(sm)
{ 
}


AlnScoreType Alignment::globalAlignment(const std::string& consensus,
							 			const std::vector<std::string>& reads)
{
	AlnScoreType finalScore = 0;
	for (size_t readId = 0; readId < _forwardScores.size(); ++readId)
	{
		unsigned int x = consensus.size() + 1;
		unsigned int y = reads[readId].size() + 1;
		ScoreMatrix scoreMat(x, y, 0);

		AlnScoreType score = this->getScoringMatrix(consensus, reads[readId],scoreMat);
		_forwardScores[readId] = std::move(scoreMat);

		//The reverse alignment is similar, but we need
		//the scoring matrix with of the reverse alignment (I guess)
		std::string revConsensus(consensus.rbegin(), consensus.rend());
		std::string revRead(reads[readId].rbegin(), reads[readId].rend());

		ScoreMatrix scoreMatRev(x, y, 0);
		this->getScoringMatrix(revConsensus, revRead, scoreMatRev);
		_reverseScores[readId] = std::move(scoreMatRev);

		finalScore += score;
	}

	return finalScore;
}


AlnScoreType Alignment::addDeletion(unsigned int letterIndex) const
{
	AlnScoreType finalScore = 0;
	for (size_t readId = 0; readId < _forwardScores.size(); ++readId)
	{
		const ScoreMatrix& forwardScore = _forwardScores[readId];
		const ScoreMatrix& reverseScore = _reverseScores[readId];

		//Note: We subtract 2 because of zero indexing and an extra added row and column count
		//unsigned int index = (reverseScore.nrows() - 1) - letterIndex;
		size_t frontRow = letterIndex - 1;
		size_t revRow = reverseScore.nrows() - 1 - letterIndex;

		AlnScoreType maxVal = std::numeric_limits<AlnScoreType>::lowest();
		for (size_t col = 0; col < forwardScore.ncols(); ++col)
		{
			size_t backCol = forwardScore.ncols() - col - 1;
			AlnScoreType sum = forwardScore.at(frontRow, col) +
							reverseScore.at(revRow, backCol);
			maxVal = std::max(maxVal, sum);
		}
		finalScore += maxVal;
	}
	return finalScore;
}


AlnScoreType Alignment::addSubstitution(unsigned int letterIndex, char base,
										const std::vector<std::string>& reads) const
{
	AlnScoreType finalScore = 0;
    size_t frontRow = letterIndex - 1;
    AlnScoreType baseScoreWithGap = _subsMatrix.getScore(base, '-');

    for (size_t readId = 0; readId < reads.size(); ++readId)
	{
		const ScoreMatrix& forwardScore = _forwardScores[readId];
		const ScoreMatrix& reverseScore = _reverseScores[readId];

		size_t revRow = reverseScore.nrows() - 1 - letterIndex;

        size_t cols = reads[readId].size();
        AlnScoreType maxVal = forwardScore.at(frontRow, 0) + reverseScore.at(revRow, cols) + baseScoreWithGap;
		for (size_t col = 0; col < cols; ++col)
		{
            char readBase = reads[readId][col];
            AlnScoreType match = forwardScore.at(frontRow, col) + _subsMatrix.getScore(base, readBase);
			AlnScoreType ins = forwardScore.at(frontRow, col + 1) + baseScoreWithGap;
            maxVal = std::max(maxVal, std::max(match, ins) + reverseScore.at(revRow, cols - col - 1));
        }
		finalScore += maxVal;
	}
	return finalScore;
}


AlnScoreType Alignment::addInsertion(unsigned int pos, char base, const std::vector<std::string>& reads) const
{
    AlnScoreType finalScore = 0;
    size_t frontRow = pos - 1;
    AlnScoreType baseScoreWithGap = _subsMatrix.getScore(base, '-');

    for (size_t readId = 0; readId < reads.size(); ++readId)
    {
        const ScoreMatrix& forwardScore = _forwardScores[readId];
        const ScoreMatrix& reverseScore = _reverseScores[readId];

        size_t revRow = reverseScore.nrows() - pos;

        size_t cols = reads[readId].size();
        AlnScoreType maxVal = forwardScore.at(frontRow, 0) + reverseScore.at(revRow, cols) + baseScoreWithGap;

        for (size_t col = 0; col < cols; ++col)
        {
            char readBase = reads[readId][col];
            AlnScoreType match = forwardScore.at(frontRow, col) + _subsMatrix.getScore(base, readBase);
            AlnScoreType ins = forwardScore.at(frontRow, col+1) + baseScoreWithGap;
            maxVal = std::max(maxVal, std::max(match, ins) + reverseScore.at(revRow, cols - col - 1));
        }

		finalScore += maxVal;
    }
    return finalScore;
}


AlnScoreType Alignment::getScoringMatrix(const std::string& v,
										 const std::string& w,
								  		 ScoreMatrix& scoreMat)
{
	AlnScoreType score = 0;

	for (size_t i = 0; i < v.size(); i++)
	{
		AlnScoreType score = _subsMatrix.getScore(v[i], '-');
		scoreMat.at(i + 1, 0) = scoreMat.at(i, 0) + score;
	}


	for (size_t i = 0; i < w.size(); i++) {
		AlnScoreType score = _subsMatrix.getScore('-', w[i]);
		scoreMat.at(0, i + 1) = scoreMat.at(0, i) + score;
	}


	for (size_t i = 1; i < v.size() + 1; i++)
	{
		char key1 = v[i - 1];
		for (size_t j = 1; j < w.size() + 1; j++)
		{
			char key2 = w[j - 1];

			AlnScoreType left = scoreMat.at(i, j - 1) + _subsMatrix.getScore('-', key2);
			AlnScoreType up = scoreMat.at(i - 1, j) + _subsMatrix.getScore(key1, '-');
			score = std::max(left, up);

			AlnScoreType cross = scoreMat.at(i - 1, j - 1) + _subsMatrix.getScore(key1, key2);
			score = std::max(score, cross);
			scoreMat.at(i, j) = score;
		}
	}

	return score;
}