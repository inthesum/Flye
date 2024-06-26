//(c) 2016 by Authors
//This file is a part of ABruijn program.
//Released under the BSD license (see LICENSE file)

#pragma once

#include "subs_matrix.h"
#include "bubble.h"
#include "score_memory_pool.h"

class DinucleotideFixerAVX
{
public:
	DinucleotideFixerAVX(const SubstitutionMatrix& subsMatrix):
		_subsMatrix(subsMatrix)
	{}
	void fixBubble(Bubble& bubble,
                   size_t batchSize,
                   ScoreMemoryPool& memoryPool) const;

private:
	std::pair<int, int> getDinucleotideRuns(const std::string& sequence) const;

	const SubstitutionMatrix& _subsMatrix;
};
