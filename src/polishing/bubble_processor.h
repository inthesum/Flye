//(c) 2016 by Authors
//This file is a part of ABruijn program.
//Released under the BSD license (see LICENSE file)

#pragma once

#include <string>
#include <vector>
#include <queue>
#include <cmath>
#include <mutex>
#include <fstream>

#include "subs_matrix.h"
#include "bubble.h"
#include "general_polisher.h"
#include "homo_polisher.h"
#include "utility.h"
#include "../common/progress_bar.h"
#include "dinucleotide_fixer.h"


class BubbleProcessor 
{
public:
	BubbleProcessor(const std::string& subsMatPath,
					const std::string& hopoMatrixPath,
					bool  showProgress, bool hopoEndabled);
	void polishAll(const std::string& inBubbles, const std::string& outConsensus,
				   int numThreads);
	void enableVerboseOutput(const std::string& filename);

private:
	void parallelWorker(const std::string inFile, const std::string outFile);
    void cacheBubbles(std::ifstream& bubblesFile, std::queue<Bubble>& bubbles, int maxRead);

	const SubstitutionMatrix  _subsMatrix;
	const HopoMatrix 		  _hopoMatrix;
	const GeneralPolisher 	  _generalPolisher;
	const HomoPolisher 		  _homoPolisher;
	const DinucleotideFixer	  _dinucFixer;

    std::mutex                _Mutex;

    std::ifstream			  _bubblesFile;
    std::ofstream			  _logFile;
	bool					  _verbose;
	bool 					  _showProgress;
	bool					  _hopoEnabled;
};
