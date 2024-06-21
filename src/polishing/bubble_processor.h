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
#include "dinucleotide_fixer_avx.h"


class BubbleProcessor
{
public:
    BubbleProcessor(const std::string& subsMatPath,
                    const std::string& hopoMatrixPath,
                    bool  showProgress, bool hopoEndabled,
                    int numThreads);
    void polishAll(const std::string& inBubbles, const std::string& outConsensus);
    void enableVerboseOutput(const std::string& filename);

private:
    void parallelWorker(const std::string outFile);
    void cacheBubbles(int numBubbles);
    void writeBubbles(const std::vector<Bubble>& bubbles);
    void writeLog(const std::vector<Bubble>& bubbles);

    const SubstitutionMatrix        _subsMatrix;
    const HopoMatrix 		        _hopoMatrix;
    const GeneralPolisher 	        _generalPolisher;
    const HomoPolisher 		        _homoPolisher;
    const DinucleotideFixerAVX      _dinucFixer;

    ProgressPercent 		            _progress;
    std::mutex                          _readMutex;
    std::queue<std::unique_ptr<Bubble>> _preprocessBubbles;

    std::ifstream			            _bubblesFile;
    std::ofstream			            _logFile;
    bool					            _verbose;
    bool 					            _showProgress;
    bool					            _hopoEnabled;

    int                                 _batchSize = 250000;
    bool                                _done = false;
    const int                           _numThreads;
};