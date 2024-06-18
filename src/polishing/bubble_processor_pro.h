//(c) 2016 by Authors
//This file is a part of ABruijn program.
//Released under the BSD license (see LICENSE file)

#pragma once

#include <string>
#include <vector>
#include <queue>
#include <cmath>
#include <mutex>
#include <condition_variable>
#include <fstream>

#include "subs_matrix.h"
#include "bubble.h"
#include "general_polisher.h"
#include "homo_polisher.h"
#include "utility.h"
#include "../common/progress_bar.h"
#include "dinucleotide_fixer_avx.h"


class BubbleProcessorPro
{
public:
    BubbleProcessorPro(const std::string& subsMatPath,
                    const std::string& hopoMatrixPath,
                    bool  showProgress, bool hopoEndabled,
                    int numThreads);
    void polishAll(const std::string& inBubbles, const std::string& outConsensus);
    void enableVerboseOutput(const std::string& filename);

private:
    void readThread(const std::string& inBubbles, const std::string outConsensus, const int id);
    void processThread(const std::string outConsensus, const int id);
    void cacheBubbles(std::ifstream& bubbleFile, std::queue<std::unique_ptr<Bubble>>& bubbles, int& batchSize);
    void writeBubbles(const std::vector<Bubble>& bubbles);
    void writeLog(const std::vector<Bubble>& bubbles);

    const SubstitutionMatrix            _subsMatrix;
    const HopoMatrix 		            _hopoMatrix;
    const GeneralPolisher 	            _generalPolisher;
    const HomoPolisher 		            _homoPolisher;
    const DinucleotideFixerAVX	        _dinucFixer;

    ProgressPercent 		            _progress;
    static std::mutex                   _mtx;
    static std::condition_variable      _cv_reader;
    static std::condition_variable      _cv_processor;
    static bool                         _ready_to_read;
    static bool                         _ready_to_process;
    static bool                         _done1;
    static bool                         _done2;
    static int                          _batchSize;

    const int                           _numThreads;

    //    std::queue<Bubble>                  _preprocessBubbles;
    std::queue<std::unique_ptr<Bubble>> _preprocessBubbles;

//    std::ifstream			            _bubblesFile;
    std::ofstream			            _logFile;
    bool					            _verbose;
    bool 					            _showProgress;
    bool					            _hopoEnabled;
};