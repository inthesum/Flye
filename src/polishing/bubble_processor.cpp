//(c) 2016 by Authors
//This file is a part of ABruijn program.
//Released under the BSD license (see LICENSE file)

#include <chrono>
#include <thread>
#include <sys/stat.h>

#include "bubble_processor.h"

namespace
{
	size_t fileSize(const std::string& filename)
	{
		struct stat st;
		if (stat(filename.c_str(), &st) != 0) return 0;
		return st.st_size;
	}
}

BubbleProcessor::BubbleProcessor(const std::string& subsMatPath,
								 const std::string& hopoMatrixPath,
								 bool showProgress, bool hopoEnabled):
	_subsMatrix(subsMatPath),
	_hopoMatrix(hopoMatrixPath),
	_generalPolisher(_subsMatrix),
	_homoPolisher(_subsMatrix, _hopoMatrix),
	_dinucFixer(_subsMatrix),
	_verbose(false),
	_showProgress(showProgress),
	_hopoEnabled(hopoEnabled)
{
}


void BubbleProcessor::polishAll(const std::string& inBubbles,
								const std::string& outConsensus,
			   					int numThreads)
{
	_cachedBubbles.clear();
	_cachedBubbles.reserve(BUBBLES_CACHE);

	size_t fileLength = fileSize(inBubbles);
	if (!fileLength)
	{
		throw std::runtime_error("Empty bubbles file!");
	}
	_bubblesFile.open(inBubbles);
	if (!_bubblesFile.is_open())
	{
		throw std::runtime_error("Error opening bubbles file");
	}

	_progress.setFinalCount(fileLength);

	_consensusFile.open(outConsensus);
	if (!_consensusFile.is_open())
	{
		throw std::runtime_error("Error opening consensus file");
	}

	std::vector<std::thread> threads(numThreads);
	for (size_t i = 0; i < threads.size(); ++i)
	{
		threads[i] = std::thread(&BubbleProcessor::parallelWorker, this);
	}
	for (size_t i = 0; i < threads.size(); ++i)
	{
		threads[i].join();
	}
	if (_showProgress) _progress.setDone();
}


void BubbleProcessor::parallelWorker()
{
    auto start = std::chrono::high_resolution_clock::now();

    std::thread::id threadId = std::this_thread::get_id();

    std::chrono::duration<double> duration(0);
    std::chrono::duration<double> cacheBubblesDuration(0);
    std::chrono::duration<double> generalPolisherDuration(0);
    std::chrono::duration<double> homoPolisherDuration(0);
    std::chrono::duration<double> fixerDuration(0);
    std::chrono::duration<double> waitReadDuration(0);
    std::chrono::duration<double> writeBubblesDuration(0);

    int64_t alignmentNum = 0;
    int64_t deletionNum = 0;
    int64_t insertionNum = 0;
    int64_t substitutionNum = 0;

    std::chrono::duration<double> optimizeDuration(0);
    std::chrono::duration<double> makeStepDuration(0);
    std::chrono::duration<double> alignmentDuration(0);
    std::chrono::duration<double> deletionDuration(0);
    std::chrono::duration<double> insertionDuration(0);
    std::chrono::duration<double> substitutionDuration(0);

    const int MAX_BUBBLE = 5000;
    int numBubbles = 0;
    int numBubblesPolished = 0;

    auto startWaiting = std::chrono::high_resolution_clock::now();
    _stateMutex.lock();
    auto endWaiting = std::chrono::high_resolution_clock::now();
    waitReadDuration += endWaiting - startWaiting;

	while (true)
	{
		if (_cachedBubbles.empty())
		{
            auto cacheBubblesStart = std::chrono::high_resolution_clock::now();
            this->cacheBubbles(BUBBLES_CACHE);
            auto cacheBubblesEnd = std::chrono::high_resolution_clock::now();
            cacheBubblesDuration += cacheBubblesEnd - cacheBubblesStart;

            if(_cachedBubbles.empty())
			{
                auto end = std::chrono::high_resolution_clock::now(); // End timer
                duration = end - start;
                std::cout << std::endl;
                std::cout << "thread id: " << threadId << std::endl;
                std::cout << "number of bubbles: " << numBubbles << std::endl;
                std::cout << "number of polished bubbles: " << numBubblesPolished << std::endl;
                std::cout << "parallelWorker: " << std::fixed << std::setprecision(2) << duration.count() << " seconds" << std::endl;
                std::cout << "waiting for read: " << std::fixed << std::setprecision(2) << waitReadDuration.count() << " seconds" << std::endl;
                std::cout << "cache bubbles: " << std::fixed << std::setprecision(2) << cacheBubblesDuration.count() << " seconds" << std::endl;
                std::cout << "write bubbles: " << std::fixed << std::setprecision(2) << writeBubblesDuration.count() << " seconds" << std::endl;
                std::cout << "_generalPolisher: " << std::fixed << std::setprecision(2) << generalPolisherDuration.count() << " seconds" << std::endl;
                std::cout << "_homoPolisher: " << std::fixed << std::setprecision(2) << homoPolisherDuration.count() << " seconds" << std::endl;
                std::cout << "_dinucFixer: " << std::fixed << std::setprecision(2) << fixerDuration.count() << " seconds" << std::endl;

                std::cout << "alignmentNum: " << alignmentNum << std::endl;
                std::cout << "deletionNum: " << deletionNum << std::endl;
                std::cout << "insertionNum: " << insertionNum << std::endl;
                std::cout << "substitutionNum: " << substitutionNum << std::endl;
                std::cout << "optimize: " << std::fixed << std::setprecision(2) << optimizeDuration.count() << " seconds" << std::endl;
                std::cout << "makeStep: " << std::fixed << std::setprecision(2) << makeStepDuration.count() << " seconds" << std::endl;
                std::cout << "alignment: " << std::fixed << std::setprecision(2) << alignmentDuration.count() << " seconds" << std::endl;
                std::cout << "deletion: " << std::fixed << std::setprecision(2) << deletionDuration.count() << " seconds" << std::endl;
                std::cout << "insertion: " << std::fixed << std::setprecision(2) << insertionDuration.count() << " seconds" << std::endl;
                std::cout << "substitution: " << std::fixed << std::setprecision(2) << substitutionDuration.count() << " seconds" << std::endl;

                _stateMutex.unlock();
				return;
			}
		}

		Bubble bubble = _cachedBubbles.back();
		_cachedBubbles.pop_back();
        numBubbles++;

        if (bubble.candidate.size() < MAX_BUBBLE &&
            bubble.branches.size() > 1)
        {
            _stateMutex.unlock();

            numBubblesPolished++;

            auto generalPolisherStart = std::chrono::high_resolution_clock::now();
            _generalPolisher.polishBubble(bubble,
                                          alignmentNum,
                                          deletionNum,
                                          insertionNum,
                                          substitutionNum,
                                          optimizeDuration,
                                          makeStepDuration,
                                          alignmentDuration,
                                          deletionDuration,
                                          insertionDuration,
                                          substitutionDuration);
            auto generalPolisherEnd = std::chrono::high_resolution_clock::now();
            generalPolisherDuration += generalPolisherEnd - generalPolisherStart;

            auto homoPolisherStart = std::chrono::high_resolution_clock::now();
            if (_hopoEnabled) {
                _homoPolisher.polishBubble(bubble);
            }
            auto homoPolisherEnd = std::chrono::high_resolution_clock::now();
            homoPolisherDuration += homoPolisherEnd - homoPolisherStart;

            auto fixerStart = std::chrono::high_resolution_clock::now();
            _dinucFixer.fixBubble(bubble);
            auto fixerEnd = std::chrono::high_resolution_clock::now();
            fixerDuration += fixerEnd - fixerStart;

            auto startWaiting = std::chrono::high_resolution_clock::now();
            _stateMutex.lock();
            auto endWaiting = std::chrono::high_resolution_clock::now();
            waitReadDuration += endWaiting - startWaiting;
        }

        auto startWriting = std::chrono::high_resolution_clock::now();
        this->writeBubbles({bubble});
        auto endWriting = std::chrono::high_resolution_clock::now();
        writeBubblesDuration += endWriting - startWriting;

        if (_verbose) this->writeLog({bubble});
	}
}


void BubbleProcessor::writeBubbles(const std::vector<Bubble>& bubbles)
{
	for (auto& bubble : bubbles)
	{
		_consensusFile << ">" << bubble.header << " " << bubble.position
			 		   << " " << bubble.branches.size() << " " << bubble.subPosition << std::endl
			 		   << bubble.candidate << std::endl;
	}
}

void BubbleProcessor::enableVerboseOutput(const std::string& filename)
{
	_verbose = true;
	_logFile.open(filename);
	if (!_logFile.is_open())
	{
		throw std::runtime_error("Error opening log file");
	}
}

void BubbleProcessor::writeLog(const std::vector<Bubble>& bubbles)
{
	std::vector<std::string> methods = {"None", "Insertion", "Substitution",
										"Deletion", "Homopolymer"};

	for (auto& bubble : bubbles)
	{
		for (auto& stepInfo : bubble.polishSteps)
		{
			 _logFile << std::fixed
				 << std::setw(22) << std::left << "Consensus: " 
				 << std::right << stepInfo.sequence << std::endl
				 << std::setw(22) << std::left << "Score: " << std::right 
				 << std::setprecision(2) << stepInfo.score << std::endl;

			_logFile << std::endl;
		}
		_logFile << "-----------------\n";
	}
}


void BubbleProcessor::cacheBubbles(int maxRead)
{
	std::string buffer;
	std::string candidate;

	int readBubbles = 0;
	while (!_bubblesFile.eof() && readBubbles < maxRead)
	{
		std::getline(_bubblesFile, buffer);
		if (buffer.empty()) break;

		std::vector<std::string> elems = splitString(buffer, ' ');
		if (elems.size() != 4 || elems[0][0] != '>')
		{
			throw std::runtime_error("Error parsing bubbles file");
		}
		std::getline(_bubblesFile, candidate);
		std::transform(candidate.begin(), candidate.end(), 
				       candidate.begin(), ::toupper);
		
		Bubble bubble;
		bubble.candidate = candidate;
		bubble.header = elems[0].substr(1, std::string::npos);
		bubble.position = std::stoi(elems[1]);
		int numOfReads = std::stoi(elems[2]);
		bubble.subPosition = std::stoi(elems[3]);

		int count = 0;
		while (count < numOfReads) 
		{
			if (buffer.empty()) break;

			std::getline(_bubblesFile, buffer);
			std::getline(_bubblesFile, buffer);
			std::transform(buffer.begin(), buffer.end(), 
				       	   buffer.begin(), ::toupper);
			bubble.branches.push_back(buffer);
			count++;
		}
		if (count != numOfReads)
		{
			//std::cerr << buffer << " " << count << " " << numOfReads << std::endl;
			throw std::runtime_error("Error parsing bubbles file");
		}

		_cachedBubbles.push_back(std::move(bubble));
		++readBubbles;
	}

	int64_t filePos = _bubblesFile.tellg();
	if (_showProgress && filePos > 0)
	{
		_progress.setValue(filePos);
	}
}
