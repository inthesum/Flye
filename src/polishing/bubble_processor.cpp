//(c) 2016 by Authors
//This file is a part of ABruijn program.
//Released under the BSD license (see LICENSE file)

#include <chrono>
#include <thread>
#include <iomanip>
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
	_preprocessBubbles.clear();
	_preprocessBubbles.reserve(BUBBLES_CACHE);

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

	std::vector<std::thread> threads(numThreads);
	for (size_t i = 0; i < threads.size(); ++i)
	{
        std::string filename = outConsensus;
        size_t dotPos = filename.find('.');
        filename.insert(dotPos, "_" + std::to_string(i));
        threads[i] = std::thread(&BubbleProcessor::parallelWorker, this, filename);
	}
	for (size_t i = 0; i < threads.size(); ++i)
	{
		threads[i].join();
	}
	if (_showProgress) _progress.setDone();
}


void BubbleProcessor::parallelWorker(const std::string outFile)
{
    auto start = std::chrono::high_resolution_clock::now();

    std::ofstream consensusFile(outFile);
    if (!consensusFile.is_open())
    {
        throw std::runtime_error("Error opening output file: " + outFile);
    }

    std::thread::id threadId = std::this_thread::get_id();

    std::chrono::duration<double> duration(0);
    std::chrono::duration<double> cacheBubblesDuration(0);
    std::chrono::duration<double> generalPolisherDuration(0);
    std::chrono::duration<double> homoPolisherDuration(0);
    std::chrono::duration<double> fixerDuration(0);
    std::chrono::duration<double> waitReadDuration(0);
    std::chrono::duration<double> writeBubblesDuration(0);

    std::chrono::duration<double> polishClosestBranchesDuration(0);
    std::chrono::duration<double> polishAllBranchesDuration(0);

    const int MAX_BUBBLE = 5000;
    const int BATCH_SIZE = 100;
    int numBubbles = 0;
    int numBubblesPolished = 0;
    int counter = 0;
    std::ostringstream bufferedBubbles;

    auto startWaiting = std::chrono::high_resolution_clock::now();
    _readMutex.lock();
    auto endWaiting = std::chrono::high_resolution_clock::now();
    waitReadDuration += endWaiting - startWaiting;

    while (true)
    {
        if (_preprocessBubbles.empty())
        {
            auto cacheBubblesStart = std::chrono::high_resolution_clock::now();
            this->cacheBubbles(BUBBLES_CACHE);
            auto cacheBubblesEnd = std::chrono::high_resolution_clock::now();
            cacheBubblesDuration += cacheBubblesEnd - cacheBubblesStart;

            if(_preprocessBubbles.empty())
            {
                if (counter != 0) {
                    auto startWriting = std::chrono::high_resolution_clock::now();
                    consensusFile << bufferedBubbles.str();
                    auto endWriting = std::chrono::high_resolution_clock::now();
                    writeBubblesDuration += endWriting - startWriting;
                }

                auto end = std::chrono::high_resolution_clock::now(); // End timer
                duration = end - start;
                std::cout << std::endl;
                std::cout << "thread id: " << threadId << std::endl;
//                std::cout << "output file: " << outFile << std::endl;
                std::cout << "number of bubbles: " << numBubbles << std::endl;
                std::cout << "number of polished bubbles: " << numBubblesPolished << std::endl;
                std::cout << "parallelWorker: " << std::fixed << std::setprecision(2) << duration.count() << " seconds" << std::endl;
                std::cout << "waiting for read: " << std::fixed << std::setprecision(2) << waitReadDuration.count() << " seconds" << std::endl;
                std::cout << "cache bubbles: " << std::fixed << std::setprecision(2) << cacheBubblesDuration.count() << " seconds" << std::endl;
                std::cout << "write bubbles: " << std::fixed << std::setprecision(2) << writeBubblesDuration.count() << " seconds" << std::endl;
                std::cout << "_generalPolisher: " << std::fixed << std::setprecision(2) << generalPolisherDuration.count() << " seconds" << std::endl;
                std::cout << "_homoPolisher: " << std::fixed << std::setprecision(2) << homoPolisherDuration.count() << " seconds" << std::endl;
                std::cout << "_dinucFixer: " << std::fixed << std::setprecision(2) << fixerDuration.count() << " seconds" << std::endl;

//                std::cout << "polish closest branches: " << std::fixed << std::setprecision(2) << polishClosestBranchesDuration.count() << " seconds" << std::endl;
//                std::cout << "polish all branches: " << std::fixed << std::setprecision(2) << polishAllBranchesDuration.count() << " seconds" << std::endl;

                _readMutex.unlock();

                return;
            }
        }

        Bubble bubble = _preprocessBubbles.back();
        _preprocessBubbles.pop_back();
        numBubbles++;
        counter++;

        _readMutex.unlock();

        if (bubble.candidate.size() < MAX_BUBBLE &&
            bubble.branches.size() > 1)
        {
            numBubblesPolished++;

            auto generalPolisherStart = std::chrono::high_resolution_clock::now();
            _generalPolisher.polishBubble(bubble, polishClosestBranchesDuration, polishAllBranchesDuration);
            auto generalPolisherEnd = std::chrono::high_resolution_clock::now();
            generalPolisherDuration += generalPolisherEnd - generalPolisherStart;

            auto homoPolisherStart = std::chrono::high_resolution_clock::now();
            if (_hopoEnabled)
            {
                _homoPolisher.polishBubble(bubble);
            }
            auto homoPolisherEnd = std::chrono::high_resolution_clock::now();
            homoPolisherDuration += homoPolisherEnd - homoPolisherStart;

            auto fixerStart = std::chrono::high_resolution_clock::now();
            _dinucFixer.fixBubble(bubble);
            auto fixerEnd = std::chrono::high_resolution_clock::now();
            fixerDuration += fixerEnd - fixerStart;
        }

        bufferedBubbles << ">" << bubble.header << " " << bubble.position
                        << " " << bubble.branches.size() << " " << bubble.subPosition << std::endl
                        << bubble.candidate << std::endl;

        if (counter >= BATCH_SIZE) {
            auto startWriting = std::chrono::high_resolution_clock::now();
            consensusFile << bufferedBubbles.str();
            auto endWriting = std::chrono::high_resolution_clock::now();
            writeBubblesDuration += endWriting - startWriting;

            counter = 0;
            bufferedBubbles.str("");
        }

        startWaiting = std::chrono::high_resolution_clock::now();
        _readMutex.lock();
        endWaiting = std::chrono::high_resolution_clock::now();
        waitReadDuration += endWaiting - startWaiting;
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

		_preprocessBubbles.push_back(std::move(bubble));
		++readBubbles;
	}

	int64_t filePos = _bubblesFile.tellg();
	if (_showProgress && filePos > 0)
	{
		_progress.setValue(filePos);
	}
}
