//(c) 2016 by Authors
//This file is a part of ABruijn program.
//Released under the BSD license (see LICENSE file)

#include <chrono>
#include <thread>
#include <iomanip>
#include <sys/stat.h>

#include "bubble_processor.h"


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
	std::vector<std::thread> threads(numThreads);
	for (size_t i = 0; i < threads.size(); ++i)
	{
        std::string inFile = inBubbles;
        size_t dotPos = inFile.find('.');
        inFile.insert(dotPos, "_" + std::to_string(i));

        std::string outFile = outConsensus;
        dotPos = outFile.find('.');
        outFile.insert(dotPos, "_" + std::to_string(i));

        threads[i] = std::thread(&BubbleProcessor::parallelWorker, this, inFile, outFile);
	}
	for (size_t i = 0; i < threads.size(); ++i)
	{
		threads[i].join();
	}
}


void BubbleProcessor::parallelWorker(const std::string inFile, const std::string outFile)
{
    auto start = std::chrono::high_resolution_clock::now();

    std::ifstream bubblesFile(inFile);
    if (!bubblesFile.is_open())
    {
        throw std::runtime_error("Error opening output file: " + outFile);
    }

    std::ofstream consensusFile(outFile);
    if (!consensusFile.is_open())
    {
        throw std::runtime_error("Error opening output file: " + outFile);
    }

    std::thread::id threadId = std::this_thread::get_id();

    std::chrono::duration<double> duration(0);
    std::chrono::duration<double> cacheBubblesDuration(0);
    std::chrono::duration<double> writeBubblesDuration(0);
    std::chrono::duration<double> generalPolisherDuration(0);
    std::chrono::duration<double> homoPolisherDuration(0);
    std::chrono::duration<double> fixerDuration(0);

//    std::chrono::duration<double> polishClosestBranchesDuration(0);
//    std::chrono::duration<double> polishAllBranchesDuration(0);

    const int MAX_BUBBLE = 5000;
    const int BATCH_SIZE = 10;
    int numBubbles = 0;
    int numBubblesPolished = 0;
    int counter = 0;
    std::queue<Bubble> bubbles;
    std::ostringstream bufferedBubbles;

    while (true)
    {
        if (bubbles.empty())
        {
            auto cacheBubblesStart = std::chrono::high_resolution_clock::now();
            this->cacheBubbles(bubblesFile, bubbles, BATCH_SIZE);
            auto cacheBubblesEnd = std::chrono::high_resolution_clock::now();
            cacheBubblesDuration += cacheBubblesEnd - cacheBubblesStart;

            if(bubbles.empty())
            {
                if (counter != 0) {
                    auto startWriting = std::chrono::high_resolution_clock::now();
                    consensusFile << bufferedBubbles.str();
                    auto endWriting = std::chrono::high_resolution_clock::now();
                    writeBubblesDuration += endWriting - startWriting;
                }

                bubblesFile.close();
                consensusFile.close();

                auto end = std::chrono::high_resolution_clock::now(); // End timer
                duration = end - start;

                _Mutex.lock();

                std::cout << std::endl;
                std::cout << "thread id: " << threadId << std::endl;
                std::cout << "number of bubbles: " << numBubbles << std::endl;
                std::cout << "number of polished bubbles: " << numBubblesPolished << std::endl;
                std::cout << "parallelWorker: " << std::fixed << std::setprecision(2) << duration.count() << " seconds" << std::endl;
                std::cout << "cache bubbles: " << std::fixed << std::setprecision(2) << cacheBubblesDuration.count() << " seconds" << std::endl;
                std::cout << "write bubbles: " << std::fixed << std::setprecision(2) << writeBubblesDuration.count() << " seconds" << std::endl;
                std::cout << "_generalPolisher: " << std::fixed << std::setprecision(2) << generalPolisherDuration.count() << " seconds" << std::endl;
                std::cout << "_homoPolisher: " << std::fixed << std::setprecision(2) << homoPolisherDuration.count() << " seconds" << std::endl;
                std::cout << "_dinucFixer: " << std::fixed << std::setprecision(2) << fixerDuration.count() << " seconds" << std::endl;

//                std::cout << "polish closest branches: " << std::fixed << std::setprecision(2) << polishClosestBranchesDuration.count() << " seconds" << std::endl;
//                std::cout << "polish all branches: " << std::fixed << std::setprecision(2) << polishAllBranchesDuration.count() << " seconds" << std::endl;

                _Mutex.unlock();

                return;
            }
        }

        while(!bubbles.empty()) {
            Bubble bubble = bubbles.front();
            bubbles.pop();
            numBubbles++;
            counter++;
            
            if (bubble.candidate.size() < MAX_BUBBLE &&
                bubble.branches.size() > 1)
            {
                numBubblesPolished++;

                auto generalPolisherStart = std::chrono::high_resolution_clock::now();
                _generalPolisher.polishBubble(bubble);
//            _generalPolisher.polishBubble(bubble, polishClosestBranchesDuration, polishAllBranchesDuration);
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

            auto startWriting = std::chrono::high_resolution_clock::now();
            bufferedBubbles << ">" << bubble.header << " " << bubble.position
                            << " " << bubble.branches.size() << " " << bubble.subPosition << std::endl
                            << bubble.candidate << std::endl;
            auto endWriting = std::chrono::high_resolution_clock::now();
            writeBubblesDuration += endWriting - startWriting;
        }

        if (counter >= BATCH_SIZE * 10) {
            auto startWriting = std::chrono::high_resolution_clock::now();
            consensusFile << bufferedBubbles.str();
            auto endWriting = std::chrono::high_resolution_clock::now();
            writeBubblesDuration += endWriting - startWriting;

            counter = 0;
            bufferedBubbles.str("");
        }
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


void BubbleProcessor::cacheBubbles(std::ifstream& bubblesFile, std::queue<Bubble>& bubbles, int maxRead)
{
	std::string buffer;
	std::string candidate;

	int readBubbles = 0;
	while (!bubblesFile.eof() && readBubbles < maxRead)
	{
		std::getline(bubblesFile, buffer);
		if (buffer.empty()) break;

		std::vector<std::string> elems = splitString(buffer, ' ');
		if (elems.size() != 4 || elems[0][0] != '>')
		{
			throw std::runtime_error("Error parsing bubbles file");
		}
		std::getline(bubblesFile, candidate);
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

			std::getline(bubblesFile, buffer);
			std::getline(bubblesFile, buffer);
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

        bubbles.push(std::move(bubble));
		++readBubbles;
	}
}