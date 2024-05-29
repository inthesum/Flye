//(c) 2016 by Authors
//This file is a part of ABruijn program.
//Released under the BSD license (see LICENSE file)

#include <chrono>
#include <thread>
#include <iomanip>
#include <sys/stat.h>

#include "bubble_processor_pro.h"

// Define and initialize static members
std::mutex BubbleProcessorPro::_mtx;
std::condition_variable BubbleProcessorPro::_cv_reader;
std::condition_variable BubbleProcessorPro::_cv_processor;
bool BubbleProcessorPro::_ready_to_read = true;
bool BubbleProcessorPro::_ready_to_process = false;
bool BubbleProcessorPro::_done = false;

namespace
{
    size_t fileSize(const std::string& filename)
    {
        struct stat st;
        if (stat(filename.c_str(), &st) != 0) return 0;
        return st.st_size;
    }
}


BubbleProcessorPro::BubbleProcessorPro(const std::string& subsMatPath,
                                 const std::string& hopoMatrixPath,
                                 bool showProgress, bool hopoEnabled,
                                 int numThreads):
        _subsMatrix(subsMatPath),
        _hopoMatrix(hopoMatrixPath),
        _generalPolisher(_subsMatrix),
        _homoPolisher(_subsMatrix, _hopoMatrix),
        _dinucFixer(_subsMatrix),
        _verbose(false),
        _showProgress(showProgress),
        _hopoEnabled(hopoEnabled),
        _numThreads(numThreads)
{}


void BubbleProcessorPro::polishAll(const std::string& inBubbles,
                                const std::string& outConsensus)
{
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

    std::vector<std::thread> threads(_numThreads);
    threads[0] = std::thread(&BubbleProcessorPro::readThread, this);
    for (size_t i = 1; i < threads.size(); ++i)
    {
        std::string filename = outConsensus;
        size_t dotPos = filename.find('.');
        filename.insert(dotPos, "_" + std::to_string(i));
        threads[i] = std::thread(&BubbleProcessorPro::processThread, this, filename);
    }
    for (size_t i = 0; i < threads.size(); ++i)
    {
        threads[i].join();
    }

    if (_showProgress) _progress.setDone();
}


void BubbleProcessorPro::processThread(const std::string outFile)
{
    auto start = std::chrono::high_resolution_clock::now();

    std::ofstream consensusFile(outFile);
    if (!consensusFile.is_open())
    {
        throw std::runtime_error("Error opening output file: " + outFile);
    }

    std::chrono::duration<double> duration(0);
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
    int counter = 0;
    std::queue<std::unique_ptr<Bubble>> bubbles;
    std::ostringstream bufferedBubbles;

    while (true)
    {
        auto startWaiting = std::chrono::high_resolution_clock::now();

        std::unique_lock<std::mutex> lock(_mtx);
        _cv_processor.wait(lock, []{ return _ready_to_process || _done; });

        auto endWaiting = std::chrono::high_resolution_clock::now();
        waitReadDuration += endWaiting - startWaiting;

        if (_done && _preprocessBubbles.empty()) {
            if (counter != 0) {
                auto startWriting = std::chrono::high_resolution_clock::now();
                consensusFile << bufferedBubbles.str();
                auto endWriting = std::chrono::high_resolution_clock::now();
                writeBubblesDuration += endWriting - startWriting;
            }

            consensusFile.close();

            auto end = std::chrono::high_resolution_clock::now(); // End timer
            duration = end - start;
            std::cout << std::endl;
            std::cout << "thread id: " << std::this_thread::get_id() << std::endl;
            std::cout << "number of bubbles: " << numBubbles << std::endl;
            std::cout << "number of polished bubbles: " << numBubblesPolished << std::endl;
            std::cout << "live: " << std::fixed << std::setprecision(2) << duration.count() << " seconds" << std::endl;
            std::cout << "waiting for read: " << std::fixed << std::setprecision(2) << waitReadDuration.count() << " seconds" << std::endl;
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

            lock.unlock();

            return;
        }

        if (!_preprocessBubbles.empty()) {
            for(int i=0; i<_batchSize && !_preprocessBubbles.empty(); i++) {
                bubbles.push(std::move(_preprocessBubbles.front()));
                _preprocessBubbles.pop();
                numBubbles++;
                counter++;
            }

            if (_preprocessBubbles.empty() && !_done) {
                _ready_to_read = true;
                _ready_to_process = false;
                _cv_reader.notify_one();
            }

            lock.unlock();

            while(!bubbles.empty()) {
                std::unique_ptr<Bubble> bubble = std::move(bubbles.front());
                bubbles.pop();

                if (bubble->candidate.size() < MAX_BUBBLE &&
                    bubble->branches.size() > 1)
                {
                    numBubblesPolished++;

                    auto generalPolisherStart = std::chrono::high_resolution_clock::now();
//                _generalPolisher.polishBubble(bubble);

                    _generalPolisher.polishBubble(*bubble,
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
                    if (_hopoEnabled)
                    {
                        _homoPolisher.polishBubble(*bubble);
                    }
                    auto homoPolisherEnd = std::chrono::high_resolution_clock::now();
                    homoPolisherDuration += homoPolisherEnd - homoPolisherStart;

                    auto fixerStart = std::chrono::high_resolution_clock::now();
                    _dinucFixer.fixBubble(*bubble);
                    auto fixerEnd = std::chrono::high_resolution_clock::now();
                    fixerDuration += fixerEnd - fixerStart;
                }

                auto startWriting = std::chrono::high_resolution_clock::now();
                bufferedBubbles << ">" << bubble->header << " " << bubble->position
                                << " " << bubble->branches.size() << " " << bubble->subPosition << std::endl
                                << bubble->candidate << std::endl;
                auto endWriting = std::chrono::high_resolution_clock::now();
                writeBubblesDuration += endWriting - startWriting;
            }

            if (counter >= _batchSize * 100) {
                auto startWriting = std::chrono::high_resolution_clock::now();
                consensusFile << bufferedBubbles.str();
                auto endWriting = std::chrono::high_resolution_clock::now();
                writeBubblesDuration += endWriting - startWriting;

                counter = 0;
                bufferedBubbles.str("");
            }
        }
    }
}


void BubbleProcessorPro::enableVerboseOutput(const std::string& filename)
{
    _verbose = true;
    _logFile.open(filename);
    if (!_logFile.is_open())
    {
        throw std::runtime_error("Error opening log file");
    }
}


void BubbleProcessorPro::readThread() {
    auto start = std::chrono::high_resolution_clock::now();

    std::queue<std::unique_ptr<Bubble>> bubbles;
    std::chrono::duration<double> duration(0);
    std::chrono::duration<double> cacheBubblesDuration(0);
    std::chrono::duration<double> waitDuration(0);

    while (!_bubblesFile.eof()) {
        auto cacheBubblesStart = std::chrono::high_resolution_clock::now();

        this->cacheBubbles(bubbles, _batchSize * _numThreads - 1);

        auto cacheBubblesEnd = std::chrono::high_resolution_clock::now();
        cacheBubblesDuration += cacheBubblesEnd - cacheBubblesStart;

        auto waitStart = std::chrono::high_resolution_clock::now();

        std::unique_lock<std::mutex> lock(_mtx);
        _cv_reader.wait(lock, []{ return _ready_to_read; });

        auto waitEnd = std::chrono::high_resolution_clock::now();
        waitDuration += waitEnd - waitStart;

        while (!bubbles.empty()) {
            _preprocessBubbles.push(std::move(bubbles.front()));
            bubbles.pop();
        }

        _ready_to_read = false;
        _ready_to_process = true;
        _cv_processor.notify_all();
    }

    _done = true;
    _ready_to_process = true;
    _cv_processor.notify_all();
    _bubblesFile.close();

    auto end = std::chrono::high_resolution_clock::now(); // End timer
    duration = end - start;

    std::cout << "thread id: " << std::this_thread::get_id() << std::endl;
    std::cout << "live: " << std::fixed << std::setprecision(2) << duration.count() << " seconds" << std::endl;
    std::cout << "wait: " << std::fixed << std::setprecision(2) << waitDuration.count() << " seconds" << std::endl;
    std::cout << "cache bubbles: " << std::fixed << std::setprecision(2) << cacheBubblesDuration.count() << " seconds" << std::endl;
}


void BubbleProcessorPro::cacheBubbles(std::queue<std::unique_ptr<Bubble>>& bubbles, int maxRead)
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

        bubbles.push(std::make_unique<Bubble>(bubble));
        ++readBubbles;
    }

    int64_t filePos = _bubblesFile.tellg();
    if (_showProgress && filePos > 0)
    {
        _progress.setValue(filePos);
    }
}
