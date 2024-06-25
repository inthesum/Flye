//(c) 2016 by Authors
//This file is a part of ABruijn program.
//Released under the BSD license (see LICENSE file)

#include <ctime>
#include <iostream>
#include <getopt.h>
#include <cstring>
#include <thread>
#include "../common/logger.h"
#include "../common/memory_info.h"

#include "../polishing/bubble_processor.h"


bool parseArgs(int argc, char** argv, std::string& bubblesFile, 
			   std::string& scoringMatrix, std::string& hopoMatrix,
			   std::string& outConsensus, std::string& outVerbose,
               std::string& logFile, int& numThreads, bool& quiet, bool& enableHopo)
{
	auto printUsage = [argv]()
	{
		std::cerr << "Usage: flye-polish "
				  << " --bubbles path --subs-mat path --hopo-mat size --out path\n"
				  << "\t\t[--log path] [--treads num] [--enable-hopo] [--quiet] [--debug] [-h]\n\n"
				  << "Required arguments:\n"
				  << "  --bubbles path\tpath to bubbles file\n"
				  << "  --subs-mat path\tpath to substitution matrix\n"
				  << "  --hopo-mat size\tpath to homopolymer matrix\n"
				  << "  --out path\tpath to output file\n\n"
				  << "Optional arguments:\n"
				  << "  --quiet \t\tno terminal output "
				  << "[default = false] \n"
				  << "  --enable-hopo \t\tenable homopolymer polishing "
				  << "[default = false] \n"
				  << "  --debug \t\textra debug output "
				  << "[default = false] \n"
                  << "  --log log_file\toutput log to file "
                  << "[default = not set] \n"
				  << "  --threads num_threads\tnumber of parallel threads "
				  << "[default = 1] \n";
	};
	
	int optionIndex = 0;
	static option longOptions[] =
	{
		{"bubbles", required_argument, 0, 0},
		{"subs-mat", required_argument, 0, 0},
		{"hopo-mat", required_argument, 0, 0},
		{"out", required_argument, 0, 0},
        {"log", required_argument, 0, 0},
        {"threads", required_argument, 0, 0},
		{"debug", no_argument, 0, 0},
		{"quiet", no_argument, 0, 0},
		{"enable-hopo", no_argument, 0, 0},
		{0, 0, 0, 0}
	};

	int opt = 0;
	while ((opt = getopt_long(argc, argv, "h", longOptions, &optionIndex)) != -1)
	{
		switch(opt)
		{
		case 0:
			if (!strcmp(longOptions[optionIndex].name, "threads"))
				numThreads = atoi(optarg);
			else if (!strcmp(longOptions[optionIndex].name, "debug"))
				outVerbose = true;
			else if (!strcmp(longOptions[optionIndex].name, "enable-hopo"))
				enableHopo = true;
			else if (!strcmp(longOptions[optionIndex].name, "quiet"))
				quiet = true;
			else if (!strcmp(longOptions[optionIndex].name, "bubbles"))
				bubblesFile = optarg;
			else if (!strcmp(longOptions[optionIndex].name, "subs-mat"))
				scoringMatrix = optarg;
			else if (!strcmp(longOptions[optionIndex].name, "hopo-mat"))
				hopoMatrix = optarg;
			else if (!strcmp(longOptions[optionIndex].name, "out"))
				outConsensus = optarg;
            else if (!strcmp(longOptions[optionIndex].name, "log"))
                logFile = optarg;
			break;

		case 'h':
			printUsage();
			exit(0);
		}
	}
	if (bubblesFile.empty() || scoringMatrix.empty() || 
		hopoMatrix.empty() || outConsensus.empty())
	{
		printUsage();
		return false;
	}

	return true;
}

int polisher_main(int argc, char* argv[]) 
{
	std::string bubblesFile;
	std::string scoringMatrix;
	std::string hopoMatrix;
	std::string outConsensus;
	std::string outVerbose;
    std::string logFile;
    int  numThreads = 1;
	bool quiet = false;
	bool enableHopo = false;
    bool debugging = false;

    if (!parseArgs(argc, argv, bubblesFile, scoringMatrix,
				   hopoMatrix, outConsensus, outVerbose,
                   logFile, numThreads, quiet, enableHopo))
		return 1;

    Logger::get().setDebugging(debugging);
    if (!logFile.empty()) Logger::get().setOutputFile(logFile);
    Logger::get().debug() << "Build date: " << __DATE__ << " " << __TIME__;
    std::ios::sync_with_stdio(false);

    Logger::get().debug() << "Total RAM: "
                          << getMemorySize() / 1024 / 1024 / 1024 << " Gb";
    Logger::get().debug() << "Available RAM: "
                          << getFreeMemorySize() / 1024 / 1024 / 1024 << " Gb";
    Logger::get().debug() << "Total CPUs: " << std::thread::hardware_concurrency();

    if (numThreads <= 16) {
        BubbleProcessor bp(scoringMatrix, hopoMatrix, !quiet, enableHopo, numThreads);
        bp.polishAll(bubblesFile, outConsensus);
    } else {
        int numThreads1 = numThreads / 2;
        int numThreads2 = numThreads - numThreads1;

        std::string bubblesFile1 = bubblesFile;
        size_t dotPos1 = bubblesFile1.find('.');
        bubblesFile1.insert(dotPos1, "_a");

        std::string outConsensus1 = outConsensus;
        dotPos1 = outConsensus1.find('.');
        outConsensus1.insert(dotPos1, "_a");

        std::string bubblesFile2 = bubblesFile;
        size_t dotPos2 = bubblesFile2.find('.');
        bubblesFile2.insert(dotPos2, "_b");

        std::string outConsensus2 = outConsensus;
        dotPos2 = outConsensus2.find('.');
        outConsensus2.insert(dotPos2, "_b");

        BubbleProcessor bp1(scoringMatrix, hopoMatrix, !quiet, enableHopo, numThreads1);
        BubbleProcessor bp2(scoringMatrix, hopoMatrix, !quiet, enableHopo, numThreads2);

        std::ifstream in_file(bubblesFile1, std::ios::binary);
        in_file.seekg(0, std::ios::end);
        int file_size = in_file.tellg();
        Logger::get().debug()<<"Size of the file is"<<" "<< file_size<<" "<<"bytes";

        std::thread t1(&BubbleProcessor::polishAll, &bp1, bubblesFile1, outConsensus1);
        std::thread t2(&BubbleProcessor::polishAll, &bp2, bubblesFile2, outConsensus2);

        t1.join();
        t2.join();
    }

    Logger::get().debug() << "Peak RAM usage: "
                          << getPeakRSS() / 1024 / 1024 / 1024 << " Gb";

	return 0;
}
