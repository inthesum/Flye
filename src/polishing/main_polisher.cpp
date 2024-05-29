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
#include "../polishing/bubble_processor_pro.h"


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

    if(numThreads < 10) {
        BubbleProcessor bp(scoringMatrix, hopoMatrix, !quiet, enableHopo, numThreads);
        if (!outVerbose.empty()) bp.enableVerboseOutput(outVerbose);
        bp.polishAll(bubblesFile, outConsensus);
    } else {
        BubbleProcessorPro bp(scoringMatrix, hopoMatrix, !quiet, enableHopo, numThreads);
        if (!outVerbose.empty()) bp.enableVerboseOutput(outVerbose);
        bp.polishAll(bubblesFile, outConsensus);
    }

    Logger::get().debug() << "Peak RAM usage: "
                          << getPeakRSS() / 1024 / 1024 / 1024 << " Gb";

	return 0;
}
