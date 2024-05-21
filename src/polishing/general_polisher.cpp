//(c) 2016 by Authors
//This file is a part of ABruijn program.
//Released under the BSD license (see LICENSE file)

#include "general_polisher.h"

constexpr size_t batchSize = 8;

void GeneralPolisher::polishBubble(Bubble& bubble,
                                   int64_t& alignmentNum,
                                   int64_t& deletionNum,
                                   int64_t& insertionNum,
                                   int64_t& substitutionNum,
                                   std::chrono::duration<double>& optimizeDuration,
                                   std::chrono::duration<double>& makeStepDuration,
                                   std::chrono::duration<double>& alignmentDuration,
                                   std::chrono::duration<double>& deletionDuration,
                                   std::chrono::duration<double>& insertionDuration,
                                   std::chrono::duration<double>& substitutionDuration) const
{
	auto optimize = [this] (const std::string& candidate,
							const std::vector<std::string>& branches,
                            const size_t readsNum,
							std::vector<StepInfo>& polishSteps,
//                            ScoreMemoryPool& memoryPool,
                            int64_t& alignmentNum,
                            int64_t& deletionNum,
                            int64_t& insertionNum,
                            int64_t& substitutionNum,
                            std::chrono::duration<double>& makeStepDuration,
                            std::chrono::duration<double>& alignmentDuration,
                            std::chrono::duration<double>& deletionDuration,
                            std::chrono::duration<double>& insertionDuration,
                            std::chrono::duration<double>& substitutionDuration)
	{
		std::string prevCandidate = candidate;
        AlignmentAVX align(branches.size(), _subsMatrix, branches);
        size_t iterNum = 0;
		while(true)
		{
//            memoryPool.reset();

            auto makeStepStart = std::chrono::high_resolution_clock::now();

            StepInfo rec = this->makeStep(prevCandidate,
                                          branches,
                                          readsNum,
                                          align,
//                                          memoryPool,
                                          alignmentNum,
                                          deletionNum,
                                          insertionNum,
                                          substitutionNum,
                                          alignmentDuration,
                                          deletionDuration,
                                          insertionDuration,
                                          substitutionDuration);

            auto makeStepEnd = std::chrono::high_resolution_clock::now();
            makeStepDuration += makeStepEnd - makeStepStart;

			polishSteps.push_back(rec);
			if (prevCandidate == rec.sequence) break;
			if (rec.score > 0)
			{
				std::cerr << "Overflow!\n";
				break;
			}
			if (iterNum++ > 10 * candidate.size())
			{
				std::cerr << "Too many iters!\n";
				break;
			}
			prevCandidate = rec.sequence;
		}
		return prevCandidate;
	};

	//first, select closest X branches (by length) and polish with them
    const int PRE_POLISH = 5;
    std::string prePolished = bubble.candidate;

    std::sort(bubble.branches.begin(), bubble.branches.end(),
              [](const std::string& s1, const std::string& s2)
              {return s1.length() < s2.length();});

    if (bubble.branches.size() > PRE_POLISH * 2)
    {
        size_t left = bubble.branches.size() / 2 - PRE_POLISH / 2;
        size_t right = left + PRE_POLISH;
        std::vector<std::string> reducedSet(bubble.branches.begin() + left,
                                            bubble.branches.begin() + right);

        const size_t readsNum = PRE_POLISH;
        size_t extendedReadsNum = 0;
        std::vector <std::string> &extendedReads = reducedSet;
        if(readsNum % batchSize != 0) {
            const size_t extendedReadsNum = batchSize - readsNum % batchSize;
            std::string lastRead = reducedSet[readsNum - 1];
            for (size_t i = 0; i < extendedReadsNum; i++) extendedReads.push_back(lastRead);
        }

//        ScoreMemoryPool memoryPool(4 * prePolished.size() * lastRead.size() * extendedReads.size());

        auto optimizeStart = std::chrono::high_resolution_clock::now();

        prePolished = optimize(prePolished,
                               extendedReads,
                               readsNum,
                               bubble.polishSteps,
//                               memoryPool,
                               alignmentNum,
                               deletionNum,
                               insertionNum,
                               substitutionNum,
                               makeStepDuration,
                               alignmentDuration,
                               deletionDuration,
                               insertionDuration,
                               substitutionDuration);

        auto optimizeEnd = std::chrono::high_resolution_clock::now();
        optimizeDuration += optimizeEnd - optimizeStart;
    }

	//then, polish with all branches
    const size_t readsNum = bubble.branches.size();
    size_t extendedReadsNum = 0;
    std::vector<std::string>& extendedReads = bubble.branches;
    if(readsNum % batchSize != 0) {
        extendedReadsNum = batchSize - readsNum % batchSize;
        std::string lastRead = bubble.branches[readsNum - 1];
        for (size_t i = 0; i < extendedReadsNum; i++) extendedReads.push_back(lastRead);
    }

//    ScoreMemoryPool memoryPool(4 * prePolished.size() * lastRead.size() * extendedReads.size());

    auto optimizeStart = std::chrono::high_resolution_clock::now();

    bubble.candidate = optimize(prePolished,
                                extendedReads,
                                readsNum,
                                bubble.polishSteps,
//                                memoryPool,
                                alignmentNum,
                                deletionNum,
                                insertionNum,
                                substitutionNum,
                                makeStepDuration,
                                alignmentDuration,
                                deletionDuration,
                                insertionDuration,
                                substitutionDuration);

    auto optimizeEnd = std::chrono::high_resolution_clock::now();
    optimizeDuration += optimizeEnd - optimizeStart;

    for (size_t i = 0; i < extendedReadsNum; i++) extendedReads.pop_back();
}

StepInfo GeneralPolisher::makeStep(const std::string& candidate,
                                   const std::vector<std::string>& branches,
                                   const size_t readsNum,
                                   AlignmentAVX& align,
//                                   ScoreMemoryPool& memoryPool,
                                   int64_t& alignmentNum,
                                   int64_t& deletionNum,
                                   int64_t& insertionNum,
                                   int64_t& substitutionNum,
                                   std::chrono::duration<double>& alignmentDuration,
                                   std::chrono::duration<double>& deletionDuration,
                                   std::chrono::duration<double>& insertionDuration,
                                   std::chrono::duration<double>& substitutionDuration) const
{
    static char alphabet[] = {'A', 'C', 'G', 'T'};
    StepInfo stepResult;

    //Alignment
    auto alignmentStart = std::chrono::high_resolution_clock::now();

    AlnScoreType score = align.globalAlignmentAVX(candidate, branches, readsNum);
//    AlnScoreType score = align.globalAlignmentAVX(candidate, branches, readsNum, memoryPool);
//    AlnScoreType score = align.globalAlignmentAVX(candidate, branches, readsNum, alignmentDuration);
//    AlnScoreType score = align.globalAlignmentAVX(candidate, branches, readsNum, memoryPool, alignmentDuration);

    stepResult.score = score;
    stepResult.sequence = candidate;
    alignmentNum++;

    auto alignmentEnd = std::chrono::high_resolution_clock::now();
    alignmentDuration += alignmentEnd - alignmentStart;

    //Deletion
    auto deletionStart = std::chrono::high_resolution_clock::now();

    bool improvement = false;
    for (size_t pos = 0; pos < candidate.size(); ++pos)
    {
        AlnScoreType score = align.addDeletionAVX(pos + 1, readsNum);
//        AlnScoreType score = align.addDeletionAVX(pos + 1, readsNum, deletionDuration);

        if (score > stepResult.score)
        {
            stepResult.score = score;
            stepResult.sequence = candidate;
            stepResult.sequence.erase(pos, 1);
            improvement = true;
        }
    }
    deletionNum++;

    auto deletionEnd = std::chrono::high_resolution_clock::now();
    deletionDuration += deletionEnd - deletionStart;

    if (improvement) return stepResult;

    //Insertion
    auto insertionStart = std::chrono::high_resolution_clock::now();

    for (size_t pos = 0; pos < candidate.size() + 1; ++pos)
    {
        for (char letter : alphabet)
        {
            AlnScoreType score = align.addInsertionAVX(pos + 1, letter, branches, readsNum);
            if (score > stepResult.score)
            {
                stepResult.score = score;
                stepResult.sequence = candidate;
                stepResult.sequence.insert(pos, 1, letter);
                improvement = true;
            }
        }
    }
    insertionNum++;

    auto insertionEnd = std::chrono::high_resolution_clock::now();
    insertionDuration += insertionEnd - insertionStart;

    if (improvement) return stepResult;

    //Substitution
    auto substitutionStart = std::chrono::high_resolution_clock::now();

    for (size_t pos = 0; pos < candidate.size(); ++pos)
    {
        for (char letter : alphabet)
        {
            if (letter == candidate[pos]) continue;

            AlnScoreType score = align.addSubstitutionAVX(pos + 1, letter, branches, readsNum);
            if (score > stepResult.score)
            {
                stepResult.score = score;
                stepResult.sequence = candidate;
                stepResult.sequence[pos] = letter;
            }
        }
    }
    substitutionNum++;

    auto substitutionEnd = std::chrono::high_resolution_clock::now();
    substitutionDuration += substitutionEnd - substitutionStart;

    return stepResult;
}