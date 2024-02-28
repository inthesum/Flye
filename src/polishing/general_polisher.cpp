//(c) 2016 by Authors
//This file is a part of ABruijn program.
//Released under the BSD license (see LICENSE file)

#include "general_polisher.h"
#include "alignment.h"

void GeneralPolisher::polishBubble(Bubble& bubble) const
{
    auto optimize = [this] (const std::string& candidate,
                            const std::vector<std::string>& branches,
                            std::vector<StepInfo>& polishSteps)
    {
        std::string prevCandidate = candidate;
        Alignment align(branches.size(), _subsMatrix);
        size_t iterNum = 0;
        while(true)
        {
            StepInfo rec = this->makeStep(prevCandidate, branches, align);
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
    if (bubble.branches.size() > PRE_POLISH * 2)
    {
        std::sort(bubble.branches.begin(), bubble.branches.end(),
                  [](const std::string& s1, const std::string& s2)
                  {return s1.length() < s2.length();});
        size_t left = bubble.branches.size() / 2 - PRE_POLISH / 2;
        size_t right = left + PRE_POLISH;
        std::vector<std::string> reducedSet(bubble.branches.begin() + left,bubble.branches.begin() + right);
        prePolished = optimize(prePolished, reducedSet,bubble.polishSteps);
    }

    //then, polish with all branches
    bubble.candidate = optimize(prePolished, bubble.branches,bubble.polishSteps);
}

void GeneralPolisher::polishBubble(Bubble& bubble,
                                   std::chrono::duration<double>& optimizeDuration,
                                   std::chrono::duration<double>& makeStepDuration,
                                   std::chrono::duration<double>& alignmentDuration,
                                   std::chrono::duration<double>& deletionDuration,
                                   std::chrono::duration<double>& insertionDuration,
                                   std::chrono::duration<double>& substitutionDuration) const
{
	auto optimize = [this] (const std::string& candidate,
							const std::vector<std::string>& branches,
							std::vector<StepInfo>& polishSteps,
                            std::chrono::duration<double>& makeStepDuration,
                            std::chrono::duration<double>& alignmentDuration,
                            std::chrono::duration<double>& deletionDuration,
                            std::chrono::duration<double>& insertionDuration,
                            std::chrono::duration<double>& substitutionDuration)
	{
		std::string prevCandidate = candidate;
		Alignment align(branches.size(), _subsMatrix);
		size_t iterNum = 0;
		while(true)
		{
            auto makeStepStart = std::chrono::high_resolution_clock::now();

            StepInfo rec = this->makeStep(prevCandidate,
                                          branches,
                                          align,
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
                std::cerr << rec.score << std::endl;

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
    if (bubble.branches.size() > PRE_POLISH * 2)
    {
        std::sort(bubble.branches.begin(), bubble.branches.end(),
                  [](const std::string& s1, const std::string& s2)
                  {return s1.length() < s2.length();});
        size_t left = bubble.branches.size() / 2 - PRE_POLISH / 2;
        size_t right = left + PRE_POLISH;
        std::vector<std::string> reducedSet(bubble.branches.begin() + left,
                                            bubble.branches.begin() + right);

        auto optimizeStart = std::chrono::high_resolution_clock::now();

        prePolished = optimize(prePolished,
                               reducedSet,
                               bubble.polishSteps,
                               makeStepDuration,
                               alignmentDuration,
                               deletionDuration,
                               insertionDuration,
                               substitutionDuration);

        auto optimizeEnd = std::chrono::high_resolution_clock::now();
        optimizeDuration += optimizeEnd - optimizeStart;
    }

	//then, polish with all branches
    auto optimizeStart = std::chrono::high_resolution_clock::now();

    bubble.candidate = optimize(prePolished,
                                bubble.branches,
								bubble.polishSteps,
                                makeStepDuration,
                                alignmentDuration,
                                deletionDuration,
                                insertionDuration,
                                substitutionDuration);

    auto optimizeEnd = std::chrono::high_resolution_clock::now();
    optimizeDuration += optimizeEnd - optimizeStart;
}

StepInfo GeneralPolisher::makeStep(const std::string& candidate,
                                   const std::vector<std::string>& branches,
                                   Alignment& align,
                                   std::chrono::duration<double>& alignmentDuration,
                                   std::chrono::duration<double>& deletionDuration,
                                   std::chrono::duration<double>& insertionDuration,
                                   std::chrono::duration<double>& substitutionDuration) const
{
    static char alphabet[] = {'A', 'C', 'G', 'T'};
    StepInfo stepResult;

    //Alignment
    auto alignmentStart = std::chrono::high_resolution_clock::now();

    AlnScoreType score = align.globalAlignment(candidate, branches);
    stepResult.score = score;
    stepResult.sequence = candidate;

    auto alignmentEnd = std::chrono::high_resolution_clock::now();
    alignmentDuration += alignmentEnd - alignmentStart;

    //Deletion
    auto deletionStart = std::chrono::high_resolution_clock::now();

    bool improvement = false;
    for (size_t pos = 0; pos < candidate.size(); ++pos)
    {
        AlnScoreType score = align.addDeletion(pos + 1);

        if (score > stepResult.score)
        {
            stepResult.score = score;
            stepResult.sequence = candidate;
            stepResult.sequence.erase(pos, 1);
            improvement = true;
        }
    }

    auto deletionEnd = std::chrono::high_resolution_clock::now();
    deletionDuration += deletionEnd - deletionStart;

    if (improvement) return stepResult;

    //Insertion
    auto insertionStart = std::chrono::high_resolution_clock::now();

    for (size_t pos = 0; pos < candidate.size() + 1; ++pos)
    {
        for (char letter : alphabet)
        {
            AlnScoreType score = align.addInsertion(pos + 1, letter, branches);
            if (score > stepResult.score)
            {
                stepResult.score = score;
                stepResult.sequence = candidate;
                stepResult.sequence.insert(pos, 1, letter);
                improvement = true;
            }
        }
    }

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

            AlnScoreType score = align.addSubstitution(pos + 1, letter,
                                                       branches);
            if (score > stepResult.score)
            {
                stepResult.score = score;
                stepResult.sequence = candidate;
                stepResult.sequence[pos] = letter;
            }
        }
    }

    auto substitutionEnd = std::chrono::high_resolution_clock::now();
    substitutionDuration += substitutionEnd - substitutionStart;

    return stepResult;
}

StepInfo GeneralPolisher::makeStep(const std::string& candidate, 
				   				   const std::vector<std::string>& branches,
								   Alignment& align) const
{
	static char alphabet[] = {'A', 'C', 'G', 'T'};
	StepInfo stepResult;
	
	//Alignment
	AlnScoreType score = align.globalAlignment(candidate, branches);
	stepResult.score = score;
	stepResult.sequence = candidate;

	//Deletion
	bool improvement = false;
	for (size_t pos = 0; pos < candidate.size(); ++pos) 
	{
		AlnScoreType score = align.addDeletion(pos + 1);

		if (score > stepResult.score) 
		{
			stepResult.score = score;
			stepResult.sequence = candidate;
			stepResult.sequence.erase(pos, 1);
			improvement = true;
		}
	}
	if (improvement) return stepResult;

	//Insertion
	for (size_t pos = 0; pos < candidate.size() + 1; ++pos) 
	{
		for (char letter : alphabet)
		{
			AlnScoreType score = align.addInsertion(pos + 1, letter, branches);
			if (score > stepResult.score)
			{
				stepResult.score = score;
				stepResult.sequence = candidate;
				stepResult.sequence.insert(pos, 1, letter);
				improvement = true;
			}
		}
	}	
	if (improvement) return stepResult;

	//Substitution
	for (size_t pos = 0; pos < candidate.size(); ++pos) 
	{
		for (char letter : alphabet)
		{
			if (letter == candidate[pos]) continue;

			AlnScoreType score = align.addSubstitution(pos + 1, letter, 
											   		   branches);
			if (score > stepResult.score) 
			{
				stepResult.score = score;
				stepResult.sequence = candidate;
				stepResult.sequence[pos] = letter;
			}
		}
	}

	return stepResult;
}
