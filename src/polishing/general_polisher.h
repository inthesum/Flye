//(c) 2016 by Authors
//This file is a part of ABruijn program.
//Released under the BSD license (see LICENSE file)

#pragma once

#include <chrono>

#include "bubble.h"
#include "subs_matrix.h"
#include "alignment.h"

class GeneralPolisher
{
public:
	GeneralPolisher(const SubstitutionMatrix& subsMatrix):
		_subsMatrix(subsMatrix)
	{}
    void polishBubble(Bubble& bubble) const;
    void polishBubble(Bubble& bubble,
                      std::chrono::duration<double>& optimizeDuration,
                      std::chrono::duration<double>& makeStepDuration,
                      std::chrono::duration<double>& alignmentDuration,
                      std::chrono::duration<double>& deletionDuration,
                      std::chrono::duration<double>& insertionDuration,
                      std::chrono::duration<double>& substitutionDuration) const;

private:
	StepInfo makeStep(const std::string& candidate, 
					  const std::vector<std::string>& branches,
					  Alignment& align) const;
    StepInfo makeStep(const std::string& candidate,
                      const std::vector<std::string>& branches,
                      Alignment& align,
                      std::chrono::duration<double>& alignmentDuration,
                      std::chrono::duration<double>& deletionDuration,
                      std::chrono::duration<double>& insertionDuration,
                      std::chrono::duration<double>& substitutionDuration) const;

	const SubstitutionMatrix& _subsMatrix;
};
