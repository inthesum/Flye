//(c) 2016 by Authors
//This file is a part of ABruijn program.
//Released under the BSD license (see LICENSE file)

#include "alignment.h"
#include <immintrin.h> // Include SIMD intrinsics header


Alignment::Alignment(size_t size, const SubstitutionMatrix& sm):
	_forwardScores(size),
	_reverseScores(size),
	_subsMatrix(sm)
{ 
}


AlnScoreType Alignment::globalAlignment(const std::string& consensus,
							 			const std::vector<std::string>& reads)
{
	AlnScoreType finalScore = 0;
	for (size_t readId = 0; readId < _forwardScores.size(); ++readId)
	{
		unsigned int x = consensus.size() + 1;
		unsigned int y = reads[readId].size() + 1;
		ScoreMatrix scoreMat(x, y, 0);
			
		AlnScoreType score = this->getScoringMatrix(consensus, reads[readId],scoreMat);
		_forwardScores[readId] = std::move(scoreMat);

		//The reverse alignment is similar, but we need
		//the scoring matrix with of the reverse alignment (I guess)
		std::string revConsensus(consensus.rbegin(), consensus.rend());
		std::string revRead(reads[readId].rbegin(), reads[readId].rend());

		ScoreMatrix scoreMatRev(x, y, 0);
		this->getScoringMatrix(revConsensus, revRead, scoreMatRev);
//        scoreMatRev.reverseRows();
		_reverseScores[readId] = std::move(scoreMatRev);

		finalScore += score;
	}

	return finalScore;
}


__m256i mm256_max_epi64(__m256i a, __m256i b) {
    // Compare a and b to get a mask of elements where a > b
    __m256i cmp_mask = _mm256_cmpgt_epi64(a, b);
    // Use the mask to select the maximum value from a and b
    __m256i result = _mm256_blendv_epi8(b, a, cmp_mask);
    return result;
}

//AlnScoreType Alignment::addDeletion(unsigned int letterIndex) const
//{
//	AlnScoreType finalScore = 0;
//	for (size_t readId = 0; readId < _forwardScores.size(); ++readId)
//	{
//		const ScoreMatrix& forwardScore = _forwardScores[readId];
//		const ScoreMatrix& reverseScore = _reverseScores[readId];
//
//		//Note: We subtract 2 because of zero indexing and an extra added row and column count
//		//unsigned int index = (reverseScore.nrows() - 1) - letterIndex;
//		size_t frontRow = letterIndex - 1;
//		size_t revRow = reverseScore.nrows() - 1 - letterIndex;
//
//		AlnScoreType maxVal = std::numeric_limits<AlnScoreType>::lowest();
//		for (size_t col = 0; col < forwardScore.ncols(); ++col)
//		{
//			size_t backCol = forwardScore.ncols() - col - 1;
//			AlnScoreType sum = forwardScore.at(frontRow, col) +
//							reverseScore.at(revRow, backCol);
//			maxVal = std::max(maxVal, sum);
//		}
//		finalScore += maxVal;
//	}
//	return finalScore;
//}

AlnScoreType Alignment::addDeletion(unsigned int letterIndex) const
{
    AlnScoreType finalScore = 0;
    constexpr size_t batchSize = 4; // Use constexpr for batch size
    size_t frontRow = letterIndex - 1;

    for (size_t readId = 0; readId < _forwardScores.size(); ++readId)
    {
        const ScoreMatrix& forwardScore = _forwardScores[readId];
        const ScoreMatrix& reverseScore = _reverseScores[readId];
        size_t revRow = reverseScore.nrows() - 1 - letterIndex;

        AlnScoreType maxVal = std::numeric_limits<AlnScoreType>::lowest();
        __m256i maxValues = _mm256_set1_epi64x(maxVal);

        size_t cols = forwardScore.ncols();
        const size_t alignedReadsN = cols - cols % batchSize;
        for (size_t col = 0; col < alignedReadsN; col += batchSize)
        {
             // Load elements into SIMD vectors
//            __m256i forwardVals = _mm256_set_epi64x(
//                    forwardScore.at(frontRow, col + 3),
//                    forwardScore.at(frontRow, col + 2),
//                    forwardScore.at(frontRow, col + 1),
//                    forwardScore.at(frontRow, col)
//            );

            __m256i reverseVals = _mm256_set_epi64x(
                    reverseScore.at(revRow, cols - col - 4),
                    reverseScore.at(revRow, cols - col - 3),
                    reverseScore.at(revRow, cols - col - 2),
                    reverseScore.at(revRow, cols - col - 1)
            );

            __m256i forwardVals = _mm256_loadu_si256((__m256i*)(forwardScore.data() + frontRow * cols + col));
//            __m256i reverseVals = _mm256_loadu_si256((__m256i*)(reverseScore.data() + revRow * cols + col));

            // Perform element-wise addition
            __m256i sum = _mm256_add_epi64(forwardVals, reverseVals);

            // Find the maximum value within the SIMD vector
            maxValues = mm256_max_epi64(maxValues, sum);
        }

        // Store the result
        alignas(32) AlnScoreType maxValuesArray[batchSize];
        _mm256_store_si256((__m256i*)maxValuesArray, maxValues);

        // Update maxVal
        for (size_t i = 0; i < batchSize; ++i)
        {
            maxVal = std::max(maxVal, maxValuesArray[i]);
        }

        for (size_t col = alignedReadsN; col < cols; ++col) {
			AlnScoreType sum = forwardScore.at(frontRow, col) +  reverseScore.at(revRow, cols - col - 1);
//			AlnScoreType sum = forwardScore.at(frontRow, col) +  reverseScore.at(revRow, col);
			maxVal = std::max(maxVal, sum);
        }

        finalScore += maxVal;
    }
    return finalScore;
}


//AlnScoreType Alignment::addSubstitution(unsigned int letterIndex, char base,
//										const std::vector<std::string>& reads) const
//{
//	AlnScoreType finalScore = 0;
//	for (size_t readId = 0; readId < reads.size(); ++readId)
//	{
//		//LetterIndex must start with 1 and go until (row.size - 1)
//		const ScoreMatrix& forwardScore = _forwardScores[readId];
//		const ScoreMatrix& reverseScore = _reverseScores[readId];
//
//		size_t frontRow = letterIndex - 1;
//		size_t revRow = reverseScore.nrows() - 1 - letterIndex;
//
//		std::vector<AlnScoreType> sub(reads[readId].size() + 1);
//		sub[0] = forwardScore.at(frontRow, 0) + _subsMatrix.getScore(base, '-');
//		for (size_t i = 0; i < reads[readId].size(); ++i)
//		{
//			AlnScoreType match = forwardScore.at(frontRow, i) +
//							_subsMatrix.getScore(base, reads[readId][i]);
//			AlnScoreType ins = forwardScore.at(frontRow, i + 1) +
//							_subsMatrix.getScore(base, '-');
//			sub[i + 1] = std::max(match, ins);
//		}
//
//		AlnScoreType maxVal = std::numeric_limits<AlnScoreType>::lowest();
//		for (size_t col = 0; col < forwardScore.ncols(); ++col)
//		{
//			size_t backCol = forwardScore.ncols() - col - 1;
//			AlnScoreType sum = sub[col] + reverseScore.at(revRow, backCol);
//			maxVal = std::max(maxVal, sum);
//		}
//		finalScore += maxVal;
//	}
//	return finalScore;
//}

//AlnScoreType Alignment::addSubstitution(unsigned int letterIndex, char base,
//										const std::vector<std::string>& reads) const
//{
//	AlnScoreType finalScore = 0;
//    size_t frontRow = letterIndex - 1;
//    AlnScoreType baseScoreWithGap = _subsMatrix.getScore(base, '-');
//
//    for (size_t readId = 0; readId < reads.size(); ++readId)
//	{
//		const ScoreMatrix& forwardScore = _forwardScores[readId];
//		const ScoreMatrix& reverseScore = _reverseScores[readId];
//
//		size_t revRow = reverseScore.nrows() - 1 - letterIndex;
//
//        size_t cols = reads[readId].size();
//        AlnScoreType maxVal = forwardScore.at(frontRow, 0) + reverseScore.at(revRow, cols) + baseScoreWithGap;
//		for (size_t col = 0; col < cols; ++col)
//		{
//            char readBase = reads[readId][col];
//            AlnScoreType match = forwardScore.at(frontRow, col) + _subsMatrix.getScore(base, readBase);
//			AlnScoreType ins = forwardScore.at(frontRow, col + 1) + baseScoreWithGap;
//            maxVal = std::max(maxVal, std::max(match, ins) + reverseScore.at(revRow, cols - col - 1));
//        }
//		finalScore += maxVal;
//	}
//	return finalScore;
//}

AlnScoreType Alignment::addSubstitution(unsigned int letterIndex, char base,
										const std::vector<std::string>& reads) const
{
    AlnScoreType finalScore = 0;
    size_t frontRow = letterIndex - 1;
    constexpr size_t batchSize = 4; // Use constexpr for batch size
    AlnScoreType baseScoreWithGap = _subsMatrix.getScore(base, '-');
    __m256i baseScores = _mm256_set1_epi64x(baseScoreWithGap);

    for (size_t readId = 0; readId < reads.size(); ++readId) {
        const ScoreMatrix& forwardScore = _forwardScores[readId];
        const ScoreMatrix& reverseScore = _reverseScores[readId];
        size_t revRow = reverseScore.nrows() - 1 - letterIndex;

        size_t cols = reads[readId].size();
        const size_t alignedReadsN = cols - cols % batchSize;
        AlnScoreType maxVal = forwardScore.at(frontRow, 0) + reverseScore.at(revRow, cols) + baseScoreWithGap;
//        AlnScoreType maxVal = forwardScore.at(frontRow, 0) + reverseScore.at(revRow, 0) + baseScoreWithGap;
        __m256i maxValues = _mm256_set1_epi64x(maxVal);
        for (size_t col = 0; col < alignedReadsN; col += batchSize)
        {
            // Load base scores
            __m256i scores = _mm256_set_epi64x(
                    _subsMatrix.getScore(base, reads[readId][col + 3]),
                    _subsMatrix.getScore(base, reads[readId][col + 2]),
                    _subsMatrix.getScore(base, reads[readId][col + 1]),
                    _subsMatrix.getScore(base, reads[readId][col])
            );

//            __m256i forwardScoreCurrent = _mm256_set_epi64x(
//                    forwardScore.at(frontRow, col + 3),
//                    forwardScore.at(frontRow, col + 2),
//                    forwardScore.at(frontRow, col + 1),
//                    forwardScore.at(frontRow, col));
//
//            __m256i forwardScoreNext = _mm256_set_epi64x(
//                    forwardScore.at(frontRow, col + 4),
//                    forwardScore.at(frontRow, col + 3),
//                    forwardScore.at(frontRow, col + 2),
//                    forwardScore.at(frontRow, col + 1));

            __m256i reverseScoreNext = _mm256_set_epi64x(
                    reverseScore.at(revRow, cols - col - 4),
                    reverseScore.at(revRow, cols - col - 3),
                    reverseScore.at(revRow, cols - col - 2),
                    reverseScore.at(revRow, cols - col - 1));

            __m256i forwardScoreCurrent = _mm256_loadu_si256((__m256i*)(forwardScore.data() + frontRow * (cols + 1) + col));
            __m256i forwardScoreNext = _mm256_loadu_si256((__m256i*)(forwardScore.data() + frontRow * (cols + 1) + col + 1));
//            __m256i reverseScoreNext = _mm256_loadu_si256((__m256i*)(reverseScore.data() + revRow * (cols + 1) + col + 1));

            // Compute match scores
            __m256i matchScores = _mm256_add_epi64(forwardScoreCurrent, scores);

            // Compute insertion scores
            __m256i insertScores = _mm256_add_epi64(forwardScoreNext, baseScores);

            // Compute max scores
            __m256i maxScores = mm256_max_epi64(matchScores, insertScores);

            // Compute sum
            __m256i sum = _mm256_add_epi64(reverseScoreNext, maxScores);

            // Update max values
            maxValues = mm256_max_epi64(maxValues, sum);
        }

        alignas(32) AlnScoreType maxValuesArray[batchSize];
        _mm256_store_si256((__m256i*)maxValuesArray, maxValues);
        for (size_t i = 0; i < batchSize; ++i)
            maxVal = std::max(maxVal, maxValuesArray[i]);

        // Process remaining elements without vectorization
        for (size_t col = alignedReadsN; col < cols; ++col) {
            char readBase = reads[readId][col];
            AlnScoreType match = forwardScore.at(frontRow, col) + _subsMatrix.getScore(base, readBase);
            AlnScoreType ins = forwardScore.at(frontRow, col+1) + baseScoreWithGap;
            maxVal = std::max(maxVal, std::max(match, ins) + reverseScore.at(revRow, cols - col - 1));
//            maxVal = std::max(maxVal, std::max(match, ins) + reverseScore.at(revRow, col+1));
        }

        finalScore += maxVal;
    }

    return finalScore;
}


//AlnScoreType Alignment::addInsertion(unsigned int pos, char base,
//									 const std::vector<std::string>& reads) const
//{
//	AlnScoreType finalScore = 0;
//	for (size_t readId = 0; readId < reads.size(); ++readId)
//	{
//		//LetterIndex must start with 1 and go until (row.size - 1)
//		const ScoreMatrix& forwardScore = _forwardScores[readId];
//		const ScoreMatrix& reverseScore = _reverseScores[readId];
//
//		size_t frontRow = pos - 1;
//		size_t revRow = reverseScore.nrows() - pos;
//
//		std::vector<AlnScoreType> sub(reads[readId].size() + 1);
//		sub[0] = forwardScore.at(frontRow, 0) + _subsMatrix.getScore(base, '-');
//
//		for (size_t i = 0; i < reads[readId].size(); ++i)
//		{
//			AlnScoreType match = forwardScore.at(frontRow, i) + _subsMatrix.getScore(base, reads[readId][i]);
//			AlnScoreType ins = forwardScore.at(frontRow, i + 1) + _subsMatrix.getScore(base, '-');
//			sub[i + 1] = std::max(match, ins);
//		}
//
//		AlnScoreType maxVal = std::numeric_limits<AlnScoreType>::lowest();
//		for (size_t col = 0; col < forwardScore.ncols(); ++col)
//		{
//			size_t backCol = forwardScore.ncols() - col - 1;
//			AlnScoreType sum = sub[col] + reverseScore.at(revRow, backCol);
//			maxVal = std::max(maxVal, sum);
//		}
//		finalScore += maxVal;
//	}
//	return finalScore;
//}

//AlnScoreType Alignment::addInsertion(unsigned int pos, char base, const std::vector<std::string>& reads) const
//{
//    AlnScoreType finalScore = 0;
//    size_t frontRow = pos - 1;
//    AlnScoreType baseScoreWithGap = _subsMatrix.getScore(base, '-');
//
//    for (size_t readId = 0; readId < reads.size(); ++readId)
//    {
//        const ScoreMatrix& forwardScore = _forwardScores[readId];
//        const ScoreMatrix& reverseScore = _reverseScores[readId];
//
//        size_t revRow = reverseScore.nrows() - pos;
//
//        size_t cols = reads[readId].size();
//        AlnScoreType maxVal = forwardScore.at(frontRow, 0) + reverseScore.at(revRow, cols) + baseScoreWithGap;
//
//        for (size_t col = 0; col < cols; ++col)
//        {
//            char readBase = reads[readId][col];
//            AlnScoreType match = forwardScore.at(frontRow, col) + _subsMatrix.getScore(base, readBase);
//            AlnScoreType ins = forwardScore.at(frontRow, col+1) + baseScoreWithGap;
//            maxVal = std::max(maxVal, std::max(match, ins) + reverseScore.at(revRow, cols - col - 1));
//        }
//
//		finalScore += maxVal;
//    }
//    return finalScore;
//}

AlnScoreType Alignment::addInsertion(unsigned int pos, char base, const std::vector<std::string>& reads) const {
    AlnScoreType finalScore = 0;
    size_t frontRow = pos - 1;
    constexpr size_t batchSize = 4; // Use constexpr for batch size
    AlnScoreType baseScoreWithGap = _subsMatrix.getScore(base, '-');
    __m256i baseScores = _mm256_set1_epi64x(baseScoreWithGap);

    for (size_t readId = 0; readId < reads.size(); ++readId) {
        const ScoreMatrix& forwardScore = _forwardScores[readId];
        const ScoreMatrix& reverseScore = _reverseScores[readId];
        size_t revRow = reverseScore.nrows() - pos;

        size_t cols = reads[readId].size();
        const size_t alignedReadsN = cols - cols % batchSize;
        AlnScoreType maxVal = forwardScore.at(frontRow, 0) + reverseScore.at(revRow, cols) + baseScoreWithGap;
//        AlnScoreType maxVal = forwardScore.at(frontRow, 0) + reverseScore.at(revRow, 0) + baseScoreWithGap;
        __m256i maxValues = _mm256_set1_epi64x(maxVal);
        for (size_t col = 0; col < alignedReadsN; col += batchSize)
        {
            // Load base scores
            __m256i scores = _mm256_set_epi64x(
                    _subsMatrix.getScore(base, reads[readId][col + 3]),
                    _subsMatrix.getScore(base, reads[readId][col + 2]),
                    _subsMatrix.getScore(base, reads[readId][col + 1]),
                    _subsMatrix.getScore(base, reads[readId][col])
            );

//            __m256i forwardScoreCurrent = _mm256_set_epi64x(
//                    forwardScore.at(frontRow, col + 3),
//                    forwardScore.at(frontRow, col + 2),
//                    forwardScore.at(frontRow, col + 1),
//                    forwardScore.at(frontRow, col));
//
//            __m256i forwardScoreNext = _mm256_set_epi64x(
//                    forwardScore.at(frontRow, col + 4),
//                    forwardScore.at(frontRow, col + 3),
//                    forwardScore.at(frontRow, col + 2),
//                    forwardScore.at(frontRow, col + 1));

            __m256i reverseScoreNext = _mm256_set_epi64x(
                    reverseScore.at(revRow, cols - col - 4),
                    reverseScore.at(revRow, cols - col - 3),
                    reverseScore.at(revRow, cols - col - 2),
                    reverseScore.at(revRow, cols - col - 1));

            __m256i forwardScoreCurrent = _mm256_loadu_si256((__m256i*)(forwardScore.data() + frontRow * (cols + 1) + col));
            __m256i forwardScoreNext = _mm256_loadu_si256((__m256i*)(forwardScore.data() + frontRow * (cols + 1) + col + 1));
//            __m256i reverseScoreNext = _mm256_loadu_si256((__m256i*)(reverseScore.data() + revRow * (cols + 1) + col + 1));

            // Compute match scores
            __m256i matchScores = _mm256_add_epi64(forwardScoreCurrent, scores);

            // Compute insertion scores
            __m256i insertScores = _mm256_add_epi64(forwardScoreNext, baseScores);

            // Compute max scores
            __m256i maxScores = mm256_max_epi64(matchScores, insertScores);

            // Compute sum
            __m256i sum = _mm256_add_epi64(reverseScoreNext, maxScores);

            // Update max values
            maxValues = mm256_max_epi64(maxValues, sum);
        }

        alignas(32) AlnScoreType maxValuesArray[batchSize];
        _mm256_store_si256((__m256i*)maxValuesArray, maxValues);
        for (size_t i = 0; i < batchSize; ++i)
            maxVal = std::max(maxVal, maxValuesArray[i]);

        // Process remaining elements without vectorization
        for (size_t col = alignedReadsN; col < cols; ++col) {
            char readBase = reads[readId][col];
            AlnScoreType match = forwardScore.at(frontRow, col) + _subsMatrix.getScore(base, readBase);
            AlnScoreType ins = forwardScore.at(frontRow, col+1) + baseScoreWithGap;
            maxVal = std::max(maxVal, std::max(match, ins) + reverseScore.at(revRow, cols - col - 1));
//            maxVal = std::max(maxVal, std::max(match, ins) + reverseScore.at(revRow, col+1));
        }

        finalScore += maxVal;
    }

    return finalScore;
}


AlnScoreType Alignment::getScoringMatrix(const std::string& v,
										 const std::string& w,
								  		 ScoreMatrix& scoreMat)
{
	AlnScoreType score = 0;

	for (size_t i = 0; i < v.size(); i++)
	{
		AlnScoreType score = _subsMatrix.getScore(v[i], '-');
		scoreMat.at(i + 1, 0) = scoreMat.at(i, 0) + score;
	}


	for (size_t i = 0; i < w.size(); i++) {
		AlnScoreType score = _subsMatrix.getScore('-', w[i]);
		scoreMat.at(0, i + 1) = scoreMat.at(0, i) + score;
	}


	for (size_t i = 1; i < v.size() + 1; i++)
	{
		char key1 = v[i - 1];
		for (size_t j = 1; j < w.size() + 1; j++)
		{
			char key2 = w[j - 1];

			AlnScoreType left = scoreMat.at(i, j - 1) + _subsMatrix.getScore('-', key2);
			AlnScoreType up = scoreMat.at(i - 1, j) + _subsMatrix.getScore(key1, '-');
			score = std::max(left, up);

			AlnScoreType cross = scoreMat.at(i - 1, j - 1) + _subsMatrix.getScore(key1, key2);
			score = std::max(score, cross);
			scoreMat.at(i, j) = score;
		}
	}

	return score;
}