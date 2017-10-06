//(c) 2016 by Authors
//This file is a part of ABruijn program.
//Released under the BSD license (see LICENSE file)

#pragma once

#include "repeat_graph.h"

struct Contig
{
	Contig(const GraphPath& path, FastaRecord::Id id = FastaRecord::ID_NONE,
		   bool circular = false, int length = 0, int meanCoverage = 0):
		   	path(path), id(id), circular(circular), repetitive(false), 
			length(length), meanCoverage(meanCoverage) {}

	std::string name() const
	{
		std::string nameTag = circular ? "circular" : "linear";
		return nameTag + "_" + std::to_string(id.signedId());
	}

	std::string nameUnsigned() const
	{
		std::string nameTag = circular ? "circular" : "linear";
		std::string idTag = id.strand() ? std::to_string(id.signedId()) : 
										  std::to_string(id.rc().signedId());
		return nameTag + "_" + idTag;
	}

	GraphPath path;
	FastaRecord::Id id;
	std::string sequence;
	bool circular;
	bool repetitive;
	int length;
	int meanCoverage;
};

class OutputGenerator
{
public:
	OutputGenerator(RepeatGraph& graph, const SequenceContainer& asmSeqs,
				    const SequenceContainer& readSeqs):
		_graph(graph), _asmSeqs(asmSeqs), _readSeqs(readSeqs) {}

	void generateContigs();
	void dumpRepeats(const std::vector<GraphAlignment>& readAlignments,
					 const std::string& outFile);

	void outputDot(bool contigs, const std::string& filename);
	void outputGfa(bool contigs, const std::string& filename);
	void outputFasta(bool contigs, const std::string& filename);

private:
	void generateContigSequences(std::vector<Contig>& paths) const;
	void outputEdgesDot(const std::vector<Contig>& paths,
						const std::string& filename);
	void outputEdgesGfa(const std::vector<Contig>& paths,
						const std::string& filename);
	void outputEdgesFasta(const std::vector<Contig>& paths,
						  const std::string& filename);
	std::vector<Contig> edgesPaths() const;

	RepeatGraph& _graph;
	const SequenceContainer& _asmSeqs;
	const SequenceContainer& _readSeqs;

	std::vector<Contig> _contigs;
};