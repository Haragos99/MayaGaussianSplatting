#pragma once
#include <maya/MPxNode.h>
#include "data.h"

class GaussiansplattingNode : public MPxNode
{
	public:
		GaussiansplattingNode() = default;
		void loadPointCloud(const std::string& filePath);
		static void* creator();
		static MStatus initialize();
		static MTypeId id;
		~GaussiansplattingNode() = default;
	private:
		std::vector<Splat> splats;

};

