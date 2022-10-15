#pragma once

#include <fstream>
#include <sstream>
#include <random>
#include <string>

using std::ofstream;
using std::string;
using std::stringstream;

static stringstream s_stream;

//Singleton
class ProgramLog {
public:
	ProgramLog(const ProgramLog&) = delete;

	~ProgramLog();

	static ProgramLog& Get();

	static void OutputLine(string line, bool is_separate = true);

	static void OutputLine(std::stringstream& line, bool is_separate = true);

	static void Close();

private:
	ProgramLog() = default;

	ofstream file_log;
	const std::string file_name = "example.txt";
	unsigned int line_count = 0;
};

class RandomFloat {
public:
	RandomFloat(float low = 0.0f, float high = 10.0f, unsigned long seed_in = 0);

	float Generate();

private:
	std::mt19937 generator;
	std::uniform_real_distribution<float> distribution;
	unsigned long seed;
};