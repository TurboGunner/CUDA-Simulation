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

	~ProgramLog() {
		Close();
	}

	static ProgramLog& Get() {
		static ProgramLog instance_;
		return instance_;
	}

	static void OutputLine(string line, bool is_separate = true) {
		if (Get().line_count == 0) {
			Get().file_log.open(Get().file_name);
		}

		Get().file_log << line;
		Get().line_count++;
		if (is_separate) {
			Get().file_log << std::endl;
		}
	}

	static void OutputLine(std::stringstream& line, bool is_separate = true) {
		if (Get().line_count == 0) {
			Get().file_log.open(Get().file_name);
		}

		Get().file_log << line.str();
		Get().line_count++;
		if (is_separate) {
			Get().file_log << std::endl;
		}
		line.clear();
		s_stream.clear();
	}

	static void Close() {
		Get().file_log.close();
	}

private:
	ProgramLog() = default;

	ofstream file_log;
	const std::string file_name = "example.txt";
	unsigned int line_count = 0;
};

class RandomFloat {
public:
	RandomFloat(float low = 0.0f, float high = 10.0f, unsigned long seed_in = 0) {
		generator = std::mt19937(0);
		distribution = std::uniform_real_distribution<float>(low, high);
		seed = seed_in;
	}

	float Generate() {
		float random = distribution(generator);

		seed++;
		generator.seed(seed);

		return random;
	}

private:
	std::mt19937 generator;
	std::uniform_real_distribution<float> distribution;
	unsigned long seed;
};