#pragma once

#include <fstream>
#include <random>
#include <string>

using std::ofstream;
using std::string;

//Singleton
class ProgramLog {
public:
	ProgramLog(const ProgramLog&) = delete;

	~ProgramLog() {
		Close();
	}

	static ProgramLog& Get() {
		return instance_;
	}

	static void OutputLine(string line, bool is_separate) {
		Get().file_log << line;
		if (is_separate) {
			Get().file_log << std::endl;
		}
	}

	static void Close() {
		Get().file_log.close();
	}

private:
	ProgramLog() = default;
	static ProgramLog instance_;

	ofstream file_log;
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
