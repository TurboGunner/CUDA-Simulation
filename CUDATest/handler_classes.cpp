#include "handler_classes.hpp"

ProgramLog::~ProgramLog() {
	Close();
}

ProgramLog& ProgramLog::Get() {
	static ProgramLog instance_;
	return instance_;
}

void ProgramLog::OutputLine(string line, bool is_separate) {
	if (Get().line_count == 0) {
		Get().file_log.open(Get().file_name);
	}

	Get().file_log << line;
	Get().line_count++;
	if (is_separate) {
		Get().file_log << std::endl;
	}
}

void ProgramLog::OutputLine(std::stringstream& line, bool is_separate) {
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
	s_stream.str(string());
}

void ProgramLog::Close() {
	Get().file_log.close();
}



RandomFloat::RandomFloat(float low, float high, unsigned long seed_in) {
	generator = std::mt19937(0);
	distribution = std::uniform_real_distribution<float>(low, high);
	seed = seed_in;
}

float RandomFloat::Generate() {
	float random = distribution(generator);

	seed++;
	generator.seed(seed);

	return random;
}