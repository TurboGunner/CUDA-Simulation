#include "handler_classes.hpp"

ProgramLog::~ProgramLog() {
	Close();
}

ProgramLog& ProgramLog::Get() {
	static ProgramLog instance_;
	return instance_;
}

void ProgramLog::OutputLine(const string& line, const bool is_separate) {
	Get().OutputLineString(line, is_separate);
}

void ProgramLog::OutputLine(stringstream& line, const bool is_separate) {
	Get().OutputLineStream(line, is_separate);
}

void ProgramLog::Close() {
	Get().file_log.close();
}

void ProgramLog::OutputLineString(const string& line, const bool is_separate) {
	if (line_count == 0) {
		file_log.open(file_name);
	}

	file_log << line;
	line_count++;
	if (is_separate) {
		file_log << std::endl;
	}
}

void ProgramLog::OutputLineStream(stringstream& line, const bool is_separate) {
	if (line_count == 0) {
		file_log.open(file_name);
	}

	file_log << line.str();
	line_count++;
	if (is_separate) {
		file_log << std::endl;
	}

	line.clear();
	line.str(string());

	s_stream.clear();
	s_stream.str(string());
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