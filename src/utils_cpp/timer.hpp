#include <iostream>
#include <chrono>
#include <cmath>

#ifndef TIMER
#define TIMER

typedef std::chrono::seconds s;
typedef std::chrono::milliseconds ms;
typedef std::chrono::microseconds mus;

#define get_name(var) #var

template <typename TD = std::chrono::milliseconds>
class Timer {
public:
	Timer(): point(std::chrono::steady_clock::now()), val(0) {}

	void start() {
		point = std::chrono::steady_clock::now();
	}

	void stop() {
		val += std::chrono::duration_cast<TD>(std::chrono::steady_clock::now() - point);
	}

	void reset() {
		val.zero();
	}

	auto get_time() {
		return val.count();
	}

	~Timer() {
	}
private:
	std::chrono::steady_clock::time_point point;
	TD val;
};

#endif
