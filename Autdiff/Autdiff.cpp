#include <iostream>
#include <tuple>

// Product rule
std::pair<double, double> adf(const double _x) {
	double f1{ _x - 1 };
	double f2{ _x + 3 };
	double f3{ _x + 2 };
	double f{ f1 * f2 / f3 };

	double df1{ f1 * 1.0 + 1.0 * f2 };
	double df2{ 1.0*1.0 };

	double df{ (df1 - f * df2) / f3 };
	return std::make_pair(f, df);
}

double f(const double x) {
	return (x - 1) * (x + 3) / (x + 2);
}

double df(const double x) {
	return (x * x + 4.0 * x + 7.0) / (x + 2) / (x + 2);
}

int main() {
	for (double x{ 0.0 }, h{ 1.0e-1 }; x < 5.0;) {
		auto res = adf(x);
		std::cout << "x: " << x << "f: (" << res.first << ", " << f(x) << ")\t" << "df: (" << res.second << ", " << df(x) << "), ddf: " << res.second -df(x) << "\n";
		x += h;
	}
	return 0;
}
//
//using namespace autodiff;
//int main() {
//	
//
//	dual2nd x = 1.0;
//	auto f = [](dual2nd x) { return x * x * x * x; };
//
//	dual2nd y = f(x);
//	double d4 = derivatives(f, wrt(x), at(x), 4);  // 4. Ableitung
//
//	return 0;
//}