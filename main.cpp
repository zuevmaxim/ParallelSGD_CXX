#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>
#include <thread>
#include <utility>
#include <chrono>

typedef double fp_type;

struct Point {
    std::vector<int> indices;
    std::vector<fp_type> xs;
    fp_type y{};
};

struct DataSet {
    std::vector<Point> points;
    int features{};
};

DataSet loadDataSet(const std::string& name) {
    DataSet dataSet;
    std::ifstream in;
    in.open(name);
    if (!in) {
        std::cerr << "Failed to load dataset from " << name << std::endl;
    }
    std::string str;
    int features = 0;
    while (std::getline(in, str)) {
        std::stringstream ss(str);
        Point p;
        fp_type x;
        int index;
        char c;
        ss >> p.y;
        if (p.y != 1) p.y = 0;
        while (ss >> index >> c >> x) {
            p.indices.push_back(index);
            p.xs.push_back(x);
            if (features < index) {
                features = index;
            }
        }
        dataSet.points.push_back(p);
    }

    in.close();
    dataSet.features = features;

    std::cout << "Dataset " << name << " loaded" << std::endl;
    return dataSet;
}


std::pair<DataSet, DataSet> loadBinaryDataSet(const std::string& name) {
    DataSet train = loadDataSet(name);
    DataSet test = loadDataSet(name + ".t");
    return {train, test};
}

class Solver {
public:
    Solver(fp_type learningRate,
           fp_type stepDecay,
           int threads,
           int iterations) : learningRate(learningRate), stepDecay(stepDecay), threads(threads),
                             iterations(iterations) {
    }

    void solve(const DataSet& train) {
        int features = train.features + 1;
        std::vector<fp_type> w(features, 0);

        std::vector<std::thread> tasks;
        tasks.reserve(threads);
        for (int i = 0; i < threads; ++i) {
            tasks.emplace_back(&Solver::threadSolve, this, i, &w[0], features, &train);
        }

        for (auto& task : tasks) {
            task.join();
        }
    }

private:
    void threadSolve(int threadId, fp_type* w, int wSize, const DataSet* train) const {
        const Point* points = &train->points[0];
        const int n = train->points.size();
        const int block = n / threads;
        const int start = threadId * block;
        const int end = threadId == threads ? n : std::min(n, start + block);
        fp_type alpha = learningRate;
        for (int iteration = iterations; iteration-- > 0;) {
            for (int i = start; i < end; ++i) {
                gradientStep(points[i], w, wSize, alpha);
            }
            alpha *= stepDecay;
        }
    }

    inline static fp_type
    dot(const fp_type* w, int wSize, const std::vector<int>& indices, const std::vector<fp_type>& xs) {
        fp_type s = 0;
        const int size = indices.size();
        for (int i = 0; i < size; ++i) {
            s += xs[i] * w[indices[i]];
        }
//        s += w[wSize];
        return s;
    }

    inline static void gradientStep(const Point& p, fp_type* w, const int wSize, const fp_type learningRate) {
        const auto& indices = p.indices;
        const auto& xs = p.xs;
        const fp_type e = exp(-dot(w, wSize, indices, xs));
        const fp_type grad = learningRate * (1 / (1 + e) - p.y);
        const int size = indices.size();
        for (int i = 0; i < size; ++i) {
            w[indices[i]] -= xs[i] * grad;
        }
//        w[wSize] -= grad;
    }


    const fp_type learningRate;
    const fp_type stepDecay;
    const int threads;
    const int iterations;
};


int main() {
    auto p = loadBinaryDataSet("/Users/Maksim.Zuev/Documents/datasets/rcv1");
    std::vector<int> time;
    for (int threads = 1; threads <= 16; threads *= 2) {
        for (int i = 0; i < 5; ++i) {
            Solver solver(0.5, 0.8, threads, 10);
            solver.solve(p.first);
        }
        std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
        for (int i = 0; i < 10; ++i) {
            Solver solver(0.5, 0.8, threads, 10);
            solver.solve(p.first);
        }
        std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
        int time_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() / 10.0;
        time.push_back(time_ms);
        std::cout << threads << ' ' << time_ms << ' ' << (double) time[0] / time_ms << std::endl;
    }

    return 0;
}
