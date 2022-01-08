#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>
#include <thread>
#include <utility>
#include <chrono>
#include <algorithm>
#include <random>
#include <libc.h>
#include <memory>

typedef double fp_type;

/*! \brief Get the current nano-seconds since some epoch
 * This is portable to OSX as well as Linux
 * \return nsec since epoch
 */
inline unsigned long long CurrentNSec() {
#ifdef __MACH__ // OS X does not have clock_gettime, use clock_get_time
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec * 1000 *1000 *1000UL + tv.tv_usec*1000UL;
#else
    struct timespec ts;
  clock_gettime(CLOCK_REALTIME, &ts);
  return (unsigned long long) ts.tv_nsec +
      (unsigned long long) ts.tv_sec * 1000 * 1000 * 1000;
#endif
}

/*! \brief A simple timer which can be started and stopped
 * Only read Clock.value after a stop or pause, otherwise use Read()
 */
class Clock {
public:
    double value; //< the elapsed time in seconds

    Clock() : value(0), start(0), stop(0),  paused_(false) { }

    /*! \brief Start counting.
     */
    inline void Start() {
        start = CurrentNSec();
        if (!paused_) {
            value = 0.0;
        }
    }

    /*! \brief Return the delta from the last call to Start()
     */
    inline double Read() {
        if (paused_) {
            return value;
        } else {
            stop = CurrentNSec();
            value = (double) (stop - start) * 1e-9;
            return value;
        }
    }

    /*! \brief Pause counting, keep a running sum of time
     */
    inline double Pause() {
        stop = CurrentNSec();
        value += (double) (stop - start) * 1e-9;
        paused_ = true;
        return value;
    }

    /*! \brief Stop counting, next call to Start() will reset the time
     */
    inline double Stop() {
        stop = CurrentNSec();
        value = (double) (stop - start) * 1e-9;
        paused_ = false;
        return value;
    }

private:
    unsigned long long start;
    unsigned long long stop;
    bool paused_;
};

struct Point {
    fp_type y{};
    std::vector<int> indices;
    std::vector<fp_type> xs;

    explicit Point(const int size) : indices(size, 0), xs(size, 0) {}
};

struct DataSet {
    int features{};
    std::vector<Point> points;
};

DataSet loadDataSet(const std::string& name) {
    std::vector<Point> points;
    std::ifstream in;
    in.open(name);
    if (!in) {
        std::cerr << "Failed to load dataset from " << name << std::endl;
    }
    std::string str;
    int features = 0;
    std::vector<int> indices;
    std::vector<fp_type> values;
    fp_type y;
    while (std::getline(in, str)) {
        indices.clear();
        values.clear();
        std::stringstream ss(str);
        fp_type x{};
        int index{};
        char c{};
        ss >> y;
        while (ss >> index >> c >> x) {
            indices.push_back(index);
            values.push_back(x);
            if (features < index) {
                features = index;
            }
        }
        Point p(indices.size());
        p.y = y;
        for (int i = 0; i < indices.size(); ++i) {
            p.indices[i] = indices[i];
            p.xs[i] = values[i];
        }
        points.push_back(p);
    }

    in.close();
    auto rng = std::default_random_engine{};
    std::shuffle(std::begin(points), std::end(points), rng);
    DataSet dataSet;
    dataSet.features = features;
    dataSet.points.reserve(points.size());
    for (const auto& point : points) {
        dataSet.points.push_back(point);
    }
    std::cout << "Dataset " << name << " loaded, size=" << dataSet.points.size() << std::endl;
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

    std::vector<fp_type> solve(DataSet& train, const int* degrees, Clock& clock) {
        this->degrees = degrees;
        int features = train.features;
        std::vector<fp_type> w(features, 0);
        {
            std::vector<std::atomic<int>> local_status(threads);
            status.swap(local_status);
        }
        std::fill(status.begin(), status.end(), 2);
        std::vector<std::thread> tasks;
        tasks.reserve(threads);
        for (int i = 0; i < threads; ++i) {
            tasks.emplace_back(&Solver::threadRoutine, this, i, &w[0], &train);
        }

        auto rng = std::default_random_engine{};
        for (int iteration = 0; iteration < iterations; ++iteration) {
            std::shuffle(std::begin(train.points), std::end(train.points), rng);
            clock.Start();
            std::fill(status.begin(), status.end(), 1);

            for (int i = 0; i < threads; ++i) {
                while (status[i] != 2);
            }
            clock.Pause();
            learningRate *= stepDecay;
        }

        std::fill(status.begin(), status.end(), 0);

        for (auto& task : tasks) {
            task.join();
        }
        return w;
    }

    static fp_type accuracy(const std::vector<fp_type>& w, const DataSet& test) {
        int s = 0;
        for (const auto& point : test.points) {
            const fp_type x = dot(&w[0], point.indices.size(), &point.indices[0], &point.xs[0]);
            if (x * point.y > 0.0) {
                s++;
            }
        }
        return (fp_type) s / test.points.size();
    }

private:
    std::vector<std::atomic<int>> status;

    void threadRoutine(int threadId, fp_type* w, const DataSet* train) {
        while (true) {
            if (status[threadId] == 0) return;
            if (status[threadId] == 1) {
                threadSolve(threadId, w, train);
                status[threadId] = 2;
            }
        }
    }

    void threadSolve(int threadId, fp_type* w, const DataSet* train) const {
        const Point* points = &train->points[0];
        const int n = train->points.size();
        const int block = n / threads;
        const int start = threadId * block;
        const int end = threadId == threads - 1 ? n : std::min(n, start + block);
        const fp_type alpha = learningRate;
        for (int i = start; i < end; ++i) {
            gradientStep(points[i], w, alpha);
        }
    }

    inline static fp_type
    dot(const fp_type* w,
        const int size,
        const int* indices,
        const fp_type* xs) {
        fp_type s = 0;
        for (int i = 0; i < size; ++i) {
            s += xs[i] * w[indices[i]];
        }
        return s;
    }

    inline void
    gradientStep(const Point& p, fp_type* w, const fp_type learningRate) const {
        const int size = p.indices.size();
        const int* indices = &p.indices[0];
        const fp_type* xs = &p.xs[0];

        const fp_type wxy = dot(w, size, indices, xs) * p.y;

        if (wxy < 1) { // hinge is active.
            const fp_type e = p.y * learningRate;
            for (int i = 0; i < size; ++i) {
                w[indices[i]] += xs[i] * e;
            }
        }

        int const* const degs = degrees;

        // update based on the evaluation
        const fp_type scalar = learningRate * 1.0;
        for (int i = 0; i < size; ++i) {
            const int j = indices[i];
            w[j] *= 1 - scalar / degs[j];
        }

    }


    fp_type learningRate;
    const fp_type stepDecay;
    const int threads;
    const int iterations;
    const int* degrees{};
};


int main() {
    auto p = loadBinaryDataSet("../datasets/rcv1");
    std::vector<int> degrees(p.first.features, 0);
    for (const auto& point : p.first.points) {
        for (int i : point.indices) {
            degrees[i]++;
        }
    }

    int base_ms = 0;
    const int iterations = 100;
    for (int threads = 1; threads <= 16; threads *= 2) {
        Solver solver(0.5, 0.8, threads, iterations);
        Clock clock;
        auto w = solver.solve(p.first, &degrees[0], clock);
        int time_ms = clock.Read() * 1e3 / iterations;
        if (threads == 1) {
            base_ms = time_ms;
        }
        std::cout << threads << ' ' << time_ms << ' ' << (double) base_ms / time_ms << ' '
                  << Solver::accuracy(w, p.second) << std::endl;
    }

    return 0;
}
