#pragma once
#include <atomic>
#include <condition_variable>
#include <deque>
#include <iostream>
#include <memory>
#include <mutex>
#include <string>
#include <thread>

namespace hy4 {

// A STOP cancels the current request and any earlier GEN still waiting to be parsed.
class LineInput {
public:
    std::atomic<bool> stop{false};

    static std::shared_ptr<LineInput> start() {
        auto input = std::make_shared<LineInput>();
        // On a fatal inference error stdin may still be open. The reader owns its state until EOF or QUIT.
        std::thread([input] { input->read_lines(); }).detach();
        return input;
    }

    bool next(std::string& out) {
        std::unique_lock<std::mutex> lock(mutex_);
        ready_.wait(lock, [this] { return !lines_.empty() || eof_; });
        if (lines_.empty()) return false;
        stop.store(cancelled_pending_ != 0);
        if (cancelled_pending_) --cancelled_pending_;
        out = std::move(lines_.front());
        lines_.pop_front();
        return true;
    }

private:
    void read_lines() {
        std::string line;
        while (std::getline(std::cin, line)) {
            if (!line.empty() && line.back() == '\r') line.pop_back();
            std::lock_guard<std::mutex> lock(mutex_);
            if (line == "STOP") {
                cancelled_pending_ = lines_.size();
                stop.store(true);
                continue;
            }
            lines_.push_back(line);
            ready_.notify_one();
            if (line == "QUIT") break;
        }
        std::lock_guard<std::mutex> lock(mutex_);
        eof_ = true;
        ready_.notify_one();
    }

    std::mutex mutex_;
    std::condition_variable ready_;
    std::deque<std::string> lines_;
    size_t cancelled_pending_ = 0;
    bool eof_ = false;
};

} // namespace hy4
