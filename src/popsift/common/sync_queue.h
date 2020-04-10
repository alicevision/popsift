#pragma once

#include <condition_variable>
#include <mutex>
#include <queue>

namespace popsift {

/*************************************************************
 * SyncQueue
 * This is a basic alternative to the Boost sync_queue class.
 * It lets threads push and pull items off a queue in a thread
 * safe manner.
 *************************************************************/
template<typename T>
class SyncQueue {
 public:
  SyncQueue() = default;

  /* Push an item onto the queue and signal it's available. */
  void push(const T& value) {
    std::unique_lock<std::mutex> lock(mtx_);
    items_.push(value);
    lock.unlock();
    signal_.notify_one();
  }

  /* Check if the queue is empty - thread safety via mutex. */
  bool empty() {
    std::unique_lock<std::mutex> lock(mtx_);
    return items_.empty();
  }

  /* BLOCKING. Pull an item off the queue, or, wait until one arrives. */
  T pull() {
    std::unique_lock<std::mutex> lock(mtx_);
    signal_.wait(lock, [this] { return !items_.empty(); });
    auto ans = items_.front();
    items_.pop();
    return ans;
  }

 private:
  std::mutex mtx_;
  std::queue<T> items_;
  std::condition_variable signal_;
};

}  // namespace popsift