#pragma once

#include <condition_variable>
#include <mutex>
#include <queue>

namespace popsift {

/**
 * @brief A thread safe wrapper around std::queue (replaces boost::sync_queue).
 * @tparam T the value type that's stored in the queue.
 */
template<typename T>
class SyncQueue {
public:
  SyncQueue() = default;

  /**
   * @brief Push an item onto the queue and signal it's available.
   * @param[in] value the item to add to the queue.
   */
  void push(const T& value) {
    std::unique_lock<std::mutex> lock(mtx_);
    items_.push(value);
    lock.unlock();
    signal_.notify_one();
  }

  /**
   * @brief Check if the queue is empty - thread safety via mutex.
   * @return True if the queue is empty.
   */
  bool empty() {
    std::unique_lock<std::mutex> lock(mtx_);
    return items_.empty();
  }

  /**
   * @brief Pull an item off the queue, or, wait until one arrives. Blocking.
   * @return The front item that was popped off the queue.
   */
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