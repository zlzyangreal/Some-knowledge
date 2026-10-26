# 1.函数
## 1.1[[lock()和unlock()]]
# 
## [[流水线特性的动态线程池任务调度系统]]

## 2. 基本概念

- **线程**：`std::thread`，创建即启动；必须 `join()`（等待结束）或 `detach()`（分离），否则析构时 `std::terminate`。
- **互斥量**：`std::mutex` + `std::lock_guard`（RAII，作用域结束自动解锁）/ `std::unique_lock`（可手动解锁、可配合条件变量）。
- **条件变量**：`std::condition_variable`，必须和 `unique_lock` 配合，等待时用 `wait(lock, pred)` 防虚假唤醒。
- **原子量**：`std::atomic<T>`，简单计数/标志位用它，比加锁便宜。
- **异步任务**：`std::async` + `std::future`，拿返回值比手工管理线程省事。

## 3. 最小可用示例

```cpp
#include <thread>
#include <mutex>
#include <vector>
#include <iostream>

std::mutex mtx;
long long counter = 0;

void worker(int n) {
    for (int i = 0; i < n; ++i) {
        std::lock_guard<std::mutex> lock(mtx);   // 作用域结束自动 unlock
        ++counter;
    }
}

int main() {
    std::vector<std::thread> pool;
    for (int i = 0; i < 4; ++i) pool.emplace_back(worker, 100000);
    for (auto& t : pool) t.join();
    std::cout << counter << std::endl;
}
```

条件变量版本（生产者—消费者骨架）：

```cpp
std::mutex mtx;
std::condition_variable cv;
std::queue<int> q;

// 消费者
std::unique_lock<std::mutex> lk(mtx);
cv.wait(lk, [] { return !q.empty(); });   // 带谓词，防虚假唤醒
int v = q.front(); q.pop();

// 生产者
{
    std::lock_guard<std::mutex> lk(mtx);
    q.push(1);
}
cv.notify_one();
```

## 4. 常见坑

1. 忘了 `join`/`detach` → 程序直接 abort。
2. 两个锁交叉加锁 → 死锁；固定加锁顺序，或用 `std::scoped_lock(lock1, lock2)`。
3. 锁的粒度太大（把 IO 放进临界区）→ 并行退化成串行。
4. 迭代器/引用在线程间共享容器被改 → 崩溃，容器操作也要加锁。
5. 只读共享数据不需要锁；要改就考虑 `atomic` 或“每线程一份 + 最后合并”。

## 相关

- [[lock()和unlock()]]
- [[流水线特性的动态线程池任务调度系统]]
- [[构造函数]]、[[移动构造函数]]（线程对象不可拷贝，只能 move）
