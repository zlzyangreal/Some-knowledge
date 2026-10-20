多线程加锁
```cpp
#include <iostream>
#include <mutex>
#include <thread>

std::mutex _mutex; // 全局互斥锁

void sharedFunction() {
    _mutex.lock(); // 显式加锁
    // ... 访问或修改共享资源 ...
    _mutex.unlock(); // 显式解锁
}

int main() {
    std::thread t1(sharedFunction);
    std::thread t2(sharedFunction);

    t1.join();
    t2.join();

    return 0;
}
```