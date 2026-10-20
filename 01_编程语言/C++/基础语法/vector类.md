`vector`是一个动态数组，它是标准模板库（STL）的一部分。`vector`提供了一种灵活的方式来存储和管理元素集合，它在内存中连续存储元素，支持随机访问，并且可以动态地增加或减少其大小。
以下是`vector`的一些主要特性和操作：
# 1. **动态大小**
`vector`的大小可以根据需要动态地增长或缩小，不需要事先指定大小。
# 2. **连续内存**
`vector`中的元素存储在连续的内存块中，这使得访问元素时可以像访问普通数组一样快速。
# 3. **随机访问迭代器**
`vector`提供了随机访问迭代器，这意味着你可以使用下标操作符`[]`来访问元素，也可以使用迭代器进行遍历。
# 4. **内存管理**
`vector`自动管理其存储空间，当元素数量超过当前容量时，`vector`会自动重新分配更大的内存空间，并复制现有元素到新内存。
# 5. **构造函数**
可以创建空的`vector`或使用初始化列表、另一个`vector`或特定数量的元素来初始化。
# 6. **添加元素**
- `push_back()`：在`vector`的末尾添加一个元素。
- `insert()`：在指定位置插入一个或多个元素。
# 7. **删除元素**：
- `pop_back()`：删除`vector`末尾的元素。
- `erase()`：删除指定位置的一个或多个元素。
# 8. **大小和容量**
- `size()`：返回`vector`中元素的数量。
- `capacity()`：返回`vector`当前分配的内存可以容纳的元素数量。
- `resize()`：调整`vector`的大小，可以增加或减少元素。
# 9. **访问元素**：
- 使用下标操作符`[]`或`at()`方法可以访问或修改元素。
# 10. **清空内容**
`clear()`方法可以移除`vector`中的所有元素。
# 11. **数据交换**
`swap()`方法可以交换两个`vector`的内容。
# 12. **类型模板**
`vector`是一个模板类，可以存储任何类型的数据，例如`vector<int>`、`vector<string>`等。
# `vector`的使用示例：

```cpp
#include <vector> 
int main() {     
	std::vector<int> myVector;     // 添加元素     
	myVector.push_back(10);     
	myVector.push_back(20);     // 访问元素     
	int firstElement = myVector[0]; // 获取第一个元素     
	// 遍历vector     
	for (int i : myVector) {         
		std::cout << i << " ";     
	}     
	// 删除元素     
	myVector.pop_back();     
	// 清空vector     
	myVector.clear();     
	return 0; 
}
```
```cpp
vector<int> num1;
//一维int动态大小数组
vector<vector<int>> num2;
// 二维int动态大小数组
```
`vector`是C++中非常常用的容器，适用于需要动态数组的场景。