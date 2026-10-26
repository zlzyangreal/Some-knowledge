```cpp
#include <iostream>
#include <vector>
#include <string>
#include <sstream>
#include <algorithm>
#include <queue>

class NAryTree {
private:
    int _val;
    std::vector<NAryTree*> _children;

    //这里必须加 const，因为被 const 函数调用了
    void _collectPreorder(std::vector<int>& result) const; 

public:
    NAryTree(int val) : _val(val) {}
    NAryTree(const std::string& str);
    
    // 析构函数（可选，但在 C++ 中 new 出来的对象最好 delete）
    ~NAryTree() { for (auto c : _children) delete c; }

    void printNAryTree() const;
};

//初始化列表加上 : _val(0)，防止解析失败时 _val 是随机乱码
NAryTree::NAryTree(const std::string& str) : _val(0) {
    std::string cleaned = str;
    cleaned.erase(std::remove_if(cleaned.begin(), cleaned.end(), ::isspace), cleaned.end());

    const size_t pos = cleaned.find("root=[");
    const size_t end = cleaned.find("]", pos);

    std::vector<int> values_temp;
    
    if (pos != std::string::npos && end != std::string::npos) {
        std::string num = cleaned.substr(pos + 6, end - (pos + 6));
        std::stringstream ss(num);
        std::string token;
        while (std::getline(ss, token, ',')) {
            if(token == "null") values_temp.push_back(-1);
            else values_temp.push_back(std::stoi(token));
        }
    }

    if (values_temp.empty()) return;

    this->_val = values_temp[0]; 

    std::queue<NAryTree*> q;
    q.push(this);

    size_t i = 2; 
    while (i < values_temp.size()) {
        NAryTree* parent = q.front();
        q.pop();

        while (i < values_temp.size() && values_temp[i] != -1) {
            NAryTree* child = new NAryTree(values_temp[i]);
            parent->_children.push_back(child); 
            q.push(child);
            i++;
        }
        i++; 
    }
}

//这里实现也必须加上 const
void NAryTree::_collectPreorder(std::vector<int>& result) const {
    result.push_back(_val);

    for (auto child : _children) {
        child->_collectPreorder(result);
    }
}

void NAryTree::printNAryTree() const {
    std::vector<int> result;

    _collectPreorder(result);
    
    std::cout << "[";
    for (size_t i = 0; i < result.size(); i++) {
        std::cout << result[i];
        if (i != result.size() - 1) std::cout << ",";
    }
    std::cout << "]" << std::endl;
}

int main() {
    std::string line;
    if (std::getline(std::cin, line)) {
        NAryTree tree(line);
        tree.printNAryTree();
    }
    return 0;
}
```