#include <iostream>

using std::string;

class AbstractEmployee { // Abstract class
    virtual void AskForPromotion() = 0; // Pure virtual function (bcs = 0): any class that need to inherit from this class needs to implement this function
};


class Employee:AbstractEmployee {
public:
    string Name;
    string Company;
    int Age;

    void AskForPromotion() {
        if (Age > 30) {
            std::cout << Name << " got promoted!" << std::endl;
        }
        else {
            std::cout << Name << ", sorry no promotion for you!" << std::endl;
        }
    }
};

int main() {
    Employee employee1;
    employee1.Name = "Saldina";
    employee1.Company = "YT-CodeBeauty";
    employee1.Age = 25;

    employee1.AskForPromotion();

    Employee employee2;
    employee2.Name = "John";
    employee2.Company = "Amazon";
    employee2.Age = 35;

    employee2.AskForPromotion();
}

