#include <iostream>

using std::string;

class Employee{
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
    virtual void Work() { // needed to have that to have pointer polymorphism
        std::cout << Name << " is checking email, task backlog, performing tasks..." << std::endl;
    }
};

class Developer:public Employee { // Inheritance
public:
    string FavProgrammingLanguage;
    // void Work() {
    //     std::cout << Name << " is writing " << FavProgrammingLanguage << " code." << std::endl;
    // }
};

class Teacher:public Employee {
public:
    string Subject;
    void Work() {
        std::cout << Name << " is teaching " << Subject << "." << std::endl;
    }
};

int main() {
    Developer d;
    d.Name = "Saldina";
    d.FavProgrammingLanguage = "C++";

    Teacher t;
    t.Name = "John";
    t.Subject = "History";

    Employee* e1 = &d;
    Employee* e2 = &t;

    e1->Work();
    e2->Work(); 
}