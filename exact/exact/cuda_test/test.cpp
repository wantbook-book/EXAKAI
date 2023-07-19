#include <iostream>
using namespace std;
int main(){
    const float alpha = 1.0;
    const float e = 2.0;
    float res = (((alpha * (e - 1.0)<0))? (alpha*e) : 0.0);
    cout<<res<<endl;
    res = ((alpha * (e - 1.0)<0))? (alpha*e) : 0.0;
    cout<<res<<endl;
    res = (alpha * (e - 1.0)<0)? (alpha*e) : 0.0;
    cout<<res<<endl;
    res = alpha * (e - 1.0)<0? alpha*e : 0.0;
    cout<<res<<endl;
    return 0;
}