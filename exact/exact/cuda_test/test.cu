#include <iostream>

using namespace std;
int main(){
    int2 a = make_int2(1,2);
    cout<<a.x<<endl;
    cout<<sizeof(a.x)<<endl;
    return 0;
}
