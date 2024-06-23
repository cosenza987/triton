#include <bits/stdc++.h>

using namespace std;

random_device rd;
mt19937 gen(rd());
uniform_int_distribution<long long> distrib(0, 100);

int main() {
    freopen("in.txt", "w", stdout);
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);
    int n;
    cin >> n;
    for(int i = 0; i < n; i++) {
        cout << distrib(gen) << " \n"[i == n - 1];
    }
    for(int i = 0; i < n; i++) {
        cout << distrib(gen) << " \n"[i == n - 1];
    }
    return 0;
}