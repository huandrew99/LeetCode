"""
https://leetcode.com/playground/QarZVLpM

Given tiles of dimensions 1 x 1 x 2, determine how many ways they can be arranged to form a
rectangular solid of dimensions 2 x 2 x n.
For example, if n = 1, there are two ways of doing so:

The number of permutations increases quickly, so the return value should be modulo (109+7).
See the samples below for the n = 2 solution diagrams.
Function Description
Complete the function ways in the editor below. The function should return an array of integers
that represent the number of ways each solid can be formed.
"""
def count3d(n):
    if n == 1:
        return 2
    
    MOD = int(1e9 + 7)

    # full
    # f(n) = 2 * f(n - 1) + 5 * f(n - 2) + 4 * p(n - 1)
    # 2: vertical + horizontal
    # 5: all vertical + 4 * (2 ver, 2 hor)
    # 4: (1 hor), 4 orientations
    # cannot use 4 * p(n) because there are overlaps between f(n - 2)
    dp = [0] * (n + 2)
    dp2 = [0] * (n + 2)
    dp[0] = 1
    dp[1] = 2
    dp2[0] = 0
    dp2[1] = 1
    for i in range(2, n + 1):
        dp2[i] = dp2[i - 1] + dp[i - 1] % MOD
        dp[i] = (2 * dp2[i] + 2 * dp2[i - 1] + dp[i - 2]) % MOD
    return dp[n]
    
    # partial, one orientation, top layer is vertical
    # p(n) = f(n - 2) + p(n - 1)
    
                
print(count3d(100))  # 828630254


"""
Time O(N)
Space O(1)
"""
