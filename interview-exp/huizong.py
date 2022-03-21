def sortAncestral(names):
    d = {'I': 1, 'V': 5, 'X': 10, 'L': 50}
    names = [name.split() for name in names]
    for n in names:
        rom = n[1]
        num = 0
        for i, c in enumerate(rom):
            if i < len(rom) - 1 and d[c] < d[rom[i + 1]]:
                num -= d[c]
            else:
                num += d[c]
        n.append(num)
    names.sort(key=lambda x: (x[0], x[-1]))
    for i, n in enumerate(names):
        names[i] = ' '.join([n[0], n[1]])
    return names

#circular printer
def minDist2(s):
    res = 0
    s = 'A' + s
    for a, b in zip(s, s[1:]):
        dist = min((ord(a) - ord(b)) % 26, 26 - (ord(a) - ord(b)) % 26)
        res += dist
    return res


class Solution:
    def countSmaller(self, nums: List[int]) -> List[int]:
        # binary index tree
        # unique values
        values = sorted(set(nums))
        v2treeid = {v: i + 1 for i, v in enumerate(values)}

        bit = [0] * (len(values) + 1)  # want an 1-indexed array

        def update(v):
            treeid = v2treeid[v]
            while treeid < len(bit):
                bit[treeid] += 1
                treeid += treeid & -treeid

        def query(v):
            treeid = v2treeid[v]
            ans = 0
            while treeid > 0:
                # query a smaller one
                ans += bit[treeid]
                treeid -= treeid & -treeid
            return ans

        # get result
        ans = [0] * len(nums)
        for i in reversed(range(len(nums))):
            n = nums[i]
            update(n)
            if v2treeid[n] > 1:
                ans[i] = query(values[v2treeid[n] - 2])
        return ans

class CustomStack:

    def __init__(self, maxSize: int):
        self.max_size = maxSize
        self.stack = []

    def push(self, x: int) -> None:
        if len(self.stack) < self.max_size:
            self.stack.append([x, 0])

    def pop(self) -> int:
        if not self.stack:
            return -1
        x, v = self.stack.pop()
        if self.stack:
        	# inherit the value to add
            self.stack[-1][-1] += v
        return x + v

    def increment(self, k: int, val: int) -> None:
        if not self.stack:
            return
        k = min(k, len(self.stack))
        # maintain the value on the top
        self.stack[k - 1][1] += val


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

#find size 3 inversions in a list
def find_inv2(l):
    sorted_unique_values = sorted(set(l))
    # BIT index starts from 1
    v2treeid = {v: i + 1 for i, v in enumerate(sorted_unique_values)}

    def update(v):
        treeid = v2treeid[v]
        while treeid < len(bit):
            # number of unique values
            bit[treeid] += 1
            treeid += treeid & -treeid

    def query(v):
        treeid = v2treeid[v]
        ans = 0
        while treeid > 0:
            # number of unique values
            ans += bit[treeid]
            treeid -= treeid & -treeid
        return ans

    # unique smaller values on the right
    smaller_right = [0] * len(l)
    bit = [0] * (len(sorted_unique_values) + 1)
    visited = set()
    for i in reversed(range(len(l))):
        v = l[i]
        if v not in visited:
            update(v)
        visited.add(v)

        if v != sorted_unique_values[0]:
            # smallest value has count = 0
            # query the nearest value smaller than v
            smaller_right[i] = query(sorted_unique_values[v2treeid[v] - 1 - 1])
    print(smaller_right)

    # unique larger values on the left
    larger_left = [0] * len(l)
    bit = [0] * (len(sorted_unique_values) + 1)
    visited = set()
    for i, n in enumerate(l):
        v = l[i]
        if v not in visited:
            update(v)
        visited.add(v)

        if v != sorted_unique_values[-1]:
            # largest value has count = 0
            larger_left[i] = query(sorted_unique_values[-1]) - query(sorted_unique_values[v2treeid[v] - 1])
    print(larger_left)

    # unique triplet
    visited2largeleft = {}
    cnt = 0
    for i, (v, n_large, n_small) in enumerate(zip(l, larger_left, smaller_right)):
        if v in visited2largeleft:
            # compared with the last occurrence, the new encountered larger value on the left
            # paired with the smaller values on the right make up of new triplets
            cnt += (n_large - visited2largeleft[v]) * n_small
        else:
            cnt += n_large * n_small
        visited2largeleft[v] = n_large

    return cnt

def increasing_subseq(arr):
    f = [0] * len(arr)

    for k in range(1, 4):
        for i in range(len(arr)):
            if k == 1:
                f[i] = 1
            else:
                f[i] = 0
                for j in range(i + 1, len(arr)):
                    if arr[j] < arr[i]:
                        f[i] += f[j]
    return sum(f)

from functools import cache
def solution(arr):
  if len(arr) < 3:
    return 0
  @cache
  def rec(idx, length):
    if length == 2:
      return 1
    if length >2:
      return 0
    total = 0
    for idx2 in range(len(arr)):
      if idx < idx2 and arr[idx] > arr[idx2]:
        total += rec(idx2, length+1)
    return total
  answer = 0
  for idx in range(len(arr)):
    answer += rec(idx, 0)
  return answer

#global local inversion
class Solution(object):
    def isIdealPermutation(self, A):
        for i,x in enumerate(A):
            if abs(i - x) > 1:
                return False
        return True

    def isIdealPermutation2(self, A):
        cmax = 0
        for i in range(len(A) - 2):
            cmax = max(cmax, A[i])
            if cmax > A[i + 2]:
                return False
        return True

# k smallest and largest aubstring
def min_max(s, k, n):
    # k: length of substring
    # n: number of 1
    substrlist = []
    cnt = 0  # counter for 1
    for i in range(len(s) - k + 1):
        sub_s = s[i: i + k]
        if i == 0:
            cnt = sub_s.count('1')
        else:
            if sub_s[-1] == '1':
                cnt += 1

        if cnt == n:
            substrlist.append(sub_s)
        if sub_s[0] == '1':
            cnt -= 1

    M = m = substrlist[0]
    for sub_s in substrlist:
        M = max(M, sub_s)
        m = min(m, sub_s)
    return M, m

#number of sets of k non overlapping line segments
class Solution2:
    def numberOfSets(self, n, k):
        res = 1
        for i in range(1, k * 2 + 1):
            res *= n + k - i
            res //= i
        return res % (10**9 + 7)

#reaching points
class Solution(object):
    def reachingPoints(self, sx, sy, tx, ty):
        while sx < tx and sy < ty:
            if tx < ty:
                ty %= tx
            else:
                tx %= ty
        if sx == tx and sy <= ty and (ty - sy) % sx == 0:
            return True
        return sy == ty and sx <= tx and (tx - sx) % sy == 0

#reformate dates
class Solution:
    def reformatDate(self, date: str) -> str:
        month = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
        d, m, y = date.split()
        m = str(month.index(m) + 1).rjust(2, '0')
        d = d[:-2].rjust(2, '0')

        return "-".join([y, m, d])

# shared interest

def find_max(friends_from, friends_to, friends_weight):
    # adjacency matrix
    weight2adjacent_mat = defaultdict(lambda: defaultdict(set))
    for f, t, w in zip(friends_from, friends_to, friends_weight):
        weight2adjacent_mat[w][f].add(t)
        weight2adjacent_mat[w][t].add(f)

    def dfs(node):
        if node in vis:
            return
        vis.add(node)
        connected_component.append(node)
        for n in adjacent_mat[node]:
            dfs(n)

    # connected component for each interest
    pair2cnt = defaultdict(int)
    for w, adjacent_mat in weight2adjacent_mat.items():
        vis = set()
        for node in adjacent_mat:
            connected_component = []
            dfs(node)

            for i in range(len(connected_component) - 1):
                for j in range(i + 1, len(connected_component)):
                    a = connected_component[i]
                    b = connected_component[j]
                    pair2cnt[(min(a, b), max(a, b))] += 1

    # count max
    max_interest_n = max(pair2cnt.values())
    return max(f * t for (f, t), cnt in pair2cnt.items() if cnt == max_interest_n)

#smallest subarray with given sum
class Solution:
    def minSubArrayLen(self, target: int, nums: List[int]) -> int:

        n = len(nums)
        ans = float('inf')
        left = sum = 0
        for i in range(n):
            sum += nums[i]
            while sum >= target:
                ans = min(i + 1 - left, ans)
                sum -= nums[left]
                left += 1
        return 0 if ans == float('inf') else ans

class Solution:
    def uniquePaths(self, m: int, n: int) -> int:
        dp = [[1] * n for _ in range(m)]
        for i in range(1, m):
            for j in range(1, n):
                dp[i][j] = dp[i - 1][j] + dp[i][j - 1]
        return dp[m - 1][n - 1]

#start and bars
from itertools import *
from bisect import *


def cnt(s, l, r):
    # 0-indexed
    l -= 1
    r -= 1

    # binary index tree
    # s = '|**|*|*'
    # star_cnt = [2, 1, 1]
    # bar_idx = [0, 3, 5]

    def update(idx, v):
        while idx < len(bit):
            bit[idx] += v
            idx += idx & -idx

    def query(idx):
        ans = 0
        while idx > 0:
            ans += bit[idx]
            idx -= idx & -idx
        return ans

    # preprocess
    bar_idx = [i for i, c in enumerate(s) if c == '|']
    star_cnt = [r_bar - l_bar - 1 for l_bar, r_bar in zip(bar_idx, bar_idx[1:])]
    print(star_cnt)

    # build BIT
    bit = [0] * (1 + len(star_cnt))
    for i, cnt in enumerate(star_cnt, 1):
        update(i, cnt)

    # query
    l = min(bisect_left(bar_idx, l), len(bar_idx) - 1)  # closest index >= l
    r = max(bisect_right(bar_idx, r) - 1, 0)  # closest index <= r
    if l >= r:
        return 0
    else:
        return query(r) - query(l)
