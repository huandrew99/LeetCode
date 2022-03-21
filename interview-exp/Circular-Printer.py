"""
https://leetcode.com/playground/SRN4cLVP

A company has invented a new type of printing technology-a circular printer that
ooks like this:

It is a circular printer wheel with the letters A through Z in sequence. It wraps so A
and Z are adjacent. The printer has a pointer that is initially at 'A'. Moving from any
character to any adjacent character takes 1 second. It can move in either direction.
Given a string of letters, what is the minimum time needed to print the string?
(Note: Assume that printing does not take any time. Only consider the time it takes
for the pointer to move.)
"""


def minDist(s):
    d = 0
    s = 'A' + s

    for a, b in zip(s, s[1:]):
        d += pair_dist(a, b)
    return d


def pair_dist(a, b):
    d = (ord(a) - ord(b)) % 26
    return min(d, 26 - d)

def minDist2(s):
    res = 0
    s = 'A' + s
    for a, b in zip(s, s[1:]):
        dist = min((ord(a) - ord(b)) % 26, 26 - (ord(a) - ord(b)) % 26)
        res += dist
    return res

s = "AZGB"  # 13
print(minDist2(s))
s = 'ZNMD'  # 23
print(minDist2(s))


"""
Time O(N)
Space O(1)
"""
