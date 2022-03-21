"""
LC 209
Given an array of positive numbers and a positive number ‘S,’ find the length of the smallest contiguous subarray whose sum is greater than or equal to ‘S’. Return 0 if no such subarray exists.

Example 1:

Input: [2, 1, 5, 2, 3, 2], S=7 
Output: 2
Explanation: The smallest subarray with a sum greater than or equal to '7' is [5, 2].
Example 2:

Input: [2, 1, 5, 2, 8], S=7 
Output: 1
Explanation: The smallest subarray with a sum greater than or equal to '7' is [8].
Example 3:

Input: [3, 4, 1, 1, 6], S=8 
Output: 3
Explanation: Smallest subarrays with a sum greater than or equal to '8' are [3, 4, 1] 
or [1, 1, 6].
"""


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
            

"""
Time O(N)
Space O(1)
"""
