# gpt-oss-20b — HumanEval (humaneval)

- Source run: `/Users/yenhaohuang/Desktop/model-tester/gpt-oss-20b-gguf/runs/20260307_145155_fasttextgenevalset2_humaneval`
- Accuracy: **96.0%** (96/100)
- Rows: 100

## First 20 predictions (for quick review)

| idx | gold | pred | passed | response (truncated) |
|---:|---|---|:---:|---|
| 1 |  |  | True | def max_fill(grid, capacity):     return sum((sum(row) + capacity - 1) // capacity for row in grid) |
| 2 |  |  | True | def get_positive(l: list):     """Return only positive numbers in the list.     >>> get_positive([-1, 2, -4, 5, 6])  ... |
| 3 |  |  | True | def fibfib(n: int) -> int:     """     Compute the n-th element of the FibFib sequence efficiently.          The FibF... |
| 4 |  |  | True | def monotonic(l: list):     """Return True if list elements are monotonically increasing or decreasing."""     if len... |
| 5 |  |  | True | def count_distinct_characters(string: str) -> int:     """ Given a string, find out how many distinct characters (reg... |
| 6 |  |  | True | def anti_shuffle(s):     result = []     word = []     for ch in s:         if ch == ' ':             if word:       ... |
| 7 |  |  | True | from typing import List   def intersperse(numbers: List[int], delimeter: int) -> List[int]:     """ Insert a number '... |
| 8 |  |  | True | def x_or_y(n, x, y):     """A simple program which should return the value of x if n is      a prime number and shoul... |
| 9 |  |  | True | def file_name_check(file_name):     # Condition 1: no more than three digits in the entire file name     if sum(ch.is... |
| 10 |  |  | True | def common(l1: list, l2: list):     """Return sorted unique common elements for two lists.     >>> common([1, 4, 3, 3... |
| 11 |  |  | True | def int_to_mini_roman(number):     """     Given a positive integer, obtain its roman numeral equivalent as a string,... |
| 12 |  |  | True | def correct_bracketing(brackets: str):     """ brackets is a string of "<" and ">".     return True if every opening ... |
| 13 |  |  | True | from typing import List  def separate_paren_groups(paren_string: str) -> List[str]:     groups: List[str] = []     cu... |
| 14 |  |  | True | def how_many_times(string: str, substring: str) -> int:     """ Find how many times a given substring can be found in... |
| 15 |  |  | True | def string_sequence(n: int) -> str:     """ Return a string containing space-delimited numbers starting from 0 upto n... |
| 16 |  |  | True | def cycpattern_check(a, b):     """Return True if any rotation of b is a substring of a."""     if not b:         ret... |
| 17 |  |  | True | from typing import List  def parse_music(music_string: str) -> List[int]:     """Parse a string of musical notes into... |
| 18 |  |  | True | def sum_product(numbers: List[int]) -> Tuple[int, int]:     total_sum = sum(numbers)     prod = 1     for n in number... |
| 19 |  |  | True | def solve(N):     total = sum(int(d) for d in str(N))     return bin(total)[2:] |
| 20 |  |  | True | def Strongest_Extension(class_name, extensions):     best_ext = None     best_strength = None     for ext in extensio... |

> Full predictions are in `predictions.csv`.
