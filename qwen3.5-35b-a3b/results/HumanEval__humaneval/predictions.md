# qwen3.5-35b-a3b — HumanEval (humaneval)

- Source run: `/Users/yenhaohuang/Desktop/model-tester/qwen3.5_35b_a3b/runs/20260307_162542_fasttextgenevalset2_humaneval`
- Accuracy: **92.0%** (92/100)
- Rows: 100

## First 20 predictions (for quick review)

| idx | gold | pred | passed | response (truncated) |
|---:|---|---|:---:|---|
| 1 |  |  | True | def max_fill(grid, capacity):     import math     total_drops = 0     for well in grid:         water_units = sum(wel... |
| 2 |  |  | True | def get_positive(l: list):     return [x for x in l if x > 0] |
| 3 |  |  | True | def fibfib(n: int):     if n == 0:         return 0     if n == 1:         return 0     if n == 2:         return 1  ... |
| 4 |  |  | True | def monotonic(l: list):     if len(l) <= 1:         return True          increasing = True     decreasing = True     ... |
| 5 |  |  | True | def count_distinct_characters(string: str) -> int:     """ Given a string, find out how many distinct characters (reg... |
| 6 |  |  | True | def anti_shuffle(s):     words = s.split(' ')     result = []     for word in words:         sorted_word = ''.join(so... |
| 7 |  |  | True | def intersperse(numbers: List[int], delimeter: int) -> List[int]:     if not numbers:         return []     result = ... |
| 8 |  |  | True | def x_or_y(n, x, y):     def is_prime(num):         if num < 2:             return False         if num == 2:        ... |
| 9 |  |  | True | def file_name_check(file_name):     if file_name.count('.') != 1:         return 'No'          name, ext = file_name.... |
| 10 |  |  | True | def common(l1: list, l2: list):     return sorted(set(l1) & set(l2)) |
| 11 |  |  | True | def int_to_mini_roman(number):     val_map = [         (1000, 'm'), (900, 'cm'), (500, 'd'), (400, 'cd'),         (10... |
| 12 |  |  | True | def correct_bracketing(brackets: str):     """ brackets is a string of "<" and ">".     return True if every opening ... |
| 13 |  |  | True | def separate_paren_groups(paren_string: str) -> List[str]:     result = []     current = []     depth = 0          fo... |
| 14 |  |  | True | def how_many_times(string: str, substring: str) -> int:     if not substring:         return 0     count = 0     star... |
| 15 |  |  | True | def string_sequence(n: int) -> str:     return " ".join(str(i) for i in range(n + 1)) |
| 16 |  |  | True | def cycpattern_check(a, b):     if len(b) > len(a):         return False     for i in range(len(b)):         rotated ... |
| 17 |  |  | True | def parse_music(music_string: str) -> List[int]:     notes = music_string.split()     result = []     for note in not... |
| 18 |  |  | True | def sum_product(numbers: List[int]) -> Tuple[int, int]:     total_sum = sum(numbers)     total_product = 1     for nu... |
| 19 |  |  | True | def solve(N):     digit_sum = sum(int(d) for d in str(N))     return bin(digit_sum)[2:] |
| 20 |  |  | True | def Strongest_Extension(class_name, extensions):     strongest_ext = None     max_strength = None          for ext in... |

> Full predictions are in `predictions.csv`.
