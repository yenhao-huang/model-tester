# lfm2-24b — HumanEval (humaneval)

- Source run: `/Users/yenhaohuang/Desktop/model-tester/lfm2-24b/runs/20260306_114503_fasttextgenevalset2_500q`
- Accuracy: **80.0%** (80/100)
- Rows: 100

## First 20 predictions (for quick review)

| idx | gold | pred | passed | response (truncated) |
|---:|---|---|:---:|---|
| 1 |  |  | False | def max_fill(grid, capacity):     total_water = 0     for row in grid:         total_water += sum(row)     return (to... |
| 2 |  |  | True | def get_positive(l: list):     return [x for x in l if x > 0] |
| 3 |  |  | True | def fibfib(n: int):     if n == 0:         return 0     elif n == 1:         return 0     elif n == 2:         return... |
| 4 |  |  | True | def monotonic(l: list):     if len(l) <= 1:         return True     increasing = all(l[i] <= l[i+1] for i in range(le... |
| 5 |  |  | True | def count_distinct_characters(string: str) -> int:     return len(set(string.lower())) |
| 6 |  |  | False | def anti_shuffle(s):     words = s.split(' ')     ordered_words = []     for word in words:         if word.isalpha()... |
| 7 |  |  | True | def intersperse(numbers: List[int], delimeter: int) -> List[int]:     result = []     for i, num in enumerate(numbers... |
| 8 |  |  | True | def x_or_y(n, x, y):     def is_prime(num):         if num <= 1:             return False         if num <= 3:       ... |
| 9 |  |  | True | def file_name_check(file_name):     if file_name.count('.') != 1:         return 'No'     parts = file_name.split('.'... |
| 10 |  |  | True | def common(l1: list, l2: list):     return sorted(set(l1) & set(l2)) |
| 11 |  |  | True | def int_to_mini_roman(number):     val = [         (1000, 'm'), (900, 'cm'), (500, 'd'), (400, 'cd'),         (100, '... |
| 12 |  |  | True | def correct_bracketing(brackets: str):     stack = []     for char in brackets:         if char == '<':             s... |
| 13 |  |  | True | def separate_paren_groups(paren_string: str) -> List[str]:     result = []     current = ""     balance = 0     for c... |
| 14 |  |  | True | def how_many_times(string: str, substring: str) -> int:     count = 0     start = 0     while start < len(string):   ... |
| 15 |  |  | True | def string_sequence(n: int) -> str:     return ' '.join(str(i) for i in range(n + 1)) |
| 16 |  |  | True | def cycpattern_check(a , b):     if len(b) > len(a):         return False     for i in range(len(b)):         rotated... |
| 17 |  |  | True | ``` def parse_music(music_string: str) -> List[int]:     note_map = {         'o': 4,         'o\|': 2,         '.\|': ... |
| 18 |  |  | True | def sum_product(numbers: List[int]) -> Tuple[int, int]:     total_sum = sum(numbers)     total_product = 1     for nu... |
| 19 |  |  | True | def solve(N):     digit_sum = sum(int(d) for d in str(N))     return bin(digit_sum)[2:] |
| 20 |  |  | True | def Strongest_Extension(class_name, extensions):     max_strength = -float('inf')     best_extension = None     for e... |

> Full predictions are in `predictions.csv`.
