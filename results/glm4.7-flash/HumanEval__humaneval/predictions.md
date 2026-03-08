# glm4.7-flash — HumanEval (humaneval)

- Source run: `/Users/yenhaohuang/Desktop/model-tester/glm-4.7-flash-fp4/runs/20260307_154444_fasttextgenevalset2_humaneval`
- Accuracy: **76.0%** (76/100)
- Rows: 100

## First 20 predictions (for quick review)

| idx | gold | pred | passed | response (truncated) |
|---:|---|---|:---:|---|
| 1 |  |  | False | total_water = sum(sum(row) for row in grid)     return math.ceil(total_water / capacity) |
| 2 |  |  | True | def get_positive(l: list):     return [x for x in l if x > 0] |
| 3 |  |  | True | if n == 0:         return 0     elif n == 1:         return 0     elif n == 2:         return 1     a, b, c = 0, 0, 1... |
| 4 |  |  | True | if len(l) <= 2:         return True     increasing = decreasing = True     for i in range(len(l) - 1):         if l[i... |
| 5 |  |  | True | return len({c.lower() for c in string}) |
| 6 |  |  | True | def anti_shuffle(s):     return ' '.join(''.join(sorted(word)) for word in s.split(' ')) |
| 7 |  |  | True | result = []     for i, num in enumerate(numbers):         result.append(num)         if i < len(numbers) - 1:        ... |
| 8 |  |  | True | def x_or_y(n, x, y):     if n < 2:         return y     for i in range(2, int(n**0.5) + 1):         if n % i == 0:   ... |
| 9 |  |  | False | def file_name_check(file_name):     if file_name.count('.') != 1:         return 'No'     name, ext = file_name.split... |
| 10 |  |  | True | return sorted(set(l1) & set(l2)) |
| 11 |  |  | True | values = [         (1000, 'm'),         (900, 'cm'),         (500, 'd'),         (400, 'cd'),         (100, 'c'),    ... |
| 12 |  |  | True | stack = []     for char in brackets:         if char == "<":             stack.append(char)         elif char == ">":... |
| 13 |  |  | True | result = []     current = []     depth = 0     for char in paren_string.replace(' ', ''):         if char == '(':    ... |
| 14 |  |  | True | count = 0     start = 0     while True:         start = string.find(substring, start)         if start == -1:        ... |
| 15 |  |  | True | return ' '.join(str(i) for i in range(n + 1)) |
| 16 |  |  | True | def cycpattern_check(a , b):     b_len = len(b)     if b_len == 0:         return False     for i in range(b_len):   ... |
| 17 |  |  | True | return [4 if note == 'o' else 2 if note == 'o\|' else 1 for note in music_string.split()] |
| 18 |  |  | True | total = 0     product = 1     for num in numbers:         total += num         product *= num     return (total, prod... |
| 19 |  |  | False | return bin(sum(int(d) for d in bin(N)[2:]))[2:] |
| 20 |  |  | True | def Strongest_Extension(class_name, extensions):         max_strength = -float('inf')         strongest_extension = N... |

> Full predictions are in `predictions.csv`.
