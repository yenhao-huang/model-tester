# gpt-oss-20b — GSM8K (gsm8k)

- Source run: `/Users/yenhaohuang/Desktop/model-tester/gpt-oss-20b-gguf/runs/20260307_131727_fasttextgenevalset2_gsm8k`
- Accuracy: **94.0%** (94/100)
- Rows: 100

## First 20 predictions (for quick review)

| idx | gold | pred | passed | response (truncated) |
|---:|---|---|:---:|---|
| 1 | 79 | 79 | True | Alisa biked 12 mph × 4.5 h = 54 miles.   Stanley biked 10 mph × 2.5 h = 25 miles.    Total distance = 54 miles + 25 m... |
| 2 | 35 | 35 | True | The number of cars at the first break was \(50 + 20 = 70\).   During lunch, half of those cars left, so the remaining... |
| 3 | 100 | 100 | True | Mark lost a total of \(10 \text{ lb/month} \times 3 \text{ months} = 30 \text{ lb}\).   His final weight was 70 lb, s... |
| 4 | 40 | 40 | True | Brian had 20 video games, but after losing 5 he had \(20-5 = 15\) video games at the time of comparison.   Bobby has ... |
| 5 | 30 | 30 | True | Prince Bruce slew 30 dragons.   #### 30 |
| 6 | 170 | 170 | True | Abraham sold half of his 80 m², which is 40 m², for $50.   After a month he sold another ¼ of the remaining land:   r... |
| 7 | 29 | 29 | True | Steve’s wife is 4 years older than him, so she is \(60 + 4 = 64\) years old.   The son is half as old as his mother: ... |
| 8 | 5 | 5 | True | Each friend received 5 lollipops.   #### 5 |
| 9 | 12 | 12 | True | Teresa’s small stuffed animals cost \$4 each, so a large one costs 3 × \$4 = \$12.   Let \(L\) be the number of large... |
| 10 | 310 | 310 | True | The number of students doubles each month. Starting with 10 students:  - End of January: \(10 \times 2 = 20\) - End o... |
| 11 | 230 | 230 | True | Mark paid a total of $230 for the radiator and its installation.   #### 230 |
| 12 | 650 | 650 | True | The total number of letters is 650.   #### 650 |
| 13 | 75 | 75 | True | Susan works for a total of \(180\text{ min} - 30\text{ min} = 150\text{ min}\).   She earns \(\$5\) every 10 minutes,... |
| 14 | 7 | 7 | True | Verna’s total cost is   - Apples: \(3 \times \$1.50 = \$4.50\)   - Oranges: \(5 \times \$0.80 = \$4.00\)   - Peaches:... |
| 15 | 21 | 21 | True | Ann is 9 years old, so her brother’s current age is twice that:   \(2 \times 9 = 18\) years.   In 3 years, his age wi... |
| 16 | 1600 | 1600 | True | The most efficient choice is to buy the 2 × 8 lumber, which yields 400 sticks for $6.   With $24 he can purchase \( \... |
| 17 | 93 | 93 | True | Jen spends $3 per day on food for her 3 fish.   May has 31 days, so the total cost is    \[ 3 \text{ dollars/day} \ti... |
| 18 | 90 | 90 | True | The total number of jellyfish \(N\) satisfies  \[ \frac{1}{3}\left(\frac{N}{5}\right)=6 \]  because one‑third of the ... |
| 19 | 18 | 18 | True | She has already made \(18 + 21 + 15 = 54\) vlogs.   To reach the monthly goal of 72, she needs \(72 - 54 = 18\) more ... |
| 20 | 2400 | 2400 | True | John spends a total of $2400 on everything.   #### 2400 |

> Full predictions are in `predictions.csv`.
