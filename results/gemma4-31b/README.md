總覽
Benchmark	總題數	答錯數	準確率
mmlu	100	4	96.0%
geo-mmlu-high-school	100	8	92.0%
law-mmlu-professional	100	27	73.0%
gsm8k	100	11	89.0%
humaneval	100	100	0.0%
MMLU (4 題錯)
idx	正解	模型答	題目摘要
8	D	C	Ring homomorphism kernel / Q as ideal
25	B	A	Maximal ideal & prime ideal statements
35	A	D	Linear transformation injectivity statements
63	C	A	Quotient ring commutativity / ideal statements
Geo-MMLU High School (8 題錯)
idx	正解	模型答	題目摘要
1	B	A	Subsistence economies — main barrier
4	A	B	Burgess concentric zone — low-income slums zone
28	B	A	Population pyramid information displayed
42	B	A	Immigrants learning new country values (assimilation)
56	C	D	Burgess zone — better houses/single-family
61	D	A	New stores around shopping mall concept
75	C	A	Country preserving native language purity
81	A	B	Crop that began Third Agricultural Revolution
Law-MMLU Professional (27 題錯)
錯誤最多，且有多題空白回答（pred 為空）：

idx	正解	模型答	備註
2	B	(空)	拒答或格式錯誤
10	C	(空)	
13	A	(空)	
26	B	(空)	
37	A	(空)	
66	C	(空)	
79	D	(空)	
86	B	(空)	
90	A	(空)	
3,7,25,27,33,35,38,46,49,58,59,68,69,76,81,84,91,98	—	答錯	
Law 共有 9 題空白，是模型對法律題拒答或無法萃取答案的問題。

GSM8K (11 題錯)
idx	正解	模型答	備註
13	13	(空)	
18	57500	(空)	
25	26	26.00	格式問題（整數 vs 浮點數）
40	18	(空)	
45	20	(空)	
66	36	(空)	
67	48	(空)	
79	6	6.00	格式問題
86	44	(空)	
88	9360	(空)	
94	36	(空)	
GSM8K 有 9 題空白，另有 2 題因浮點數格式（26.00 vs 26）被判錯，這可能是評分邏輯需修正的問題。

HumanEval (全部 100 題錯)
96 題：NameError: name 'check' is not defined — 代表執行時找不到 test harness 的 check 函式，屬於執行環境問題，不是模型本身的 coding 能力問題。
4 題：empty response（空白回答）
HumanEval 的 0% 準確率不代表模型不會寫程式，而是測試框架的 check() 函式沒有正確注入，建議檢查執行環境設定。