# Introduction

The Simple QA dataset is a dataset of questions which contains questions that are easy to grade and have a single answer. We want to evaluate different language models on their ability to answer these questions correctly and whether function calling degrades performance

## Instructions

1. First, install the dependencies

```
pip install -r pyproject.toml
```

2. First, you'll need to create a new Braintrust dataset that will store your evaluation data. You can do this by running the following command

```
python3 ./generate_qa_dataset.py
```

3. Next, you'll be able to evaluate your model by running the `evaluate.py` script.

```
python3 ./evaluate.py
```

4. We then analyze the results in a notebook `analyze_results.ipynb`. We look at the relative accuracy of each model and whether function calling degrades performance. The bootstrapping analysis shows that there's no probably no statistical significance.

{
metadata: {
"topic": "Science and technology",
"answer_type": "Person",
"urls": [
"https://en.wikipedia.org/wiki/IEEE_Frank_Rosenblatt_Award",
"https://ieeexplore.ieee.org/author/37271220500",
"https://en.wikipedia.org/wiki/IEEE_Frank_Rosenblatt_Award",
"https://www.nxtbook.com/nxtbooks/ieee/awards_2010/index.php?startid=21#/p/20"
]
},
question : "Who received the IEEE Frank Rosenblatt Award in 2010?",
answer : "Michio Sugeno"
}
