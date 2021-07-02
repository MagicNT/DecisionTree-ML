# Decision Tree Algorithm

## **Implementation**

### Scripting Language

- Python 3

### Dataset

- The banknote dataset involves predicting whether a given banknote is authentic given a number of measures taken from a photograph. This dataset contains 1,372 rows with 5 numeric variables (variance of Wavelet Transformed image (continuous), skewness of Wavelet Transformed image (continuous), kurtosis of Wavelet Transformed image (continuous), entropy of image (continuous), class (integer)). It is a binary classification problem with two classes. Below provides a list of the five variables in the dataset.
- Dataset link: http://archive.ics.uci.edu/ml/datasets/banknote+authentication

- Dataset excerpt:

```
0.01727,8.693,1.3989,-3.9668,0
3.2414,0.40971,1.4015,1.1952,0
2.2504,3.5757,0.35273,0.2836,0
-1.3971,3.3191,-1.3927,-1.9948,1
0.39012,-0.14279,-0.031994,0.35084,1
-1.6677,-7.1535,7.8929,0.96765,1
-3.8483,-12.8047,15.6824,-1.281,1
-3.5681,-8.213,10.083,0.96765,1
-2.2804,-0.30626,1.3347,1.3763,1
```

### Import Libraries

- We import the csv Python library to read in the dataset saved inside a CSV file
- We import the random Python library to generate random values
- We import the prettytable Python library to present the output results in a nice tabular layout
- We import the sklearn Python library to utilize its metric calculation functions


```python
import csv
import random
import prettytable
import sklearn.metrics
```

### Algorithm Evaluation

We will evaluate the algorithm via cross-validation by splitting the data into 7 folds (k=7) and computing the accuracy, precision, recall and F1 scores of predictions.

The general procedure for this evaluation is:
1. Shuffle the dataset randomly.
2. Split the dataset into k groups
3. For each unique group:
        - Take group as a test dataset
        - Take remaining groups as a training dataset
        - Fit a model on the training dataset and evaluate it on the test dataset
        - Retain the evaluation accuracy score and discard the model
4. Summarize the skill of the model using the sample of model evaluation scores


```python
class Evaluate:

	def __init__(self, dataset, func, k=7):
		self.k = k
		self.folds = []
		self.accuracy_metric = []
		self.f1_metric = []
		self.precision_metric = []
		self.recall_metric = []
		self.cross_validate(dataset)
		self.evaluate(dataset, func)


	def cross_validate(self, dataset):
		fold_size = int(len(dataset)/self.k)
		for i in range(self.k):
			fold = []
			while len(fold) < fold_size:
				index = random.randrange(len(dataset))
				fold.append(dataset.pop(index))
			self.folds.append(fold)


	def average(self, array):
		return round(sum(array)/len(array), 3)


	def accuracy(self, originalset, prediction):
		hits = 0
		for a, b in zip(originalset, prediction):
			if a == b:
				hits += 1
		return hits/len(originalset)


	def evaluate(self, dataset, func):
		for fold in list(self.folds):
			trainset = list(self.folds)
			trainset.remove(fold)
			trainset = sum(trainset, [])
			testset = [row for row in fold]
			originalset = [row[-1] for row in fold]
			prediction = func(trainset, testset)
			self.accuracy_metric.append(self.accuracy(originalset, prediction))
			self.f1_metric.append(sklearn.metrics.f1_score(originalset, prediction, average='macro'))
			self.precision_metric.append(sklearn.metrics.precision_score(originalset, prediction, average='macro'))
			self.recall_metric.append(sklearn.metrics.recall_score(originalset, prediction, average='macro'))


	def display(self):
		t = prettytable.PrettyTable(["Fold #", "Accuracy", "F1", "Precision", "Recall"])
		print(" [+] Evaluation Metric Scores: \n")
		for index in range(self.k):
			accuracy = round(self.accuracy_metric[index], 5)
			f1 = round(self.f1_metric[index], 5)
			precision = round(self.precision_metric[index], 5)
			recall = round(self.recall_metric[index], 5)
			t.add_row([index+1, accuracy, f1, precision, recall])
		t.add_row(["AVERAGE", self.average(self.accuracy_metric), self.average(self.f1_metric), self.average(self.precision_metric), self.average(self.recall_metric)])
		print(t)
```

### Decision Tree Algorithm Class

We implemented the algorithm in a modular form for better data management as well as better organization. In the initialization function of the class, we specify the required local variables (CSV dataset file name, maximum depth, minimum size) and trigger the dataset importation process.


```python
class Decisiontree:

	def __init__(self, filename, max_depth=5, min_size=10):
		self.dataset = None
		self.filename = filename
		self.max_depth = max_depth
		self.min_size = min_size
		self.import_dataset()
```

### Importing/Pre-Processing Dataset

In the below routine we import the corresponding dataset file and directly convert the read strings into numeric values


```python
	def import_dataset(self):
		with open(self.filename, "rt") as dataset_csvfile:
			dataset_reader = csv.reader(dataset_csvfile, delimiter=",", quoting=csv.QUOTE_NONNUMERIC)
			self.dataset = list(dataset_reader)
```

### Splitting Dataset
Below is the implementation of routine to split dataset into 2 lists of rows having index of attribute and its split value. With these 2 groups, we use the Gini score to evaluate the cost of the split. To split dataset we iterate over each row, to check if the attribute value is less/greater than the split value, then assign it to the left or right group.


```python
	def test_split(self, index, value, dataset):
		[L, R] = [[], []]
		for row in dataset:
			L.append(row) if row[index] < value else R.append(row)
		return [L, R]
```

### Gini Score

The Gini score is the cost function to evaluate splits in the dataset. A split in the dataset involves one input attribute and one value for that attribute. It is used to divide training patterns into 2 groups of rows. A Gini score describes the quality of the split by how mixed the classes are in the two groups created by the split. A perfect separation results in a Gini score of 0, whereas the worst case split results in a Gini score of 0.5 (i.e., 50/50 classes in each group).


```python
	def gini_score(self, groups, classes):
		n_instances = float(sum([len(group) for group in groups]))
		gini = 0.0
		for group in groups:
			if group == []:
				continue
			score = 0.0
			group_size = len(group)
			for class_val in classes:
				p = [row[-1] for row in group].count(class_val)/group_size
				score += p * p
			gini += (1.0-score)*(group_size/n_instances)
		return gini
```

### Select Best Split

We check every value on each attribute as a potential split, evaluate the split cost and find the best possible split to make, then use it as a node in the decision tree. Each node in the tree is a dictionary; when selecting the best split for use as new node for the tree we will store the index of the chosen attribute, the value of that attribute by which to split and the 2 groups of data split by the chosen split point. Below is the corresponding routine implementation, in which the best split is recorded then returned.


```python
	def select_best_split(self, dataset):
		class_values = list(set(row[-1] for row in dataset))
		b_index = b_value = b_score = 999
		b_groups = None
		for index in range(len(dataset[0])-1):
			for row in dataset:
				groups = self.test_split(index, row[index], dataset)
				gini = self.gini_score(groups, class_values)
				if gini < b_score:
					b_index = index
					b_value = row[index]
					b_score = gini
					b_groups = groups
		return {
			"index": b_index,
			"value": b_value,
			"groups": b_groups
		}
```

### Terminal Nodes

To stop adding nodes to tree, we use depth limit (i.e., maximum number of nodes from the root node of the tree) and number of rows limit (i.e., minimum number of training patterns) that the node is responsible for in the training dataset. When we stop growing tree, that node is called terminal and it is used to make a final prediction, by taking the group of rows assigned to it and selecting the most common class value in the group. The below routine will select a class value for a group of rows then return the most common output value in the list of rows.


```python
	def create_terminal_node(self, group):
		outcomes = [row[-1] for row in group]
		return max(set(outcomes), key=outcomes.count)
```

### Recursive Splitting

New nodes added to an existing node are called child nodes. A node can have 0 children (i.e., terminal node), 1 child or 2 child nodes. We will refer to the child nodes as left and right. Once a node is created, we can create child nodes recursively on each group of data from the split by calling the routine itself again. The below routine implements this recursive procedure.
1. Extract the 2 groups of data split by the node and delete them from the node
2. Check if either left or right group of rows is empty; if so create a terminal node
3. Check if maximum depth is reached; if so we create a terminal node
4. Process left child, and create terminal node if the group of rows is small, else create the left node in a depth first until the end of the tree is reached on this branch
5. Process right child in the same way, and rise the constructed tree to the root


```python
	def split_helper(self, node, side, value, depth):
		if len(value) <= self.min_size:
			node[side] = self.create_terminal_node(value)
		else:
			node[side] = self.select_best_split(value)
			self.split(node[side], depth+1)


	def split(self, node, depth):
		L, R = node["groups"]
		del(node["groups"])

		if not L or not R:
			node["L"] = node["R"] = self.create_terminal_node(L + R)
			return

		if depth >= self.max_depth:
			node["L"] = self.create_terminal_node(L)
			node["R"] = self.create_terminal_node(R)
			return

		self.split_helper(node, "L", L, depth)
		self.split_helper(node, "R", R, depth)

```

### Make Prediction

To make prediction we need a recursive routine, which is called with the left or the right child nodes where we check if the child node is either a terminal value to be returned as the prediction, or if it is a dictionary node containing another level of the tree. Below is the implemented routine, where the index and value of a given node are used to evaluate whether the row of provided data is located on the left or the right of the split.


```python
	def predict(self, node, row):
		node_child = node["R"]
		if row[node["index"]] < node["value"]:
			node_child = node["L"]
		if isinstance(node_child, dict):
			return self.predict(node_child, row)
		else:
			return node_child
```

### Decision Tree Algorithm Trigger Routine

Building tree requires creating the main node then calling for split() routine that works recursively to build the entine tree. Below is the routine that implements this part.


```python
	def decision_tree(self, train, test):
		tree = self.select_best_split(train)
		self.split(tree, 1)
		predictions = []
		for row in test:
			prediction = self.predict(tree, row)
			predictions.append(prediction)
		return predictions
```

### Driver Routine

Below is the main driver routine which creates a Decisiontree object that will import the dataset, then passes the dataset as well as the trigger decision tree routine over to the evaluation class to initiate the algorithm and measure its performance. In return, the Evaluate object will provide the accuracy results from this performance runtime, which we print in a nice way and provide their representative overall average score.


```python
def main():
	print("\n" + "="*70)
	print(" Decision Tree Algrithm")
	print("="*70 + "\n")
	random.seed(1)
	DECISIONTREE = Decisiontree("data_banknote_authentication.txt")
	EVALUATE = Evaluate(DECISIONTREE.dataset, DECISIONTREE.decision_tree)
	EVALUATE.display()
```


### Output
    
    ======================================================================
     Decision Tree Algrithm
    ======================================================================
    
     [+] Evaluation Metric Scores: 
    
    +---------+----------+---------+-----------+---------+
    |  Fold # | Accuracy |    F1   | Precision |  Recall |
    +---------+----------+---------+-----------+---------+
    |    1    | 0.97959  | 0.97959 |   0.9802  |  0.9798 |
    |    2    | 0.98469  | 0.98425 |  0.98342  | 0.98513 |
    |    3    | 0.95918  | 0.95915 |  0.95896  | 0.95953 |
    |    4    | 0.97449  | 0.97419 |  0.97748  | 0.97222 |
    |    5    | 0.97449  | 0.97344 |   0.9767  | 0.97069 |
    |    6    | 0.98469  | 0.98425 |  0.98513  | 0.98342 |
    |    7    | 0.96939  | 0.96875 |  0.96875  | 0.96875 |
    | AVERAGE |  0.975   |  0.975  |   0.976   |  0.974  |
    +---------+----------+---------+-----------+---------+



