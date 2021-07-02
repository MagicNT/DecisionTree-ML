import csv
import random
import prettytable
import sklearn.metrics



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


class Decisiontree:

	def __init__(self, filename, max_depth=5, min_size=10):
		self.dataset = None
		self.filename = filename
		self.max_depth = max_depth
		self.min_size = min_size
		self.import_dataset()


	def import_dataset(self):
		with open(self.filename, "rt") as dataset_csvfile:
			dataset_reader = csv.reader(dataset_csvfile, delimiter=",", quoting=csv.QUOTE_NONNUMERIC)
			self.dataset = list(dataset_reader)


	def test_split(self, index, value, dataset):
		[L, R] = [[], []]
		for row in dataset:
			L.append(row) if row[index] < value else R.append(row)
		return [L, R]


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


	def create_terminal_node(self, group):
		outcomes = [row[-1] for row in group]
		return max(set(outcomes), key=outcomes.count)


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


	def predict(self, node, row):
		node_child = node["R"]
		if row[node["index"]] < node["value"]:
			node_child = node["L"]
		if isinstance(node_child, dict):
			return self.predict(node_child, row)
		else:
			return node_child


	def decision_tree(self, train, test):
		tree = self.select_best_split(train)
		self.split(tree, 1)
		predictions = []
		for row in test:
			prediction = self.predict(tree, row)
			predictions.append(prediction)
		return predictions


def main():
	print("\n" + "="*70)
	print(" Decision Tree Algrithm")
	print("="*70 + "\n")
	random.seed(1)
	DECISIONTREE = Decisiontree("data_banknote_authentication.txt")
	EVALUATE = Evaluate(DECISIONTREE.dataset, DECISIONTREE.decision_tree)
	EVALUATE.display()


main()