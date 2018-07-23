import numpy as np
from sklearn.datasets import load_iris

iris_dataset = load_iris()
X = iris_dataset.data
y = iris_dataset.target
dataset = np.column_stack((X,y))

def gini_index(groups, classes):
	n_samples = sum(len(group) for group in groups)
	gini_score = 0
	for group in groups:
		group_size = len(group)
		score = 0
		if group_size == 0:
			continue
		for class_value in classes:
			p = [row[-1] for row in group].count(class_value)/group_size
			score+=p*p
		gini_score += (1-score)*(group_size/n_samples)
	return gini_score

def split(feature, value, dataset):
	Xi_left = np.array([]).reshape(0, dataset.shape[-1])
	Xi_right = np.array([]).reshape(0, dataset.shape[-1]) 

	for Xi in dataset:
		if Xi[feature] <= value:
			Xi_left = np.vstack((Xi_left, Xi))
		else:
			Xi_right = np.vstack((Xi_right, Xi))

	return Xi_left, Xi_right

def best_split(dataset):
	classes = np.unique(dataset[:,-1])
	best_feature, best_value, best_score, best_groups = 1000, 1000, 1000, None
	for feature in range(dataset.shape[1]-1):
		for Xi in dataset:
			groups=split(feature, Xi[feature], dataset)
			gini_score = gini_index(groups, classes)
			if gini_score < best_score:
				best_feature, best_value, best_score, best_groups = feature, Xi[feature], gini_score, groups
	output = {}
	output["feature"] = best_feature
	output["groups"] = best_groups
	output["value"] = best_value
	return output

def terminal_node(group):
	classes, counts = np.unique(group[:,-1],return_counts=True)
	return classes[np.argmax(counts)]

def split_branch(node, max_depth, min_num_sample, depth):
    left_node, right_node = node['groups']
    del(node['groups'])
    if not isinstance(left_node,np.ndarray) or not isinstance(right_node,np.ndarray):
        node['left'] = node['right'] = terminal_node(left_node + right_node)
        return
    if depth >= max_depth:
        node['left'] = terminal_node(left_node)
        node['right'] = terminal_node(right_node)
        return
    if len(left_node) <= min_num_sample:
        node['left'] = terminal_node(left_node)
    else:
        node['left'] = best_split(left_node)
        split_branch(node['left'], max_depth, min_num_sample, depth+1)
    if len(right_node) <= min_num_sample:
        node['right'] = terminal_node(right_node)
    else:
        node['right'] = best_split(right_node)
        split_branch(node['right'], max_depth, min_num_sample, depth+1)

split_dictionary = best_split(dataset)
left, right = split_dictionary['groups']

def build_tree(dataset, max_depth, min_num_sample):
    root = best_split(dataset)
    split_branch(root, max_depth, min_num_sample, 1)
    return root

dataset = np.column_stack((X,y))
tree = build_tree(dataset, 2, 30)

def display_tree(node, depth=0):
    if isinstance(node,dict):
        print('{}[feat{} < {:.2f}]'.format(depth*'\t',(node['feature']+1), node['value']))
        display_tree(node['left'], depth+1)
        display_tree(node['right'], depth+1)
    else:
        print('{}[{}]'.format(depth*'\t', node))

display_tree(tree)


def predict_sample(node,sample):
    print(node)
    if sample[node['feature']] < node['value']:
        if isinstance(node['left'],dict):
            return predict_sample(node['left'],sample)
        else:
            return node['left']
    else:
        if isinstance(node['right'],dict):
            return predict_sample(node['right'],sample)
        else:
            return node['right']


def predict(X):
    y_pred = np.array([])
    for i in X:
        y_pred = np.append(y_pred,predict_sample(tree,i))
    return y_pred

y_pred = predict(X)

accuracy = 0.0
for i in range(y.shape[0]):
	if y[i] == y_pred[i]:
		accuracy+=1
accuracy/=y.shape[0]
print("Accuracy = ", accuracy)
