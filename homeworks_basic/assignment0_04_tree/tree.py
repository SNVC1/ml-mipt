import numpy as np
from sklearn.base import BaseEstimator


def entropy(y):  
    """
    Computes entropy of the provided distribution. Use log(value + eps) for numerical stability
    
    Parameters
    ----------
    y : np.array of type float with shape (n_objects, n_classes)
        One-hot representation of class labels for corresponding subset
    
    Returns
    -------
    float
        Entropy of the provided subset
    """
    EPS = 5e-4
    
    if len(y) == 0:
        return 1
    probs = y.mean(axis=0)
    return -np.sum(probs * np.log2(probs + (probs == 0) * EPS))
    
def gini(y):
    """
    Computes the Gini impurity of the provided distribution
    
    Parameters
    ----------
    y : np.array of type float with shape (n_objects, n_classes)
        One-hot representation of class labels for corresponding subset
    
    Returns
    -------
    float
        Gini impurity of the provided subset
    """
    if len(y) == 0:
        return 1
    probs = y.mean(axis=0)
    return 1 - np.sum(probs**2)
    
def variance(y):
    """
    Computes the variance the provided target values subset
    
    Parameters
    ----------
    y : np.array of type float with shape (n_objects, 1)
        Target values vector
    
    Returns
    -------
    float
        Variance of the provided target vector
    """
    
    if len(y) == 0:
        return 1
    return np.mean((y - np.mean(y))**2)

def mad_median(y):
    """
    Computes the mean absolute deviation from the median in the
    provided target values subset
    
    Parameters
    ----------
    y : np.array of type float with shape (n_objects, 1)
        Target values vector
    
    Returns
    -------
    float
        Mean absolute deviation from the median in the provided vector
    """
    
    if len(y) == 0:
        return 1
    return np.mean(abs(y - np.median(y)))


def one_hot_encode(n_classes, y):
    y_one_hot = np.zeros((len(y), n_classes), dtype=float)
    y_one_hot[np.arange(len(y)), y.astype(int)[:, 0]] = 1.
    return y_one_hot


def one_hot_decode(y_one_hot):
    return y_one_hot.argmax(axis=1)[:, None]


class Node:
    def __init__(self, feature_index, threshold, depth=1):
        self.feature_index = feature_index
        self.value = threshold
        self.depth = depth
        self.left_child = None
        self.right_child = None
        
        
class DecisionTree(BaseEstimator):
    all_criterions = {
        'gini': (gini, True), # (criterion, classification flag)
        'entropy': (entropy, True),
        'variance': (variance, False),
        'mad_median': (mad_median, False)
    }

    def __init__(self, n_classes=None, max_depth=np.inf, min_samples_split=2, 
                 criterion_name='gini', debug=False):

        assert criterion_name in self.all_criterions.keys(), 'Criterion name must be on of the following: {}'.format(self.all_criterions.keys())
        
        self.n_classes = n_classes
        self.max_depth = max_depth
        self.current_depth = 1
        self.min_samples_split = min_samples_split
        self.criterion_name = criterion_name

        self.root = None # Use the Node class to initialize it later
        self.debug = debug

        
        
    def make_split(self, feature_index, threshold, X_subset, y_subset):
        """
        Makes split of the provided data subset and target values using provided feature and threshold
        
        Parameters
        ----------
        feature_index : int
            Index of feature to make split with

        threshold : float
            Threshold value to perform split

        X_subset : np.array of type float with shape (n_objects, n_features)
            Feature matrix representing the selected subset

        y_subset : np.array of type float with shape (n_objects, n_classes) in classification 
                   (n_objects, 1) in regression 
            One-hot representation of class labels for corresponding subset
        
        Returns
        -------
        (X_left, y_left) : tuple of np.arrays of same type as input X_subset and y_subset
            Part of the providev subset where selected feature x^j < threshold
        (X_right, y_right) : tuple of np.arrays of same type as input X_subset and y_subset
            Part of the providev subset where selected feature x^j >= threshold
        """

        right_indices = np.argwhere(X_subset.T[feature_index] >= threshold).ravel()
        left_indices = np.argwhere(X_subset.T[feature_index] < threshold).ravel()
        
        X_left = X_subset[left_indices]
        y_left = y_subset[left_indices]
        X_right = X_subset[right_indices]
        y_right = y_subset[right_indices]
        
        return (X_left, y_left), (X_right, y_right)
    
    def make_split_only_y(self, feature_index, threshold, X_subset, y_subset):
        """
        Split only target values into two subsets with specified feature and threshold
        
        Parameters
        ----------
        feature_index : int
            Index of feature to make split with

        threshold : float
            Threshold value to perform split

        X_subset : np.array of type float with shape (n_objects, n_features)
            Feature matrix representing the selected subset

        y_subset : np.array of type float with shape (n_objects, n_classes) in classification 
                   (n_objects, 1) in regression 
            One-hot representation of class labels for corresponding subset
        
        Returns
        -------
        y_left : np.array of type float with shape (n_objects_left, n_classes) in classification 
                   (n_objects, 1) in regression 
            Part of the provided subset where selected feature x^j < threshold

        y_right : np.array of type float with shape (n_objects_right, n_classes) in classification 
                   (n_objects, 1) in regression 
            Part of the provided subset where selected feature x^j >= threshold
        """

        right_indices = np.argwhere(X_subset.T[feature_index] >= threshold).ravel()
        left_indices = np.argwhere(X_subset.T[feature_index] < threshold).ravel()
        
        y_left = y_subset[left_indices]
        y_right = y_subset[right_indices]
        
        return y_left, y_right

    def choose_best_split(self, X_subset, y_subset):
        """
        Greedily select the best feature and best threshold w.r.t. selected criterion
        
        Parameters
        ----------
        X_subset : np.array of type float with shape (n_objects, n_features)
            Feature matrix representing the selected subset

        y_subset : np.array of type float with shape (n_objects, n_classes) in classification 
                   (n_objects, 1) in regression 
            One-hot representation of class labels or target values for corresponding subset
        
        Returns
        -------
        feature_index : int
            Index of feature to make split with

        threshold : float
            Threshold value to perform split

        """
        assert len(y_subset) > 1
        G_opt = None
        threshold_opt = 0
        feature_index_opt = 0
        
        for feature_index in range(X_subset.shape[1]):
            for threshold in np.unique(X_subset.T[feature_index]):
                y_left, y_right = self.make_split_only_y(feature_index, threshold, X_subset, y_subset)
                G = (len(y_left) * self.criterion(y_left) + len(y_right) * self.criterion(y_right)) / len(y_subset)
                if G_opt is None or G < G_opt:
                    G_opt = G
                    threshold_opt = threshold
                    feature_index_opt = feature_index
        
        return feature_index_opt, threshold_opt
    
    def make_tree(self, X_subset, y_subset):
        """
        
        Parameters
        ----------
        X_subset : np.array of type float with shape (n_objects, n_features)
            Feature matrix representing the selected subset

        y_subset : np.array of type float with shape (n_objects, n_classes) in classification 
                   (n_objects, 1) in regression 
            One-hot representation of class labels or target values for corresponding subset
        
        Returns
        -------
        root_node : Node class instance
            Node of the root of the fitted tree
        """
        
        feasible_criterion = 5e-2
        assert len(y_subset) != 0
        if self.current_depth <= self.max_depth and \
           len(y_subset) >= self.min_samples_split and \
           self.criterion(y_subset) >= feasible_criterion: # children required
            
            feature_index, threshold = self.choose_best_split(X_subset, y_subset)
            new_node = Node(feature_index, threshold, self.current_depth)
            (X_left, y_left), (X_right, y_right) = self.make_split(feature_index, threshold, X_subset, y_subset)
            self.current_depth = new_node.depth + 1
            new_node.left_child = self.make_tree(X_left, y_left)
            self.current_depth = new_node.depth + 1
            new_node.right_child = self.make_tree(X_right, y_right)
            self.current_depth = new_node.depth
        else: # decision required
            new_node = Node(0, 0, self.current_depth)
            if self.classification:
                new_node.probs = y_subset.mean(axis=0)
            else:
                new_node.decision = np.mean(y_subset)
        
        return new_node
        
    def fit(self, X, y):
        """
        Fit the model from scratch using the provided data
        
        Parameters
        ----------
        X : np.array of type float with shape (n_objects, n_features)
            Feature matrix representing the data to train on

        y : np.array of type int with shape (n_objects, 1) in classification 
                   of type float with shape (n_objects, 1) in regression 
            Column vector of class labels in classification or target values in regression
        
        """
        assert len(y.shape) == 2 and len(y) == len(X), 'Wrong y shape'
        self.criterion, self.classification = self.all_criterions[self.criterion_name]
        if self.classification:
            if self.n_classes is None:
                self.n_classes = len(np.unique(y))
            y = one_hot_encode(self.n_classes, y)
        self.root = self.make_tree(X, y)
    
    def predict(self, X):
        """
        Predict the target value or class label the model from scratch using the provided data
        
        Parameters
        ----------
        X : np.array of type float with shape (n_objects, n_features)
            Feature matrix representing the data the predictions should be provided for

        Returns
        -------
        y_predicted : np.array of type int with shape (n_objects, 1) in classification 
                   (n_objects, 1) in regression 
            Column vector of class labels in classification or target values in regression
        
        """
        
        if self.classification:
            y_predicted = np.argmax(self.predict_proba(X), axis=1)
        else:
            y_predicted = np.zeros(len(X))
            for i, x in enumerate(X):
                current_node = self.root
                while not (current_node.left_child is None):
                    feature_index = current_node.feature_index
                    threshold = current_node.value
                    current_node = current_node.left_child if x[feature_index] < threshold else current_node.right_child
                y_predicted[i] = current_node.decision
    
        return y_predicted[:, np.newaxis]
        
    def predict_proba(self, X):
        """
        Only for classification
        Predict the class probabilities using the provided data
        
        Parameters
        ----------
        X : np.array of type float with shape (n_objects, n_features)
            Feature matrix representing the data the predictions should be provided for

        Returns
        -------
        y_predicted_probs : np.array of type float with shape (n_objects, n_classes)
            Probabilities of each class for the provided objects
        
        """
        assert self.classification, 'Available only for classification problem'

        y_predicted_probs = np.zeros((len(X), self.n_classes))
        for i, x in enumerate(X):
            current_node = self.root
            while not (current_node.left_child is None):
                feature_index = current_node.feature_index
                threshold = current_node.value
                current_node = current_node.left_child if x[feature_index] < threshold else current_node.right_child
            y_predicted_probs[i] = current_node.probs
     
        return y_predicted_probs
