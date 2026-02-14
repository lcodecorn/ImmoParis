import numpy as np
from collections import Counter


class StandardScalerNumPy:
    """Standardize features by removing mean and scaling to unit variance"""
    
    def __init__(self):
        self.mean_ = None
        self.scale_ = None
        
    def fit(self, X):
        """Compute mean and std for scaling"""
        X = np.asarray(X)
        self.mean_ = np.mean(X, axis=0)
        self.scale_ = np.std(X, axis=0)
        self.scale_[self.scale_ == 0] = 1.0
        return self
    
    def transform(self, X):
        """Scale features"""
        X = np.asarray(X)
        return (X - self.mean_) / self.scale_
    
    def fit_transform(self, X):
        """Fit and transform"""
        return self.fit(X).transform(X)


class SimpleImputerNumPy:
    """Impute missing values"""
    
    def __init__(self, strategy='mean'):
        self.strategy = strategy
        self.statistics_ = None
        
    def fit(self, X):
        """Compute statistics for imputation"""
        X = np.asarray(X)
        
        if self.strategy == 'mean':
            self.statistics_ = np.nanmean(X, axis=0)
        elif self.strategy == 'median':
            self.statistics_ = np.nanmedian(X, axis=0)
        elif self.strategy == 'most_frequent':
            self.statistics_ = np.array([self._most_frequent(X[:, i]) for i in range(X.shape[1])])
        
        return self
    
    def _most_frequent(self, column):
        """Get most frequent value in column"""
        valid = column[~np.isnan(column)]
        if len(valid) == 0:
            return 0
        counter = Counter(valid)
        return counter.most_common(1)[0][0]
    
    def transform(self, X):
        """Impute missing values"""
        X = np.asarray(X, dtype=float).copy()
        
        for i in range(X.shape[1]):
            mask = np.isnan(X[:, i])
            X[mask, i] = self.statistics_[i]
        
        return X
    
    def fit_transform(self, X):
        """Fit and transform"""
        return self.fit(X).transform(X)


class DecisionTreeRegressorNumPy:
    """Fast Decision Tree Regressor using optimized NumPy operations"""
    
    def __init__(self, max_depth=None, min_samples_split=2, min_samples_leaf=1, 
                 max_features=None, random_state=None, criterion='squared_error'):
        self.max_depth = max_depth if max_depth is not None else 999
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.random_state = random_state
        self.criterion = criterion
        self.tree_ = None
        self.feature_importances_ = None
        
    def _compute_error(self, y):
        """Compute error metric for split evaluation"""
        if len(y) == 0:
            return 0
        
        if self.criterion == 'absolute_error':
            median = np.median(y)
            return np.mean(np.abs(y - median))
        else:
            mean = np.mean(y)
            return np.mean((y - mean) ** 2)
    
    def _best_split_fast(self, X, y, feature_indices, indices):
        """Fast best split using vectorized operations"""
        best_gain = -np.inf
        best_feature = None
        best_threshold = None
        
        current_error = self._compute_error(y[indices])
        n_total = len(indices)
        
        for feature in feature_indices:
            X_feat = X[indices, feature]
            
            unique_vals = np.unique(X_feat)
            if len(unique_vals) > 20:
                percentiles = np.linspace(5, 95, 20)
                thresholds = np.percentile(X_feat, percentiles)
            else:
                thresholds = unique_vals[:-1]
            
            for threshold in thresholds:
                left_mask = X_feat <= threshold
                n_left = np.sum(left_mask)
                n_right = n_total - n_left
                
                if n_left < self.min_samples_leaf or n_right < self.min_samples_leaf:
                    continue
                
                left_error = self._compute_error(y[indices][left_mask])
                right_error = self._compute_error(y[indices][~left_mask])
                
                weighted_error = (n_left * left_error + n_right * right_error) / n_total
                gain = current_error - weighted_error
                
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature
                    best_threshold = threshold
        
        return best_feature, best_threshold
    
    def _build_tree_fast(self, X, y, indices, depth=0):
        """Build tree using indices"""
        n_samples = len(indices)
        
        if self.criterion == 'absolute_error':
            leaf_value = np.median(y[indices])
        else:
            leaf_value = np.mean(y[indices])
        
        if depth >= self.max_depth or n_samples < self.min_samples_split or np.std(y[indices]) < 1e-7:
            return {'type': 'leaf', 'value': leaf_value, 'n_samples': n_samples}
        
        n_features = X.shape[1]
        if self.max_features == 'sqrt':
            n_select = max(1, int(np.sqrt(n_features)))
            feature_indices = self.rng.choice(n_features, n_select, replace=False)
        elif isinstance(self.max_features, int):
            feature_indices = self.rng.choice(n_features, min(self.max_features, n_features), replace=False)
        else:
            feature_indices = np.arange(n_features)
        
        feature, threshold = self._best_split_fast(X, y, feature_indices, indices)
        
        if feature is None:
            return {'type': 'leaf', 'value': leaf_value, 'n_samples': n_samples}
        
        left_mask = X[indices, feature] <= threshold
        left_indices = indices[left_mask]
        right_indices = indices[~left_mask]
        
        return {
            'type': 'node',
            'feature': feature,
            'threshold': threshold,
            'n_samples': n_samples,
            'left': self._build_tree_fast(X, y, left_indices, depth + 1),
            'right': self._build_tree_fast(X, y, right_indices, depth + 1)
        }
    
    def fit(self, X, y):
        """Build decision tree"""
        X = np.asarray(X)
        y = np.asarray(y)
        
        if self.random_state is not None:
            self.rng = np.random.RandomState(self.random_state)
        else:
            self.rng = np.random.RandomState()
        
        self.n_features_ = X.shape[1]
        indices = np.arange(len(X))
        self.tree_ = self._build_tree_fast(X, y, indices)
        self.feature_importances_ = self._compute_feature_importances()
        
        return self
    
    def _compute_feature_importances(self):
        """Compute feature importances"""
        importances = np.zeros(self.n_features_)
        
        def traverse(node):
            if node['type'] == 'leaf':
                return
            importances[node['feature']] += node['n_samples']
            traverse(node['left'])
            traverse(node['right'])
        
        traverse(self.tree_)
        
        if importances.sum() > 0:
            importances = importances / importances.sum()
        
        return importances
    
    def _predict_sample(self, x, node):
        """Predict single sample"""
        if node['type'] == 'leaf':
            return node['value']
        
        if x[node['feature']] <= node['threshold']:
            return self._predict_sample(x, node['left'])
        else:
            return self._predict_sample(x, node['right'])
    
    def predict(self, X):
        """Predict target values"""
        X = np.asarray(X)
        return np.array([self._predict_sample(x, self.tree_) for x in X])


class RandomForestRegressorNumPy:
    """Random Forest Regressor using NumPy"""
    
    def __init__(self, n_estimators=100, max_depth=None, min_samples_split=2,
                 min_samples_leaf=1, max_features='sqrt', random_state=None,
                 criterion='squared_error', n_jobs=1):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.random_state = random_state
        self.criterion = criterion
        self.n_jobs = n_jobs
        self.trees_ = []
        self.feature_importances_ = None
        
    def fit(self, X, y):
        """Fit Random Forest"""
        X = np.asarray(X)
        y = np.asarray(y)
        
        n_samples = X.shape[0]
        
        if self.random_state is not None:
            base_rng = np.random.RandomState(self.random_state)
        else:
            base_rng = np.random.RandomState()
        
        self.trees_ = []
        
        for i in range(self.n_estimators):
            tree_seed = base_rng.randint(0, 2**31)
            
            bootstrap_indices = base_rng.choice(n_samples, n_samples, replace=True)
            X_bootstrap = X[bootstrap_indices]
            y_bootstrap = y[bootstrap_indices]
            
            tree = DecisionTreeRegressorNumPy(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                max_features=self.max_features,
                random_state=tree_seed,
                criterion=self.criterion
            )
            
            tree.fit(X_bootstrap, y_bootstrap)
            self.trees_.append(tree)
        
        importances = np.zeros(X.shape[1])
        for tree in self.trees_:
            importances += tree.feature_importances_
        self.feature_importances_ = importances / len(self.trees_)
        
        return self
    
    def predict(self, X):
        """Predict using average of all trees"""
        X = np.asarray(X)
        predictions = np.array([tree.predict(X) for tree in self.trees_])
        return np.mean(predictions, axis=0)
