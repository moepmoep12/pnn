class NormalScaler():
    def fit(self, X, has_bias_column=False):
        self.min = X.min(axis=0)
        self.max = X.max(axis=0) - self.min

        if has_bias_column:
            self.min[0] = 0

    def transform(self, X):
        return (X - self.min) / self.max

    def inverse_transform(self, X_scaled):
        return X_scaled * self.max + self.min


class StandardScaler():
    def fit(self, X):
        self.mean = X.mean(axis=0)
        self.std = X.std(axis=0)

        # the standard deviation can be 0, which provokes
        # devision-by-zero errors; let's omit that:
        self.std[self.std == 0] = 0.00001

    def transform(self, X):
        return (X - self.mean) / self.std

    def inverse_transform(self, X_scaled):
        return X_scaled * self.std + self.mean
