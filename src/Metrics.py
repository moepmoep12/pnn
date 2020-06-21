import numpy as np


class Metric:
    """
    A metric measures the value of the predicted data
    """

    def score(self, predicted, actual):
        """
        :param predicted: The predicted classes/labels
        :param actual: The ground truth
        """
        pass


class Accuracy(Metric):
    """
    Calculates the accuracy of the prediction
    """

    def score(self, predicted, actual):
        num_correct_predicted = np.array([np.allclose(p, a) for p, a in zip(predicted, actual)]).sum()
        curr_accuracy = num_correct_predicted / len(predicted)
        return curr_accuracy


class Precision(Metric):
    """
    Calculates the precision of the prediction
    """

    def score(self, predicted, actual):
        true_positives = (predicted == 1) & (actual == 1)
        false_positives = (predicted == 1) & (actual == 0)
        return np.sum(true_positives) / (np.sum(true_positives) + np.sum(false_positives))


class Recall(Metric):
    """
    Calculates the recall of the prediction
    """

    def score(self, predicted, actual):
        true_positives = (predicted == 1) & (actual == 1)
        false_negatives = (predicted == 0) & (actual == 1)
        return np.sum(true_positives) / (np.sum(true_positives) + np.sum(false_negatives))


class F1(Metric):
    """
    Calculates the f1 score for two classes
    """

    def score(self, predicted, actual):
        precision = Precision().score(predicted, actual)
        recall = Recall().score(predicted, actual)
        if precision + recall == 0 or precision != precision or recall != recall:
            return 0
        else:
            return 2 * (precision * recall) / (precision + recall)


class MultiF1(Metric):
    """
    Calculates the f1 score for multiple classes
    """

    def score(self, predicted, actual):
        h_one_hot = np.eye(actual.shape[1])[np.argmax(predicted, axis=1)]
        f1_metric = F1()
        f1_scores = []

        labels = range(actual.shape[1])
        for label in labels:
            f1_scores.append(f1_metric.score(h_one_hot[:, label], actual[:, label]))

        return np.array(f1_scores)


class MultiF1Mean(Metric):

    def score(self, predicted, actual):
        return MultiF1().score(predicted, actual).mean()
