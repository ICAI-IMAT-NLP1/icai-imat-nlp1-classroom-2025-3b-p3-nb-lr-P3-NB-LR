import torch

try:
    from src.utils import SentimentExample
    from src.data_processing import bag_of_words
except ImportError:
    from utils import SentimentExample
    from data_processing import bag_of_words


class LogisticRegression:
    def __init__(self, random_state: int):
        self._weights: torch.Tensor = None
        self.random_state: int = random_state

    def fit(
        self,
        features: torch.Tensor,
        labels: torch.Tensor,
        learning_rate: float,
        epochs: int,
    ):
        """
        Train the logistic regression model using pre-processed features and labels.

        Args:
            features (torch.Tensor): The bag of words representations of the training examples.
            labels (torch.Tensor): The target labels.
            learning_rate (float): The learning rate for gradient descent.
            epochs (int): The number of iterations over the training dataset.

        Returns:
            None: The function updates the model weights in place.
        """
        # TODO: Implement gradient-descent algorithm to optimize logistic regression weights

        weights = self.initialize_parameters(features.shape[1], self.random_state)
        for epoch in range(epochs):
            features_with_bias = torch.cat((features, torch.ones((features.shape[0], 1))), dim=1)
            #y_pred = self.sigmoid(torch.matmul(features_with_bias, weights[:-1])) + weights[-1]
            y_pred = self.sigmoid(torch.matmul(features_with_bias, weights)) 
            loss = self.binary_cross_entropy_loss(y_pred, labels)
            # gradient en weights = (y_pred - y) * x
            gradient = torch.matmul(features_with_bias.T, (y_pred - labels)) / features.shape[0]
            # weights[:-1] -= learning_rate * gradient
            # weights[-1] -= learning_rate * torch.sum(y_pred - labels)
            weights -= learning_rate * gradient
            self._weights = weights
            print(f"Epoch {epoch + 1}, Loss: {loss.item()}")

        return

    def predict(self, features: torch.Tensor, cutoff: float = 0.5) -> torch.Tensor:
        """
        Predict class labels for given examples based on a cutoff threshold.

        Args:
            features (torch.Tensor): The bag of words representations of the input examples.
            cutoff (float): The threshold for classifying a sample as positive. Defaults to 0.5.

        Returns:
            torch.Tensor: Predicted class labels (0 or 1).
        """
        """
        1. Use the probabilities from predict_proba and classify each sample.
        2. Return class labels (0 or 1) based on a threshold, typically 0.5, turning probabilities into definitive
        predictions.
        """
        decisions: torch.Tensor = torch.tensor([0] * len(features))
        probs = self.predict_proba(features)
        for i, prob in enumerate(probs):
            if prob > cutoff:
                decisions[i] = 1
            else:
                decisions[i] = 0
        return decisions

    def predict_proba(self, features: torch.Tensor) -> torch.Tensor:
        """
        Predicts the probability of each sample belonging to the positive class using pre-processed features.

        Args:
            features (torch.Tensor): The bag of words representations of the input examples.

        Returns:
            torch.Tensor: A tensor of probabilities for each input sample being in the positive class.

        Raises:
            ValueError: If the model weights are not initialized (model not trained).
        """
        if self.weights is None:
            raise ValueError("Model not trained. Call the 'train' method first.")
        
        probabilities: torch.Tensor = torch.tensor([0.0] * len(features))
        for i, feature in enumerate(features):
            probabilities[i] = self.sigmoid(torch.matmul(feature, self._weights))
        
        return probabilities

    def initialize_parameters(self, dim: int, random_state: int) -> torch.Tensor:
        """
        Initialize the weights for logistic regression using a normal distribution.

        This function initializes the weights (and bias as the last element) with values drawn from a normal distribution.
        The use of random weights can help in breaking the symmetry and improve the convergence during training.

        Args:
            dim (int): The number of features (dimension) in the input data.
            random_state (int): A seed value for reproducibility of results.

        Returns:
            torch.Tensor: Initialized weights as a tensor with size (dim + 1,).
        """
        torch.manual_seed(random_state)
        
        params: torch.Tensor = torch.rand(dim + 1) * 0.01
        
        return params

    @staticmethod
    def sigmoid(z: torch.Tensor) -> torch.Tensor:
        """
        Compute the sigmoid of z.

        This function applies the sigmoid function, which is defined as 1 / (1 + exp(-z)).
        It is used to map predictions to probabilities in logistic regression.

        Args:
            z (torch.Tensor): A tensor containing the linear combination of weights and features.

        Returns:
            torch.Tensor: The sigmoid of z.
        """
        result: torch.Tensor = 1 / (1 + torch.exp(-z))
        return result

    @staticmethod
    def binary_cross_entropy_loss(
        predictions: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute the binary cross-entropy loss.

        The binary cross-entropy loss is a common loss function for binary classification. It calculates the difference
        between the predicted probabilities and the actual labels.

        Args:
            predictions (torch.Tensor): Predicted probabilities from the logistic regression model.
            targets (torch.Tensor): Actual labels (0 or 1).

        Returns:
            torch.Tensor: The computed binary cross-entropy loss.
        """
        y = targets
        #y_pred = predictions
        epsilon = 1e-15
        y_pred = torch.clamp(predictions, epsilon, 1 - epsilon)
        N = len(y)
        ce_loss: torch.Tensor = - (1/N) * torch.sum(y * torch.log(y_pred) + (1-y) * torch.log(1- y_pred) )
        return ce_loss

    @property
    def weights(self):
        """Get the weights of the logistic regression model."""
        return self._weights

    @weights.setter
    def weights(self, value):
        """Set the weights of the logistic regression model."""
        self._weights: torch.Tensor = value

