import torch
from torch import nn
from torch.utils.data import DataLoader
import pandas as pd
from torchmetrics import Accuracy, ConfusionMatrix, ROC
from torchinfo import summary
from sklearn.metrics import classification_report
from typing import *
from tqdm.auto import tqdm
import random
import matplotlib.pyplot as plt
import seaborn as sns

# Define Main Function
def main():
    ...


# Define Additional Functions
# A PyTorch Wrapper class for Classification models.
class CModel:
    """
    Class to wrap PyTorch models for classification tasks.
    Args:
        model: PyTorch model to be wrapped.
        num_classes: Number of classes for classification. If None, it will be inferred from the model.
        m_device: Device to run the model on.
        input_size: Input size of the model. Required for model summary. Default to [1, 1, 28, 28] if not set.
    """
    def __init__(self, model: nn,
                 num_classes: int | None = None,
                 m_device: torch.device | str | None = None,
                 input_size: Optional[torch.Size | List[int] | Tuple[int]] = None):
        """
        Initialize the CModel class.
        :param model: PyTorch model to be wrapped.
        :param num_classes: Number of classes for classification. If None, it will be inferred from the model.
        :param m_device: Device to run the model on. If None, it will use the "cuda" if available.
        :param input_size: Input size of the model. Required for model summary. Default to [1, 1, 28, 28] if not set.
        """

        # Check device and set it.
        if not m_device:
            self.m_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            if not isinstance(m_device, torch.device):
                # Check for valid device strings and convert to torch.device.
                if m_device in ["cuda", "cpu", "xpu", "mps", "xla", "meta"]:
                    m_device = torch.device(m_device)
                else:
                    raise TypeError(
                        "The provided device must be an instance of torch.device or a valid string alternative.")
            else:
                self.m_device = m_device

        with torch.device(self.m_device):
            # Check if the model is an instance of nn.Module.
            if not isinstance(model, nn.Module):
                raise TypeError("The provided model must be an instance of torch.nn.Module.")

            # Check the number of classes. Set default to binary classification if not provided.
            if not num_classes:
                num_classes = 2
                print("Defaulting to Binary Classification as \"num_classes\" was not provided.")

            # Check if the input size is valid and set to default if not provided.
            if input_size and (isinstance(input_size, torch.Size) or
                               isinstance(input_size, list) or
                               isinstance(input_size, tuple)):
                self._input_size = input_size
            else:
                print("Invalid input size provided.")
                self._input_size = torch.Size([1, 1, 28, 28]) # Default input size for MNIST

            try:
                if num_classes < 2:
                    # If the number of classes is less than 2, it's a regression problem.
                    # Raise an error to let user know.
                    raise TypeError("Invalid number of classes for Classification model. "
                                     "Choose RModel class for Regression problems.")
                elif num_classes == 2:
                    # If the number of classes is 2, it's a binary classification problem.
                    # Set loss and accuracy for Binary Classification.
                    self._loss_fn = nn.BCEWithLogitsLoss()
                    self._acc_fn = Accuracy(task="binary")
                    self._num_classes = num_classes
                else:
                    # If the number of classes in more than 2, it's a multiclass problem.
                    # Set loss and accuracy for Multiclass Classification.
                    self._loss_fn = nn.CrossEntropyLoss()
                    self._acc_fn = Accuracy(task="multiclass", num_classes=num_classes)
                    self._num_classes = num_classes
            except ValueError:
                print(f"Invalid value for \"num_classes\". Expected <int>, got <{type(num_classes)}>")
            except NameError:
                print(f"One or more modules required for the class may be missing."
                      f" Please check the requirements and installed packages.")

            try:
                self._model = model.to(m_device)
                self._optimizer = torch.optim.Adam(self._model.parameters(), lr=0.001)
                self._tracker = pd.DataFrame(columns=["epoch", "train_accuracy",
                                                      "val_accuracy", "train_loss", "val_loss"])
            except NameError:
                print(f"One or more modules required for the class may be missing. "
                      f"Please check the requirements and installed packages.")

            # Data Variables
            self._complete_dataset = None
            self._train_dataset = None
            self._test_dataset = None
            self._train_dl = None
            self._test_dl = None
            self._X_train = None
            self._y_train = None
            self._X_test = None
            self._y_test = None

            # Variable to store list of class names.
            self._class_names = None

            # Predicted and target data variables
            self._predicted_logits = None
            self._predictions = None
            self._targets = None

    def load_data(self,
                  train_data: torch.utils.data.Dataset | torch.utils.data.DataLoader,
                  test_data: torch.utils.data.Dataset | torch.utils.data.DataLoader,
                  class_names: list[str] = None,
                  batch_size=32,
                  shuffle=True,
                  num_workers=0):
        """
        Create PyTorch DataLoaders for train and test sets of both features and targets.

        Parameters:
        - train_data: PyTorch Dataset/DataLoader. Train data to be used as the input dataset.
        - test_data: PyTorch Dataset/DataLoader. Test data to be used as the target dataset.
        - batch_size (int): Batch size for the DataLoader. Default is 32.
        - shuffle (bool): Whether to shuffle the batches in each epoch. Default is True.
        - num_workers (int): Number of subprocesses to use for data loading. Default is 0.
        """

        if not self._class_names and class_names:
            self._class_names = class_names

        # TODO: Change to generic function that determines what sort of data is provided.
        with torch.device(self.m_device):
            # If train_data is not a DataLoader, wrap it in one.
            if not isinstance(train_data, DataLoader):
                if self._class_names is None:
                    # If class_names is present, infer from the dataset.
                    self._class_names = train_data.classes
                if class_names:
                    # If class_names is present, check for consistency with the dataset.
                    assert len(class_names) == len(train_data.classes)
                self._train_dataset = train_data
                self._train_dl = DataLoader(
                    dataset=train_data,
                    batch_size=batch_size,
                    shuffle=shuffle,
                    num_workers=num_workers,
                    generator=torch.Generator(device=self.m_device)
                )
            else:
                self._train_dl = train_data

            # If target_data is not a DataLoader, wrap it in one.
            if not isinstance(test_data, DataLoader):
                self._test_dataset = test_data
                self._test_dl = DataLoader(
                    dataset=test_data,
                    batch_size=batch_size,
                    shuffle=False,
                    num_workers=num_workers,
                    generator=torch.Generator(device=self.m_device)
                )
            else:
                self._test_dl = test_data

    def _load_train_test_data(self,
                              train_data: torch.utils.data.Dataset | torch.utils.data.DataLoader,
                              test_data: torch.utils.data.Dataset | torch.utils.data.DataLoader,
                              class_names: list[str] = None,
                              batch_size=32,
                              shuffle=True,
                              num_workers=0):
        # TODO: Make this the function that handles train test data.
        ...

    def _load_feature_target_data(self):
        # TODO: Make this the function that handles X-y split data. Add train test split functionality.
        ...

    @property
    def loss_fn(self):
        return self._loss_fn

    @loss_fn.setter
    def loss_fn(self, loss_fn):
        with torch.device(self.m_device):
            if not isinstance(loss_fn, nn.Module):
                raise TypeError("The provided loss function must be an instance of a valid PyTorch loss module.")
            self._loss_fn = loss_fn

    @property
    def optimizer(self):
        return self._optimizer

    @optimizer.setter
    def optimizer(self, optimizer):
        with torch.device(self.m_device):
            if not hasattr(torch.optim, optimizer.__class__.__name__):
                raise ValueError(f"Invalid optimizer: {optimizer.__class__.__name__}")
            self._optimizer = optimizer

    @property
    def num_classes(self) -> int:
        return self._num_classes

    @property
    def tracker(self) -> pd.DataFrame:
        return self._tracker

    @property
    def model(self) -> nn.Module:
        return self._model

    @property
    def device(self) -> torch.device:
        return self.m_device

    @property
    def train_dataset(self) -> torch.utils.data.Dataset:
        self._dataset_check()
        return self._train_dataset

    @property
    def test_dataset(self) -> torch.utils.data.Dataset:
        self._dataset_check()
        return self._test_dataset

    @property
    def train_dl(self):
        self._dataloader_check()
        return self._train_dl

    @property
    def test_dl(self):
        self._dataloader_check()
        return self._test_dl

    @property
    def class_names(self) -> List[str]:
        if self._class_names:
            return self._class_names
        else:
            raise NameError("Class names have not been set.")

    @property
    def predicted_logits(self) -> torch.Tensor:
        if isinstance(self._predicted_logits, torch.Tensor):
            return self._predicted_logits
        else:
            raise ValueError("Run predict() to calculate predicted logits before accessing them.")

    @property
    def predictions(self) -> torch.Tensor:
        if isinstance(self._predictions, torch.Tensor):
            return self._predictions
        else:
            raise ValueError("Run predict() to calculate predictions and targets before accessing them.")

    @property
    def targets(self) -> torch.Tensor:
        if isinstance(self._targets, torch.Tensor):
            return self._targets
        else:
            raise ValueError("Run predict() to calculate predictions and targets before accessing them.")

    def fit(self, epochs: int = 100):
        """
        Fits the model to the training data for a specified number of epochs.
        :param epochs: Number of epochs to train the model. Default is 100.
        :return:
        """
        self._dataloader_check()

        with torch.device(self.m_device):
            if not self._tracker.empty:
                # If there are any previous epochs stored in the tracker, start from the last one.
                start = int(self._tracker.at[len(self._tracker) - 1, "epoch"])
            else:
                # Otherwise, start from the first epoch.
                start = 0

            for epoch in tqdm(range(start, start + epochs)):
                # Training and testing steps are performed here.
                print(f"Epoch {epoch + 1} / {epochs}")
                train_loss, train_acc = self._train_step()
                test_loss, test_acc = self._test_step()

                # Store the results for each epoch in the tracker DataFrame.
                self._tracker.loc[len(self._tracker)] = {
                    'epoch': epoch + 1,
                    'train_loss': train_loss,
                    'train_accuracy': train_acc,
                    'val_loss': test_loss,
                    'val_accuracy': test_acc
                }

                # Print progress every 5% of the epochs or every 1 epoch if less than 20 epochs.
                # This helps in keeping track of the progress without overwhelming the console.
                if epoch % round(epochs/(20 if epochs >= 20 else epochs)) == 0:
                    print(f"Epoch {epoch + 1} / {epochs}: Train Loss: {train_loss:.4f}, Train Accuracy: "
                          f"{train_acc:.4f}, Val Loss: {test_loss}, Val Accuracy: {test_acc:.4f}")

    def _train_step(self):
        """
        Perform a single training step.
        :return:
        """
        with torch.device(self.m_device):
            # Set the model to training mode
            self._model.train()

            # Initialize variables to accumulate the training loss and accuracy over the batch.
            train_loss, train_acc = 0, 0

            # Iterate over the training data loader in batches.
            for batch, (X, y) in enumerate(tqdm(self._train_dl)):
                X = torch.Tensor(X).to(self.m_device)
                y = torch.Tensor(y).to(self.m_device)

                # Calculate the loss and accuracy of the current batch.
                loss, acc = self.calculate_loss_acc(X, y)

                # Backpropagate the loss and update the model parameters.
                loss.backward()

                # Update the optimizer's parameters.
                self._optimizer.step()

                # Zero the gradients to prevent accumulation of gradients from previous batch.
                self._optimizer.zero_grad()

                # Accumulate the training loss and accuracy over the entire dataset.
                train_loss += loss.item()
                train_acc += acc.item()

            # Calculate the average training loss and accuracy over the entire dataset.
            train_loss /= len(self._train_dl)
            train_acc /= len(self._train_dl)

            # Return the training loss and accuracy for each epoch.
            return train_loss, train_acc

    def _test_step(self):
        """
        Test the model on the validation dataset.
        :return:
        """
        with torch.device(self.m_device):
            # Initialize variables to accumulate the loss and accuracy over the validation dataset.
            test_loss, test_acc = 0, 0

            # Set the model to evaluation mode and disable gradient calculation.
            self._model.eval()
            with torch.inference_mode():
                for batch, (X, y) in enumerate(tqdm(self._test_dl)):
                    X = torch.Tensor(X).to(self.m_device)
                    y = torch.Tensor(y).to(self.m_device)

                    # Calculate the loss and accuracy for the current batch.
                    loss, acc = self.calculate_loss_acc(X, y)

                    # Accumulate the loss and accuracy over the validation dataset.
                    test_loss += loss.item()
                    test_acc += acc.item()

            # Calculate the average loss and accuracy over the validation dataset
            test_loss /= len(self._test_dl)
            test_acc /= len(self._test_dl)

            # Return the average loss and accuracy over the validation dataset
            return test_loss, test_acc

    # TODO: Early Stopping Mechanism

    def predict(self, X: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Predicts the labels for a given dataset
        :param X: The input data to predict on. If not provided the model will predict on the test set.
        :return: The predicted labels for the input data
        """
        self._dataloader_check()

        with torch.device(self.m_device):
            # Clear the previous predictions and targets.
            self._predictions = None
            self._targets = None

            if not X:
                # Use the test dataset if no input data is provided.
                X = self._test_dl

            # Initialize lists to store the predicted logits, predictions, and targets.
            pred_logits = []
            predictions = []
            targets = []

            # Set the model to evaluation mode and turn off gradient tracking.
            self._model.eval()
            with torch.inference_mode():
                for batch, (X, y) in enumerate(tqdm(X)):
                    X = torch.Tensor(X).to(self.m_device)
                    y = torch.Tensor(y).to(self.m_device)

                    y_pred = self._model(X)

                    # Store the predicted logits, predictions, and targets for later use.
                    pred_logits.append(y_pred)
                    predictions.append(y_pred.argmax(dim=1))
                    targets.append(y)

            # Concatenate all predicted logits, predictions and targets into a single tensor
            self._predicted_logits = torch.cat(pred_logits)
            self._predictions = torch.cat(predictions)
            self._targets = torch.cat(targets)

            # Return the concatenated predictions.
            return torch.cat(predictions)

    def _dataset_check(self):
        """
        Check if both train_dataset and test_dataset are not None. Raise an error if either is missing.
        :return:
        """
        if not self._train_dataset or not self._test_dataset:
            raise ValueError(f"A Dataset is missing. Use load_data() to load data into the class.")

    def _dataloader_check(self):
        """
        Check if both train_dl and test_dl are not None. Raise an error if either is missing.
        :return:
        """
        if not self._train_dl or not self._test_dl:
            raise ValueError(f"A DataLoader is missing. Use load_data() to load data into the class.")

    def calculate_loss_acc(self, features, target):
        """
        Calculate the loss and accuracy for a given set of features and target values.
        :param features: features to perform the predictions on.
        :param target: target values to compare the predictions against.
        :returns: a tuple containing the loss and accuracy.
        """
        with torch.device(self.m_device):
            predictions = self._model(features)
            loss = self.calculate_loss(predictions, target)
            acc = self.calculate_acc(predictions, target)
            return loss, acc

    def calculate_loss(self, predictions, target):
        """
        Calculate the loss using the model's loss
        :param predictions: predictions from the model.
        :param target: target values.
        :returns: loss value.
        """
        with torch.device(self.m_device):
            return self._loss_fn(predictions, target)

    def calculate_acc(self, predictions, target):
        """
        Calculate the accuracy of predictions.
        :param predictions: predictions from the model.
        :param target: target values.
        :returns: accuracy value.
        """
        with torch.device(self.m_device):
            return self._acc_fn(predictions.argmax(dim=1), target)


    def clear_tracker(self):
        """
        Clears the tracker DataFrame.
        :return:
        """
        self._tracker.drop(self._tracker.index, inplace=True)

    def save_model(self, path):
        """
        Saves the model's state dictionary to a file.
        :param path: path to save the model.
        :return:
        """
        torch.save(self._model.state_dict(), path)

    def load_model(self, path):
        """
        Loads the model's state dictionary from a file.
        :param path: path to load the model.
        :return:
        """
        self._model.load_state_dict(torch.load(path))

    def summary(self):
        """
        Prints a summary of the model's architecture.
        Uses torchinfo library.
        :return:
        """
        if self._input_size:
            # Use the input size if provided. The printed summary will include the output size for each layer.
            print(summary(self._model, input_size=self._input_size))
        else:
            # If input size is not provided, use the model's summary is printed without the output size.
            print(summary(self._model))


class CMetrics:
    """
    Class to calculate and store metrics for CModel Classification models.
    """
    def __init__(self, model:CModel):
        """
        Initializes the CMetrics object with a CModel instance.
        :param model: CModel instance.
        """
        # Ensure the model is on the correct device before using it.
        self.m_device = model.m_device

        # Convert frequently used CModel attributes to CMetrics attributes.
        with torch.device(self.m_device):
            self._model_wrap = model
            self._tracker = self._model_wrap.tracker

            try:
                # Attempt to access targets and predictions directly from the model wrap.
                self._targets = self._model_wrap.targets
                self._predictions = self._model_wrap.predictions
            except ValueError:
                # If targets and predictions are not directly accessible, call the predict method to generate them.
                self._model_wrap.predict()
                self._targets = self._model_wrap.targets
                self._predictions = self._model_wrap.predictions

        try:
            # Attempt to access class names directly from the model wrap.
            self._class_names = self._model_wrap.class_names
        except NameError:
            # If class names are not directly accessible, set them to None and print a warning.
            self._class_names = None
            print("Class names not found. Please set them in the model class or "
                  "use the \"class_names\" property to set them.")
        self._conf_matrix = None

    @property
    def class_names(self):
        return self._class_names

    @class_names.setter
    def class_names(self, names: list[str]):
        self._class_names = names

    def loss(self):
        """
        Calculate and print the loss value.
        :return:
        """
        with torch.device(self.m_device):
            print(f"Loss: {self._model_wrap.calculate_loss(self._predictions, self._targets)}")

    def accuracy(self):
        """
        Calculate and print the accuracy value.
        :return:
        """
        with torch.device(self.m_device):
            print(f"Accuracy: {self._model_wrap.calculate_acc(self._predictions, self._targets)}")

    def plot_loss_vs_accuracy(self):
        """
        Plot a graph that shows the change of loss and accuracy with each epoch.
        :return:
        """
        with torch.device(self.m_device):
            fig, ax1 = plt.subplots(figsize=(13, 6))

            ax1.plot(self._tracker['epoch'],
                     self._tracker['train_accuracy'],
                     label='Training Accuracy',
                     color='limegreen',
                     alpha=0.5, lw=6)
            ax1.plot(self._tracker['epoch'],
                     self._tracker['val_accuracy'],
                     label=' Validation Accuracy',
                     color='green')

            ax2 = ax1.twinx()

            ax2.plot(self._tracker['epoch'],
                     self._tracker['train_loss'],
                     label='Training Loss', color='coral',
                     alpha=0.5, lw=6)
            ax2.plot(self._tracker['epoch'],
                     self._tracker['val_loss'],
                     label='Validation Loss', color='red')

            plt.title('Model Performance Over Epochs')
            ax1.set_xlabel('Epoch')
            ax1.set_ylabel('Accuracy')
            ax2.set_ylabel('Loss')

            epochs = int(self._tracker.at[len(self._tracker) - 1, "epoch"])

            plt.xlim(left = -epochs / 50, right = epochs + (epochs / 50))

            lines, labels = ax1.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax2.legend(lines + lines2, labels + labels2, loc="center right")

            plt.show()

    def classification_report(self, class_names=None):
        """
        Generate a classification report for the model's predictions.
        Uses scikit-learn's classification_report function.
        :param class_names: Optional list of class names to use in the report.
        If not provided, uses the class names from the model or runs the function without class names.
        :return:
        """
        if not class_names:
            if not self._class_names:
                print(classification_report(self._targets.cpu().numpy(),
                                            self._predictions.cpu().numpy()))
            else:
                print(classification_report(self._targets.cpu().numpy(),
                                            self._predictions.cpu().numpy(),
                                            target_names=self._class_names))
        else:
            print(classification_report(self._targets.cpu().numpy(),
                                        self._predictions.cpu().numpy(),
                                        target_names=class_names))

    def _calculate_confusion_matrix(self):
        """
        Calculate and store the confusion matrix for the model's predictions.
        :return:
        """
        with torch.device(self.m_device):
            if self._model_wrap.num_classes == 2:
                conf_matrix = ConfusionMatrix("binary")
            else:
                conf_matrix = ConfusionMatrix("multiclass", num_classes=self._model_wrap.num_classes)

            self._conf_matrix = conf_matrix(self._predictions, self._targets)

    def confusion_matrix(self):
        """
        Generate a confusion matrix for the model's predictions.
        :return:
        """
        with torch.device(self.m_device):
            # If the confusion matrix is not already present, and a tensor, calculate it first.
            if not isinstance(self._conf_matrix, torch.Tensor):
                self._calculate_confusion_matrix()
            print(self._conf_matrix)

    def plot_confusion_matrix(self):
        """
        Plot a confusion matrix for the model's predictions.
        Uses Seaborn for plotting.
        :return:
        """
        with torch.device(self.m_device):
            # If the confusion matrix is not already present, and a tensor, calculate it first.
            if not isinstance(self._conf_matrix, torch.Tensor):
                self._calculate_confusion_matrix()

            # Convert to NumPy for plotting
            confusion_matrix_np = self._conf_matrix.cpu().numpy()

            plt.figure(figsize=(8, 6))

            # Use seaborn to create a heatmap of the confusion matrix.
            sns.heatmap(confusion_matrix_np, annot=True, fmt='d', cmap='Blues', cbar=False,
                        xticklabels=self._class_names,
                        yticklabels=self._class_names)
            plt.xlabel('Predicted Label')
            plt.ylabel('True Label')
            plt.title('Confusion Matrix')
            plt.show()

    def plot_roc(self):
        """
        Compute and plot the Receiver Operating Characteristic (ROC) Curve for the model.
        Uses torchmetrics for ROC computation.
        :return:
        """
        with torch.device(self.m_device):
            # Initialize ROC metric based on the number of classes.
            if self._model_wrap.num_classes == 2:
                roc = ROC(task="binary")
            else:
                roc = ROC(task="multiclass", num_classes=self._model_wrap.num_classes)

            fpr, tpr, thresholds = roc(self._model_wrap.predicted_logits, self._targets)

            plt.figure(figsize = (8, 8))
            if self._class_names:
                # Plot ROC curve with class names if available.
                fig, ax = roc.plot(score=True, labels = self._class_names)
            else:
                fig, ax = roc.plot(score=True)
            plt.title('ROC Curve')
            plt.show()

    def image_prediction_vs_truth(self, random_state=None, rows: int = 3, cols: int = 3):
        """
        Display a grid of images with their predicted labels and true labels.
        :param random_state: Random state for reproducibility. Default is 42.
        :param rows: Number of rows in the grid. Default is 3.
        :param cols: Number of columns in the grid. Default is 3.
        :return:
        """

        if random_state:
            random.seed(random_state)

        # Pick random image indices
        rand_indices = random.choices(range(len(self._model_wrap.test_dataset)), k=rows * cols)

        # Create a figure using rows and columns. This ensures the figure scales with number of images.
        plt.figure(figsize=(cols * 3, rows * 3))
        for i, idx in enumerate(rand_indices):
            # Get the image and truth label from the dataset
            img, lbl = self._model_wrap.test_dataset[idx]

            # Get the predicted label from the model's predictions
            pred = self._model_wrap.predictions[idx]

            # Plot the image with predicted and true labels as a subplot according to the row and column index.
            plt.subplot(rows, cols, i + 1)
            plt.imshow(img.squeeze(), cmap="gray")
            title = f"Truth: {self.class_names[lbl]}, Predicted: {self.class_names[pred]}"

            if lbl == pred:
                # If the predicted label matches the true label, color the title green.
                plt.title(title, fontsize=9, c="green")
            else:
                # If the predicted label does not match the true label, color the title red.
                plt.title(title, fontsize=9, c="red")
            plt.axis(False)

        plt.show()


# Main Function
if __name__ == '__main__':
    main()
