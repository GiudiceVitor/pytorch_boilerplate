import torch
from source.callbacks import CallbackHandler
import torch.backends.cudnn as cudnn
 
from torch.utils.data import DataLoader
 
 
class Project():
    '''
    A keras-like class to control your project.
    The most high level class of the project.
    Everything else should be changed in the dependencies.
    This class creates the training loop and handles the callbacks.
    Use callbacks to change the training loop, like changing the learning rate or saving the model.
    '''
 
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
 
    def compile(self, model):
        '''
        Compiles the model by casting it to the correct device
 
        Args:
            model (torch.nn.Module): your neural network model
        '''
        self.model = model.to(self.device)
 
    def predict(self, inputs):
        '''
        Method to predict the output of the model for a single input batch.
 
        Args:
            inputs (_type_): your input data
 
        Returns:
            (_type_): output of the model
        '''
 
        self.model.eval()
        with torch.no_grad():
            inputs = inputs.float().to(self.device)
            outputs = self.model(inputs)
        return outputs
   
    def evaluate(self, val_loader):
        '''
        Method for model evaluation of validation loss and metrics
 
        Args:
            val_loader (Dataloader): data loader for the test set
 
        Returns:
            float: validation loss
            list of floats: list of validation metrics
        '''
 
        val_loss = 0
        metric_loss = torch.zeros(len(self.metrics)).to(self.device)
        self.model.eval()
 
        with torch.no_grad():
            for (inputs, targets) in val_loader:
 
                # forward pass
                inputs = inputs.to(self.device).float()
                targets = targets.to(self.device)
                outputs = self.model(inputs)
               
                # calculate loss
                val_loss += self.loss_function(outputs, targets)
 
                # calculate metrics
                for i, metric in enumerate(self.metrics):
                    metric_loss[i] += metric(outputs, targets)
               
            # average loss and metrics
            for i, metric in enumerate(self.metrics):
                metric_loss[i] /= len(val_loader)
            val_loss /= len(val_loader)
 
        return val_loss, metric_loss
 
    def fit(self,
            train_dataset,
            val_dataset=None,
            optimizer=None,
            epochs=5,
            batch_size=1,
            shuffle=True,
            learning_rate=0.001,
            callbacks=[],
            loss = None,
            metrics=[],
            ):
        '''
        Method to train the model.
        Use this to change the training loop if needed.
        Notice that any change that is not structural should be done using callbacks.
        For example, if you want to change the learning rate during training, you should use a callback.
        Otherwise, if you are training for example a GAN, you should change the training loop.
 
        Args:
            train_dataset (Dataset): dataset for the training set.
            val_dataset (Dataset optional): dataset for the evaluation set. Defaults to None. If None, the training set is used.
            optimizer (optimizer, optional): optimizer used to train the model. Defaults to None.
            epochs (int, optional): number of epochs to train the model for. Defaults to 10.
            batch_size (int, optional): batch size for training. Defaults to 1.
            shuffle (bool, optional): whether to shuffle or not the data before training and testing. Defaults to True.
            learning_rate (float, optional): learning rate. Defaults to 0.001.
            callbacks (list of Callbacks, optional): callbacks to run during trainig. Defaults to [].
            loss (loss function, optional): loss function to use for training. Defaults to None. Not actually optional.
            metrics (list of metrics, optional): metrics to use for evaluation. Defaults to [].
       
        Returns:
            dict: dictionary containing history of the training
        '''
 
        self.train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, pin_memory=True)
        if (val_dataset != None):
            self.val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=shuffle, pin_memory=True)
        else:
            self.val_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, pin_memory=True)
       
        self.epochs = epochs
        self.batch_size = batch_size
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.callbacks = callbacks
        self.loss_function = loss
        self.metrics = metrics
 
        self.optimizer.param_groups[0]['lr'] = self.learning_rate
 
        self.train_loss = []
        self.val_loss = []
        self.metrics_loss = []
        self.epoch_loss = torch.tensor(0.0).to(self.device)
        self.stop_training = False
 
        callback_handler = CallbackHandler(callbacks, learner=self)
        callback_handler.on_train_begin()
 
        for self.epoch in range(1, epochs+1):
            callback_handler.on_epoch_begin()
            self.model.train()
           
            for self.batch, (inputs, targets) in enumerate(self.train_loader):
                callback_handler.on_batch_begin()
 
                # forward pass
                inputs = inputs.to(self.device).float()
                targets = targets.to(self.device)
                outputs = self.model(inputs)
 
                # calculate loss
                self.batch_loss = self.loss_function(outputs, targets)
 
                # backward pass
                self.optimizer.zero_grad()
                self.batch_loss.backward()
                self.optimizer.step()
                self.epoch_loss += self.batch_loss
               
                callback_handler.on_batch_end()
 
            val, metric = self.evaluate(self.val_loader)
           
            self.train_loss.append((self.epoch_loss).item()/len(self.train_loader))
            self.val_loss.append(val.item())
            self.metrics_loss.append(metric.cpu().numpy().tolist())
 
            self.epoch_loss = torch.tensor(0.0).to(self.device)
           
            callback_handler.on_epoch_end()
 
            if (self.stop_training): break
 
        callback_handler.on_train_end()
        self.history = {'train_loss': self.train_loss,
                        'val_loss': self.val_loss,
                        'metrics': self.metrics_loss}
 
        return self.history