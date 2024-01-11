import torch
from torch.nn import DataParallel
import time
import os
 
class Callback():
    '''
    Base class for callbacks to extend.
    '''
 
    def __init__(self): pass
    def on_train_begin(self, learner): pass
    def on_train_end(self, learner): pass
    def on_epoch_begin(self, learner): pass
    def on_epoch_end(self, learner): pass
    def on_batch_begin(self, learner): pass
    def on_batch_end(self, learner): pass
    def on_loss_begin(self, learner): pass
    def on_loss_end(self, learner): pass
    def on_step_begin(self, learner): pass
    def on_step_end(self, learner): pass
 
 
class CallbackHandler():
    def __init__(self, callbacks, learner):
       
        '''
        A class to handle callbacks.
 
        Args:
            callbacks (list): list of callbacks to handle
            learner (Project): learner to pass to callbacks to access and modify its attributes
        '''
        self.callbacks = callbacks
        self.learner = learner
 
    def on_train_begin(self):
        for callback in self.callbacks:
            callback.on_train_begin(self.learner)
 
    def on_train_end(self):
        for callback in self.callbacks:
            callback.on_train_end(self.learner)
 
    def on_epoch_begin(self):
        for callback in self.callbacks:
            callback.on_epoch_begin(self.learner)
 
    def on_epoch_end(self):
        for callback in self.callbacks:
            callback.on_epoch_end(self.learner)
 
    def on_batch_begin(self):
        for callback in self.callbacks:
            callback.on_batch_begin(self.learner)
 
    def on_batch_end(self):
        for callback in self.callbacks:
            callback.on_batch_end(self.learner)
 
 
class Logger(Callback):
    def __init__(self):
        '''
        A callback to print metrics.
        '''
        self.elapsed_time = 0
        self.elapsed_epoch_time = 0
        super().__init__()
 
    def on_train_begin(self, learner):
        gpus = int(os.environ.get('WORLD_SIZE', 1))
        print(f'Running on {gpus} GPUs', end = '. ')
        print(f'Training on {len(learner.train_loader.dataset)} samples', end = ' -- ')
        print(f'Validating on {len(learner.val_loader.dataset)} samples', end = '\n\n')
        self.initial_time = time.time()

    def on_epoch_begin(self, learner):
        self.initial_epoch_time = time.time()
 
    def on_batch_end(self, learner):
        self.elapsed_time = time.time() - self.initial_time
        self.elapsed_epoch_time = time.time() - self.initial_epoch_time
        print(f'Epoch {learner.epoch}/{learner.epochs} >>> Batch {learner.batch+1}/{len(learner.train_loader)} -- Loss: {learner.epoch_loss/(learner.batch+1):.5f}, Elapsed Time: {self.elapsed_epoch_time:.1f}', end='\r')
   
    def on_epoch_end(self, learner):
        self.elapsed_time = time.time() - self.initial_time
        self.elapsed_epoch_time = time.time() - self.initial_epoch_time
        print(f'Epoch {learner.epoch}/{learner.epochs} >>> Batch {learner.batch+1}/{len(learner.train_loader)} -- Loss: {learner.train_loss[-1]:.5f},  Val Loss: {learner.val_loss[-1]:.5f},  Elapsed Time: {self.elapsed_epoch_time:.1f}')
   
    def on_train_end(self, learner):
        print(f'\nFinished training. Total elapsed time: {self.elapsed_time:.1f} seconds')
 
 
class ReduceLROnPlateau(Callback):
    def __init__(self, patience=1, min_lr=0, verbose=1, factor=0.1, min_delta=1e-4):
        '''
        A callback to reduce the learning rate when the validation loss has stopped improving.
 
        Args:
            patience (int, optional): number of epochs to wait to see improvement. Defaults to 1.
            min_lr (int, optional): mininum learning rate to reduce to. Defaults to 0.
            verbose (int, optional): whether or not to see visual feedback of the callback. Defaults to 1.
                                     verbose == 0: don't show anything
                                     verbose == 1: show a message when the learning rate is reduced
                                     verbose == 2: for debugging purposes
            factor (float, optional): factor to multiply the learning rate by. Defaults to 0.1.
            min_delta (_type_, optional): minimum difference between best metric and current metric to consider improvement. Defaults to 1e-4.
        '''
 
        super().__init__()
        self.patience = patience
        self.min_lr = min_lr
        self.verbose = verbose
        self.factor = factor
        self.min_delta = min_delta
        self.best_loss = 1e10
        self.wait = 0
 
    def on_epoch_end(self, learner):
        if(self. verbose == 2):
            print(f'loss: {learner.val_loss[-1]}')
            print(f'best_loss: {self.best_loss}')
 
        if self.best_loss - learner.val_loss[-1] < self.min_delta:
            self.wait += 1
        else:
            self.wait = 0
            if (self.verbose == 2): print(f'\n[ReduceLROnPlateau] wait = 0 because diff is {self.best_loss - learner.val_loss[-1]} and min_delta is {self.min_delta}')
            self.best_loss = learner.val_loss[-1]
           
        if self.wait >= self.patience:
            new_lr = max(learner.optimizer.param_groups[0]['lr'] * self.factor, self.min_lr)
            learner.optimizer.param_groups[0]['lr'] = new_lr
            learner.learning_rate = new_lr
            self.wait = 0
            if (self.verbose == 1) or (self.verbose == 2):
                print('\nReducing learning rate to {}\n'.format(new_lr))
        else:
            if (self.verbose == 2): print(f'\n[ReduceLROnPlateau] not reducing learning rate because wait is {self.wait} and patience is {self.patience}')
 
 
class ModelCheckpoint(Callback):
    def __init__(self, save_best_only=True, save_path=os.getcwd()):
        '''
        A callback to save the model.
 
        Args:
            save_best_only (bool, optional): whether or not to save only the best model. Defaults to True.
            save_path (String optional): path to save the model. Defaults to the current directory.
        '''
 
        super().__init__()
        self.save_best_only = save_best_only
        self.save_path = save_path
        self.best_loss = 1e10
 
    def on_epoch_end(self, learner):
        if self.save_best_only:
            if not learner.val_loss[-1] < self.best_loss:
                return
            self.best_loss = learner.val_loss[-1]
        torch.save(learner.model.state_dict(), self.save_path + '/model.pth')
 
 
class EarlyStopping(Callback):
    def __init__(self, patience=1, min_delta=1e-4, verbose=1):
        '''
        A callback to stop training when a metric has stopped improving.
 
        Args:
            patience (int, optional): number of epochs to wait to see improvement. Defaults to 1.
            min_delta (int, optional): minimum difference between best metric and current metric to consider improvement. Defaults to 1e-4.
            verbose (int, optional): whether or not to see visual feedback of the callback. Defaults to 1.
                                     verbose == 0: don't show anything;
                                     verbose == 1: show a message when the learning rate is reduced;
                                     verbose == 2: for debugging purposes.
        '''
 
        super().__init__()
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.wait = 0
        self.best_loss = 1e10
        self.best_model = None
 
    def on_epoch_end(self, learner):
        if self.best_loss - learner.val_loss[-1] < self.min_delta:
            self.wait += 1
        else:
            self.wait = 0
            if (self.verbose == 2): print(f'\n[EarlyStopping] wait = 0 because diff is {self.best_loss - learner.val_loss[-1]} and min_delta is {self.min_delta}\n')
            self.best_loss = learner.val_loss[-1]
 
        if self.wait >= self.patience:
            learner.stop_training = True
            if (self.verbose == 1) or (self.verbose == 2):
                learner.stop_training = True
                print('\nEarly stopping\n')
        else:
            if (self.verbose == 2): print(f'\n[EarlyStopping] not stopping training because wait is {self.wait} and patience is {self.patience}\n')
