import torch 
from tqdm import tqdm 
from sklearn.metrics import (
    accuracy_score, 
    classification_report,
    f1_score,
)

import numpy as np 
import matplotlib.pyplot as plt 

class Trainer:

    def __init__(self, epochs):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.epochs = epochs
        
        self.Train_results = {
            'Losses' : [],
            'Accuracies': []
        }
        self.Validation_results = {
            'Losses' : [],
            'Accuracies': []
        }

        self.model = None 
        self.input_transformer = None 
        self.optimizer = None 
        self.loss_function = None 
        self.train_loader = None 
        self.val_loader = None 
    
    def all_param_init(self):
        all_init = True 

        if self.model is None :
            print("Set model using instance method Trainer::set_model")
            all_init = False 
        
        if self.input_transformer is None :
            print("Set input_transformer using instance method Trainer::set_model")
            all_init = False 
        
        if self.optimizer is None :
            print("Set optimizer using instance method Trainer::set_optimizer")
            all_init = False 
        
        if self.loss_fn is None :
            print("Set loss_function using instance method Trainer::set_optimizer")
            all_init = False 

        if self.train_loader is None:
            print("Set train_loader using instance method Trainer::set_data")
            all_init = False

        if self.val_loader is None:
            print("Set val_loader using instance method Trainer::set_data")
            all_init = False

        return all_init

    def set_model(self, model, input_transformer):
        self.model = model  
        self.input_transformer = input_transformer

    def set_optimizer( self, optimizer, loss_function):
        self.optimizer = optimizer 
        self.loss_fn = loss_function

    def set_data(self, train_loader, val_loader):
        self.train_loader = train_loader 
        self.val_loader = val_loader
    
    def train(self, debug_model_layers=False, l2_r=1e-4):
        
        if not self.all_param_init():
            exit(-1)

        for epoch in range(self.epochs):
            
            self.model.train()
            train_loss = 0
            train_accuracy = 0
            for x,y in tqdm(self.train_loader):
                
                x,y_true = self.input_transformer.transform(x,y)
                y_hat, y_pred = self.model(*x)
                if debug_model_layers:
                    return
                batch_accuracy = accuracy_score(y_pred.cpu().detach().numpy(), y_true.cpu().detach().numpy())
                batch_loss = self.loss_fn(y_hat, y_true) + self.model.l2_norm(l2_r)

                train_loss += batch_loss.item()
                train_accuracy += batch_accuracy.item()
                batch_loss.backward()
                self.optimizer.step()
            
            avg_train_loss = train_loss / len(self.train_loader)
            self.Train_results['Losses'].append(avg_train_loss)

            avg_train_accuracy = train_accuracy / len(self.train_loader)
            self.Train_results['Accuracies'].append(avg_train_accuracy)

            self.model.eval()
            val_loss = 0
            val_accuracy = 0 
            with torch.no_grad():
                for x,y in tqdm(self.val_loader):
                
                    x,y_true = self.input_transformer.transform(x,y)
                    y_hat, y_pred = self.model(*x)
                    
                    batch_accuracy = accuracy_score(y_pred.cpu().detach().numpy(), y_true.cpu().detach().numpy())
                    batch_loss = self.loss_fn(y_hat, y_true)

                    val_accuracy += batch_accuracy.item()
                    val_loss += batch_loss.item()
                
                avg_val_loss = val_loss/len(self.val_loader)
                self.Validation_results['Losses'].append(avg_val_loss)

                avg_val_acccuracy = val_accuracy/len(self.val_loader)
                self.Validation_results['Accuracies'].append(avg_val_acccuracy)
            
            print("\n")
            print(f"For epoch = {epoch}")
            print(f"Training Loss = {avg_train_loss} | Training Accuracy = {avg_train_accuracy}")
            print(f"Validation Loss = {avg_val_loss}|Validation Accuracy = {avg_val_acccuracy}")
            print("\n")

    def plot(self):
        SMALL_SIZE = 8
        MEDIUM_SIZE = 10
        BIGGER_SIZE = 12

        plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
        plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
        plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
        plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
        plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
        plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
        plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
        Epochs = [i for i in range(self.epochs)]

        plt.figure(figsize=(20,20))
        plt.plot(Epochs,self.Train_results['Losses'],label = "Training Loss")
        plt.plot(Epochs,self.Validation_results['Losses'],label = "Validation Loss")
        plt.title("Loss vs Epoch curve")
        plt.xlabel("Epoch number")
        plt.ylabel("Loss")
        plt.legend()
        plt.show()

        plt.figure(figsize=(20,20))
        plt.plot(Epochs,self.Train_results['Accuracies'],label = "Training Accuracy")
        plt.plot(Epochs,self.Validation_results['Accuracies'],label = "Validation Accuracy")
        plt.title("Accuracy vs Epoch curve")
        plt.xlabel("Epochs number")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.show() 

    def test(self, test_loader, print_classification_report=False):
        self.model.eval()
        actual = None 
        predicted = None 

        with torch.no_grad():           
            
            for x,y in tqdm(self.train_loader):
                #There is only one batch
                    
                x,y_true = self.input_transformer.transform(x,y)
                y_hat, y_pred = self.model(*x)
                
                if actual is None:
                    actual = y_true.cpu().detach().numpy()
                else:
                    y_true = y_true.cpu().detach().numpy()
                    actual = np.concatenate( [actual, y_true] )
                
                if predicted is None:
                    predicted = y_pred.cpu().detach().numpy() 
                else:
                    y_pred = y_pred.cpu().detach().numpy() 
                    predicted = np.concatenate( [predicted, y_pred] )
            
            print(f"Accuracy Score = {accuracy_score(predicted, actual)*100}")
            print(f"Macro F1 score Score = {f1_score(predicted,actual,average='macro')*100}")
            print(f"Micro F1 score Score = {f1_score(predicted,actual,average='micro')*100}")
            if print_classification_report:
                print("Classification Report\n\n")
                print(classification_report(predicted, actual))