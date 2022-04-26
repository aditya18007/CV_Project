import torch 
from tqdm import tqdm 
from sklearn.metrics import (
    accuracy_score, 
    classification_report,
    f1_score,
)

import numpy as np 

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
        
        if self.loss_function is None :
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
        self.loss_function = loss_function

    def set_data(self, train_loader, val_loader):
        self.train_loader = train_loader 
        self.val_loader = val_loader
    
    def train(self, debug_model_layers=False):
        
        if not self.all_param_init():
            exit(-1)

        for epoch in range(self.epochs):
            
            self.model.train()
            train_loss = 0
            train_accuracy = 0
            for x,y in tqdm(self.train_loader):
                
                x,y_true = self.input_transformer.transform(x,y)
                y_hat, y_pred = self.model(x)
                if debug_model_layers:
                    return
                batch_accuracy = accuracy_score(y_pred, y_true)
                batch_loss = self.loss_fn(y_hat, y_true) + self.model.l2_norm(1e-4)

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
                    y_hat, y_pred = self.model(x)
                    
                    batch_accuracy = accuracy_score(y_pred, y_true)
                    batch_loss = self.loss_fn(y_hat, y_true)

                    val_accuracy += batch_accuracy.item()
                    val_loss += batch_loss.item()
                
                avg_val_loss = val_loss/len(self.val_loader)
                self.Validation_results['Losses'].append(avg_val_loss)

                avg_val_acccuracy = val_accuracy/len(self.val_loader)
                self.Validation_results['Accuracies'].append(avg_val_acccuracy)
            
            print("\n")
            print(f"For epoch = {epoch}")
            print("Training Loss = {avg_train_loss} | Training Accuracy = {avg_train_accuracy}")
            print("Validation Loss = {avg_val_loss}|Validation Accuracy = {avg_val_acccuracy}")
            print("\n")

    def test(self, test_loader):
        self.model.eval()
        actual = None 
        predicted = None 

        with torch.no_grad():           
            
            for x,y in tqdm(self.train_loader):
                #There is only one batch
                    
                x,y_true = self.input_transformer.transform(x,y)
                y_hat, y_pred = self.model(x)
                
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
            
            print(f"Accuracy Score = {accuracy_score(predicted, actual)}")
            print(f"Macro F1 score Score = {f1_score(predicted,actual,average='macro')}")
            print(f"Micro F1 score Score = {f1_score(predicted,actual,average='micro')}")
            print("Classification Report\n\n")
            print(classification_report(predicted, actual))