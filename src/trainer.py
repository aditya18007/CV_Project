import torch 
from tqdm import tqdm 

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
    
    def train(self):

        for epoch in range(self.epochs):
            self.model.train()
            train_loss = 0
            train_accuracy = 0
            for x,y in tqdm(self.train_loader):
                x = self.input_transformer(x)
                y_hat = self.model(x)
