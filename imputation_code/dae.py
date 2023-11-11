import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.utils.data
import torch.optim as optim
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer



class Autoencoder(nn.Module):
    def __init__(self, dim , theta,dropout):
        super(Autoencoder, self).__init__()
        self.dim = dim

        self.drop_out = nn.Dropout(p=dropout)

        self.encoder = nn.Sequential(
            nn.Linear(dim+(theta*0), dim+(theta*1)),
            nn.Tanh(),
            nn.Linear(dim+(theta*1), dim+(theta*2)),
            nn.Tanh(),
            nn.Linear(dim+(theta*2), dim+(theta*3))
        )

        self.decoder = nn.Sequential(
            nn.Linear(dim+(theta*3), dim+(theta*2)),
            nn.Tanh(),
            nn.Linear(dim+(theta*2), dim+(theta*1)),
            nn.Tanh(),
            nn.Linear(dim+(theta*1), dim+(theta*0))
        )

    def forward(self, x):
        
        x = x.view(-1, self.dim)
        
        x_missed = self.drop_out(x)

        z = self.encoder(x_missed)
        out = self.decoder(z)

        out = out.view(-1, self.dim)

        return out



class DAE:

    def __init__(self,parameters: dict,
                 missing_values=np.nan):


        self.theta = parameters.get("theta",7)
        self.drop_out = parameters.get("dropout",0.5)
        self.batch_size = parameters.get("batch_size",64)
        self.epochs = parameters.get("epochs",500)
        self.lr = parameters.get("lr",0.01)

        self.model = None


        torch.manual_seed(0)


        self.missing_values = missing_values


    def _initial_imputation(self, X):


        self.dim = X.shape[1]
        

        X_filled = X.copy()

        
        self.initial_imputer_Mean = SimpleImputer(missing_values=self.missing_values,strategy='constant',fill_value=0)
        self.initial_imputer_Mean.fit(X)
        X_filled = self.initial_imputer_Mean.transform(X)

        return X_filled



    def fit_transform(self, X, y=None):
        """Fits the imputer on X and return the transformed X.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Input data, where "n_samples" is the number of samples and
            "n_features" is the number of features.

        y : ignored.

        Returns
        -------
        Xt : array-like, shape (n_samples, n_features)
            The imputed input data.
        """
        device = torch.device('cpu')

        self.initial_imputer_Mean = None


        data_m = 1-np.isnan(X)

        X_conc = self._initial_imputation(X)


        train_data = torch.from_numpy(X_conc).float()

        train_loader = torch.utils.data.DataLoader(dataset=train_data,batch_size=min(self.batch_size, X.shape[0]),shuffle=True)


        cost_list = []
        early_stop = False

        self.dim = X_conc.shape[1]

        self.model = Autoencoder(dim = self.dim,theta = self.theta,dropout=self.drop_out).to(device)
        self.loss = nn.MSELoss()
        self.optimizer = optim.SGD(self.model.parameters(), momentum=0.99, lr=self.lr, nesterov=True)


        for epoch in range(self.epochs):

            total_batch = len(train_data)//min(self.batch_size, X.shape[0])

            for i, batch_data in enumerate(train_loader):

                batch_data = batch_data.to(device)
                reconst_data = self.model(batch_data)

                cost = self.loss(reconst_data, batch_data)

                self.optimizer.zero_grad()
                cost.backward()
                self.optimizer.step()

                #if (i+1) % (total_batch//2) == 0:
                #   print('Epoch [%d/%d], lter [%d/%d], Loss: %.6f'%(epoch+1, self.epochs, i+1, total_batch, cost.item()))

                # early stopping rule 1 : MSE < 1e-06
                if cost.item() < 1e-06 :
                    early_stop = True
                    break

                cost_list.append(cost.item())

            if early_stop :
                break

        #Evaluate
        self.model.eval()
        filled_data = self.model(train_data.to(device))
        filled_data_train = filled_data.cpu().detach().numpy()

        return filled_data_train

    def transform(self, X):
        """Imputes all missing values in X.

        Note that this is stochastic, and that if random_state is not fixed,
        repeated calls, or permuted input, will yield different results.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input data to complete.

        Returns
        -------
        Xt : array-like, shape (n_samples, n_features)
             The imputed input data.
        """
        device = torch.device('cpu')
        #check_is_fitted(self)

        #1 if not missing , 0 if missing.
        data_m = 1-np.isnan(X)
        data_nm = np.isnan(X)-0
        #keeps nan
        #gets original values. - > (data_m)*X
        #gets imputed values. - > (1-data_m)*X

        X_filled = self._initial_imputation(X)

        X_orig = X_filled.copy()


        X_filled = torch.from_numpy(X_filled).float()
        #Evaluate
        self.model.eval()

        #Transform Test set
        filled_data = self.model(X_filled.to(device))
        X_r = filled_data.cpu().detach().numpy()

        #add mask
        #Keep the original values and add the. imputations through X_R
        X_r= (data_m*X_orig) + (data_nm*X_r)

        return X_r

    def fit(self, X, y=None):
        """Fits the imputer on X and return self.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Input data, where "n_samples" is the number of samples and
            "n_features" is the number of features.

        y : ignored

        Returns
        -------
        self : object
            Returns self.
        """

        self.fit_transform(X)
        return self

