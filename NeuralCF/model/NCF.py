import torch
import torch.nn as nn
import torch.nn.functional as F

class NCF(nn.Module):
    def __init__(
            self, user_num, item_num, factor_num,
            num_layers, dropout, model,
            GMF_model=None, MLP_model=None):
        super(NCF, self).__init__()
        # set the category of model, model : {'MLP', 'GMF', 'NeuMF-end', 'NeuMF-pre'}
        self.model = model

        # Set the dropout rate
        self.dropout = dropout

        # Pretrained models
        self.GMF_model = GMF_model
        self.MLP_model = MLP_model

        # Set the embedding vectors
        self.embed_user_GMF = nn.Embedding(user_num, factor_num)
        self.embed_item_GMF = nn.Embedding(item_num, factor_num)
        # the input is the cancat of embedding vectors from 1 user and 1 item respectively
        embedding_size_MLP = factor_num * (2 ** (num_layers - 1))
        self.embed_user_MLP = nn.Embedding(user_num, embedding_size_MLP)
        self.embed_item_MLP = nn.Embedding(item_num, embedding_size_MLP)

        self.MLP_layers = self._make_layer(embedding_size_MLP, num_layers)

        if self.model in ['MLP', 'GMF']:
            # single tower
            predict_size = factor_num
        else:
            # concate
            predict_size = factor_num * 2
        # If we use the model of single MLP and singl MF, then the input size of the predict layer
        # is 1 side of factor_num. If we use the double tower strcuture of the model, the the input
        # size of the model is the concated size (twice of the factor_num).
        self.predict_layer = nn.Linear(predict_size, 1)
        self._init_weight()


    def _make_layer(
            self, embedding_size_MLP, num_layers):
        
        MLP_layers = []
        mlp_input_size = embedding_size_MLP * 2
        for i in range(num_layers):
            mlp_output_size = mlp_input_size // 2
            MLP_layers.append(nn.Dropout(p=self.dropout))
            MLP_layers.append(nn.Linear(mlp_input_size, mlp_output_size))
            MLP_layers.append(nn.BatchNorm1d(num_features=mlp_output_size))
            MLP_layers.append(nn.ReLU())
            mlp_input_size = mlp_output_size

        return nn.Sequential(*MLP_layers)

    def _init_weight(self):
        if not self.model == "NeuMF-pre":
            # default weight initialization
            nn.init.normal_(self.embed_item_GMF.weight, std=0.01)
            nn.init.normal_(self.embed_user_GMF.weight, std=0.01)
            nn.init.normal_(self.embed_item_MLP.weight, std=0.01)
            nn.init.normal_(self.embed_user_MLP.weight, std=0.01)

            for m in self.modules():
                if isinstance(m, nn.Linear):
                    # nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))
                    nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')
            
            # NOTE: for predict layer, since we should expect a sigmoid, we use this
            nn.init.kaiming_uniform_(self.predict_layer.weight, a=1, nonlinearity='sigmoid')

            # for m in self.modules():
            #     if isinstance(m, nn.Linear):
            #         # initialization of bias
            #         m.bias.data.zero_()

        else:
            # Copy pretrained weight of embedding weights (Embedding Layer on has weight)
            # NOTE: Also keep in mind that for the embedding layer, we can only use 
            # SGD, SparseAdam and Adagrad(CPU only) as optimizer
            self.embed_item_GMF.weight.data.copy_(self.GMF_model.embed_item_GMF.weight)
            self.embed_user_GMF.weight.data.copy_(self.GMF_model.embed_user_GMF.weight)
            self.embed_item_MLP.weight.data.copy_(self.MLP_model.embed_item_MLP.weight)
            self.embed_user_MLP.weight.data.copy_(self.MLP_model.embed_user_MLP.weight)
            
            # Copy pretrained weight and bias of MLP layers
            for (m, m_pre) in zip(self.MLP_layers, self.MLP_model.MLP_layers):
                if isinstance(m, nn.Linear) and isinstance(m_pre, nn.Linear):
                    m.weight.data.copy_(m_pre.weight)
                    m.bias.data.copy_(m_pre.bias)

            # Copy predict layer's weight
            # NOTE: here we need to concat the weight for MF and MLP
            predict_weight = torch.cat([self.GMF_model.predict_layer.weight, self.MLP_model.predict_layer.weight], dim=1)
            predict_bias = self.GMF_model.predict_layer.bias + self.MLP_model.predict_layer.bias
            self.predict_layer.weight.data.copy_(0.5 * predict_weight)
            self.predict_layer.bias.data.copy_(0.5 * predict_bias)

    def forward(self, user, item):
        if self.model == 'GMF':
            # NOTE: here the input of embedding layer is a LongTensor 
            # arbitrary shape containing the indices to extract
            embed_user_GMF = self.embed_user_GMF(user)
            embed_item_GMF = self.embed_item_GMF(item)
            # The output for this is simply the inner product of the embedding vectors
            output_GMF = embed_user_GMF * embed_item_GMF
            return self.predict_layer(output_GMF).view(-1)
        
        elif self.model == 'MLP':
            embed_user_MLP = self.embed_user_MLP(user)
            embed_item_MLP = self.embed_item_MLP(item)
            output_MLP = self.MLP_layers(torch.cat((embed_user_MLP, embed_item_MLP), -1))
            # output_MLP = self.MLP_layers(torch.cat([embed_user_MLP, embed_item_MLP], dim=1))
            return self.predict_layer(output_MLP).view(-1)
        
        else:
            # Matrix Factorization side of the model
            embed_user_GMF = self.embed_user_GMF(user)
            embed_item_GMF = self.embed_item_GMF(item)
            output_GMF = embed_user_GMF * embed_item_GMF
            # MLP side of the model
            embed_user_MLP = self.embed_user_MLP(user)
            embed_item_MLP = self.embed_item_MLP(item)
            output_MLP = self.MLP_layers(torch.cat((embed_user_MLP, embed_item_MLP), -1))
            # Concat these 2 latent vectors together
            concat = torch.cat((output_GMF, output_MLP), -1)
            return self.predict_layer(concat).view(-1)