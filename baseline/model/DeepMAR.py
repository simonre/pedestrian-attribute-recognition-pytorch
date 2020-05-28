import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from .resnet import resnet50
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data, Batch



class DeepMAR_ResNet50(nn.Module):
    def __init__(
        self, 
        **kwargs
    ):
        super(DeepMAR_ResNet50, self).__init__()
        self.device = torch.device('cuda:0')
        # init the necessary parameter for netwokr structure
        if 'num_att' in kwargs:
            self.num_att = kwargs['num_att'] 
        else:
            self.num_att = 35
        if 'last_conv_stride' in kwargs:
            self.last_conv_stride = kwargs['last_conv_stride']
        else:
            self.last_conv_stride = 2
        if 'drop_pool5' in kwargs:
            self.drop_pool5 = kwargs['drop_pool5']
        else:
            self.drop_pool5 = True 
        if 'drop_pool5_rate' in kwargs:
            self.drop_pool5_rate = kwargs['drop_pool5_rate']
        else:
            self.drop_pool5_rate = 0.5
        if 'pretrained' in kwargs:
            self.pretrained = kwargs['pretrained'] 
        else:
            self.pretrained = True

        self.base = resnet50(pretrained=self.pretrained, last_conv_stride=self.last_conv_stride)
        
        self.classifier = nn.Linear(2048, self.num_att)
        init.normal(self.classifier.weight, std=0.001)
        init.constant(self.classifier.bias, 0)

        self.d = 32
        # One feature extractor per Class
        for i in range(self.num_att):
            feature = nn.Sequential(
                nn.Linear(2048, 1024),
                nn.LeakyReLU(0.2),
                nn.Linear(1024, self.d)
            )
            setattr(self, "feature%d" % i, feature)

        second_layer = 32
        self.gc1 = GCNConv(self.d, second_layer, normalize=True)
        self.relu = nn.LeakyReLU(0.2)
        self.gc2 = GCNConv(second_layer, 1)
        self.relu2 = nn.LeakyReLU(0.2)

        for i in range(self.num_att):
            classifier = nn.Sequential(
                nn.Linear(self.d, 1)
            )
            setattr(self, "classifier%d" % i, classifier)

        self.bn_features = torch.nn.BatchNorm1d(self.num_att)
        self.sig_features = torch.nn.Sigmoid()

        self.bn = torch.nn.BatchNorm1d(self.num_att)
        self.sigmoid = torch.nn.Sigmoid()

    def get_data_from_outputs(self, x, adjacency, edge_weight):
        data = []
        for el in x:
            datapoint = Data(x=el, edge_index=adjacency, edge_attr=edge_weight)
            data.append(datapoint)
        return data

    def forward(self, x, adjacency, edge_weight):
        x = self.base(x)
        x = F.avg_pool2d(x, x.shape[2:])
        x = x.view(x.size(0), -1)
        features = torch.zeros(x.shape[0], self.num_att, self.d).to(self.device)
        feature_outputs = torch.zeros([x.shape[0], self.num_att]).to(self.device)
        features = torch.transpose(features, 0, 1)
        for i in range(self.num_att):
            temp = getattr(self, "feature%d" % i)(x)
            features[i] = temp

        feature_outputs = torch.transpose(feature_outputs, 0, 1)
        features = self.relu(features)
        features = torch.transpose(features, 0, 1)

        features = self.bn_features(features)
        features = torch.transpose(features, 0, 1)

        for i in range(self.num_att):
            inp = features[i]
            temp_classif = getattr(self, "classifier%d" % i)(inp)
            temp_classif = temp_classif.view(-1)
            feature_outputs[i] = temp_classif
        feature_outputs = feature_outputs.transpose(0, 1)
        features = torch.transpose(features, 0, 1)

        dl = self.get_data_from_outputs(features, adjacency, edge_weight)
        b = Batch.from_data_list(dl)

        x, edge_index, edge_weight = b.x, b.edge_index.long(), b.edge_attr

        x = x.to(self.device)
        edge_index = edge_index.to(self.device)
        if edge_weight is not None:
            edge_weight = edge_weight.to(self.device)
        x = self.gc1(x, edge_index, edge_weight=edge_weight)
        x = self.relu(x)
        x = self.gc2(x, edge_index, edge_weight=edge_weight)

        x = x.view(feature_outputs.shape[0], -1)

        return (x, feature_outputs)

class DeepMAR_ResNet50_ExtractFeature(object):
    """
    A feature extraction function
    """
    def __init__(self, model, **kwargs):
        self.model = model

    def __call__(self, imgs, edge_index, edge_weights):
        old_train_eval_model = self.model.training

        # set the model to be eval
        self.model.eval()

        # imgs should be Variable
        if not isinstance(imgs, Variable):
            print('imgs should be type: Variable')
            raise ValueError
        score = self.model(imgs, edge_index, edge_weights)
        score = score.data.cpu().numpy()

        self.model.train(old_train_eval_model)

        return score
