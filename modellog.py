import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.metrics import accuracy_score

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import DataStructs

from modellogger.modellogger import ModelLogger
mlog = ModelLogger('mllog.db')

def mol2arr(mol, radi=2, nBits=1024):
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radi, nBits=nBits)
    arr = np.zeros((0,))
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr

class RDKit_sol_data():
    def __init__(self, trainsdf, testsdf):
        self.trainsdf = trainsdf
        self.testsdf = testsdf
        self.cls_dict = {
            '(A) low': 0,
            '(B) medium': 1,
            '(C) high' :2
        }
        self.target = 'SOL_classification'
        self.train_mols = [m for m in Chem.SDMolSupplier(self.trainsdf) if m is not None]
        self.test_mols =  [m for m in Chem.SDMolSupplier(self.testsdf) if m is not None]
    
    def get_dataset(self):
        train_y = [self.cls_dict[m.GetProp(self.target)] for m in self.train_mols]
        test_y =  [self.cls_dict[m.GetProp(self.target)] for m in self.test_mols]
        return {
            'train_mols': self.train_mols,
            'train_y': np.array(train_y),
            'test_mols':self.test_mols,
            'test_y': np.array(test_y)
            }

def scoringfunc(test_Y, pred_Y):
    return accuracy_score(test_Y, pred_Y)

## Data preparation
radi = 2
nBits = 1024

train_path = './data/solubility.train.sdf'
test_path = './data/solubility.test.sdf'

loader = RDKit_sol_data(train_path, test_path)
dataset = loader.get_dataset()

featlist = [f'fp_{i}' for i in range(nBits)]

trainFP = [mol2arr(m, radi=radi, nBits=nBits) for m in dataset['train_mols']]
train_X = np.array(trainFP)
train_X = pd.DataFrame(train_X, columns=featlist)
train_Y = dataset['train_y']

testFP = [mol2arr(m, radi=radi, nBits=nBits) for m in dataset['test_mols']]
test_X = np.array(testFP)
test_X = pd.DataFrame(test_X, columns=featlist)
test_Y = dataset['test_y']

rf_cls = RandomForestClassifier()
sv_cls = SVC(C=100, gamma='auto')
gp_cls = GaussianProcessClassifier()
modelnames = ['rf_cls', 'sv_c', 'gp_c']


for idx, mdl in enumerate([rf_cls, sv_cls, gp_cls]):
    mdl.fit(train_X, train_Y)
    pred_Y = mdl.predict(test_X)
    mlog.store_model(modelnames[idx], mdl, train_X, scoringfunc(test_Y, pred_Y))

mlog.model_profiles()