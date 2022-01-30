# %%
from collections import OrderedDict
from re import S
from numpy import dtype, pad
from sklearn.preprocessing import OneHotEncoder
import torch
import torch.nn as nn
import torch.nn.functional as f
import torch.optim as optim
from torch.utils.data import Dataset
from torch.autograd import Variable
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data.dataset import Dataset
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from xgboost import train

torch.manual_seed(1)
_device = (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))

class SignalPeptidesDataset(Dataset):
  def __init__(self, data, labels, transform=None, target_transform=None):
    self.data = data
    self.labels = labels
    self.transform = transform
    self.target_transform = target_transform

  def __len__(self):
    return len(self.labels)
  
  def __getitem__(self, idx):
    d = self.data[idx]
    l = self.labels[idx]
    if self.transform:
      d = self.transform(d)
    if self.target_transform:
      l = self.target_transform(l)
    return d, l

class SPCNN(nn.Module):
    def __init__(self) :
        super(SPCNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=5,
                               kernel_size=6, stride=1, padding=3).to(device=_device)
        self.conv2 = nn.Conv1d(in_channels=self.conv1.out_channels, out_channels=self.conv1.out_channels * 2,
                               kernel_size=6, stride=1, padding=3).to(device=_device)
        self.conv3 = nn.Conv1d(in_channels=self.conv2.out_channels, out_channels=self.conv2.out_channels //
                               2, kernel_size=2, stride=1, padding=1).to(device=_device)

        self.act1 = nn.ReLU().to(device=_device)
        self.act2 = nn.ReLU().to(device=_device)
        self.act3 = nn.ReLU().to(device=_device)

        self.pool1 = nn.MaxPool1d(2).to(device=_device)
        self.pool2 = nn.MaxPool1d(2).to(device=_device)
        
        self.batch_norm1 = nn.BatchNorm1d(self.conv1.out_channels).to(device=_device)
        self.batch_norm2 = nn.BatchNorm1d(self.conv2.out_channels).to(device=_device)
        self.batch_norm3 = nn.BatchNorm1d(self.conv3.out_channels).to(device=_device)
        
        self.drop1 = nn.Dropout(p=0.1).to(device=_device)
        self.drop2 = nn.Dropout(p=0.1).to(device=_device)
        self.drop3 = nn.Dropout(p=0.1).to(device=_device)

        self.lin = nn.Linear(5 * 14, 5).to(device=_device)


    def forward(self,x):

        out = self.drop1(
            self.pool1(
                self.batch_norm1(
                    self.conv1(x))))
        out = self.drop2(
            self.pool2(
                self.batch_norm2(
                    self.conv2(out))))
        out = self.lin(
            self.batch_norm3(
                self.conv3(out)).view(-1, 5 * 14))

        return out

def training_loop(n_epochs: int, model, train_loader: DataLoader, 
  optimizer, loss_fn, device, val_loader=False, verbose=False, **kwargs) -> OrderedDict:
  best = OrderedDict()
  last_val_loss = 10e20
  for i in range(1, n_epochs+1):
    loader_iter = zip(train_loader, val_loader) if val_loader else train_loader
    for x in loader_iter:

      train_inputs, train_labels = x[0]
      test_inputs, test_labels = x[1] if len(x) > 1 else (None, None)

      #pytorch is converting my list of tuples nx2 into a len 2 list of n size tensors
      #train_labels = torch.stack(train_labels, dim=1).to(device=device)
      #test_labels = torch.stack(test_labels, dim=1).to(device=device) if test_labels else None
      model = model.train()
      train_output = model(train_inputs)
      train_loss = loss_fn(train_output, train_labels)

      if test_inputs is not None:
        with torch.no_grad():
          model=model.eval()
          val_output = model(test_inputs)
          val_loss = loss_fn(val_output, test_labels)

      model = model.train()
      optimizer.zero_grad()
      train_loss.backward()
      optimizer.step()
    
    if val_loss < last_val_loss:
      best = model.state_dict()
      last_val_loss = val_loss

    if verbose and (i % round(n_epochs/10) == 0):
      print(f"Epoch: {i}, Train Loss: {train_loss}, Val Loss: {val_loss}")
  return best

df = pd.read_excel("Secretion Database List.xlsx")
df['Reliability (IH)'].unique()
encoding = {'Enhanced':4, 'Uncertain':1, 'Approved':2, 'Supported':3, 0:0}
df['Reliability (IH)'].fillna(0, inplace=True)
df['ordinal_reliability'] = df['Reliability (IH)'].map(encoding)
x = df['AA Sequence'].to_numpy()
y = df['ordinal_reliability'].to_numpy()
ohe = OneHotEncoder(sparse=False)
y = ohe.fit_transform(y.reshape(-1, 1))
#%%
_alphabet = set()
for i in df['AA Sequence'].to_numpy():
  for j in i:
    _alphabet.add(j)

alphabet = {v:i for i, v in enumerate(_alphabet)}
encodedrows = [list(map(lambda x: float(alphabet[x]), row)) for row in x]
max_length = max([len(y) for y in x])
passed_values = [row + [0.0]*(max_length - len(row)) for row in encodedrows]
X = torch.tensor(passed_values)[:600]
y = torch.tensor(y)[:600]

X_train, X_val, y_train, y_val = train_test_split(X, y)

training_dataset = SignalPeptidesDataset(X_train, y_train,
                                         transform=lambda x: x.unsqueeze(0).to(device=_device),
                                         target_transform=lambda x: x.unsqueeze(0).to(device=_device, dtype=torch.float32))
validation_dataset = SignalPeptidesDataset(X_val, y_val, 
                                         transform=lambda x: x.unsqueeze(0).to(device=_device), 
                                         target_transform=lambda x: x.unsqueeze(0).to(device=_device, dtype=torch.float32))

training_dataloader = DataLoader(training_dataset, batch_size=50, shuffle=True)
validation_dataloader = DataLoader(validation_dataset, batch_size=50)
#%%

model = SPCNN().to(device=_device)

num_epochs = 1000
learning_rate = 0.01

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

result = training_loop(
  n_epochs=num_epochs,
  train_loader=training_dataloader,
  val_loader=validation_dataloader,
  model=model,
  loss_fn=criterion,
  optimizer=optimizer,
  device=_device,
  verbose=True
)
result
