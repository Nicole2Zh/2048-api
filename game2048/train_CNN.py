import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from sklearn.model_selection import train_test_split
from CNN_Model import CNN
import numpy as np

batch_size = 128
NUM_EPOCHS = 80


board_data = np.load("data/data0.npy")
board_direction = np.load("data/direction0.npy")
for i in range(100):
    data_name = "data/data" + str(i) + ".npy"
    direction_name = "data/direction" + str(i) + ".npy"
    board_data_i = np.load(data_name)
    board_direction_i = np.load(direction_name)
    board_data = np.concatenate((board_data,board_data_i),axis=0)
    board_direction = np.concatenate((board_direction, board_direction_i),axis=0)
board_data = np.where(board_data>0,board_data,1)
board_data = np.log2(board_data)
print(board_data.shape)
print(board_data[100])
board_direction = board_direction.squeeze()
X = np.int64(board_data)
Y = np.int64(board_direction)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3,shuffle=False)
X_train = torch.FloatTensor(X_train)
X_test = torch.FloatTensor(X_test)
Y_train = torch.LongTensor(Y_train)
Y_test = torch.LongTensor(Y_test)

train_dataset = torch.utils.data.TensorDataset(X_train,Y_train)
test_dataset = torch.utils.data.TensorDataset(X_test,Y_test)

train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True
)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False
)

model = CNN()
# Find total parameters and trainable parameters
total_params = sum(p.numel() for p in model.parameters())
print(f'{total_params:,} total parameters.')
total_trainable_params = sum(
    p.numel() for p in model.parameters() if p.requires_grad)
print(f'{total_trainable_params:,} training parameters.')
model = model.cuda()
optimizer = optim.Adam(model.parameters(), lr = 0.001)

def train(epoch):
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = Variable(data).cuda(), Variable(target).cuda()
        data = data.unsqueeze(dim=1)
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 1000 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
    torch.save(model.state_dict(), 'model_cnn/epoch_{}.pkl'.format(epoch))

def test(epoch):
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        with torch.no_grad():
            data = Variable(data).cuda()
            target =Variable(target).cuda()
        #print(target)
        data = data.unsqueeze(dim=1)
        output = model(data)
        test_loss += F.cross_entropy(output, target, size_average=False).item()
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    test_loss /= len(test_loader.dataset)
    print('\nTest set epoch {}: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        epoch, test_loss, correct, len(test_loader.dataset),
        100. * float(correct) / len(test_loader.dataset)))


for epoch in range(100):
    model.train()
    train(epoch)
    model.eval()
    test(epoch)
