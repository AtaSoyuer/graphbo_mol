import torch
import torch.nn.functional as F

class FocalLoss(torch.nn.Module):

    def __init__(self, device, gamma = 2.0, beta=0.25):
        super(FocalLoss, self).__init__()
        self.device = device
        self.gamma = torch.tensor(gamma, dtype = torch.float32).to(self.device)
        self.beta = torch.tensor(beta, dtype = torch.float32).to(self.device)

    def forward(self, inputs, targets):
        l1_loss = F.l1_loss(inputs, targets, reduction='none').to(self.device)
        l2_loss = torch.square(l1_loss).to(self.device)
        #2_loss = torch.square(inputs-targets).to(self.device)

        #weights = torch.sigmoid(self.beta*l1_loss).to(self.device)
        #focal_loss = weights**self.gamma * l1_loss

        weights = torch.sigmoid(self.gamma*(l1_loss - self.beta)).to(self.device)
        focal_loss = weights * l2_loss #* targets

        return focal_loss.mean()    

#loss = FocalLoss()

#print(loss(torch.tensor([0,1]),torch.tensor([4,5])))
