import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, z_dim):
        super(Encoder, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc21 = nn.Linear(hidden_dim, z_dim)
        self.fc22 = nn.Linear(hidden_dim, z_dim)

    def forward(self, x):
        h = F.relu(self.fc1(x))
        return self.fc21(h), self.fc22(h)

class Decoder(nn.Module):
    def __init__(self, z_dim, hidden_dim, num_dim, cat_dim):
        super(Decoder, self).__init__()
        self.fc1 = nn.Linear(z_dim, hidden_dim)
        self.fc_num = nn.Linear(hidden_dim, num_dim)
        self.fc_cat = nn.Linear(hidden_dim, cat_dim)

    def forward(self, z):
        h = F.relu(self.fc1(z))
        num_out = self.fc_num(h)
        cat_out = torch.sigmoid(self.fc_cat(h))
        return torch.cat((num_out, cat_out), dim=1)

class CVAE(nn.Module):
    def __init__(self, num_dim, cat_dim, n_batches, hidden_dim=128, z_dim=32):
        super(CVAE, self).__init__()
        self.num_dim = num_dim
        self.cat_dim = cat_dim
        self.n_batches = n_batches
        self.encoder = Encoder(num_dim + cat_dim + n_batches, hidden_dim, z_dim)
        self.decoder = Decoder(z_dim + n_batches, hidden_dim, num_dim, cat_dim)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x, y):
        if y is None:
            # Sample y from a categorical distribution
            y = F.one_hot(torch.randint(self.n_batches, (x.size(0),)), self.n_batches).to(dtype=torch.float32, device=x.device)
        mu, logvar = self.encoder(torch.cat((x, y), dim=1))
        z = self.reparameterize(mu, logvar)
        return self.decoder(torch.cat((z, y), dim=1)), mu, logvar


    def loss_function(self, recon_x, x, mu, logvar, as_dict=False):
        mse = nn.MSELoss(reduction='sum')
        bce = nn.BCELoss(reduction='sum')
        mse_loss = mse(recon_x[:, :self.num_dim], x[:, :self.num_dim])
        bce_loss = bce(recon_x[:, self.num_dim:], x[:, self.num_dim:])  # Categorical require BCE as loss
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        total_loss = mse_loss + bce_loss + KLD
        if as_dict:
            return {'Loss': total_loss, 'MSE': mse_loss, 'KLD': KLD, 'BCE': bce_loss}
        else:
            return total_loss

    def predict(self, x, y=None, n_samples=100):
        recon_x_samples = []
        for _ in range(n_samples):
            recon_x, _, _ = self.forward(x + torch.rand_like(x), y)
            recon_x_samples.append(recon_x)
        
        mean_recon_x = torch.stack(recon_x_samples).mean(dim=0)
        return mean_recon_x


# # Example usage:
# num_dim = 10
# cat_dim = 5
# n_batches = 4
# cvae = CVAE(num_dim, cat_dim, n_batches)

# x = torch.rand(64, num_dim + cat_dim)
# y = torch.rand(64, n_batches)
# recon_x, mu, logvar = cvae(x, y)
# loss = loss_function(recon_x, x, mu, logvar, num_dim, cat_dim)
# print(recon_x, mu)
# print(loss)
