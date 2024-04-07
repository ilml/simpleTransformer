from tqdm import tqdm
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.utils import save_image
from torchvision.datasets import MNIST


class VAE(nn.Module):

    def __init__(self, in_features, latent_size):
        super(VAE, self).__init__()

        self.latent_size = latent_size

        self.encoder_forward = nn.Sequential(
            nn.Linear(in_features, in_features),
            nn.LeakyReLU(),
            nn.Linear(in_features, in_features),
            nn.LeakyReLU(),
            nn.Linear(in_features, latent_size * 2)
        )

        self.decoder_forward = nn.Sequential(
            nn.Linear(latent_size, in_features),
            nn.LeakyReLU(),
            nn.Linear(in_features, in_features),
            nn.LeakyReLU(),
            nn.Linear(in_features, in_features),
            nn.Sigmoid()
        )

    def encoder(self, X):
        out = self.encoder_forward(X)
        mu = out[:, :self.latent_size]
        log_var = out[:, self.latent_size:]
        return mu, log_var

    def decoder(self, z):
        mu_prime = self.decoder_forward(z)
        return mu_prime

    def reparameterization(self, mu, log_var):
        epsilon = torch.randn_like(log_var)
        z = mu + epsilon * torch.sqrt(log_var.exp())
        return z

    def loss(self, X, mu_prime, mu, log_var):
        # reconstruction_loss = F.mse_loss(mu_prime, X, reduction='mean') is wrong!
        reconstruction_loss = torch.mean(torch.square(X - mu_prime).sum(dim=1))

        latent_loss = torch.mean(0.5 * (log_var.exp() + torch.square(mu) - log_var).sum(dim=1))
        return reconstruction_loss + latent_loss

    def forward(self, X, *args, **kwargs):
        mu, log_var = self.encoder(X)
        z = self.reparameterization(mu, log_var)
        mu_prime = self.decoder(z)
        return mu_prime, mu, log_var



def train(model, optimizer, data_loader, device):
    model.train()

    total_loss = 0
    pbar = tqdm(data_loader)
    for X, y in pbar:
        batch_size = X.shape[0]
        X = X.view(batch_size, -1).to(device)
        model.zero_grad()

        mu_prime, mu, log_var = model(X)

        loss = model.loss(X.view(batch_size, -1), mu_prime, mu, log_var)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        pbar.set_description('Loss: {loss:.4f}'.format(loss=loss.item()))

    return total_loss / len(data_loader)


@torch.no_grad()
def save_res(vae, data, latent_size, device):
    num_classes = len(data.classes)

    # raw samples from dataset
    out = []
    for i in range(num_classes):
        img = data.data[torch.where(data.targets == i)[0][:num_classes]]
        out.append(img)
    out = torch.stack(out).transpose(0, 1).reshape(-1, 1, 28, 28)
    save_image(out.float(), './img/raw_samples.png', nrow=num_classes, normalize=True)

    z = torch.randn(num_classes ** 2, latent_size).to(device)
    out = vae.decoder(z)
    save_image(out.view(-1, 1, 28, 28), './img/vae_samples.png', nrow=num_classes)


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)

    batch_size = 256 * 4
    epochs = 50
    latent_size = 64
    in_features = 28 * 28
    lr = 0.001

    data = MNIST('dataset', download=True, transform=transforms.ToTensor())
    data_loader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=True)

    # train VAE
    vae = VAE(in_features, latent_size).to(device)
    optimizer = torch.optim.AdamW(vae.parameters(), lr=lr)

    print('Start Training VAE...')
    for epoch in range(1, 1 + epochs):
        loss = train(vae, optimizer, data_loader, device)
        print("Epochs: {epoch}, AvgLoss: {loss:.4f}".format(epoch=epoch, loss=loss))
    print('Training for VAE has been done.')

    save_res(vae, data, latent_size, device)


if __name__ == '__main__':
    main()