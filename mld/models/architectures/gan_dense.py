import torch
import torch.nn as nn
import pytorch_lightning as pl

class Generator(nn.Module):

    def __init__(self, latent_in, text_in, latent_out):
        super().__init__()
        self.latent_in = latent_in
        self.text_in = text_in
        self.latent_out = latent_out

        self.linear1 = nn.Linear(latent_in + text_in, 1024)
        self.bn1 = nn.BatchNorm1d(1024)
        self.relu1 = nn.ReLU()

        self.residual_block1 = ResidualBlock(1024, 512)
        self.residual_block2 = ResidualBlock(512, 256)

        self.linear2 = nn.Linear(256, latent_out)

    def forward(self, z, text_emb):
        text_emb = text_emb.reshape(text_emb.shape[0], -1)
        x = torch.cat([z, text_emb], dim=-1)

        x = self.linear1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        print(x.shape)
        x = self.residual_block1(x)
        print(x.shape)
        x = self.residual_block2(x)
        print(x.shape)

        x = self.linear2(x)
        return x

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.layer_norm = nn.LayerNorm(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels)
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.layer_norm(out)
        out = self.relu(out)
        return out


# class ResidualBlock(nn.Module):
#     def __init__(self, in_channels, out_channels, stride=1):
#         super().__init__()
#         self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
#         self.bn1 = nn.BatchNorm1d(out_channels)
#         self.relu = nn.ReLU(inplace=True)
#         self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
#         self.bn2 = nn.BatchNorm1d(out_channels)
#         self.layer_norm = nn.LayerNorm(out_channels)

#         self.shortcut = nn.Sequential()
#         if stride != 1 or in_channels != out_channels:
#             self.shortcut = nn.Sequential(
#                 nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
#                 nn.BatchNorm1d(out_channels)
#             )

#     def forward(self, x):
#         batch_size, in_channels, _ = x.size()
#         out = self.relu(self.bn1(self.conv1(x.view(batch_size, in_channels, -1))))
#         out = self.bn2(self.conv2(out))
#         out += self.shortcut(x.view(batch_size, in_channels, -1))
#         out = self.layer_norm(out.view(batch_size, -1))
#         out = self.relu(out)
#         return out.view(batch_size, out.size(1), 1)
    
    
class Discriminator(nn.Module):
    """
    Discriminator class in a CGAN. Accepts a tensor of size 784 and
    a label tensor as input, and outputs a tensor of size 1,
    with the predicted class probabilities (generated or real data)
    """

    def __init__(self, latent_in, text_in, latent_out):
        super().__init__()
        self.latent_in = latent_in
        self.text_in = text_in
        self.latent_out = latent_out

        self.linear1 = nn.Linear(latent_out + text_in, 1024)
        self.bn1 = nn.BatchNorm1d(1024)
        self.relu1 = nn.ReLU()

        self.residual_block1 = ResidualBlock(1024, 512)
        self.residual_block2 = ResidualBlock(512, 256)

        self.linear2 = nn.Linear(256, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, text_emb):
        text_emb = text_emb.reshape(text_emb.shape[0], -1)
        x = torch.cat([x, text_emb], dim=-1)

        x = self.linear1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        x = self.residual_block1(x)
        x = self.residual_block2(x)

        x = self.linear2(x)
        x = self.sigmoid(x)
        return x
    
    
class CGAN(pl.LightningModule):

  def __init__(self, latent_in_dim, text_emb_dim, latent_out_dim):
    super().__init__()
    self.latent_in_dim = latent_in_dim
    self.text_emb_dim = text_emb_dim
    self.latent_out_dim = latent_out_dim
    self.generator = Generator(latent_in_dim, text_emb_dim, latent_out_dim)
    self.discriminator = Discriminator(latent_in_dim, text_emb_dim, latent_out_dim)
    self.BCE_loss = nn.BCELoss()

  def forward(self, z, text_emb):
    """
    Generates an image using the generator
    given input noise z and labels y
    """
    return self.generator(z, text_emb)

  def generator_step(self, z, text_emb):
    """
    Training step for generator
    1. Sample random noise and labels
    2. Pass noise and labels to generator to
       generate images
    3. Classify generated images using
       the discriminator
    4. Backprop loss
    """

    # Generate images
    fake_latent = self(z, text_emb)

    # Classify generated image using the discriminator
    fake_pred = torch.squeeze(self.discriminator(fake_latent, text_emb))

    # Backprop loss. We want to maximize the discriminator's
    # loss, which is equivalent to minimizing the loss with the true
    # labels flipped (i.e. y_true=1 for fake images). We do this
    # as PyTorch can only minimize a function instead of maximizing
    g_loss = self.BCE_loss(fake_pred, torch.ones_like(fake_pred))

    return g_loss

  def discriminator_step(self, z, x, text_emb):
    """
    Training step for discriminator
    1. Get actual images and labels
    2. Predict probabilities of actual images and get BCE loss
    3. Get fake images from generator
    4. Predict probabilities of fake images and get BCE loss
    5. Combine loss from both and backprop
    """
    
    # Real images
    x = x.reshape(z.shape[0], -1)
    real_pred = torch.squeeze(self.discriminator(x, text_emb))
    real_loss = self.BCE_loss(real_pred, torch.ones_like(real_pred))


    fake_latent = self(z, text_emb).detach() 
    fake_pred = torch.squeeze(self.discriminator(fake_latent, text_emb))
    fake_loss = self.BCE_loss(fake_pred, torch.zeros_like(fake_pred))


    d_loss = (real_loss + fake_loss) / 2
    return d_loss

    
  def configure_optimizers(self):
    g_optimizer = torch.optim.Adam(self.generator.parameters(), lr=0.0002)
    d_optimizer = torch.optim.Adam(self.discriminator.parameters(), lr=0.0002)
    return [g_optimizer, d_optimizer], []


# if __name__ == "__main__":
#   device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#   mnist_transforms = transforms.Compose([transforms.ToTensor(),
#                                       transforms.Normalize(mean=[0.5], std=[0.5]),
#                                       transforms.Lambda(lambda x: x.view(-1, 784)),
#                                       transforms.Lambda(lambda x: torch.squeeze(x))
#                                       ])

#   data = datasets.MNIST(root='../data/MNIST', download=True, transform=mnist_transforms)

#   mnist_dataloader = DataLoader(data, batch_size=128, shuffle=True, num_workers=0) 

#   model = CGAN()

#   trainer = pl.Trainer(max_epochs=100, gpus=1 if torch.cuda.is_available() else 0, progress_bar_refresh_rate=50)
#   trainer.fit(model, mnist_dataloader)
  
       

    
    