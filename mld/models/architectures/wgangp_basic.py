import torch
import torch.nn as nn
import pytorch_lightning as pl
import torch.nn.functional as F

def wasserstein_loss(real_pred, fake_pred):
    return torch.mean(real_pred - fake_pred)

class ResidualBlock(nn.Module):
    def __init__(self, hidden_dim):
        super(ResidualBlock, self).__init__()
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        residual = x
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        x += residual
        x = self.relu(x)
        return x

class Generator(nn.Module):
  '''
  Generator class in a CGAN. Accepts a noise tensor (latent dim 100)
  and a label tensor as input as outputs another tensor of size 784.
  Objective is to generate an output tensor that is indistinguishable 
  from the real MNIST digits.
  '''

  def __init__(self, latent_in, text_in, latent_out):
    super().__init__()
    #in_channels = latent_in + text_in
    #self.starting_constant = torch.ones((1, in_channels))
    self.layer1 = nn.Sequential(nn.Linear(in_features=latent_in+text_in, out_features=1024),
                                nn.LeakyReLU())
    self.layer2 = nn.Sequential(nn.Linear(in_features=1024, out_features=512),
                                nn.LeakyReLU())
    self.output = nn.Sequential(nn.Linear(in_features=512, out_features=latent_out),
                                nn.Tanh()
                                )

    self.residual_block = ResidualBlock(512)


  def forward(self, z, text_emb):

    # x is a tensor of size (batch_size, 110)
    # reshapeing text_emb
    #print('z, txt', z.shape, text_emb.shape)
    text_emb = text_emb.reshape(text_emb.shape[0],-1)
    x = torch.cat([z, text_emb], dim=-1)
    x = self.layer1(x)
    x = self.layer2(x)
    x = self.residual_block(x)
    x = self.output(x)
    return x


class Discriminator(nn.Module):
  '''
  Discriminator class in a CGAN. Accepts a tensor of size 784 and
  a label tensor as input and outputs a tensor of size 1,
  with the predicted class probabilities (generated or real data)
  '''

  def __init__(self, latent_in, text_in, latent_out):
    super(Discriminator, self).__init__()
    
    self.layer1 = nn.Sequential(nn.Linear(in_features=latent_out+text_in, out_features=1024),
                                nn.LeakyReLU())
    self.layer2 = nn.Sequential(nn.Linear(in_features=1024, out_features=512),
                                nn.LeakyReLU())
    self.layer3 = nn.Sequential(nn.Linear(in_features=512, out_features=256),
                                nn.LeakyReLU())
    self.output = nn.Linear(in_features=256, out_features=1)
    self.residual_block = ResidualBlock(256)

  def forward(self, z, text_emb):
    # pass the labels into a embedding layer
    # labels_embedding = self.embedding(y)
    # concat the embedded labels and the input tensor
    # x is a tensor of size (batch_size, 794)
    #print('disc inp shape', x.shape, len(self.prog_blocks), steps)
    #print('disc', z.shape, text_emb.shape)
    text_emb = text_emb.reshape(text_emb.shape[0],-1)
    x = torch.cat([z, text_emb], dim=-1)    
    x = self.layer1(x)
    x = self.layer2(x)
    x = self.layer3(x)
    x = self.residual_block(x)
    x = self.output(x)
    return x

class WGAN(pl.LightningModule):

  def __init__(self, latent_in_dim, text_emb_dim, latent_out_dim):
    super().__init__()
    self.latent_in_dim = latent_in_dim
    self.text_emb_dim = text_emb_dim
    self.latent_out_dim = latent_out_dim
    self.generator = Generator(latent_in_dim, text_emb_dim, latent_out_dim)
    self.discriminator = Discriminator(latent_in_dim, text_emb_dim, latent_out_dim)
    self.BCE_loss = nn.BCELoss()
    self.lamda = 10

  def forward(self, z, text_emb):
    """
    Generates an image using the generator
    given input noise z and labels y
    """
    return self.generator(z, text_emb)

  def adversarial_loss(self, y_hat, y):
    return F.binary_cross_entropy(y_hat, y)

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
    #print('gs step', z.shape, text_emb.shape)
    fake_latent = self(z, text_emb)

    # Classify generated image using the discriminator
    #print('fake latent shape', fake_latent.shape)
    #fake_latent_prog = self.generator(fake_latent, text_emb)
    fake_pred = torch.squeeze(self.discriminator(fake_latent, text_emb))

    # Backprop loss. We want to maximize the discriminator's
    # loss, which is equivalent to minimizing the loss with the true
    # labels flipped (i.e. y_true=1 for fake images). We do this
    # as PyTorch can only minimize a function instead of maximizing
    g_loss = -torch.mean(fake_pred)
    #print('gloss', g_loss.shape, g_loss, fake_pred.shape)

    return g_loss

  def discriminator_step(self, z_fake, z, text_emb):
    """
    Training step for discriminator
    1. Get actual images and labels
    2. Predict probabilities of actual images and get BCE loss
    3. Get fake images from generator
    4. Predict probabilities of fake images and get BCE loss
    5. Combine loss from both and backprop
    """
    
    # Real images
    #print('======================')
    #print('b4 reshape real shape', z_fake.shape, text_emb.shape)
    #x = x.reshape(z_fake.shape[0], -1)
    #print('after reshape real shape', x.shape)
    #real_pred = torch.squeeze(self.discriminator(x, 1.0, 6))
    fake_pred = self.discriminator(z_fake, text_emb)

    if len(z.size()) > 2:
        z = z.squeeze()
    real_pred = self.discriminator(z, text_emb)

    #d_loss = (real_loss + fake_loss) / 2
    d_loss = wasserstein_loss(real_pred, fake_pred)
    d_loss = self.lamda * self.gradient_penalty(z, z_fake, text_emb)
    #print('dloss', d_loss.shape, d_loss)
    return d_loss

    
  def configure_optimizers(self):

    g_optimizer = torch.optim.RMSprop(self.generator.parameters(), lr=1e-6)
    d_optimizer = torch.optim.RMSprop(self.discriminator.parameters(), lr=1e-5)
    return [g_optimizer, d_optimizer], []

  def gradient_penalty(self, real_pred, fake_pred, text_emb):
    batch_size = real_pred.size(0)
    alpha = torch.rand(batch_size, 1, device=self.device)
    alpha = alpha.expand_as(real_pred)

    interpolates = alpha * real_pred + ((1 - alpha) * fake_pred.detach())
    interpolates.requires_grad_(True)

    d_interpolates = self.discriminator(interpolates, text_emb)
    d_interpolates.requires_grad_(True)

    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=torch.ones(d_interpolates.size(), device=self.device),
        create_graph=True,
        retain_graph=True,
        materialize_grads=True,
    )[0]


    gradients = gradients.view(batch_size, -1)
    gradient_norm = gradients.norm(2, dim=1)
    gradient_penalty = ((gradient_norm - 1) ** 2).mean()
    return gradient_penalty

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
  
  