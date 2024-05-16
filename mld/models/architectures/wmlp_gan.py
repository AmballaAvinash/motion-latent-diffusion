import torch
import torch.nn as nn
import pytorch_lightning as pl
import torch.nn.functional as F

#factors = [1, 1, 1, 1, 1 / 2, 1 / 4, 1 / 8]
#factors = [1, 1, 1, 1, 1, 1, 1, 1, 1]
#factors = [2, 4, 8, 16, 32, 16, 8]
factors = [1, 1/2, 1/4, 1/8]
#factors = [1, 2, 4]

def wasserstein_loss(real_pred, fake_pred):
    return torch.mean(real_pred - fake_pred)

class PixelNorm(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        #print('pn', x.shape)
        out = x / torch.sqrt(torch.mean(x ** 2, dim=1, keepdim=True) + 1e-8)
        #print('pn out', out.shape)
        return out

class WSLinear(nn.Module):

    def __init__(self, in_features, out_features):
        super(WSLinear, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.scale = (2 / in_features) ** 0.5
        self.bias = self.linear.bias
        self.linear.bias = None

        nn.init.normal_(self.linear.weight)
        nn.init.zeros_(self.bias)

    def forward(self, x):
        #print('ws line', x.shape)
        return self.linear(x * self.scale) + self.bias


class MappingNetwork(nn.Module):

    def __init__(self, z_dim, w_dim):
        super().__init__()
        self.mapping = nn.Sequential(
            PixelNorm(),
            WSLinear(z_dim, w_dim),
            nn.ReLU(),
            WSLinear(w_dim, w_dim),
            nn.ReLU(),
            WSLinear(w_dim, w_dim),
            nn.ReLU(),
            WSLinear(w_dim, w_dim),
            nn.ReLU(),
            WSLinear(w_dim, w_dim),
            nn.ReLU(),
            WSLinear(w_dim, w_dim),
            nn.ReLU(),
            WSLinear(w_dim, w_dim),
            nn.ReLU(),
            WSLinear(w_dim, w_dim),
        )

    def forward(self, x):
        return self.mapping(x)

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
    self.map = MappingNetwork(text_in + latent_in, latent_out)


  def forward(self, z, text_emb, skip_init = False):

    # x is a tensor of size (batch_size, 110)
    # reshapeing text_emb
    #print('z, txt', z.shape, text_emb.shape)
    text_emb = text_emb.reshape(text_emb.shape[0],-1)
    x = torch.cat([z, text_emb], dim=-1)
    out = self.map(x)
    return out


class Discriminator(nn.Module):
  '''
  Discriminator class in a CGAN. Accepts a tensor of size 784 and
  a label tensor as input and outputs a tensor of size 1,
  with the predicted class probabilities (generated or real data)
  '''

  def __init__(self, text_in = 768, in_channels = 256):
    super(Discriminator, self).__init__()
    
    self.prog_blocks = nn.ModuleList([])
    self.leaky = nn.LeakyReLU(0.2)
    #c_in = in_channels * factors[-1]
    self.init_lin = nn.Linear(in_channels + text_in, in_channels)
    for i in range(len(factors) - 1):
        c_in = int(in_channels * factors[i])
        c_out = int(in_channels * factors[i + 1])
        self.prog_blocks.append(nn.Sequential(
            WSLinear(c_in, c_out),
            nn.LeakyReLU(0.2, True)
        ))
    self.prog_blocks.append(
        nn.Linear(int(in_channels * factors[-1]), 1)
    )


  def forward(self, z, text_emb):
    # pass the labels into a embedding layer
    # labels_embedding = self.embedding(y)
    # concat the embedded labels and the input tensor
    # x is a tensor of size (batch_size, 794)
    #print('disc inp shape', x.shape, len(self.prog_blocks), steps)
    #print('disc', z.shape, text_emb.shape)
    x = torch.cat([z, torch.squeeze(text_emb, 1)], dim=-1)
    out = self.init_lin(x)
    #print('disc prog init', out.shape)
    for step in range(len(self.prog_blocks)):
        out = self.prog_blocks[step](out)
        #print('prog out disc', out.shape)
    return out

class CGAN(pl.LightningModule):

  def __init__(self, latent_in_dim, text_emb_dim, latent_out_dim):
    super().__init__()
    self.latent_in_dim = latent_in_dim
    self.text_emb_dim = text_emb_dim
    self.latent_out_dim = latent_out_dim
    self.generator = Generator(latent_in_dim, text_emb_dim, latent_out_dim)
    self.discriminator = Discriminator()
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
    g_loss =  -torch.mean(fake_pred)

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
    
    
      fake_latent = self(z, text_emb).detach() 
      fake_pred = torch.squeeze(self.discriminator(fake_latent, text_emb))
    
          
      
      d_loss = torch.mean(fake_pred) - torch.mean(real_pred)
      
      return d_loss

    
  def configure_optimizers(self):
      
    g_optimizer = torch.optim.Adam(self.generator.parameters(), lr=0.0002)
    d_optimizer = torch.optim.Adam(self.discriminator.parameters(), lr=0.0002)
    
    n_critic = 5
    
    return (
            {'optimizer': g_optimizer, 'frequency': 1},
            {'optimizer': d_optimizer, 'frequency': n_critic}
        )


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
  
  