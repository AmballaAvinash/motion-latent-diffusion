import torch
import torch.nn as nn
import pytorch_lightning as pl

#factors = [1, 1, 1, 1, 1 / 2, 1 / 4, 1 / 8]
#factors = [1, 1, 1, 1, 1, 1, 1, 1, 1]
#factors = [2, 4, 8, 16, 32, 16, 8]
factors = [1, 1/2, 1/4, 1/8, 1/256]

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


class AdaIN(nn.Module):

    def __init__(self, channels, w_dim):
        super().__init__()
        self.instance_norm = nn.InstanceNorm1d(channels)
        self.style_scale = WSLinear(w_dim, channels)
        self.style_bias = WSLinear(w_dim, channels)

    def forward(self, x, w):
        #print('adain========>', x.shape, w.shape)
        x = self.instance_norm(x)
        style_scale = self.style_scale(w)
        style_bias = self.style_bias(w)
        out = style_scale * x + style_bias
        #print('st', style_scale.shape, x.shape, style_bias.shape)
        return out

class InjectNoise(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(1, channels))

    def forward(self, x):
        #print('inoise', x.shape, self.weight.shape, x.device)
        noise = torch.randn((x.shape[0], x.shape[1]), device=x.device)
        out = x + self.weight * noise
        #print('out', out.shape)
        return out


class GenBlock(nn.Module):
    def __init__(self, in_channels, out_channels, w_dim):
        super(GenBlock, self).__init__()
        #self.conv1 = WSConv1d(in_channels, out_channels)
        #self.conv2 = WSConv1d(out_channels, out_channels)
        self.lin1 = WSLinear(in_channels, out_channels)
        self.lin2 = WSLinear(out_channels, out_channels)
        self.leaky = nn.LeakyReLU(0.2, inplace=True)
        self.inject_noise1 = InjectNoise(out_channels)
        self.inject_noise2 = InjectNoise(out_channels)
        self.adain1 = AdaIN(out_channels, w_dim)
        self.adain2 = AdaIN(out_channels, w_dim)

    def forward(self, x, w):
        #print('gb inp', x.shape, w.shape, self.conv1(x).shape)
        x = self.adain1(self.leaky(self.inject_noise1(self.lin1(x))), w)
        x = self.adain2(self.leaky(self.inject_noise2(self.lin1(x))), w)
        return x

class Generator(nn.Module):
  '''
  Generator class in a CGAN. Accepts a noise tensor (latent dim 100)
  and a label tensor as input as outputs another tensor of size 784.
  Objective is to generate an output tensor that is indistinguishable 
  from the real MNIST digits.
  '''

  def __init__(self, latent_in, text_in, latent_out, in_channels = 256, init_channels = 32):
    super().__init__()
    #in_channels = latent_in + text_in
    print('args', latent_in, text_in, latent_out, in_channels)
    self.in_channels = in_channels
    #self.starting_constant = torch.ones((1, in_channels))
    self.map = MappingNetwork(text_in + latent_in, latent_out)
    self.initial_adain1 = AdaIN(in_channels, latent_out)
    self.initial_adain2 = AdaIN(in_channels, latent_out)
    self.initial_noise1 = InjectNoise(in_channels)
    self.initial_noise2 = InjectNoise(in_channels)
    #self.initial_conv = nn.Conv1d(in_channels, in_channels, kernel_size=1, stride = 1)
    self.initial_lin = nn.Linear(in_channels, in_channels)
    self.leaky = nn.LeakyReLU(0.2, inplace = True)

    '''
    self.layer1 = nn.Sequential(nn.Linear(in_features=latent_in+text_in, out_features=1024),
                                nn.LeakyReLU())
    self.layer2 = nn.Sequential(nn.Linear(in_features=1024, out_features=512),
                                nn.LeakyReLU())
    self.output = nn.Sequential(nn.Linear(in_features=512, out_features=latent_out),
                                )
    '''

  def forward(self, z, text_emb):

    # x is a tensor of size (batch_size, 110)
    # reshapeing text_emb
    #print('z, txt', z.shape, text_emb.shape)
    text_emb = text_emb.reshape(text_emb.shape[0],-1)
    x = torch.cat([z, text_emb], dim=-1)
    #print('x', x.shape, z.shape)
    noise = torch.randn(x.shape[0], x.shape[1]).to(torch.device("cuda"))
    w = self.map(noise)
    starting_constant = nn.Parameter(torch.ones(w.shape)).to(torch.device("cuda"))
    #self.starting_constant = nn.Parameter(self.starting_constant.tile((z.shape[0],)).reshape(z.shape[0], -1)).to(torch.device("cuda"))
    #print('w', w.shape, starting_constant.shape)
    x = self.initial_adain1(self.initial_noise1(starting_constant), w)
    #x = self.initial_conv(x)
    #print('x init adain1', x.shape)
    x = self.initial_lin(x)
    #print('x out', x.shape)
    out = self.initial_adain2(self.leaky(self.initial_noise2(x)), w)
    #print('out', out.shape)
    
    return out


class Discriminator(nn.Module):
  '''
  Discriminator class in a CGAN. Accepts a tensor of size 784 and
  a label tensor as input and outputs a tensor of size 1,
  with the predicted class probabilities (generated or real data)
  '''

  def __init__(self, text_in = 768, in_channels = 256):
    super(Discriminator, self).__init__()
    
    self.prog_blocks, self.xyz_layers = nn.ModuleList([]), nn.ModuleList([])
    self.leaky = nn.LeakyReLU(0.2)
    self.init_lin = nn.Linear(in_channels + text_in, in_channels)
    for i in range(len(factors) - 1):
        conv_in = int(in_channels * factors[i])
        conv_out = int(in_channels * factors[i + 1])
        #print('disc setup prog', conv_in, conv_out)
        self.prog_blocks.append(WSLinear(conv_in, conv_out))

    self.sigmoid = nn.Sigmoid()

  def forward(self, x, text_emb, steps = 4):
    # pass the labels into a embedding layer
    # labels_embedding = self.embedding(y)
    # concat the embedded labels and the input tensor
    # x is a tensor of size (batch_size, 794)
    #print('disc inp shape', x.shape, len(self.prog_blocks), steps)
    cur_step = len(self.prog_blocks) - steps
    #print('disc', z.shape, text_emb.shape)
    text_emb = text_emb.reshape(text_emb.shape[0],-1)
    x = torch.cat([x, text_emb], dim=-1)   
    out = self.init_lin(x)
    for step in range(cur_step, len(self.prog_blocks)):
        out = self.prog_blocks[step](out)
    out = self.sigmoid(out)
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
    # g_loss = self.BCE_loss(fake_pred, torch.ones_like(fake_pred))
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
      # real_loss = self.BCE_loss(real_pred, torch.ones_like(real_pred))
    
    
      fake_latent = self(z, text_emb).detach() 
      fake_pred = torch.squeeze(self.discriminator(fake_latent, text_emb))
      # fake_loss = self.BCE_loss(fake_pred, torch.zeros_like(fake_pred))
    
    
      # d_loss = (real_loss + fake_loss) / 2
      
      
      d_loss = torch.mean(fake_pred) - torch.mean(real_pred)
      
      return d_loss

    
  def configure_optimizers(self):
    g_optimizer = torch.optim.Adam(self.generator.parameters(), lr=0.0001)
    d_optimizer = torch.optim.Adam(self.discriminator.parameters(), lr=0.0001)
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
  
  