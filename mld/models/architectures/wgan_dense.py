import torch
import torch.nn as nn
import pytorch_lightning as pl

class ResidualBlockDense(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.fc1 = nn.Linear(in_features, out_features)
        self.ln1 = nn.LayerNorm(out_features)
        self.relu = nn.LeakyReLU(0.2)
        self.fc2 = nn.Linear(out_features, out_features)
        self.ln2 = nn.LayerNorm(out_features)

        self.shortcut = nn.Linear(in_features, out_features) if in_features != out_features else nn.Identity()

    def forward(self, x):
        shortcut = self.shortcut(x)
        x = self.relu(self.ln1(self.fc1(x)))
        x = self.ln2(self.fc2(x))
        return self.relu(x + shortcut)


class Generator(nn.Module):
    def __init__(self, latent_in, text_in, latent_out, num_actions = None):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Linear(latent_in + text_in, 1024),
            nn.LayerNorm(1024),
            nn.LeakyReLU(0.2),
        )
        self.residual_block1 = ResidualBlockDense(1024, 512)
        self.residual_block2 = ResidualBlockDense(512, 256)
        self.output = nn.Linear(256, latent_out)
        self.num_actions = num_actions
        if self.num_actions is not None:
           self.embedding = nn.Embedding(num_actions, text_in)


    def forward(self, z, text_emb):
        if self.num_actions is not None:
           text_emb = self.embedding(text_emb)

        text_emb = text_emb.reshape(text_emb.shape[0], -1)
        x = torch.cat([z, text_emb], dim=-1)
        x = self.layer1(x)
        x = self.residual_block1(x)
        x = self.residual_block2(x)
        x = self.output(x)
        return x


class Discriminator(nn.Module):
    def __init__(self, latent_in, text_in, latent_out, num_actions = None):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Linear(latent_out + text_in, 1024),
            nn.LayerNorm(1024),
            nn.LeakyReLU(0.2),
        )
        self.residual_block1 = ResidualBlockDense(1024, 512)
        self.residual_block2 = ResidualBlockDense(512, 256)
        self.output = nn.Linear(256, 1)
        self.num_actions = num_actions
        if self.num_actions is not None:
           self.embedding = nn.Embedding(num_actions, text_in)


    def forward(self, x, text_emb):
        if self.num_actions is not None:
           text_emb = self.embedding(text_emb)

        text_emb = text_emb.reshape(text_emb.shape[0], -1)
        x = torch.cat([x, text_emb], dim=-1)
        x = self.layer1(x)
        x = self.residual_block1(x)
        x = self.residual_block2(x)
        x = self.output(x)
        return x

    
    
class CGAN(pl.LightningModule):

  def __init__(self, latent_in_dim, text_emb_dim, latent_out_dim, num_actions = None):
    super().__init__()
    self.latent_in_dim = latent_in_dim
    self.text_emb_dim = text_emb_dim
    self.latent_out_dim = latent_out_dim
    self.generator = Generator(latent_in_dim, text_emb_dim, latent_out_dim, num_actions)
    self.discriminator = Discriminator(latent_in_dim, text_emb_dim, latent_out_dim, num_actions)
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
    g_loss = -torch.mean(fake_pred)

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


    lambda_gp = 10  # Gradient penalty coefficient
    gp = self.gradient_penalty(fake_latent, x, text_emb) 
    d_loss = torch.mean(fake_pred) - torch.mean(real_pred) + lambda_gp * gp
    return d_loss

  def gradient_penalty(self, fake_pred, real_pred, text_emb):
      batch_size = real_pred.size(0)
      alpha = torch.rand(batch_size, 1).to(real_pred.device)
      alpha = alpha.expand_as(real_pred)

      interpolates = alpha * real_pred + ((1 - alpha) * fake_pred)
      # interpolates.requires_grad_(True)
      interpolates.requires_grad = True

      # print(interpolates)

      interpolated_score = self.discriminator(interpolates, text_emb)
      interpolated_score.requires_grad_(True)

      # print(interpolated_score)
      gradients = torch.autograd.grad(
          outputs=interpolated_score,
          inputs=interpolates,
          grad_outputs=torch.ones_like(interpolated_score),
          create_graph=True,
          retain_graph=True,
          materialize_grads=True,
      )[0]


      gradients = gradients.view(gradients.size(0), -1)
      gradient_norm = gradients.norm(2, dim=1)
      gradient_penalty = ((gradient_norm - 1) ** 2).mean()
      return gradient_penalty
   
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
  
       

    
    