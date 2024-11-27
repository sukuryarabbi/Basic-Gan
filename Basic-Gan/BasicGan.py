import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision import transforms
import matplotlib.pyplot as plt
from torchvision.utils import make_grid

class Generator(nn.Module):
  def __init__(self,noise_dim,im_dim=784,hidden_layer=128):
    super().__init__()
    self.gen = nn.Sequential(
        self.get_generator_block(noise_dim,hidden_layer),
        self.get_generator_block(hidden_layer,hidden_layer*2),
        self.get_generator_block(hidden_layer*2,hidden_layer*4),
        self.get_generator_block(hidden_layer*4,hidden_layer*8),
        self.get_generator_block(hidden_layer*8,im_dim,final_layer=True),
    )

  def get_generator_block(self,input_dim,output_dim,final_layer = False):

      if not final_layer:
        return nn.Sequential(
          nn.Linear(input_dim,output_dim),
          nn.BatchNorm1d(output_dim),
          nn.ReLU()
        )
      else:
        return nn.Sequential(
            nn.Linear(input_dim,output_dim),
            nn.Sigmoid()
        )

  def get_gen_loss(self,gen,disc,criterion,noise_dim,num_images,device):
    noise = torch.randn(num_images,noise_dim,device=device)
    fake_img = gen(noise)
    fake_pred = disc(fake_img)
    fake_target = torch.ones_like(fake_pred,device=device)
    gen_loss = criterion(fake_pred,fake_target)

    return gen_loss

  def forward(self,noise):
    return self.gen(noise)

  def get_noise(self,n_samples,noise_dim,device):
    return torch.randn(n_samples,noise_dim).to(device)

class Discriminator(nn.Module):
  def __init__(self,im_dim=784,hidden_layer=128):
    super().__init__()
    self.disc = nn.Sequential(
        self.get_disc_block(im_dim,hidden_layer*4),
        self.get_disc_block(hidden_layer*4,hidden_layer*2),
        self.get_disc_block(hidden_layer*2,hidden_layer),
        self.get_disc_block(hidden_layer,1,final_layer=True)
    )

  def get_disc_block(self,input_dim,output_dim,final_layer = False):

    if not final_layer:
      return nn.Sequential(
          nn.Linear(input_dim,output_dim),
          nn.LeakyReLU(0.2)
      )

    else:
      return nn.Sequential(
          nn.Linear(input_dim,output_dim)
      )

  def get_disc_loss(self,gen,disc,noise_dim,num_images,real,criterion,device):

    noise = torch.randn(num_images,noise_dim,device=device)
    fake_img = gen(noise).detach()
    fake_img_pred = disc(fake_img)
    fake_img_target = torch.zeros_like(fake_img_pred,device=device)
    fake_img_loss = criterion(fake_img_pred,fake_img_target)

    real_img_pred = disc(real)
    real_img_target = torch.ones_like(real_img_pred,device=device)
    real_img_loss = criterion(real_img_pred,real_img_target)


    disc_loss = 0.5*(fake_img_loss+real_img_loss)

    return disc_loss

  def forward(self,img):
    return self.disc(img)

def show_tensor_images(image_tensor, num_images=25, size=(1, 28, 28)):

    image_unflat = image_tensor.detach().cpu().view(-1, *size)
    image_grid = make_grid(image_unflat[:num_images], nrow=5)
    plt.imshow(image_grid.permute(1, 2, 0).squeeze())
    plt.show()

def train() :

  criterion = nn.BCEWithLogitsLoss()
  lr = 0.00001
  n_epochs = 200
  batch_size = 32
  noise_dim = 64
  display_step = 5000
  mean_disc_loss = 0
  mean_gen_loss = 0
  cur_step = 0
  device = "cuda" if torch.cuda.is_available() else "cpu"


  data_loader = DataLoader(MNIST("." , download = True , transform=transforms.ToTensor()),batch_size=batch_size,shuffle = True)


  gen = Generator(noise_dim=noise_dim).to(device=device)
  disc = Discriminator().to(device=device)
  gen_opt = torch.optim.Adam(gen.parameters(),lr=lr)
  disc_opt = torch.optim.Adam(disc.parameters(),lr=lr)

  for epoch in range(n_epochs):
    for bathc_idx ,(img,labels) in enumerate(data_loader):

      cur_batch_size = img.size(0)
      real = img.view(cur_batch_size,-1).to(device=device)

      #Train Discriminator
      disc_opt.zero_grad()
      disc_loss = disc.get_disc_loss(gen,disc,noise_dim,cur_batch_size,real,criterion,device=device)
      disc_loss.backward(retain_graph=True)
      disc_opt.step()

      #Train Generator
      gen_opt.zero_grad()
      gen_loss = gen.get_gen_loss(gen,disc,criterion,noise_dim,cur_batch_size,device=device)
      gen_loss.backward()
      gen_opt.step()


      mean_disc_loss += disc_loss.item() / display_step
      mean_gen_loss += gen_loss.item() / display_step

      if cur_step % display_step == 0 and cur_step > 0 :
          print("step : {} and Generator loss : {} and disciriminator loss : {}".format(cur_step, mean_gen_loss,mean_disc_loss))
          fake_noise = gen.get_noise(cur_batch_size,noise_dim,device=device)
          fake = gen(fake_noise)
          show_tensor_images(fake)
          show_tensor_images(real)
          mean_gen_loss = 0
          mean_disc_loss = 0

      cur_step+=1
