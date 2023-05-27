"""

Make GAN
https://github.com/eriklindernoren/PyTorch-GAN/tree/master/implementations
"""

from torchvision import datasets
import torchvision.transforms as transforms

import numpy as np
import os
import torch
import matplotlib.pyplot as plt


class GAN(torch.nn.Module):

    def __init__(self, noise=100, image_shape=(1, 28, 28)):
        super().__init__()

        GAN.noise = noise
        GAN.image_shape = image_shape

        # setting enviroment.
        GAN.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if GAN.device.type == 'cuda':
            print(f'"Cuda" is Available')
        else:
            print(f'"CPU" is Available.')

        model_gen = self.initialize_weights(GAN.Generator().to(GAN.device))
        model_dis = self.initialize_weights(GAN.Discriminator().to(GAN.device))
        dataloader = self.MNIST()

        # train
        generator, discriminator, history = self.train(model_gen, model_dis, dataloader)

        self.plot_loss_history(
            history['gen'],history['dis'],
            label1='Gen Loss', label2='Dis Loss', 
            figsize=(16,8), image_name='Pytorch_GAN',
        )

        # torch.save(generator.state_dict(), 'pytorch_weights_gen.pt')
        # torch.save(discriminator.state_dict(), 'pytorch_weights_dis.pt')


    class Generator(torch.nn.Module):
        def __init__(self):
            super(GAN.Generator, self).__init__()

            self.model = torch.nn.Sequential(
                self.FullyConnectNet(GAN.noise, 128, normalize=True),
                self.FullyConnectNet(128, 256),
                self.FullyConnectNet(256, 512),
                torch.nn.Linear(512, int(torch.prod(torch.tensor(GAN.image_shape)))),
                torch.nn.Tanh()
            )

        def forward(self, noise):
            image = self.model(noise)
            image = image.view(image.size(0), *GAN.image_shape)
            return image
        
        def FullyConnectNet(self, input_size, output_size, normalize=True):
            layers = []
            layers.append(torch.nn.Linear(input_size, output_size))
            if normalize:
                layers.append(torch.nn.BatchNorm1d(output_size, .8))
            layers.append(torch.nn.LeakyReLU(.2)) # 이미지는 음수값이 없으므로

            return torch.nn.Sequential(*layers)


    class Discriminator(torch.nn.Module):
        def __init__(self):
            super(GAN.Discriminator, self).__init__()

            self.model = torch.nn.Sequential(
                torch.nn.Linear(int(torch.prod(torch.tensor(GAN.image_shape))), 512),
                torch.nn.LeakyReLU(.2, inplace=True),
                torch.nn.Linear(512, 256),
                torch.nn.LeakyReLU(.2, inplace=True),
                torch.nn.Linear(256, 1),
                torch.nn.Sigmoid()
            )

        def forward(self, image):
            image_flat = image.view(image.size(0), -1).to(GAN.device)
            validity = self.model(image_flat)
            return validity


    def initialize_weights(self, model):
        for m in model.modules():
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.normal_(m.weight, 0, 0.01)
                torch.nn.init.constant_(m.bias, 0)
            elif isinstance(m, torch.nn.BatchNorm1d):
                torch.nn.init.normal_(m.weight, 1.0, 0.02)
                torch.nn.init.constant_(m.bias, 0)
        
        return model

    def MNIST(self, image_size=28, batch_size=64):
        os.makedirs('../data/mnist', exist_ok=True)
        dataloader = torch.utils.data.DataLoader(
                datasets.MNIST(
                '../data/mnist',
                train=True,
                download=True,
                transform=transforms.Compose(
                    [transforms.Resize(image_size), 
                    transforms.ToTensor(), 
                    transforms.Normalize([.5],[.5])]
                )
            ),
            batch_size=batch_size,
            shuffle=True,
        )
        return dataloader

    def plot_loss_history(
            self, loss1, loss2, label1='label 1', label2='label 2', 
            figsize=(16,8), title='Title', xlabel='Batch', ylabel='Loss', image_name='temp'
            ):
        
        plt.figure(figsize=figsize)
        plt.title(title)
        plt.plot(np.arange(len(loss1)),loss1, label=label1)
        plt.plot(np.arange(len(loss1)),loss2, label=label2)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.legend()
        plt.savefig(f'{image_name}.png')

    def train(self, generator, discriminator, dataloader, num_epochs=2, noise=100, lr=2e-4, b1=0.5, b2=.999):

        loss_history = {'gen':[], 'dis':[]}

        # Initialize binary cross-entropy loss and optimizers
        adversarial_loss = torch.nn.BCELoss()
        optimizer_G = torch.optim.Adam(generator.parameters(), lr=lr, betas=(b1, b2))
        optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(b1, b2))

        # Train the GAN for the specified number of epochs
        for epoch in range(num_epochs):
            for i, (imgs, _) in enumerate(dataloader):
                batch_size = imgs.shape[0]

                valid_labels = torch.ones(batch_size, 1).to(GAN.device)
                fake_labels = torch.zeros(batch_size, 1).to(GAN.device)

                # =======================
                # Train the generator
                # =======================

                # Generate fake images from random noise
                z = torch.randn(batch_size, noise, device=GAN.device)
                gen_imgs = generator(z)

                g_loss = adversarial_loss(discriminator(gen_imgs), valid_labels)

                optimizer_G.zero_grad()
                g_loss.backward()
                optimizer_G.step()

                # =======================
                # Train the discriminator
                # =======================
                real_loss = adversarial_loss(discriminator(imgs), valid_labels)
                fake_loss = adversarial_loss(discriminator(gen_imgs.detach()), fake_labels)

                d_loss = (real_loss + fake_loss) / 2

                optimizer_D.zero_grad() # zero_grad가 없으면 이전 값이 다음 학습에 영향을 미침.
                d_loss.backward()
                optimizer_D.step()

    
                loss_history['gen'].append(g_loss.item())
                loss_history['dis'].append(d_loss.item())

                # Print the progress
                if i % 10 == 0:
                    print(
                        "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]" 
                        % (epoch, num_epochs, i, len(dataloader), d_loss.item(), g_loss.item())
                    )

        return generator, discriminator, loss_history

        
if __name__ == '__main__':
    GAN()