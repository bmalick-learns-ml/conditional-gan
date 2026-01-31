import torch
import torchvision
import torchvision.transforms as T

from src.utils import one_hot
from src.model import MnistGenerator, MnistDiscriminator
from src.train import train_gan

if __name__=="__main__":
    transform = T.Compose([
        T.ToTensor(),
        T.Normalize((0.5), (0.5)),
        T.Lambda(lambda x: torch.flatten(x)),
        ])
    target_transform = T.Compose([
        T.Lambda(one_hot)
    ])

    data = torchvision.datasets.MNIST(root="data/mnist", train=True, download=True,
                                            transform=transform, target_transform=target_transform)

    batch_size = 100
    dataloader = torch.utils.data.DataLoader(
            dataset=data,
            batch_size=batch_size, shuffle=True)


    latent_dim = 100
    generator = MnistGenerator(latent_dim=latent_dim)
    discriminator = MnistDiscriminator()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    lr_D = 0.1
    lr_G = 0.1
    # num_epochs = 480
    num_epochs = 50
    print_every = 150
    visualize_each = 24

    fixed_noise = torch.normal(0., 1., size=(10, latent_dim))
    fixed_label = torch.cat([one_hot(i).view(1,-1) for i in range(10)], dim=0)


    train_gan(device=device, D=discriminator, G=generator, dataloader=dataloader, num_epochs=num_epochs,
            lr_G=lr_G, lr_D=lr_D, latent_dim=latent_dim, fixed_noise=fixed_noise, fixed_label=fixed_label,
            visualize=True, print_every=print_every, visualize_each=visualize_each)
