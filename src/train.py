import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn


def update_discriminator(x, y, z, D, G, criterion, trainer_D):
    """Update discriminator."""
    batch_size = x.shape[0]
    ones = torch.ones((batch_size,), device=x.device)
    zeros = torch.zeros((batch_size,), device=x.device)

    trainer_D.zero_grad()

    real_y = D(x, y)
    fake_y = D(G(z, y), y)
    loss_D = (criterion(real_y, ones.reshape(real_y.shape)) +
                          criterion(fake_y, zeros.reshape(fake_y.shape))) / 2
    loss_D.backward()
    trainer_D.step()
    return loss_D


def update_generator(z, y, D, G, criterion, trainer_G):
    """Update generator."""
    batch_size = z.shape[0]
    ones = torch.ones((batch_size,), device=z.device)

    trainer_G.zero_grad()

    fake_y = D(G(z, y), y)
    loss_G = criterion(fake_y, ones.reshape(fake_y.shape))
    loss_G.backward()
    trainer_G.step()
    return  loss_G


def train_gan(device, D, G, dataloader, num_epochs, lr_G, lr_D, latent_dim, fixed_noise, fixed_label, visualize=False, visualize_each=100, print_every=25):
    loss = nn.BCEWithLogitsLoss(reduction="mean")
    for w in D.parameters(): nn.init.normal_(w, 0., 0.02)
    for w in G.parameters(): nn.init.normal_(w, 0., 0.02)
    D = D.to(device)
    G = G.to(device)
    # trainer_D = torch.optim.Adam(discriminator.parameters(), lr=lr_D, betas=(0.5,0.999))
    # trainer_G = torch.optim.Adam(generator.parameters(), lr=lr_G, betas=(0.5,0.999))
    trainer_D = torch.optim.SGD(D.parameters(), lr=lr_D, momentum=0.5)
    trainer_G = torch.optim.SGD(G.parameters(), lr=lr_G, momentum=0.5)

    scheduler_G = torch.optim.lr_scheduler.ExponentialLR(trainer_G, 1/1.00004)
    scheduler_D = torch.optim.lr_scheduler.ExponentialLR(trainer_D, 1/1.00004)

    fixed_label = fixed_label.to(device)
    fixed_noise = fixed_noise.to(device)


    metrics = []
    os.makedirs("visualizations", exist_ok=True)
    for epoch in range(num_epochs):
        G.train()
        D.train()

        loss_D = 0
        loss_G = 0
        num_instances = 0
        for step_num, batch in enumerate(dataloader):
            X, Y = batch
            X = X.to(device)
            Y = Y.to(device)
            batch_size = X.shape[0]
            num_instances += batch_size
            Z = torch.normal(0, 1, size=(batch_size, latent_dim)).to(device)
            loss_D += update_discriminator(X, Y, Z, D, G, loss, trainer_D).item() * batch_size
            loss_G += update_generator(Z, Y, D, G, loss, trainer_G).item() * batch_size
            scheduler_D.step()
            scheduler_G.step()

            # if step_num % print_every == 0:
            #     print(f"[Epoch {epoch+1}/{num_epochs}] [Step {step_num}/{len(dataloader)}] loss_D: {loss_D/num_instances:.5f}, loss_G: {loss_G/num_instances:.5f}, lr_D: {scheduler_D.get_last_lr()[0]:.5f}, lr_G:{scheduler_G.get_last_lr()[0]:.5f}")
        # scheduler_D.step()
        # scheduler_G.step()

        loss_G /= num_instances
        loss_D /= num_instances
        metrics.append([loss_D, loss_G])
        print(f"[Epoch {epoch+1}/{num_epochs}] loss_D: {loss_D:.5f}, loss_G: {loss_G:.5f}, lr_D: {scheduler_D.get_last_lr()[0]:.5f}, lr_G:{scheduler_G.get_last_lr()[0]:.5f}")

        G.eval()
        D.eval()

        if visualize:
            # Z = torch.normal(0, 1, size=(100, latent_dim))
            # fake_data = torch.sigmoid(G(fixed_noise, fixed_label)).detach().cpu().numpy()
            fake_data = G(fixed_noise, fixed_label).detach().cpu().numpy()
            fig, axes = plt.subplots(2, 5, figsize=(19.2, 10.8))
            axes = axes.ravel()
            for i in range(len(fake_data)):
                label = fixed_label[i].argmax().item()
                ax = axes[i]
                ax.imshow(fake_data[i].reshape((28,28)), cmap="gray")
                ax.axis("off")
                ax.set_title(f"Label: {label}")
            fig.suptitle(f"Epoch {epoch:03d}")
            fig.savefig(f"visualizations/{epoch:02d}.png")
            # if epoch%visualize_each==0: plt.show()
            plt.close()

    metrics = np.array(metrics)

    plt.plot(metrics[:, 0], label="discriminator")
    plt.plot(metrics[:, 1], label="generator", linestyle="--")
    plt.legend()
    plt.ylabel("loss")
    plt.show()
