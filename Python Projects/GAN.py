import random, math

# Sigmoid activation and derivative
sigmoid = lambda x: 1 / (1 + math.exp(-x))
sigmoid_derivative = lambda x: x * (1 - x)


# Simple GAN Components
class GAN:
    def __init__(self):
        self.gen_w, self.disc_w = random.random(), random.random()

    def gen(self, noise):
        return sigmoid(noise * self.gen_w)

    def disc(self, value):
        return sigmoid(value * self.disc_w)

    def train(self, real_data, epochs=1000, lr=0.01):
        for _ in range(epochs):  # fake data generation
            noise = random.random()
            fake_data = self.gen(noise)

            # Discriminator update
            real_pred, fake_pred = self.disc(real_data), self.disc(fake_data)
            d_loss = -math.log(real_pred) - math.log(1 - fake_pred)
            d_grad = (real_pred - 1) * sigmoid_derivative(
                real_pred
            ) + fake_pred * sigmoid_derivative(fake_pred)
            self.disc_w += lr * d_grad

            # Generator update
            g_loss = -math.log(fake_pred)
            g_grad = (1 - fake_pred) * sigmoid_derivative(fake_pred)
            self.gen_w += lr * g_grad

        return self.gen(random.random())


# Testing GAN
gan = GAN()
fake_output = gan.train(real_data=1)
print("Generated data after training:", fake_output)
