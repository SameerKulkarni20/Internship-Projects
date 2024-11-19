// Sigmoid activation and derivative
const sigmoid = (x) => 1 / (1 + Math.exp(-x));
const sigmoid_derivative = (x) => x * (1 - x);

// Simple GAN Components
class GAN {
    constructor() {
        this.gen_w = Math.random();
        this.disc_w = Math.random();
    }

    gen(noise) {
        return sigmoid(noise * this.gen_w);
    }

    disc(value) {
        return sigmoid(value * this.disc_w);
    }

    train(real_data, epochs = 1000, lr = 0.01) {
        for (let i = 0; i < epochs; i++) {
            // Fake data generation
            const noise = Math.random();
            const fake_data = this.gen(noise);

            // Discriminator update
            const real_pred = this.disc(real_data);
            const fake_pred = this.disc(fake_data);
            const d_grad = (real_pred - 1) * sigmoid_derivative(real_pred) + fake_pred * sigmoid_derivative(fake_pred);
            this.disc_w += lr * d_grad;

            // Generator update
            const g_grad = (1 - fake_pred) * sigmoid_derivative(fake_pred);
            this.gen_w += lr * g_grad;
        }
        return this.gen(Math.random());
    }
}

// Testing GAN
const gan = new GAN();
const fake_output = gan.train(1);
console.log("Generated data after training:", fake_output);
