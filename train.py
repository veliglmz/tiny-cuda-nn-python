import numpy as np
from timeit import default_timer as timer
import torch

from image_utils import write_image


def determine_samples(batch_size, n_coords, img):
    xs_and_ys_indices = []
    gt = []
    h, w = img.shape[:2]
    for bi in range(batch_size):
        r = np.random.randint(n_coords)
        xs_and_ys_indices.append(r * 2)
        xs_and_ys_indices.append(r * 2 + 1)
        gt.append(img[int(r // w), int(r % w)])
    return xs_and_ys_indices, np.asarray(gt)


class Trainer:
    def __init__(self, encoding, network, optimizer, criterion, n_epochs, 
                       img, xs_and_ys, n_coords, batch_size, output_folder):
        self.encoding = encoding
        self.network = network
        self.optimizer = optimizer
        self.criterion = criterion
        self.n_epochs = n_epochs
        self.img = img
        self.xs_and_ys = xs_and_ys
        self.n_coords = n_coords
        self.batch_size = 1 << batch_size
        self.output_folder = output_folder

    def train(self):
        n_levels = self.encoding.n_levels
        n_features = self.encoding.n_features
        self.network.train().cuda()
        # we only select pixels as many as the number of batch size for each epoch.
        for epoch in range(1, self.n_epochs+1):
            ts = timer()
            xs_and_ys_indices, gt = determine_samples(self.batch_size, self.n_coords, self.img)
            inputs_xs_and_ys = self.xs_and_ys[xs_and_ys_indices]

            encoding_forward_output, encoding_backward_output = self.encoding.forward(inputs_xs_and_ys, self.batch_size)
            encoding_forward_output = encoding_forward_output.copy_to_host()
            encoding_backward_output = encoding_backward_output.copy_to_host()
            # reshape the outputs of the encoding to give them to the network.
            encoding_forward_output = encoding_forward_output.reshape(-1, n_levels * n_features).astype(np.float32)
            encoding_backward_output = encoding_backward_output.reshape(-1, n_levels, self.encoding.n_backward_contents).astype(np.float32)

            self.optimizer.zero_grad()

            encoding_outputs = torch.from_numpy(encoding_forward_output).cuda()
            gt = torch.from_numpy(gt).cuda()
            # to update the grid params, we need to derivatives of the input layer.
            encoding_outputs.requires_grad = True

            network_outputs = self.network(encoding_outputs)
            loss = self.criterion(network_outputs, gt)
            loss.backward()

            inputs_grad = self.network.inputs.grad
            # backward for grids
            self.encoding.update_grid_params(self.batch_size, inputs_grad, encoding_backward_output)
            self.optimizer.step()
            te = timer()
            
            if epoch % 100 == 0:
                print(f"Epoch {epoch} | # of samples: {self.batch_size} | Time: {te - ts: .2f}")
                self.inference(name=str(epoch))

    def inference(self, name=""):
        n_levels = self.encoding.n_levels
        n_features = self.encoding.n_features
        h, w = self.img.shape[:2]
        self.network.eval()
        with torch.no_grad():
            encoding_forward_outputs, _ = self.encoding.forward(self.xs_and_ys, self.n_coords, is_inference=True)
            encoding_forward_outputs = encoding_forward_outputs.copy_to_host()

            encoding_forward_outputs = encoding_forward_outputs.reshape(-1, n_levels * n_features).astype(np.float32)
            encoding_outputs = torch.from_numpy(encoding_forward_outputs).cuda()
            
            network_outputs = []
            for b in range(0, encoding_outputs.size(0), self.batch_size):
                network_output = self.network(encoding_outputs[b:b+self.batch_size])
                network_outputs.append(network_output.cpu().detach().numpy())

            network_outputs = np.concatenate(network_outputs, axis=0)

            output_img = np.reshape(network_outputs, (h, w, 3))
            write_image(f"{self.output_folder}/inference_{name}.jpg", output_img)
        print("Inference is done.")
