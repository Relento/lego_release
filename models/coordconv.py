import torch
import torch.nn as nn


class AddCoords3D(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        """
        :param x: shape (N, C, X, Y, Z)
        :return:
        """

        # batch_size
        n, _, x, y, z = input.shape

        # Coordinate axes:
        x_coord = torch.linspace(-1, 1, x, device=input.device)
        x_coord = x_coord[None, None, :, None, None]
        x_coord = x_coord.expand(n, 1, x, y, z)

        y_coord = torch.linspace(-1, 1, y, device=input.device)
        y_coord = y_coord[None, None, None, :, None]
        y_coord = y_coord.expand(n, 1, x, y, z)

        z_coord = torch.linspace(-1, 1, z, device=input.device)
        z_coord = z_coord[None, None, None, None, :]
        z_coord = z_coord.expand(n, 1, x, y, z)

        output = torch.cat([input, x_coord, y_coord, z_coord], dim=1)
        return output


if __name__ == '__main__':
    from tqdm import tqdm

    for i in tqdm(range(10)):
        addcoord = AddCoords3D()
        size = torch.randint(5, 40, (3,))
        xs = torch.linspace(-1, 1, size[0])
        ys = torch.linspace(-1, 1, size[1])
        zs = torch.linspace(-1, 1, size[2])
        batch_num = torch.randint(5, 40, (1,)).item()
        ch_num = torch.randint(5, 40, (1,)).item()

        input = torch.rand((batch_num, ch_num) + tuple(size))
        for i in range(100):
            b_ind = torch.randint(0, batch_num, (1,))
            c_ind = torch.randint(0, ch_num, (1,))

            x_ind = torch.randint(0, size[0], (1,))
            y_ind = torch.randint(0, size[1], (1,))
            z_ind = torch.randint(0, size[2], (1,))
            o = addcoord(input)
            assert torch.allclose(o[b_ind, ch_num:, x_ind, y_ind, z_ind],
                                  torch.as_tensor([xs[x_ind], ys[y_ind], zs[z_ind]]))
