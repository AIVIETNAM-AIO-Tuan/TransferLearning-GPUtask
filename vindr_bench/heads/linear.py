# Trich NGUYEN VAN tu chexfound/eval/classification/utils.py (RPIDIAL/CheXFound @ 7e288d0, MIT — xem LICENSE): create_linear_input
# + LinearClassifier. Tep goc import chexfound.eval.utils / chexfound.distributed (khong chep) nen khong import nguyen tep duoc.
import torch
import torch.nn as nn


def create_linear_input(x_tokens_list, use_n_blocks, use_avgpool):
    intermediate_output = x_tokens_list[-use_n_blocks:]
    output = torch.cat([class_token for _, class_token in intermediate_output], dim=-1)
    if use_avgpool:
        output = torch.cat(
            (
                output,
                torch.mean(intermediate_output[-1][0], dim=1),  # patch tokens
            ),
            dim=-1,
        )
        output = output.reshape(output.shape[0], -1)
    return output.float()


class LinearClassifier(nn.Module):
    """Linear layer to train on top of frozen features"""

    def __init__(self, out_dim, use_n_blocks, use_avgpool, num_classes=1000, is_3d=False):
        super().__init__()
        self.out_dim = out_dim
        self.use_n_blocks = use_n_blocks
        self.use_avgpool = use_avgpool
        self.num_classes = num_classes
        self.linear = nn.Linear(out_dim, num_classes)
        self.linear.weight.data.normal_(mean=0.0, std=0.01)
        self.linear.bias.data.zero_()

    def forward(self, x):
        # output = torch.stack( # If 3D, take average of all slices.
        #     [create_linear_input(image, self.use_n_blocks, self.use_avgpool) for image in x]
        #     ).mean(dim=0)
        output = torch.cat(  # cat multiview cls tokens
            [create_linear_input(image, self.use_n_blocks, self.use_avgpool) for image in x],
            dim=-1
        )
        # output = output.squeeze()
        return self.linear(output).squeeze(-1)
