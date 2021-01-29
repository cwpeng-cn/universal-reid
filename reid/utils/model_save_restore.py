import os
import torch


def save_network(path, network, epoch_label, is_only_parameter=True):
    save_filename = 'net_%s.pth' % epoch_label
    save_path = os.path.join(path, save_filename)
    if is_only_parameter:
        state = network.state_dict()
        for key in state: state[key] = state[key].clone().cpu()
        torch.save(network.state_dict(), save_path)
    else:
        torch.save(network.detach.cpu(), save_path)


def restore_network(path, epoch, network=None):
    path = os.path.join(path, 'net_%s.pth' % epoch)
    if network is None:
        network = torch.load(path)
    else:
        network.load_state_dict(torch.load(path))
    if torch.cuda.is_available:
        network.cuda()
    return network
