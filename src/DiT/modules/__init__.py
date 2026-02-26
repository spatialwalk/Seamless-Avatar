
def create_denoise_net(denoise_network_type):
    if denoise_network_type == 'naive':
        from .naive_denoise_network import NaiveDenoisingNetwork
        denoising_net = NaiveDenoisingNetwork()
        
    elif denoise_network_type == 'dit':
        from .dit_denoise_network import DiTDenoiseNetwork
        denoising_net = DiTDenoiseNetwork()
    else:
        raise ValueError(
            f"Unknown denoising network type: {denoise_network_type}")
        
    return denoising_net