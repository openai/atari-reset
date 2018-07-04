from atari_reset.nn import Reshape, Conv2d, WavenetSeparableConv2d, GlobalMaxPool, ReLU, Dense, GroupNorm2d

def make_wavenet_policy(n_actions):
    # input is TxNx105x80
    model = Reshape((-1,105,80,1)) # T*N,105,80,1
    model += Conv2d(num_filters_in=1, num_filters_out=64, filter_size=[7,5], stride=[2,2]) # 53,40,64
    model += Reshape((-1, 'nenv', 53, 40, 64)) # T,N,53,40,64
    model += ReLU()
    model += WavenetSeparableConv2d(num_filters_in=64, channel_multiplier=2, filter_size=[5,5], stride=[2,2], delay=4) # T-4,N,27,20,64
    model += ReLU()
    model += WavenetSeparableConv2d(num_filters_in=128, channel_multiplier=2, filter_size=[5,5], stride=[2,2], delay=8) # T-12,N,14,10,128
    model += ReLU()
    model += WavenetSeparableConv2d(num_filters_in=256, channel_multiplier=2, filter_size=[5,5], stride=[2,2], delay=16) # T-28,N,7,5,256
    model += ReLU()
    model += WavenetSeparableConv2d(num_filters_in=512, channel_multiplier=2, filter_size=[5,3], stride=[1,1], delay=32) # T-60,N,7,5,512
    model += Reshape((-1,7,5,1024)) # (T-60)*N,7,5,1024
    model += GlobalMaxPool() # (T-60)*N,1024
    model += GroupNorm2d(C=1024)
    model += ReLU()
    model += Dense(num_in=1024, num_out=n_actions+1)
    model += Reshape((-1,'nenv',n_actions+1))

    return model
