import torch.nn.functional as F
import torch.nn as nn
import einops
import torch
        
def get_backbone(backbone, num_classes, batch_size, device):
    if backbone == 'resnet18':
        net = ResNet18(num_classes).to(device)
    elif backbone == 'mlp':
        net = MLP_400_400(num_classes).to(device)
    elif backbone == 'mobilenetv2':
        net = MobileNetV2(num_classes).to(device)
    elif backbone == 'vgg16':
        net = VGG16('like', num_classes).to(device)
    elif backbone == 'vit-base':
        net = ViT(num_classes, batch_size, patch_size=16, n_channels=3, latent_size=768, num_encoders=12, num_heads=12, dropout=0.1).to(device)
    elif backbone == 'vit-small':
        net = ViT(num_classes, batch_size, patch_size=16, n_channels=3, latent_size=384, num_encoders=12, num_heads=6, dropout=0.1).to(device)
    elif backbone == 'vit-tiny':
        net = ViT(num_classes, batch_size, patch_size=16, n_channels=3, latent_size=192, num_encoders=12, num_heads=3, dropout=0.1).to(device)
    elif backbone == 'vit-micro':
        net = ViT(num_classes, batch_size, patch_size=4, n_channels=3, latent_size=192, num_encoders=6, num_heads=3, dropout=0.1).to(device)

    return net

############################################################################################################
################################################ ResNet ####################################################
############################################################################################################

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes, affine=False, track_running_stats=False)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes, affine=False, track_running_stats=False)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes, affine=False, track_running_stats=False)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes, affine=False, track_running_stats=False)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes, affine=False, track_running_stats=False)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes, affine=False, track_running_stats=False)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes, affine=False, track_running_stats=False)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes):
        super(ResNet, self).__init__()
        _outputs = [64, 128, 256, 512]
        #_outputs = [21, 42, 85, 170]
        self.in_planes = _outputs[0]

        self.conv1 = nn.Conv2d(3, _outputs[0], kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(_outputs[0], affine=False, track_running_stats=False)
        self.layer1 = self._make_layer(block, _outputs[0], num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, _outputs[1], num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, _outputs[2], num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, _outputs[3], num_blocks[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(_outputs[3]*block.expansion, num_classes, bias=False)
        self.embedding_recorder = EmbeddingRecorder(record_embedding=False)


    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)
    
    def get_last_layer(self):
        return self.classifier
    
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        features = out.view(out.size(0), -1)
        #out = self.embedding_recorder(features)
        out = self.classifier(features)

        return out

def ResNet18(c=10):
    return ResNet(BasicBlock, [2,2,2,2], c)


############################################################################################################
################################################ ViT #######################################################
############################################################################################################

class InputEmbedding(nn.Module):
    def __init__(self, patch_size, n_channels, latent_size, batch_size):
        super(InputEmbedding, self).__init__()
        self.latent_size = latent_size
        self.patch_size = patch_size
        self.n_channels = n_channels
        self.batch_size = batch_size
        self.input_size = self.patch_size * self.patch_size * self.n_channels

        # Linear projection
        self.linearProjection = nn.Linear(self.input_size, self.latent_size)
        # Random initialization of [class] token that is prepended to the linear projection vector.
        self.class_token = nn.Parameter(torch.randn(self.batch_size, 1, self.latent_size))
        # Positional embedding
        self.pos_embedding = nn.Parameter(torch.randn(self.batch_size, 1, self.latent_size))


    def forward(self, input):
        # Re-arrange image into patches.
        patches = einops.rearrange(input, 'b c (h h1) (w w1) -> b (h w) (h1 w1 c)', 
                                   h1=self.patch_size, 
                                   w1=self.patch_size)
        linear_projection = self.linearProjection(patches)
        b, n, _ = linear_projection.shape

        # Prepend the [class] token to the original linear projection
        linear_projection = torch.cat((self.class_token, linear_projection), dim=1)
        pos_embed = einops.repeat(self.pos_embedding, 'b 1 d -> b m d', m=n+1)
        # Add positional embedding to linear projection
        linear_projection += pos_embed

        return linear_projection
    
class EncoderBlock(nn.Module):
    def __init__(self, latent_size, num_heads, dropout):
        super(EncoderBlock, self).__init__()
        self.latent_size = latent_size
        self.num_heads = num_heads
        self.dropout = dropout

        # Normalization layer
        self.norm = nn.LayerNorm(self.latent_size)
        # Multi-Head Attention layer
        self.multihead = nn.MultiheadAttention(self.latent_size, self.num_heads, dropout=self.dropout)          
        # MLP_head layer in the encoder which is latent_size*4.
        self.enc_MLP = nn.Sequential(nn.Linear(self.latent_size, self.latent_size*4),
                                     nn.GELU(),
                                     nn.Dropout(self.dropout),
                                     nn.Linear(self.latent_size*4, self.latent_size),
                                     nn.Dropout(self.dropout)
                                     )

    def forward(self, embedded_patches):
        # First sublayer: Norm + Multi-Head Attention + residual connection
        firstNorm_out = self.norm(embedded_patches)
        attention_output = self.multihead(firstNorm_out, firstNorm_out, firstNorm_out)[0]  # because returns 'Tuple[attention_output, attention_output_weights]'
        first_sublayer_output = attention_output + embedded_patches
        # Second sublayer: Norm + enc_MLP (Feed forward)
        secondNorm_out = self.norm(first_sublayer_output)
        ff_output = self.enc_MLP(secondNorm_out)
        second_sublayer_output = ff_output + first_sublayer_output
        # Return the output of the second residual connection
        return second_sublayer_output


class ViT(nn.Module):
    def __init__(self, num_classes, batch_size, patch_size=16, n_channels=3, latent_size=768, num_encoders=12, num_heads=12, dropout=0.1):
        super(ViT, self).__init__()
        #base_lr = 10e-3         Base LR
        #weight_decay = 0.03     Weight decay for ViT-Base (on ImageNet-21k)

        # Create a embeddings
        self.embedding = InputEmbedding(patch_size, n_channels, latent_size, batch_size)
        # Create a stack of encoder layers
        self.encStack = nn.ModuleList([EncoderBlock(latent_size, num_heads, dropout) for i in range(num_encoders)])
        # MLP_head at the classification stage has two version:
        #(i) 'one hidden layer and one classifier layer' -> at pre-training time,
        #(ii) ' only a one classifier layer' -> at fine-tuning time'.
        self.MLP_head = nn.Sequential(nn.LayerNorm(latent_size),
                                      nn.Linear(latent_size, latent_size),
                                      nn.Linear(latent_size, num_classes)
                                      )

    def forward(self, input):
        # Apply input embedding (patchify + linear projection + position embedding)
        # to the input image passed to the model
        enc_output = self.embedding(input)
        # Loop through all the encoder layers
        for enc_layer in self.encStack:
            enc_output = enc_layer.forward(enc_output)
        # Extract the output embedding information of the [class] token
        cls_token_embedding = enc_output[:, 0]
        # Finally, return the classification vector for all image in the batch
        return self.MLP_head(cls_token_embedding)



class AlexNet(nn.Module):
    """AlexNet with batch normalization and without pooling.

    This is an adapted version of AlexNet as taken from
    SNIP: Single-shot Network Pruning based on Connection Sensitivity,
    https://arxiv.org/abs/1810.02340

    There are two different version of AlexNet:
    AlexNet-s (small): Has hidden layers with size 1024
    AlexNet-b (big):   Has hidden layers with size 2048

    Based on https://github.com/mi-lad/snip/blob/master/train.py
    by Milad Alizadeh.
    """

    def __init__(self, config='s', num_classes=1000, save_features=False, bench_model=False):
        super(AlexNet, self).__init__()
        self.save_features = save_features
        self.feats = []
        self.densities = []
        self.bench = None if not bench_model else SparseSpeedupBench()

        factor = 1 if config=='s' else 2
        self.features = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=11, stride=2, padding=2, bias=True),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
            nn.Conv2d(96, 256, kernel_size=5, stride=2, padding=2, bias=True),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 384, kernel_size=3, stride=2, padding=1, bias=True),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 384, kernel_size=3, stride=2, padding=1, bias=True),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, stride=2, padding=1, bias=True),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.classifier = nn.Sequential(
            nn.Linear(256, 1024*factor),
            nn.BatchNorm1d(1024*factor),
            nn.ReLU(inplace=True),
            nn.Linear(1024*factor, 1024*factor),
            nn.BatchNorm1d(1024*factor),
            nn.ReLU(inplace=True),
            nn.Linear(1024*factor, num_classes),
        )

    def forward(self, x):
        for layer_id, layer in enumerate(self.features):
            if self.bench is not None and isinstance(layer, nn.Conv2d):
                x = self.bench.forward(layer, x, layer_id)
            else:
                x = layer(x)

            if self.save_features:
                if isinstance(layer, nn.ReLU):
                    self.feats.append(x.clone().detach())
                if isinstance(layer, nn.Conv2d):
                    self.densities.append((layer.weight.data != 0.0).sum().item()/layer.weight.numel())

        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

class MLP_400_400(nn.Module):
    """
    Simple NN with hidden layers [400, 400]
    """
    def __init__(self, num_classes=10, save_features=None, bench_model=False):
        super(MLP_400_400, self).__init__()
        self.fc1 = nn.Linear(28*28, 400, bias=False)
        self.fc2 = nn.Linear(400, 400, bias=False)
        self.fc3 = nn.Linear(400, num_classes, bias=False)
        self.embedding_recorder = EmbeddingRecorder(record_embedding=False)
        self.mask = None

    def get_last_layer(self):
        return self.fc3
    
    def forward(self, x):
        x0 = x.view(-1, 28*28)
        x1 = F.relu(self.fc1(x0))
        x2 = F.relu(self.fc2(x1))
        x2 = self.embedding_recorder(x2)
        x3 = self.fc3(x2)
        return x3

class MLP_CIFAR10(nn.Module):
    def __init__(self, save_features=None, bench_model=False):
        super(MLP_CIFAR10, self).__init__()

        self.fc1 = nn.Linear(3*32*32, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 10)
        self.embedding_recorder = EmbeddingRecorder(record_embedding=False)

    def get_last_layer(self):
        return self.fc3
    
    def forward(self, x):
        x0 = F.relu(self.fc1(x.view(-1, 3*32*32)))
        x1 = F.relu(self.fc2(x0))
        x1 = self.embedding_recorder(x1)
        x2 = self.fc3(x1)
        return x2

class LeNet5(nn.Module):
    def __init__(self, n_classes):
        super(LeNet5, self).__init__()
        #self.conv1 = nn.Conv2d(3, 6, 5, bias=False) #CIFAR10
        self.conv1 = nn.Conv2d(1, 6, 5, bias=False)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(6, 16, 5, bias=False)

        #self.fc1 = nn.Linear(16*5*5, 120, bias=False) #CIFAR10
        self.fc1 = nn.Linear(16 * 4 * 4, 120, bias=False)
        self.fc2 = nn.Linear(120, 84, bias=False)
        self.fc3 = nn.Linear(84, n_classes, bias=False)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))

        #x = x.view(-1, 16 * 5 * 5) #CIFAR10
        x = x.view(-1, 16 * 4 * 4)

        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)

        return x

############################################################################################################
################################################ VGG #######################################################
############################################################################################################

VGG_CONFIGS = {
    # M for MaxPool, Number for channels
    'like': [
        64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M',
        512, 512, 512, 'M'
    ],
    'D': [
        64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M',
        512, 512, 512, 'M'
    ],
    'C': [
        64, 64, 'M', 128, 128, 'M', 256, 256, (1, 256), 'M', 512, 512, (1, 512), 'M',
        512, 512, (1, 512), 'M' # tuples indicate (kernel size, output channels)
    ]
}

class VGG16(nn.Module):
    """
    This is a base class to generate three VGG variants used in SNIP paper:
        1. VGG-C (16 layers)
        2. VGG-D (16 layers)
        3. VGG-like

    Some of the differences:
        * Reduced size of FC layers to 512
        * Adjusted flattening to match CIFAR-10 shapes
        * Replaced dropout layers with BatchNorm

    Based on https://github.com/mi-lad/snip/blob/master/train.py
    by Milad Alizadeh.
    """

    def __init__(self, config, num_classes=10, save_features=False, bench_model=False):
        super().__init__()

        self.features = self.make_layers(VGG_CONFIGS[config], batch_norm=True)
        self.feats = []
        self.densities = []
        self.save_features = save_features
        self.bench = None if not bench_model else SparseSpeedupBench()

        if config == 'C' or config == 'D':
            self.classifier = nn.Sequential(
                nn.Linear((512 if config == 'D' else 2048), 512),  # 512 * 7 * 7 in the original VGG
                nn.ReLU(True),
                nn.BatchNorm1d(512),  # instead of dropout
                nn.Linear(512, 512),
                nn.ReLU(True),
                nn.BatchNorm1d(512),  # instead of dropout
                nn.Linear(512, num_classes),
            )
        else:
            self.classifier = nn.Sequential(
                nn.Linear(512, 512),  # 512 * 7 * 7 in the original VGG
                nn.ReLU(True),
                nn.BatchNorm1d(512),  # instead of dropout
                nn.Linear(512, num_classes),
            )

    @staticmethod
    def make_layers(config, batch_norm=False):
        layers = []
        in_channels = 3
        for v in config:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                kernel_size = 3
                if isinstance(v, tuple):
                    kernel_size, v = v
                conv2d = nn.Conv2d(in_channels, v, kernel_size=kernel_size, padding=1)
                if batch_norm:
                    layers += [
                        conv2d,
                        nn.BatchNorm2d(v),
                        nn.ReLU(inplace=True)
                    ]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v
        return nn.Sequential(*layers)

    def forward(self, x):
        for layer_id, layer in enumerate(self.features):
            if self.bench is not None and isinstance(layer, nn.Conv2d):
                x = self.bench.forward(layer, x, layer_id)
            else:
                x = layer(x)

            if self.save_features:
                if isinstance(layer, nn.ReLU):
                    self.feats.append(x.clone().detach())
                    self.densities.append((x.data != 0.0).sum().item()/x.numel())

        x = x.view(x.size(0), -1)
        x = self.classifier(x)

        return x

############################################################################################################
################################################ MobileNet #################################################
############################################################################################################

class InvertedResidual(nn.Module):
    def __init__(self, in_channels, out_channels, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.use_residual = (stride == 1 and in_channels == out_channels)
        hidden_dim = in_channels * expand_ratio
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, 1, 1, 0, bias=False),
            nn.BatchNorm2d(hidden_dim, affine=False, track_running_stats=False),
            nn.ReLU6(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim, affine=False, track_running_stats=False),
            nn.ReLU6(inplace=True),
            nn.Conv2d(hidden_dim, out_channels, 1, 1, 0, bias=False),
            nn.BatchNorm2d(out_channels, affine=False, track_running_stats=False),
        )

    def forward(self, x):
        if self.use_residual:
            return x + self.conv(x)
        else:
            return self.conv(x)

# Define the MobileNetV2 architecture
class MobileNetV2(nn.Module):
    def __init__(self, num_classes=100, width_mult=1.0):
        super(MobileNetV2, self).__init__()
        # Define the architecture parameters for MobileNetV2
        inverted_residual_setting = [
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]
        
        # Initial convolution layer
        input_channels = int(32 * width_mult)
        self.features = [nn.Sequential(
            nn.Conv2d(3, input_channels, 3, 2, 1, bias=False),
            nn.BatchNorm2d(input_channels, affine=False, track_running_stats=False),
            nn.ReLU6(inplace=True)
        )]
        
        # Build the MobileNetV2 backbone
        for t, c, n, s in inverted_residual_setting:
            output_channels = int(c * width_mult)
            for i in range(n):
                stride = s if i == 0 else 1
                self.features.append(InvertedResidual(input_channels, output_channels, stride, t))
                input_channels = output_channels
        
        # Final convolution layers
        self.features.append(nn.Sequential(
            nn.Conv2d(input_channels, 1280, 1, 1, 0, bias=False),
            nn.BatchNorm2d(1280, affine=False, track_running_stats=False),
            nn.ReLU6(inplace=True)
        ))
        
        self.features = nn.Sequential(*self.features)
        
        # Classifier (fully connected layer)
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(1280, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


class EmbeddingRecorder(nn.Module):
    def __init__(self, record_embedding: bool = False):
        super().__init__()
        self.record_embedding = record_embedding

    def forward(self, x):
        if self.record_embedding:
            self.embedding = x
        return x

    def __enter__(self):
        self.record_embedding = True

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.record_embedding = False


'''class SparseSpeedupBench(object):
    """Class to benchmark speedups for convolutional layers.

    Basic usage:
    1. Assing a single SparseSpeedupBench instance to class (and sub-classes with conv layers).
    2. Instead of forwarding input through normal convolutional layers, we pass them through the bench:
        self.bench = SparseSpeedupBench()
        self.conv_layer1 = nn.Conv2(3, 96, 3)

        if self.bench is not None:
            outputs = self.bench.forward(self.conv_layer1, inputs, layer_id='conv_layer1')
        else:
            outputs = self.conv_layer1(inputs)
    3. Speedups of the convolutional layer will be aggregated and print every 1000 mini-batches.
    """
    def __init__(self):
        self.layer_timings = {}
        self.layer_timings_channel_sparse = {}
        self.layer_timings_sparse = {}
        self.iter_idx = 0
        self.layer_0_idx = None
        self.total_timings = []
        self.total_timings_channel_sparse = []
        self.total_timings_sparse = []

    def get_density(self, x):
        return (x.data!=0.0).sum().item()/x.numel()

    def print_weights(self, w, layer):
        # w dims: out, in, k1, k2
        #outers = []
        #for outer in range(w.shape[0]):
        #    inners = []
        #    for inner in range(w.shape[1]):
        #        n = np.prod(w.shape[2:])
        #        density = (w[outer, inner, :, :] != 0.0).sum().item() / n
        #        #print(density, w[outer, inner])
        #        inners.append(density)
        #    outers.append([np.mean(inners), np.std(inner)])
        #print(outers)
        #print(w.shape, (w!=0.0).sum().item()/w.numel())
        pass

    def forward(self, layer, x, layer_id):
        if self.layer_0_idx is None: self.layer_0_idx = layer_id
        if layer_id == self.layer_0_idx: self.iter_idx += 1
        self.print_weights(layer.weight.data, layer)

        # calc input sparsity
        sparse_channels_in = ((x.data != 0.0).sum([2, 3]) == 0.0).sum().item()
        num_channels_in = x.shape[1]
        batch_size = x.shape[0]
        channel_sparsity_input = sparse_channels_in/float(num_channels_in*batch_size)
        input_sparsity = self.get_density(x)

        # bench dense layer
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        x = layer(x)
        end.record()
        start.synchronize()
        end.synchronize()
        time_taken_s = start.elapsed_time(end)/1000.0

        # calc weight sparsity
        num_channels = layer.weight.shape[1]
        sparse_channels = ((layer.weight.data != 0.0).sum([0, 2, 3]) == 0.0).sum().item()
        channel_sparsity_weight = sparse_channels/float(num_channels)
        weight_sparsity = self.get_density(layer.weight)

        # store sparse and dense timings
        if layer_id not in self.layer_timings:
            self.layer_timings[layer_id] = []
            self.layer_timings_channel_sparse[layer_id] = []
            self.layer_timings_sparse[layer_id] = []
        self.layer_timings[layer_id].append(time_taken_s)
        self.layer_timings_channel_sparse[layer_id].append(time_taken_s*(1.0-channel_sparsity_weight)*(1.0-channel_sparsity_input))
        self.layer_timings_sparse[layer_id].append(time_taken_s*input_sparsity*weight_sparsity)

        if self.iter_idx % 1000 == 0:
            self.print_layer_timings()
            self.iter_idx += 1

        return x

    def print_layer_timings(self):
        total_time_dense = 0.0
        total_time_sparse = 0.0
        total_time_channel_sparse = 0.0
        print('\n')
        for layer_id in self.layer_timings:
            t_dense = np.mean(self.layer_timings[layer_id])
            t_channel_sparse = np.mean(self.layer_timings_channel_sparse[layer_id])
            t_sparse = np.mean(self.layer_timings_sparse[layer_id])
            total_time_dense += t_dense
            total_time_sparse += t_sparse
            total_time_channel_sparse += t_channel_sparse

            print('Layer {0}: Dense {1:.6f} Channel Sparse {2:.6f} vs Full Sparse {3:.6f}'.format(layer_id, t_dense, t_channel_sparse, t_sparse))
        self.total_timings.append(total_time_dense)
        self.total_timings_sparse.append(total_time_sparse)
        self.total_timings_channel_sparse.append(total_time_channel_sparse)

        print('Speedups for this segment:')
        print('Dense took {0:.4f}s. Channel Sparse took {1:.4f}s. Speedup of {2:.4f}x'.format(total_time_dense, total_time_channel_sparse, total_time_dense/total_time_channel_sparse))
        print('Dense took {0:.4f}s. Sparse took {1:.4f}s. Speedup of {2:.4f}x'.format(total_time_dense, total_time_sparse, total_time_dense/total_time_sparse))
        print('\n')

        total_dense = np.sum(self.total_timings)
        total_sparse = np.sum(self.total_timings_sparse)
        total_channel_sparse = np.sum(self.total_timings_channel_sparse)
        print('Speedups for entire training:')
        print('Dense took {0:.4f}s. Channel Sparse took {1:.4f}s. Speedup of {2:.4f}x'.format(total_dense, total_channel_sparse, total_dense/total_channel_sparse))
        print('Dense took {0:.4f}s. Sparse took {1:.4f}s. Speedup of {2:.4f}x'.format(total_dense, total_sparse, total_dense/total_sparse))
        print('\n')

        # clear timings
        for layer_id in list(self.layer_timings.keys()):
            self.layer_timings.pop(layer_id)
            self.layer_timings_channel_sparse.pop(layer_id)
            self.layer_timings_sparse.pop(layer_id)
'''