import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from thop import profile
from torchsummary import summary
import os


class Reconstruct(nn.Module):
    def __init__(self, in_channel, out_channel, scale=3):
        super(Reconstruct, self).__init__()
        self.scale = scale
        self.in_channel  = in_channel
        self.out_channel = out_channel
        self.Conv = nn.Conv2d(in_channel, out_channel * (scale ** 2), kernel_size=3, stride=1, padding=1)
        self.PS   = nn.PixelShuffle(scale)

    def forward(self, x):
        x = self.Conv(x)
        x = self.PS(x)
        return x


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class WindowAttention(nn.Module):
    r""" Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.
    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)

        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4).contiguous()
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C).contiguous()
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size
    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image
    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class ChannelAttentionBlock(nn.Module):
    '''
        input   : B, H, W, C
        output  : B, H, W, C
    '''
    def __init__(self, in_channel):
        super(ChannelAttentionBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.conv1 = nn.Conv2d(in_channel, 1, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv2 = nn.Conv2d(1, in_channel, kernel_size=1, stride=1, padding=0, bias=False)
        self.LeakyReLU = nn.LeakyReLU(inplace=True)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        B, H, W, C = x.shape
        x = x.permute(0, 3, 1, 2).contiguous()  # from B, H, W, C --> B, C, H, W
        y = self.conv2(self.LeakyReLU(self.conv1(x)))
        
        avg_out = self.sigmoid(self.conv2(self.LeakyReLU(self.conv1(self.avg_pool(y)))))
        out = avg_out * y
        out = out.permute(0, 2, 3, 1).contiguous() # from B, C, H, W --> B, H, W, C
        
        return out


class HybridAttentionBlock(nn.Module):
    r""" Swin Transformer Block.
    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resulotion.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, input_resolution, num_heads=2, x_size=(256, 256), window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        if min(self.input_resolution) <= self.window_size:
            # if window size is larger than input resolution, we don't partition windows
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        if self.shift_size > 0:
            attn_mask = self.calculate_mask(self.input_resolution)
        else:
            attn_mask = None

        self.cab = ChannelAttentionBlock(in_channel=dim)

        self.register_buffer("attn_mask", attn_mask)

    def calculate_mask(self, x_size):
        # calculate attention mask for SW-MSA
        H, W = x_size
        img_mask = torch.zeros((1, H, W, 1))  # 1 H W 1
        h_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        w_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1

        mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))

        return attn_mask

    def forward(self, x, x_size):
        H, W = x_size
        B, L, C = x.shape
        # assert L == H * W, "input feature has wrong size"

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)
        y = self.cab(x) # B, H, W, C

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        # partition windows
        x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C

        # W-MSA/SW-MSA (to be compatible for testing on images whose shapes are the multiple of window size
        if self.input_resolution == x_size:
            attn_windows = self.attn(x_windows, mask=self.attn_mask)  # nW*B, window_size*window_size, C
        else:
            attn_windows = self.attn(x_windows, mask=self.calculate_mask(x_size).to(x.device))

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)  # B H' W' C

        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        x = x.view(B, H * W, C)

        # FFN
        y = y.view(B, H * W, C)
        x = shortcut + self.drop_path(x) + 0.1*y
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x


class PatchEmbed(nn.Module):
    def __init__(self, img_size=256, patch_size=4, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.embed_dim = embed_dim

        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        # B, C, H, W --> B, C, H*W --> B, H*W, C
        x = x.flatten(2).transpose(1, 2).contiguous()  # B Ph*Pw C
        if self.norm is not None:
            x = self.norm(x)
        return x


class PatchUnEmbed(nn.Module):
    def __init__(self, img_size=256, patch_size=4, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.embed_dim = embed_dim

    def forward(self, x, x_size):
        # B, H*W, C --> B, C, H*W --> B, C, H, W
        B, HW, C = x.shape
        x = x.transpose(1, 2).contiguous().view(B, self.embed_dim, x_size[0], x_size[1])  # B Ph*Pw C
        return x


class BasicBlock(nn.Module):
    """
        input size  : B, C, H, W
        output size : B, C, H, W
    """
    def __init__(self, in_channel, out_channel, depth=4, num_heads=2, window_size=8,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False,
                 img_size=256, patch_size=4):
        super(BasicBlock, self).__init__()

        self.dim = in_channel
        
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, embed_dim=self.dim, norm_layer=None)

        self.patch_unembed = PatchUnEmbed(
            img_size=img_size, patch_size=patch_size, embed_dim=self.dim, norm_layer=None)
        
        self.patches_resolution = self.patch_embed.patches_resolution

        self.blocks = nn.ModuleList([
            HybridAttentionBlock(dim=in_channel, 
                                 input_resolution=(
                                    self.patches_resolution[0],
                                    self.patches_resolution[1]
                                 ),
                                 num_heads=num_heads, window_size=window_size,
                                 shift_size=0 if (i % 2 == 0) else window_size // 2,
                                 mlp_ratio=mlp_ratio,
                                 qkv_bias=qkv_bias, qk_scale=qk_scale,
                                 drop=drop, attn_drop=attn_drop,
                                 drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                 norm_layer=norm_layer)
            for i in range(depth)])

        self.conv = nn.Conv2d(in_channel, out_channel, 3, 1, 1)

    def forward(self, x, x_size):
        shortcut = x
        x = self.patch_embed(x)
        for block in self.blocks:
            x = block(x, x_size)
        x = self.patch_unembed(x, x_size) + shortcut
        x = self.conv(x)    # from input_dim --> output_dim
        return x


class Downsample(nn.Module):
    def __init__(self, in_channel, out_channel, num_heads=2, mlp_ratio=4, patch_size=4, window_size=8, img_size=256) -> None:
        super(Downsample, self).__init__()
        self.block = BasicBlock(in_channel=in_channel, out_channel=in_channel, 
                                num_heads=num_heads,
                                mlp_ratio=mlp_ratio,
                                patch_size=patch_size,
                                window_size=window_size,
                                img_size=img_size
                                )
        self.pool = nn.Conv2d(in_channel, out_channel, kernel_size=4, stride=2, padding=1)

    def forward(self, x):
        x_size = (x.shape[2], x.shape[3])
        x = self.block(x, x_size)
        x = self.pool(x)
        return x


class Upsample(nn.Module):
    def __init__(self, in_channel, out_channel, num_heads=2, mlp_ratio=4, patch_size=4, window_size=8, img_size=256) -> None:
        super(Upsample, self).__init__()
        self.up = nn.ConvTranspose2d(in_channel, out_channel, kernel_size=2, stride=2)
        self.block = BasicBlock(in_channel=out_channel, out_channel=out_channel,
                                num_heads=num_heads,
                                mlp_ratio=mlp_ratio,
                                patch_size=patch_size,
                                window_size=window_size,
                                img_size=img_size
                                )

        self.conv = nn.Conv2d(in_channels=out_channel, out_channels=out_channel, kernel_size=3, stride=1, padding=1)

    def forward(self, skip, x):
        up_x = self.up(x)
        x_size = (skip.shape[2], skip.shape[3])
        convskip = self.conv(skip)    # calculate the difference between orignal img & upgraded img (idea from laplacian pyramid)
        x = self.block((convskip + up_x), x_size)
        return x


class LapSUNET(nn.Module):
    def __init__(self, 
                 depth=2, 
                 img_size=256,
                 downblock=Downsample, 
                 upblock=Upsample, dim=18, 
                 num_heads=[2, 4], 
                 mlp_ratio=2, 
                 patch_size=[4, 4], 
                 norm=nn.LayerNorm) -> None:
        super(LapSUNET, self).__init__()
        self.depth = depth

        # Initial convolution block (input projection)
        self.shallow_feature_extraction = nn.Conv2d(3, dim, kernel_size=3, padding=1)
        
        # Downward path
        n_dim = dim
        imgsz = img_size
        self.downs = nn.ModuleList()
        for _ in range(depth):
            self.downs.append(downblock(n_dim, n_dim*2, num_heads=num_heads[_], mlp_ratio=mlp_ratio))
            n_dim *= 2

        # Upward path
        self.ups = nn.ModuleList()
        for _ in range(depth):
            self.ups.append(upblock(n_dim, n_dim//2, num_heads=num_heads[-1*(_+1)], mlp_ratio=mlp_ratio))
            n_dim //= 2

        
        # Reconstruction
        self.Reconstruction = Reconstruct(n_dim, 3)
        self.conv = nn.Conv2d(3, 3, 3, 1, 1)



    def forward(self, x):
        # shallow feature extraction
        x = self.shallow_feature_extraction(x)
        LSC = x

        # Downward path
        skips = []
        for down in self.downs:
            skips.insert(0, x)
            x = down(x)

        # Upward path
        for up, skip in zip(self.ups, skips):
            x = up(skip, x)

        # Reconstruction
        x = x + LSC
        x = self.Reconstruction(x)
        # x = self.conv(x)

        return x



if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1, 2, 3'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = LapSUNET(dim=16, depth=2, num_heads=[2, 4], mlp_ratio=2).to(device)
    summary(model, (3, 256, 256))
    Img = torch.randn(1, 3, 256, 256).to(device)
    Out = model(Img)
    print('Input shape  : ', Img.shape)
    print('Output shape : ', Out.shape)
    macs, params = profile(model, inputs=(Img, ))
    print('macs: {} G, flops: {} G'.format(macs / 1e9, macs * 2 / 1e9))
    print(f'FLOPs for 720p --> 2160p (15 patches), 60fps: {macs * 2 * 15 * 60 / 1e9 / 1000} T')