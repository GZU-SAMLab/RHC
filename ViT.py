"""实现ViT网络架构"""
from torch import nn, optim
from collections import OrderedDict
import torch
from functools import partial


class PatchEmbed(nn.Module):
    """
    将一个2D的图片输入PatchEmbed层
    img_size；图像大小
    in_c:图像通道数
    patch_size:块的大小
    embed_dim:隐藏层的大小
    norm_layer:是否使用正则化
    """

    def __init__(self, img_size=224, patch_size=16, in_c=3, embed_dim=768, norm_layer=None):
        super(PatchEmbed, self).__init__()
        img_size = (img_size, img_size)
        patch_size = (patch_size, patch_size)

        self.image_size = img_size
        self.patch_size = patch_size
        """ 求得块的数量，相当于transformer中有效输入序列长度 """
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]

        """ 这里用了一个感受野和步长都是patch_size的卷积 """
        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=patch_size, stride=patch_size)
        """ 使用正则化或者什么都不做 """
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape
        """
        假定输入图像的分辨率和模型处理的分辨率相同，否则报错
        ViT模型并不像传统模型那样可以更改图像大小，而是必须固定的
        """
        assert H == self.image_size[0] and W == self.image_size[1], \
            f"Input image size({H}*{W} doesn't match model({self.image_size[0]}*{self.image_size[1]})."

        """
        flatten: [B, C, H, W] -> [B, C, HW] 从第二个维度开始展平
        transpose: [B, C, HW] -> [B, HW, C] 将维度一二上数据进行调换
        """
        x = self.proj(x).flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x


class Multi_Head_Attention(nn.Module):
    """
    实现多头注意力
    dim: 输入token的dimension
    num_heads: 头的数量
    qkv_bias:
    qkv_scale:
    attn_drop_ratio:
    proj_drop_ratio:
    """

    def __init__(self,
                 dim,
                 num_heads=8,
                 qkv_bias=False,
                 qk_scale=None,
                 attn_drop_ratio=0.,
                 proj_drop_ratio=0.):
        super(Multi_Head_Attention, self).__init__()
        self.num_heads = num_heads
        """ 针对每一个header的dimension """
        head_dim = dim // num_heads
        """ 对应（d)^(-0.5) """
        self.scale = qk_scale or head_dim ** -0.5
        """ 通过一个全连接层得到q、k、v"""
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop_ratio)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop_ratio)

    def forward(self, x):
        """
        此时进入多头注意力的是经过embeding层并加入位置编码的
        B: batch_size
        N: num_patches + 1(class token)
        C: total_embed_dim
        """
        B, N, C = x.shape

        """
        原先: [batch_size, num_patches + 1, total_embed_dim]
        qkv(): -> [batch_size, num_patches + 1, 3 * total_embed_dim]
        reshape: -> [batch_size, num_patches + 1, 3, num_heads, embed_dim_per_head]
        permute: -> [3, batch_size, num_heads, num_patches + 1, embed_dim_per_head]
        permute: 调整数据的位置
        """
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        """
        缩放点积注意力
        transpose: -> [batch_size, num_heads, embed_dim_per_head, num_patches + 1]
        @: multiply -> [batch_size, num_heads, num_patches + 1, num_patches + 1]
        """
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)  # 对最后一维求softmax
        attn = self.attn_drop(attn)

        """
        @: multiply -> [batch_size, num_heads, num_patches + 1, embed_dim_per_head]
        transpose: -> [batch_size, num_patches + 1, num_heads, embed_dim_per_head]
        reshape: -> [batch_size, num_patches + 1, total_embed_dim]
        """
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Mlp_Block(nn.Module):
    """
    MLP as used in Encoder block,
    Linear,GELU,Dropout,Linear,Dropout
    """

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super(Mlp_Block, self).__init__()
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


class Encoder_Block(nn.Module):
    """
    Encoder block
    dim: 输入token的dimension
    num_heads: 头的数量
    """

    def __init__(self,
                 dim,
                 num_heads,
                 mlp_ratio=4.,
                 qkv_bias=False,
                 qk_scale=None,
                 proj_drop_ratio=0.,
                 drop_ratio=0.,
                 attn_drop_ratio=0.,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm):
        super(Encoder_Block, self).__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Multi_Head_Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                                         attn_drop_ratio=attn_drop_ratio, proj_drop_ratio=proj_drop_ratio)
        self.norm2 = norm_layer(dim)
        self.drop = nn.Dropout(drop_ratio)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp_Block(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop_ratio)

    def forward(self, x):
        x = x + self.drop(self.attn(self.norm1(x)))
        x = x + self.drop(self.mlp(self.norm2(x)))
        return x


class VisionTransformer(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_c=3, num_classes=1000,
                 embed_dim=768, depth=12, num_heads=12, mlp_ratio=4.0, qkv_bias=True,
                 qk_scale=None, representation_size=None, proj_drop_ratio=0.,
                 attn_drop_ratio=0., drop_ratio=0., embed_layer=PatchEmbed, norm_layer=None,
                 act_layer=None):
        """
        Args:
            img_size (int, tuple): input image size
            patch_size (int, tuple): patch size
            in_c (int): number of input channels
            num_classes (int): number of classes for classification head
            embed_dim (int): embedding dimension
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            qk_scale (float): override default qk scale of head_dim ** -0.5 if set
            representation_size (Optional[int]): enable and set representation layer (pre-logits) to this value if set
            proj_drop_ratio (float): project dropout rate
            attn_drop_ratio (float): attention dropout rate
            drop_ratio (float): dropout rate
            embed_layer (nn.Module): patch embedding layer
            norm_layer: (nn.Module): normalization layer
        """
        super(VisionTransformer, self).__init__()
        self.num_classes = num_classes
        self.num_tokens = 1
        self.num_features = self.embed_dim = embed_dim
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU

        self.patch_embed = embed_layer(img_size=img_size, patch_size=patch_size, in_c=in_c, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_tokens, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_ratio)

        self.blocks = nn.Sequential(*[
            Encoder_Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                          proj_drop_ratio=proj_drop_ratio, drop_ratio=drop_ratio, attn_drop_ratio=attn_drop_ratio,
                          norm_layer=norm_layer, act_layer=act_layer)
            for i in range(depth)
        ])
        self.norm = norm_layer(embed_dim)

        # Representation layer
        if representation_size:
            self.has_logits = True
            self.num_features = representation_size
            self.pre_logits = nn.Sequential(OrderedDict([
                ("fc", nn.Linear(embed_dim, representation_size)),
                ("act", nn.Tanh())
            ]))
        else:
            self.has_logits = False
            self.pre_logits = nn.Identity()

        # Classifier head(s)
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()

        # Weight init
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.apply(_init_vit_weights)

    def forward_features(self, x):
        # [B, C, H, W] -> [B, num_patches, embed_dim]
        x = self.patch_embed(x)  # [B, 196, 768]
        # [1, 1, 768] -> [B, 1, 768]
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_token, x), dim=1)  # [B, 197, 768]
        # 加入位置编码后通过dropout层
        x = self.pos_drop(x + self.pos_embed)
        # 通过Transformer Encoder层
        x = self.blocks(x)
        # 层归一化
        x = self.norm(x)
        # 切片并通过MLP head，切片后的数据[1, 768]
        return self.pre_logits(x[:, 0])

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x
        
        
    def load_from(self, weights):
        with torch.no_grad():

            res_weight = weights
            #给嵌入块赋参数
            self.transformer.embeddings.patch_embeddings.weight.copy_(np2th(weights["embedding/kernel"], conv=True))
            self.transformer.embeddings.patch_embeddings.bias.copy_(np2th(weights["embedding/bias"]))
            # 给编码器赋参数
            self.transformer.encoder.encoder_norm.weight.copy_(np2th(weights["Transformer/encoder_norm/scale"]))
            self.transformer.encoder.encoder_norm.bias.copy_(np2th(weights["Transformer/encoder_norm/bias"]))
            # 给位置嵌入赋参数
            posemb = np2th(weights["Transformer/posembed_input/pos_embedding"])

            posemb_new = self.transformer.embeddings.position_embeddings
            if posemb.size() == posemb_new.size():
                #这里是true 下面全都不执行 直接到下个循环 循环那里有问题
                self.transformer.embeddings.position_embeddings.copy_(posemb)
                
            elif posemb.size()[1]-1 == posemb_new.size()[1]:
                posemb = posemb[:, 1:]
                self.transformer.embeddings.position_embeddings.copy_(posemb)
            else:
                logger.info("load_pretrained: resized variant: %s to %s" % (posemb.size(), posemb_new.size()))
                ntok_new = posemb_new.size(1)
                if self.classifier == "seg":
                    _, posemb_grid = posemb[:, :1], posemb[0, 1:]
                gs_old = int(np.sqrt(len(posemb_grid)))
                gs_new = int(np.sqrt(ntok_new))
                print('load_pretrained: grid-size from %s to %s' % (gs_old, gs_new))
                posemb_grid = posemb_grid.reshape(gs_old, gs_old, -1)
                zoom = (gs_new / gs_old, gs_new / gs_old, 1)
                posemb_grid = ndimage.zoom(posemb_grid, zoom, order=1)  # th2np
                posemb_grid = posemb_grid.reshape(1, gs_new * gs_new, -1)
                posemb = posemb_grid
                self.transformer.embeddings.position_embeddings.copy_(np2th(posemb))
            # Encoder whole
            #分别给我的编码器中每一个层赋权重和偏置
            for bname, block in self.transformer.encoder.named_children():
                print(bname)
                print(block)
                for uname, unit in block.named_children():
                    print(uname)
                    print(unit)
                    unit.load_from(weights, n_block=uname)


            if self.transformer.embeddings.hybrid:
                self.transformer.embeddings.hybrid_model.root.conv.weight.copy_(np2th(res_weight["conv_root/kernel"], conv=True))
                gn_weight = np2th(res_weight["gn_root/scale"]).view(-1)
                gn_bias = np2th(res_weight["gn_root/bias"]).view(-1)
                self.transformer.embeddings.hybrid_model.root.gn.weight.copy_(gn_weight)
                self.transformer.embeddings.hybrid_model.root.gn.bias.copy_(gn_bias)

                for bname, block in self.transformer.embeddings.hybrid_model.body.named_children():
                    for uname, unit in block.named_children():
                        unit.load_from(res_weight, n_block=bname, n_unit=uname)


def _init_vit_weights(m):
    """
    ViT weight initialization
    :param m: module
    """
    if isinstance(m, nn.Linear):
        nn.init.trunc_normal_(m.weight, std=.01)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode="fan_out")
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.LayerNorm):
        nn.init.zeros_(m.bias)
        nn.init.ones_(m.weight)


def vit_base_patch16_224_in21k(num_classes: int = 200, has_logits: bool = True):
    """
    ViT-Base model (ViT-B/16) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-21k weights @ 224x224, source https://github.com/google-research/vision_transformer.
    weights ported from official Google JAX impl:
    https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_patch16_224_in21k-e5005f0a.pth
    """
    model = VisionTransformer(img_size=224,
                              patch_size=16,
                              embed_dim=768,
                              depth=12,
                              num_heads=12,
                              representation_size=768 if has_logits else None,
                              num_classes=num_classes)
    return model


def vit_base_patch16_448(num_classes: int = 200, has_logits: bool = True):
    """
    ViT-Base model (ViT-B/16) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-21k weights @ 224x224, source https://github.com/google-research/vision_transformer.
    weights ported from official Google JAX impl:
    https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_patch16_224_in21k-e5005f0a.pth
    """
    model = VisionTransformer(img_size=448,
                              patch_size=16,
                              embed_dim=768,
                              depth=12,
                              num_heads=12,
                              representation_size=768 if has_logits else None,
                              num_classes=num_classes)
    return model
