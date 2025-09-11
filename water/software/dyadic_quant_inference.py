import torch
import torch.nn.functional as F
import math
import cv2
import numpy as np
from torch import nn
from kornia.color import rgb_to_lab, lab_to_rgb
from std_L_ch_based import ResIERes, CALayer, PALayer, Conv_Atten_Block, CPLayer
import matplotlib.pyplot as plt
from torch.serialization import add_safe_globals
import os
import torch.nn as nn
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
# 設定 `ResIERes` 為可加載的模型類別
add_safe_globals([ResIERes])
quant_val = 127
data_type = torch.int64
def float32_to_q4_32_hex(val):
    """
    將 float32 轉為 Q5.32 格式，回傳 10-digit hex 字串（37-bit）
    """
    scaled = int(round(val * (1 << 32)))  # Q5.32 仍然是乘 2^32

    if scaled < 0:
        scaled = (1 << 37) + scaled  # 轉換為 37-bit 二補數

    hex_str = format(scaled & 0x1FFFFFFFFF, '010X')  # 37-bit mask → 10 hex digits
    return hex_str
def log_approx(x):
    x = torch.clamp(x, min=1e-6)  # 避免 log(0)
    return 0.69 * (x - 1) - 0.33 * (x - 1) ** 2  # Pade Approximation
# 讀取訓練好的全精度 (float32) 模型
def load_trained_model(model_path):
    model = torch.load(model_path, map_location=torch.device("cpu"), weights_only=False)
    model.eval()
    return model
def closest_dyadic_scale(scale):
    if scale <= 0:
        return 1, 0, 1.0, float('inf')  # 避免 scale <= 0 的情況

    best_M, best_N, best_reconstructed, best_error = None, None, None, float('inf')

    # for N in range(-1, -30, -1):  # 遍歷 N：從 -1 到 -15
    N = -16
    for M in range(1, 1024*1024):  # 遍歷 M：從 1 到 255
        reconstructed = M * (2 ** N)  # 計算重建值
        error = abs(scale - reconstructed)  # 計算誤差

        if error < best_error:
            best_M, best_N, best_reconstructed, best_error = M, N, reconstructed, error
    # print(best_M)
    return best_M, best_N, best_reconstructed

def requantize(output, old_scale, bias_q, bias_M,bias_N):
    output_float = output.float() * old_scale  # 恢復原本的 float32 值
    output_float += bias_q.view(1,-1,1,1) * (bias_M * (2 ** bias_N))
    return output_float
# **Dyadic Quantization 方法**
def dyadic_conv2d(X, W_q, bias_q, stride=1, padding=1):
    output_q = F.conv2d(X, W_q, bias=bias_q, stride=stride, padding=padding)
    return output_q

class QuantizedResIERes(nn.Module):
    def __init__(self, pretrained_model, example_input_shape):
        super(QuantizedResIERes, self).__init__()

        self.model = pretrained_model
        self.quantized_weights = {}
        self.quantized_biases = {}
        self.quantized_M = {}
        self.quantized_N = {}
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.sig1 = nn.Sigmoid()#(inplace=True)
        self.relu = nn.ReLU(inplace=True)

        self.input_shapes = {}  # 存储每层 Conv2D 的输入形状

        # ✅ **修正：将 `example_input_shape` 变成 `Tensor`**
        self.example_input = torch.randn(*example_input_shape)  # 生成随机 Tensor
        print(f"✅ example_input.shape: {self.example_input.shape}")

        # **注册 Hook 以获取 `input_tensor_shape`**
        self._register_hooks()

        # **执行一次前向传播，获取所有 Conv2D 层的 `input_tensor_shape`**
        with torch.no_grad():
            _ = self.model(self.example_input)

        # # **保存 `conv_layer_info.txt`**
        # # self._save_layer_info()

        with open("quantized_W_q.txt", "w") as f_W_q, \
             open("quantized_W_M.txt", "w") as f_W_M, \
             open("quantized_W_N.txt", "w") as f_W_N, \
             open("quantized_B_q.txt", "w") as f_B_q, \
             open("quantized_B_M.txt", "w") as f_B_M, \
             open("quantized_B_N.txt", "w") as f_B_N:
        #     # print("org weight",self.model.conv1.weight)
            for name, module in self.model.named_modules():
                if isinstance(module, nn.Conv2d):
                    # **1️⃣ 计算 Weight Scale**
                    W_scale = max(module.weight.abs().max().item(), 1e-2) / 127  # 🚀 避免 scale 太小
                    W_M, W_N, best_reconstructed = closest_dyadic_scale(W_scale)
                    W_q = torch.round(module.weight * (2** (-W_N)) / W_M).to(dtype=torch.int64)  # 🚀 量化到 INT16
                    
                    # **2️⃣ 存储 INT16 量化权重 & scale**
                    self.quantized_weights[name] = W_q
                    self.quantized_M[name] = W_M
                    self.quantized_N[name] = W_N

                    # **3️⃣ 量化 Bias**
                    if module.bias is not None:
                        B_scale = max(module.bias.abs().max().item(), 1e-2) / 127
                        B_M, B_N, best_reconstructed = closest_dyadic_scale(B_scale)
                        B_q = torch.round(module.bias * (2** (-B_N)) / B_M).to(dtype=torch.int64)

                        self.quantized_biases[name] = B_q
                        self.quantized_M[name + "_bias"] = B_M
                        self.quantized_N[name + "_bias"] = B_N
                    else:
                        B_q = None

                        # f.write(f"==== Layer: {name} ====\n")
                        # f.write(f"W_M: {format(W_M & 0xFFFF, '016b')}, W_N: {format(-W_N & 0xFF, '08b')}\n")
                        # W_q_numpy = W_q.cpu().numpy().astype(np.int8)
                        # f.write("[\n")  # 开始 4D 数组
        
                        # for batch in W_q:  # 遍历 batch 维度
                        #     f.write("    [\n")  # 开始 batch 维度
                        #     for channel in batch:  # 遍历通道维度
                        #         f.write("        [\n")  # 开始 channel 维度
                        #         for row in channel:  # 遍历高度 (height)
                        #             binary_row = ["{:08b}".format(value & 0xFF) for value in row.tolist()]  # 确保 `value` 是 `int`
                        #             f.write("            [" + ", ".join(binary_row) + "],\n")  # 保留数组格式
                        #         f.write("        ],\n")  # 结束 channel 维度
                        #     f.write("    ],\n")  # 结束 batch 维度
                        # f.write("]\n")  # 结束 4D 数组



                        # if B_q is not None:
                        #     f.write(f"B_M: {format(B_M & 0xFFFF, '016b')}, B_N: {format(-B_N & 0xFF, '08b')}\n")
                        #     B_q_numpy = B_q.cpu().numpy().astype(np.int8)
                        #     f.write("[")  # 开始方括号
                        #     binary_values = ["{:08b}".format(value & 0xFF) for value in B_q_numpy]  # 8-bit 二进制格式
                        #     f.write(", ".join(binary_values))  # 用逗号分隔
                        #     f.write("]\n")  # 结束方括号
                        #     # f.write(f"B_q:\n{B_q.cpu().numpy()}\n")

                        # **🔹 逐行写入 6 个量化参数文件**
                    f_W_M.write(f"//==== Layer: {name} ====\n{format(W_M & 0xFFFF, '016b')}\n")
                    f_W_N.write(f"//==== Layer: {name} ====\n{format(-W_N & 0xFF, '08b')}\n")
                    f_W_q.write(f"//==== Layer: {name} ====\n")
                    W_q_numpy = W_q.cpu().numpy().astype(np.int8)
                    binary_representations = []
                    for value in W_q_numpy.flatten():  # 遍历所有值
                        binary = format(value & 0xFF, '08b')  # 确保是 8-bit 2 进制格式
                        binary_representations.append(binary)
                    binary_string = '\n'.join(binary_representations)
                    f_W_q.write(binary_string + '\n')

                    if B_q is not None:
                        f_B_M.write(f"//==== Layer: {name} ====\n{format(B_M & 0xFFFF, '016b')}\n")
                        f_B_N.write(f"//==== Layer: {name} ====\n{format(-B_N & 0xFF, '08b')}\n")
                        f_B_q.write(f"//==== Layer: {name} ====\n")
                        B_q_numpy = B_q.cpu().numpy().astype(np.int8)
                        binary_representations = []
                        for value in B_q_numpy.flatten():  # 遍历所有值
                            binary = format(value & 0xFF, '08b')  # 确保是 8-bit 2 进制格式
                            binary_representations.append(binary)
                        binary_string = '\n'.join(binary_representations)
                        f_B_q.write(binary_string + '\n')

    def _register_hooks(self):
        """
        在所有 Conv2D 层上注册 Forward Hook，以获取输入形状
        """
        def hook_fn(module, input, output):
            self.input_shapes[module.name] = tuple(input[0].shape)  # 记录输入张量形状

        self.hooks = []
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Conv2d):
                module.name = name  # 赋值层名称
                self.hooks.append(module.register_forward_hook(hook_fn))

    def _save_layer_info(self):
        """
        将 Conv2D 层的信息写入 `conv_layer_info.txt`
        """
        with open("conv_layer_info.txt", "w") as f_layer_info:
            # 获取模型所有层的列表，方便查找激活函数
            module_list = list(self.model.named_modules())

            for i, (name, module) in enumerate(module_list):
                if isinstance(module, nn.Conv2d):
                    # 获取 Conv2D 配置信息
                    input_tensor_shape = self.input_shapes.get(name, "Unknown")  # 获取 Hook 记录的输入形状
                    kernel_size = module.weight.shape
                    out_channels = module.out_channels
                    padding = module.padding

                    # **检查是否有 Activation Function**
                    activation_function = "0"
                    if i + 1 < len(module_list):
                        next_module = module_list[i + 1][1]  # 获取下一个层
                        if isinstance(next_module, nn.ReLU):
                            activation_function = "1"
                        elif isinstance(next_module, nn.Sigmoid):
                            activation_function = "2"

                    # **写入 `conv_layer_info.txt`，确保每个数值单独占一行**
                    f_layer_info.write(f"//{name}\n")
                    
                    # **Flatten Tuple 为单行存储**
                    for dim in input_tensor_shape[1:2]:
                        f_layer_info.write(f"{format(dim & 0b1111, '04b')}")
                    for dim in input_tensor_shape[2:3]:
                        f_layer_info.write(f"{format(dim & 0b111111111, '09b')}")
                    for dim in kernel_size[0:3]:
                        f_layer_info.write(f"{format(dim & 0b1111, '04b')}")
                    # f_layer_info.write(f"{out_channels}\n")
                    for dim in padding[1:]:
                        f_layer_info.write(f"{format(dim & 0b1, '01b')}")
                    
                    # **存储 Activation Function**
                    f_layer_info.write(f"{format(int(activation_function) & 0b11, '02b')}\n")
                

        print("✅ `conv_layer_info.txt` 已更新，包含 Activation Function 信息！")

    def __del__(self):
        """
        在对象销毁时，移除所有 Hook
        """
        for hook in self.hooks:
            hook.remove()

    def forward(self, x):
        np.set_printoptions(threshold=np.inf)
        # X_scale = max(x.abs().max().item(), 1e-2) / quant_val
        # print("max:",x.abs().max().item())
        # print("min:",x.abs().min().item())
        X_q = torch.round(x * quant_val / max(x.abs().max().item(), 1e-2)).to(torch.int64)
        with open("quant.txt", "w") as f_conv1_golden:
            for value in X_q.flatten():  # 展平 Tensor 以便逐个写入
                f_conv1_golden.write(f"{value}\n")
        
        res = x
        x1 = dyadic_conv2d(
            X_q, 
            self.quantized_weights['conv1'], 
            bias_q=self.quantized_biases['conv1'], 
            stride=1, 
            padding=1, 
        )
        old_scale = ((self.quantized_M['conv1'] ) * (2 ** (self.quantized_N['conv1']))) * max(x.abs().max().item(), 1e-2) / 127
        x1 = requantize(x1, old_scale, self.quantized_biases['conv1'],self.quantized_M['conv1_bias'],self.quantized_N['conv1_bias'])
        with open("check.txt", "w") as f:
            for val in x1.flatten():
                hex_val = float32_to_q4_32_hex(val.item())
                f.write(hex_val + "\n")
        ca1_res = x1
        # **手動拆解 ca1**
        # print("max:",x1.abs().max().item())
        # print("min:",x1.abs().min().item())
        X_scale = max(x1.abs().max().item(), 1e-2) / quant_val
        # print(X_scale)
        X_q = torch.round(x1 / X_scale).to(torch.int64)
        x2 = dyadic_conv2d(
            X_q, 
            self.quantized_weights['ca1.conv1'], 
            bias_q=self.quantized_biases['ca1.conv1'], 
            stride=1, 
            padding=1, 
        )
        old_scale = ((self.quantized_M['ca1.conv1'] ) * (2 ** (self.quantized_N['ca1.conv1']))) * X_scale
        x2 = requantize(x2, old_scale, self.quantized_biases['ca1.conv1'],self.quantized_M['ca1.conv1_bias'],self.quantized_N['ca1.conv1_bias'])
        x2 = self.relu(x2)
        ###############################################################################
        # print("max:",x2.abs().max().item())
        # print("min:",x2.min().item())
        # X_q = torch.round(x2 * (2 ** (7))).to(dtype=data_type)  # 🚀 量化到 INT16
        X_scale = max(x2.abs().max().item(), 1e-2) / quant_val
        X_q = torch.round(x2 / X_scale).to(torch.int64)
        x3 = dyadic_conv2d(
            X_q, 
            self.quantized_weights['ca1.conv2'], 
            bias_q=self.quantized_biases['ca1.conv2'], 
            stride=1, 
            padding=1, 
        )
        old_scale = ((self.quantized_M['ca1.conv2'] ) * (2 ** (self.quantized_N['ca1.conv2']))) * X_scale
        x3 = requantize(x3, old_scale, self.quantized_biases['ca1.conv2'],self.quantized_M['ca1.conv2_bias'],self.quantized_N['ca1.conv2_bias'])      
        print("x3 shape",x3.shape)
        # **拆解 CPLayer（CALayer & PALayer）**
        ##########################################################################
        #ca_layer
        x4 = self.avg_pool(x3)
        # print("avgpool:",x4)
        # print("max:",x4.abs().max().item())
        # print("min:",x4.abs().min().item())
        # X_q = torch.round(x4 * (2 ** (7))).to(dtype=data_type)  # 🚀 量化到 INT16
        X_scale = max(x4.abs().max().item(), 1e-2) / quant_val
        X_q = torch.round(x4 / X_scale).to(torch.int64)
        
        x5 = dyadic_conv2d(
            X_q, 
            self.quantized_weights['ca1.attention.ca.conv_du.0'], 
            bias_q=self.quantized_biases['ca1.attention.ca.conv_du.0'], 
            stride=1, 
            padding=0, 
        )
        old_scale = ((self.quantized_M['ca1.attention.ca.conv_du.0'] ) * (2 ** (self.quantized_N['ca1.attention.ca.conv_du.0']))) * X_scale
        x5 = requantize(x5, old_scale, self.quantized_biases['ca1.attention.ca.conv_du.0'],self.quantized_M['ca1.attention.ca.conv_du.0_bias'],self.quantized_N['ca1.attention.ca.conv_du.0_bias'])  
        x5 = self.relu(x5)
        # print("max:",x5.abs().max().item())
        # print("min:",x5.abs().min().item())
        # X_q = torch.round(x5 * (2 ** (7))).to(dtype=data_type)  # 🚀 量化到 INT16
        X_scale = max(x5.abs().max().item(), 1e-2) / quant_val
        X_q = torch.round(x5 / X_scale).to(torch.int64)
        x55 = dyadic_conv2d(
            X_q, 
            self.quantized_weights['ca1.attention.ca.conv_du.2'], 
            bias_q=self.quantized_biases['ca1.attention.ca.conv_du.2'], 
            stride=1, 
            padding=0, 
        )
        old_scale = ((self.quantized_M['ca1.attention.ca.conv_du.2'] ) * (2 ** (self.quantized_N['ca1.attention.ca.conv_du.2']))) * X_scale
        x55 = requantize(x55, old_scale, self.quantized_biases['ca1.attention.ca.conv_du.2'],self.quantized_M['ca1.attention.ca.conv_du.2_bias'],self.quantized_N['ca1.attention.ca.conv_du.2_bias'])  
        x55 = self.sig1(x55)
        
        x6 = x3 * x55#(8*1) * (256*256*8)
        out = x3 * x6#(8*1) * (256*256*8)
        
        ##########################################################
        #pa_layer
        # print("max:",out.abs().max().item())
        # print("min:",out.abs().min().item())
        # X_q = torch.round(out * (2 ** (7))).to(dtype=data_type)  # 🚀 量化到 INT16
        X_scale = max(out.abs().max().item(), 1e-2) / quant_val
        X_q = torch.round(out / X_scale).to(torch.int64)
        
        x7 = dyadic_conv2d(
            X_q, 
            self.quantized_weights['ca1.attention.pa.conv_du.0'], 
            bias_q=self.quantized_biases['ca1.attention.pa.conv_du.0'], 
            stride=1, 
            padding=0, 
        )
        old_scale = ((self.quantized_M['ca1.attention.pa.conv_du.0'] ) * (2 ** (self.quantized_N['ca1.attention.pa.conv_du.0']))) * X_scale
        x7 = requantize(x7, old_scale, self.quantized_biases['ca1.attention.pa.conv_du.0'],self.quantized_M['ca1.attention.pa.conv_du.0_bias'],self.quantized_N['ca1.attention.pa.conv_du.0_bias'])  
        x7 = self.relu(x7)
        
        ########################################################################################
        # print("max:",x7.abs().max().item())
        # print("min:",x7.abs().min().item())
        # X_q = torch.round(x7 * (2 ** (7))).to(dtype=data_type)  # 🚀 量化到 INT16
        X_scale = max(x7.abs().max().item(), 1e-2) / quant_val
        X_q = torch.round(x7 / X_scale).to(torch.int64)
        x8 = dyadic_conv2d(
            X_q, 
            self.quantized_weights['ca1.attention.pa.conv_du.2'], 
            bias_q=self.quantized_biases['ca1.attention.pa.conv_du.2'], 
            stride=1, 
            padding=0, 
        )
        old_scale = ((self.quantized_M['ca1.attention.pa.conv_du.2'] ) * (2 ** (self.quantized_N['ca1.attention.pa.conv_du.2']))) * X_scale
        x8 = requantize(x8, old_scale, self.quantized_biases['ca1.attention.pa.conv_du.2'],self.quantized_M['ca1.attention.pa.conv_du.2_bias'],self.quantized_N['ca1.attention.pa.conv_du.2_bias'])  
        x8 = self.sig1(x8)
        print("x8_shape",x8.shape)  
        x9 = out * x8  #(256*256*8) * (256*256*1)
        
        out = out * x9 #(256*256*8) * (256*256*8)
        out = out + ca1_res #ca1 finish
        # print("max:",out.abs().max().item())
        # print("min:",out.abs().min().item())
        # X_q = torch.round(out * (2 ** (7))).to(dtype=data_type)  # 🚀 量化到 INT16
        X_scale = max(out.abs().max().item(), 1e-2) / quant_val
        X_q = torch.round(out / X_scale).to(torch.int64)
        x11 = dyadic_conv2d(
            X_q, 
            self.quantized_weights['conv2'], 
            bias_q=self.quantized_biases['conv2'], 
            stride=1, 
            padding=1, 
        )
        old_scale = ((self.quantized_M['conv2'] ) * (2 ** (self.quantized_N['conv2']))) * X_scale
        x11 = requantize(x11, old_scale, self.quantized_biases['conv2'],self.quantized_M['conv2_bias'],self.quantized_N['conv2_bias'])  
        
        x = x11 + res
        # print("max:",x.abs().max().item())
        # print("min:",x.abs().min().item())
        x = self.sig1(x) #finish first resblock
       
        ##########################################################################################################
        R, G, B = x[:, 0, :, :], x[:, 1, :, :], x[:, 2, :, :]
        R_Mean, G_Mean, B_Mean = R.mean(dim=[1, 2]), G.mean(dim=[1, 2]), B.mean(dim=[1, 2])
        K = R_Mean + G_Mean + B_Mean
        # print("B mean1",B_Mean)
        # 判斷條件 G_mean / K > 2/3
        condition = G_Mean * 3 > 2 * K
        if condition.any():#clamp
            print("CLAMP")
            x = torch.clamp(x, min=0, max=1)
        else:#retinex_alpha
            R_Mean = R_Mean.unsqueeze(-1).unsqueeze(-1)
            G_Mean = G_Mean.unsqueeze(-1).unsqueeze(-1)
            B_Mean = B_Mean.unsqueeze(-1).unsqueeze(-1)
            
            alpha = 1 - log_approx(torch.clamp(G_Mean - R_Mean, min=1e-6))
            # Process beta for the second condition
            if (G_Mean - B_Mean).mean() >= 0.1:
                beta = -log_approx(torch.clamp(G_Mean - B_Mean, min=1e-6))
                if (R_Mean - B_Mean).mean() <= 0.1:
                    print("123")
                    beta = 1 / (1 + beta)
                    print(beta)
                elif (R_Mean - B_Mean).mean() > 0.1:
                    beta = 1 + beta
                    alpha = 0
                    print("456")

                # Update B channel
                B = B + (G_Mean - B_Mean) * (1 - B) * G * beta

            # Process gamma for the third condition
            if (B_Mean - G_Mean).mean() > 0.1:# not handle yet
                print("789")
                gamma = -log_approx(torch.clamp(B_Mean - R_Mean, min=1e-6))
                G = G + (B_Mean - R_Mean) * (1 - G) * B * gamma

            # Update R channel
            R = R + (G_Mean - R_Mean) * (1 - R) * G * alpha
            
            x = torch.stack([R, G, B], dim=1)
            x = torch.clamp(x, min=0, max=1)
        # print("a:",alpha)
        # print("b:",beta)
        # stdclamp47
        batch_size, channels, height, width = x.shape
        balanced_images = torch.zeros_like(x)
        
        # k = 2.0
        for b in range(batch_size):
            for c in range(channels):
                ch = x[b, c, :, :]
                mean = ch.mean()
                # print("mean",mean)
                std = ch.std()
                # print("std",std*std)
                # 計算上下限
                lower_limit = mean - 2.0 * std
                upper_limit = mean + 2.0 * std
                # print(upper_limit,lower_limit)

                # clamp + normalize
                ch = torch.clamp(ch, min=lower_limit, max=upper_limit)
                ch = (ch - lower_limit) / (upper_limit - lower_limit + 1e-8)
                ch = torch.clamp(ch, 0, 1)
                
                balanced_images[b, c, :, :] = ch
        x = torch.clamp(balanced_images, min=0, max=1)
    
        #L_ch_based
        R, G, B = x[:, 0, :, :], x[:, 1, :, :], x[:, 2, :, :]
        Y = 0.299 * R + 0.587 * G + 0.114 * B
        
        B, H, W = Y.shape
        Y_out = torch.zeros_like(Y)

        for b in range(B):
            y = Y[b]
            mean = y.mean()
            std = y.std()
            # print("y_std",std)
            lower = mean - 2.0 * std
            upper = mean + 2.0 * std
            # print(upper,lower)
            y = torch.clamp(y, lower, upper)
            y = (y - lower) / (upper - lower + 1e-8)
            Y_out[b] = y

        # 增強亮度後的 Y，用來放大 RGB（比例法）
        scale = Y_out / (Y + 1e-6)
        # print("max:",(scale).abs().max().item())
        # print("min:",(scale).abs().min().item())
        # print(B)
        # print(B*scale)
    
        R = torch.clamp(R * scale, 0, 1)
        G = torch.clamp(G * scale, 0, 1)
        B = torch.clamp(B * scale, 0, 1)
        x = torch.stack([R, G, B], dim=1)
        ###############################################################################################
        # print("max:",x.abs().max().item())
        # print("min:",x.abs().min().item())
        # X_q = torch.round(x * (2 ** (7))).to(dtype=data_type)  # 🚀 量化到 INT16
        X_scale = max(x.abs().max().item(), 1e-2) / quant_val
        X_q = torch.round(x / X_scale).to(torch.int64)
        x12 = dyadic_conv2d(
            X_q, 
            self.quantized_weights['conv3'], 
            bias_q=self.quantized_biases['conv3'], 
            stride=1, 
            padding=1, 
        )
        old_scale = ((self.quantized_M['conv3'] ) * (2 ** (self.quantized_N['conv3']))) * X_scale
        x12 = requantize(x12, old_scale, self.quantized_biases['conv3'],self.quantized_M['conv3_bias'],self.quantized_N['conv3_bias'])  
        ca2_res = x12
        # with open("quant.txt", "w") as f_conv1_golden:
        #     for value in X_q.flatten():  # 展平 Tensor 以便逐个写入
        #         f_conv1_golden.write(f"{value}\n")
        # with open("check.txt", "w") as f:
        #         for val in x12.flatten():
        #             hex_val = float32_to_q4_32_hex(val.item())
        #             f.write(hex_val + "\n")

        # # **手動拆解 ca2**
        # print("max:",x12.abs().max().item())
        # print("min:",x12.abs().min().item())
        # X_q = torch.round(x12 * (2 ** (7))).to(dtype=data_type)  # 🚀 量化到 INT16
        X_scale = max(x12.abs().max().item(), 1e-2) / quant_val
        X_q = torch.round(x12 / X_scale).to(torch.int64)
        x13 = dyadic_conv2d(
            X_q, 
            self.quantized_weights['ca2.conv1'], 
            bias_q=self.quantized_biases['ca2.conv1'], 
            stride=1, 
            padding=1, 
        )
        old_scale = ((self.quantized_M['ca2.conv1'] ) * (2 ** (self.quantized_N['ca2.conv1']))) * X_scale
        x13 = requantize(x13, old_scale, self.quantized_biases['ca2.conv1'],self.quantized_M['ca2.conv1_bias'],self.quantized_N['ca2.conv1_bias'])
        x13 = self.relu(x13)
        
        # print("max:",x13.abs().max().item())
        # print("min:",x13.abs().min().item())
        # X_q = torch.round(x13 * (2 ** (7))).to(dtype=data_type)  # 🚀 量化到 INT16
        X_scale = max(x13.abs().max().item(), 1e-2) / quant_val
        X_q = torch.round(x13 / X_scale).to(torch.int64)
        x14 = dyadic_conv2d(
            X_q, 
            self.quantized_weights['ca2.conv2'], 
            bias_q=self.quantized_biases['ca2.conv2'], 
            stride=1, 
            padding=1, 
        )
        old_scale = ((self.quantized_M['ca2.conv2'] ) * (2 ** (self.quantized_N['ca2.conv2']))) * X_scale
        x14 = requantize(x14, old_scale, self.quantized_biases['ca2.conv2'],self.quantized_M['ca2.conv2_bias'],self.quantized_N['ca2.conv2_bias'])
        
        x145 = self.avg_pool(x14)
        # print("max:",x145.abs().max().item())
        # print("min:",x145.abs().min().item())
        X_scale = max(x145.abs().max().item(), 1e-2) / quant_val
        X_q = torch.round(x145 / X_scale).to(torch.int64)
        # X_q = torch.round(x145 * (2 ** (7))).to(dtype=data_type)  # 🚀 量化到 INT16
        x15 = dyadic_conv2d(
            X_q, 
            self.quantized_weights['ca2.attention.ca.conv_du.0'], 
            bias_q=self.quantized_biases['ca2.attention.ca.conv_du.0'], 
            stride=1, 
            padding=0, 
        )
        old_scale = ((self.quantized_M['ca2.attention.ca.conv_du.0'] ) * (2 ** (self.quantized_N['ca2.attention.ca.conv_du.0']))) * X_scale
        x15 = requantize(x15, old_scale, self.quantized_biases['ca2.attention.ca.conv_du.0'],self.quantized_M['ca2.attention.ca.conv_du.0_bias'],self.quantized_N['ca2.attention.ca.conv_du.0_bias'])
        x15 = self.relu(x15)
        
        # print("max:",x15.abs().max().item())
        # print("min:",x15.abs().min().item())
        X_scale = max(x15.abs().max().item(), 1e-2) / quant_val
        X_q = torch.round(x15 / X_scale).to(torch.int64)
        # X_q = torch.round(x15 * (2 ** (7))).to(dtype=data_type)  # 🚀 量化到 INT16
        x16 = dyadic_conv2d(
            X_q, 
            self.quantized_weights['ca2.attention.ca.conv_du.2'], 
            bias_q=self.quantized_biases['ca2.attention.ca.conv_du.2'], 
            stride=1, 
            padding=0, 
        )
        old_scale = ((self.quantized_M['ca2.attention.ca.conv_du.2'] ) * (2 ** (self.quantized_N['ca2.attention.ca.conv_du.2']))) * X_scale
        x16 = requantize(x16, old_scale, self.quantized_biases['ca2.attention.ca.conv_du.2'],self.quantized_M['ca2.attention.ca.conv_du.2_bias'],self.quantized_N['ca2.attention.ca.conv_du.2_bias'])
        x16 = self.sig1(x16)
        x17 = x14 * x16 
        out = x14 * x17
        
        #################################################
        #pa_layer
        # print("max:",out.abs().max().item())
        # print("min:",out.abs().min().item())
        X_scale = max(out.abs().max().item(), 1e-2) / quant_val
        X_q = torch.round(out / X_scale).to(torch.int64)
        # X_q = torch.round(out * (2 ** (7))).to(dtype=data_type)  # 🚀 量化到 INT16
        x18 = dyadic_conv2d(
            X_q, 
            self.quantized_weights['ca2.attention.pa.conv_du.0'], 
            bias_q=self.quantized_biases['ca2.attention.pa.conv_du.0'], 
            stride=1, 
            padding=0, 
        )
        old_scale = ((self.quantized_M['ca2.attention.pa.conv_du.0'] ) * (2 ** (self.quantized_N['ca2.attention.pa.conv_du.0']))) * X_scale
        x18 = requantize(x18, old_scale, self.quantized_biases['ca2.attention.pa.conv_du.0'],self.quantized_M['ca2.attention.pa.conv_du.0_bias'],self.quantized_N['ca2.attention.pa.conv_du.0_bias'])
        x18 = self.relu(x18)
        
        # print("max:",x18.abs().max().item())
        # print("min:",x18.abs().min().item())
        X_scale = max(x18.abs().max().item(), 1e-2) / quant_val
        X_q = torch.round(x18 / X_scale).to(torch.int64)
        # X_q = torch.round(x18 * (2 ** (7))).to(dtype=data_type)  # 🚀 量化到 INT16
        x19 = dyadic_conv2d(
            X_q, 
            self.quantized_weights['ca2.attention.pa.conv_du.2'], 
            bias_q=self.quantized_biases['ca2.attention.pa.conv_du.2'], 
            stride=1, 
            padding=0, 
        )
        
        old_scale = ((self.quantized_M['ca2.attention.pa.conv_du.2'] ) * (2 ** (self.quantized_N['ca2.attention.pa.conv_du.2']))) * X_scale
        x19 = requantize(x19, old_scale, self.quantized_biases['ca2.attention.pa.conv_du.2'],self.quantized_M['ca2.attention.pa.conv_du.2_bias'],self.quantized_N['ca2.attention.pa.conv_du.2_bias'])
        x19 = self.sig1(x19)
        
        x20 = out * x19  
        out = out * x20
        out = out + ca2_res#ca1 finish
        
        # print("max:",out.abs().max().item())
        # print("min:",out.abs().min().item())
        X_scale = max(out.abs().max().item(), 1e-2) / quant_val
        X_q = torch.round(out / X_scale).to(torch.int64)
        # X_q = torch.round(out * (2 ** (7))).to(dtype=data_type)  # 🚀 量化到 INT16
        x21 = dyadic_conv2d(
            X_q, 
            self.quantized_weights['conv4'], 
            bias_q=self.quantized_biases['conv4'], 
            stride=1, 
            padding=1, 
        )
        old_scale = ((self.quantized_M['conv4'] ) * (2 ** (self.quantized_N['conv4']))) * X_scale
        x21 = requantize(x21, old_scale, self.quantized_biases['conv4'],self.quantized_M['conv4_bias'],self.quantized_N['conv4_bias'])
        
        x = x21 + res

        out = self.sig1(x)
        
        return out



# **測試 Dyadic 量化後的模型**
model = load_trained_model("std_L_ch_based.pth")  
quantized_model = QuantizedResIERes(model,example_input_shape=(1,3,256,256))

image_path = "00020.png"
input_image = cv2.imread(image_path)
input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
input_image = cv2.resize(input_image, (256, 256))
input_image = torch.tensor(input_image, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0) / 255.0

output_image = quantized_model(input_image)

plt.imshow(output_image.squeeze(0).permute(1, 2, 0).cpu().numpy())
plt.title("Quantized Output")
plt.axis("off")
plt.show()

# import torchvision.transforms as transforms

# model = load_trained_model("std_L_ch_based.pth")  
# quantized_model = QuantizedResIERes(model,example_input_shape=(1,3,256,256))

# # ✅ 設定資料夾
# input_folder = r"C:\Users\user\Desktop\underwater\final2\UIEB\test_raw"   # 替換成你的輸入資料夾
# output_folder = r"C:\Users\user\Desktop\underwater\final2\current_result\std_L_ch_based"  # 替換成你的輸出資料夾
# os.makedirs(output_folder, exist_ok=True)  # 若輸出資料夾不存在，則建立

# # ✅ 影像轉換
# img_transforms = transforms.Compose([
#     transforms.ToTensor(),
# ])

# # 🚀 **處理資料夾內所有圖片**
# for img_name in os.listdir(input_folder):
#     if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):  # 確保是圖片檔案
#         img_path = os.path.join(input_folder, img_name)
#         print(f"📌 正在處理: {img_name} ...")

#         # 讀取 & 預處理圖片
#         input_image = cv2.imread(img_path)
#         input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
#         input_image = cv2.resize(input_image, (256, 256))  # 依你的模型需求修改
#         input_tensor = torch.tensor(input_image, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0) / 255.0

#         # 🚀 **模型處理**
#         output_tensor = quantized_model(input_tensor)
#         output_image = output_tensor.detach().squeeze(0).permute(1, 2, 0).cpu().numpy()
#         output_image = np.clip(output_image * 255, 0, 255).astype(np.uint8)  # 🚀 轉回 uint8 格式

#         # 🚀 **儲存圖片**
#         output_path = os.path.join(output_folder, img_name)
#         # cv2.imwrite(output_path, cv2.cvtColor(output_image, cv2.COLOR_RGB2BGR))
#         # print(f"✅ 已儲存: {output_path}")

# print("🚀 所有圖片處理完成！")
